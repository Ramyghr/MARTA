"""
run_per_asset_regime.py
Per-asset Gaussian HMM experiment for MARTA.

Motivation: The shared HMM (trained on all 5 assets together) correctly identifies
stress/crisis regimes in GOLD but misclassifies BTC's March 2020 crash as R1/R4
because the shared embedding space is anchored by GOLD's crisis signature. This
script fits per-asset HMMs independently and checks whether BTC-specific regime
detection correctly identifies the March 12-13 2020 crash.

Experimental finding (run 2026-05-31):
  BTC March 2020 verdict: NOT_DETECTED
  The per-asset BTC HMM (n_states=6) assigns regime R1 to both the pre-crash
  window (March 1–11 2020) and the crash window (March 12–13 2020). KL divergence
  between the two probability distributions = 0.486 — just below the 0.5 threshold.

  Root cause: the `realized_vol` feature in the aligned parquet is a rolling
  720h z-score, not raw realized vol. The March 2020 crash reads as ~6.3 raw but
  still maps to the same z-scored regime that encompasses BTC's historically
  volatile periods. The per-asset HMM does produce significantly more semantically
  distinct regimes overall (vol_std 0.572 vs 0.242 for shared), but the crash window
  specifically is not assigned to a separate regime.

  Consequence: agents/context.py was NOT modified (USE_PER_ASSET_REGIME stays False).
  The debate system continues to use regime_stream.parquet (shared HMM).

Usage:
    conda run -n marta python run_per_asset_regime.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from regime.hmm_per_asset import (  # noqa: E402
    fit_per_asset_hmms,
    generate_per_asset_regime_stream,
    _build_asset_map_from_timestamps,
)
from regime.validate_per_asset import (  # noqa: E402
    validate_per_asset_semantics,
    check_btc_march_2020,
)
from regime.compare import compare_regime_streams, plot_regime_comparison  # noqa: E402
from utils.logger import logger  # noqa: E402

console = Console()
_PLOTS_DIR = _ROOT / "regime" / "plots"
_PROCESSED_DIR = _ROOT / "data" / "processed"


def _load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _load_data() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    embeddings_path = _PROCESSED_DIR / "embeddings.npy"
    if not embeddings_path.exists():
        logger.error(
            "Embeddings not found at data/processed/embeddings.npy. "
            "Run `python run_encoder.py` first."
        )
        sys.exit(1)

    embeddings = np.load(embeddings_path)
    timestamp_df = pd.read_parquet(
        _PROCESSED_DIR / "embedding_timestamps.parquet", engine="pyarrow"
    )

    asset_map_path = _PROCESSED_DIR / "embedding_asset_map.parquet"
    if asset_map_path.exists():
        asset_map_df = pd.read_parquet(asset_map_path, engine="pyarrow")
    else:
        logger.info(
            "embedding_asset_map.parquet not found — building asset map from timestamps"
        )
        asset_map_df = _build_asset_map_from_timestamps(timestamp_df)

    logger.info(
        f"Embeddings: {embeddings.shape}  "
        f"Timestamps: {timestamp_df.shape}  "
        f"Assets: {asset_map_df['asset_name'].tolist()}"
    )
    return embeddings, timestamp_df, asset_map_df


def _patch_context_for_per_asset_regime() -> None:
    """
    Modify agents/context.py to add USE_PER_ASSET_REGIME flag and conditional
    per-asset stream loading. Idempotent — safe to call multiple times.
    """
    context_path = _ROOT / "agents" / "context.py"
    content = context_path.read_text()

    if "USE_PER_ASSET_REGIME" in content:
        logger.info("agents/context.py already has USE_PER_ASSET_REGIME — skipping patch")
        return

    # Add Path import and _ROOT after the existing imports
    old_imports = "from dataclasses import dataclass\n\nimport pandas as pd"
    new_imports = (
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n\n"
        "import pandas as pd\n\n"
        "_ROOT = Path(__file__).resolve().parent.parent"
    )
    if old_imports not in content:
        logger.error("Cannot patch context.py: expected import block not found")
        return
    content = content.replace(old_imports, new_imports, 1)

    # Insert USE_PER_ASSET_REGIME, BTC descriptions, cache, and resolver
    # after _WINDOW_INDICATORS line
    old_anchor = '_WINDOW_INDICATORS = ["realized_vol", "rsi", "adx", "close_return"]'
    new_anchor = (
        '_WINDOW_INDICATORS = ["realized_vol", "rsi", "adx", "close_return"]\n\n'
        "USE_PER_ASSET_REGIME: bool = True\n\n"
        "# TODO: Fill in BTC-specific regime descriptions after running run_per_asset_regime.py.\n"
        "# Check the per-asset semantic validation table output for regime character assignments.\n"
        "# Example structure: {0: 'BTC calm/transitional', 1: 'BTC calm bull', 2: 'BTC crash/crisis', ...}\n"
        "_BTC_REGIME_DESCRIPTIONS: dict[int, str] = {}\n\n"
        "_PER_ASSET_STREAM_PATH = _ROOT / 'data' / 'processed' / 'regime_stream_per_asset.parquet'\n"
        "_per_asset_stream_cache: dict[str, pd.DataFrame] = {}\n\n\n"
        "def _resolve_regime_stream(regime_stream_df: pd.DataFrame) -> pd.DataFrame:\n"
        '    """Return per-asset stream when USE_PER_ASSET_REGIME is True and file exists."""\n'
        "    if not USE_PER_ASSET_REGIME:\n"
        "        return regime_stream_df\n"
        '    if "data" not in _per_asset_stream_cache:\n'
        "        if _PER_ASSET_STREAM_PATH.exists():\n"
        '            _per_asset_stream_cache["data"] = pd.read_parquet(\n'
        '                _PER_ASSET_STREAM_PATH, engine="pyarrow"\n'
        "            )\n"
        "        else:\n"
        "            return regime_stream_df\n"
        '    return _per_asset_stream_cache["data"]\n'
    )
    if old_anchor not in content:
        logger.error("Cannot patch context.py: _WINDOW_INDICATORS anchor not found")
        return
    content = content.replace(old_anchor, new_anchor, 1)

    # Patch build_regime_context to use _resolve_regime_stream
    old_mask = (
        '    mask = (regime_stream_df["asset"] == asset) '
        '& (regime_stream_df["timestamp"] == ts_utc)'
    )
    new_mask = (
        "    active_stream = _resolve_regime_stream(regime_stream_df)\n"
        '    mask = (active_stream["asset"] == asset) '
        '& (active_stream["timestamp"] == ts_utc)'
    )
    content = content.replace(old_mask, new_mask, 1)

    old_regime_rows = "    regime_rows = regime_stream_df[mask]"
    new_regime_rows = "    regime_rows = active_stream[mask]"
    content = content.replace(old_regime_rows, new_regime_rows, 1)

    old_asset_df = (
        "        asset_df = regime_stream_df[regime_stream_df"
        '["asset"] == asset].copy()'
    )
    new_asset_df = (
        "        asset_df = active_stream[active_stream"
        '["asset"] == asset].copy()'
    )
    content = content.replace(old_asset_df, new_asset_df, 1)

    old_prob_cols = (
        "    prob_cols = [c for c in regime_stream_df.columns "
        'if c.startswith("regime_prob_")]'
    )
    new_prob_cols = (
        "    prob_cols = [c for c in active_stream.columns "
        'if c.startswith("regime_prob_")]'
    )
    content = content.replace(old_prob_cols, new_prob_cols, 1)

    # Update regime description lookup to prefer BTC-specific descriptions
    old_desc = (
        '        regime_description=_REGIME_DESCRIPTIONS.get(regime_label, "Unknown regime"),'
    )
    new_desc = (
        "        regime_description=(\n"
        "            _BTC_REGIME_DESCRIPTIONS.get(regime_label)\n"
        "            if USE_PER_ASSET_REGIME\n"
        '            and asset == "BTC_USDT"\n'
        "            and regime_label in _BTC_REGIME_DESCRIPTIONS\n"
        "            else _REGIME_DESCRIPTIONS.get(regime_label, "
        '"Unknown regime")\n'
        "        ),"
    )
    content = content.replace(old_desc, new_desc, 1)

    context_path.write_text(content)
    logger.success("Patched agents/context.py with USE_PER_ASSET_REGIME support")


def _print_summary(
    per_asset_models: dict,
    semantic_summaries: dict[str, Any],
    crash_result: dict[str, Any],
    comparison: dict[str, Any],
    run_url: str,
) -> None:
    table = Table(
        title="MARTA Per-Asset HMM Experiment — Summary",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Asset", style="cyan", min_width=12)
    table.add_column("n_states", justify="center")
    table.add_column("Semantic", justify="center")
    table.add_column("Vol Std", justify="right")
    table.add_column("Regimes (rvol μ)", min_width=36)

    for asset, (hmm, _pca) in per_asset_models.items():
        summary = semantic_summaries.get(asset, {})
        passed = summary.get("semantic_validation_passed", False)
        vol_std = summary.get("vol_std_across_regimes", 0.0)
        regimes_dict = summary.get("regimes", {})
        regime_str = "  ".join(
            f"R{k}={v['realized_vol_mean']:.3f}"
            for k, v in sorted(regimes_dict.items(), key=lambda x: int(x[0]))
        )
        table.add_row(
            asset,
            str(hmm.n_components),
            "[green]PASS[/green]" if passed else "[red]FAIL[/red]",
            f"{vol_std:.5f}",
            regime_str if regime_str else "—",
        )

    console.print()
    console.print(table)

    # BTC March 2020 verdict
    verdict = crash_result.get("verdict", "NOT_DETECTED")
    verdict_color = "green" if verdict == "CRASH_DETECTED" else "red"
    console.print()
    console.print(
        f"  BTC March 2020 crash: [{verdict_color}]{verdict}[/{verdict_color}]  "
        f"(pre_regime=R{crash_result.get('dominant_pre_crash_regime', '?')} → "
        f"crash_regime=R{crash_result.get('dominant_crash_regime', '?')}  "
        f"KL={crash_result.get('kl_divergence', 0.0):.3f}  "
        f"rvol: {crash_result.get('pre_crash_realized_vol', 0.0):.3f} → "
        f"{crash_result.get('crash_realized_vol', 0.0):.3f})"
    )

    # Shared vs per-asset comparison
    if comparison:
        agree = comparison.get("agreement_rate", 0.0)
        sh_std = comparison.get("shared_vol_std_across_regimes", 0.0)
        pa_std = comparison.get("per_asset_vol_std_across_regimes", 0.0)
        more_distinct = comparison.get("per_asset_more_distinct", False)
        sh_n = comparison.get("shared_n_states", "?")
        pa_n = comparison.get("per_asset_n_states", "?")
        btc_sh_cat = comparison.get("btc_march2020_shared_vol_cat", "?")
        btc_pa_cat = comparison.get("btc_march2020_per_asset_vol_cat", "?")

        console.print(
            f"\n  BTC Shared HMM:    n_states={sh_n}  vol_std={sh_std:.4f}  "
            f"March2020_vol_cat={btc_sh_cat}"
        )
        console.print(
            f"  BTC Per-Asset HMM: n_states={pa_n}  vol_std={pa_std:.4f}  "
            f"March2020_vol_cat={btc_pa_cat}"
        )
        console.print(
            f"  Vol-category agreement rate: {agree:.1%}  "
            f"Per-asset more distinct: {'YES' if more_distinct else 'NO'}"
        )

    # Final verdict — based specifically on crash detection, not general distinctness
    improves = verdict == "CRASH_DETECTED"
    improve_str = "[green]YES[/green]" if improves else "[red]NO[/red]"
    console.print(
        f"\n  [bold]Per-asset HMM improves BTC crash detection:[/bold] {improve_str}"
    )
    console.print(f"\n  MLflow: {run_url}")


def main() -> None:
    config_path = _ROOT / "configs" / "assets.yaml"
    config = _load_config(config_path)

    mlflow.set_tracking_uri(str(_ROOT / "mlruns"))
    mlflow.set_experiment("marta_regime_per_asset")

    embeddings, timestamp_df, asset_map_df = _load_data()

    with mlflow.start_run(run_name="per_asset_hmm_pipeline") as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        logger.info(f"MLflow run: {run_id}")
        mlflow.log_param("config_path", str(config_path))
        mlflow.log_param("n_assets", len(asset_map_df))

        # ── Step 1: Fit per-asset HMMs ────────────────────────────────────
        logger.info("\n[Step 1] Fitting per-asset HMMs …")
        per_asset_models = fit_per_asset_hmms(
            embeddings, timestamp_df, asset_map_df, config
        )

        # ── Step 2: Generate per-asset regime stream ──────────────────────
        logger.info("\n[Step 2] Generating per-asset regime stream …")
        per_asset_stream = generate_per_asset_regime_stream(
            embeddings, timestamp_df, asset_map_df, per_asset_models
        )

        # ── Step 3: Semantic validation ───────────────────────────────────
        logger.info("\n[Step 3] Semantic validation …")
        semantic_summaries = validate_per_asset_semantics(per_asset_stream, config)

        # ── Step 4: BTC March 2020 crash check ───────────────────────────
        logger.info("\n[Step 4] BTC March 2020 crash check …")
        btc_aligned_path = _PROCESSED_DIR / "BTC_USDT_aligned.parquet"
        aligned_btc_df = pd.read_parquet(btc_aligned_path, engine="pyarrow")
        crash_result = check_btc_march_2020(per_asset_stream, aligned_btc_df)

        # ── Step 5: Compare shared vs per-asset for BTC ───────────────────
        logger.info("\n[Step 5] Comparing shared vs per-asset HMM for BTC …")
        shared_stream = pd.read_parquet(
            _PROCESSED_DIR / "regime_stream.parquet", engine="pyarrow"
        )
        comparison = compare_regime_streams(shared_stream, per_asset_stream, "BTC_USDT")

        # ── Step 6: Comparison plot for BTC ──────────────────────────────
        logger.info("\n[Step 6] Generating BTC comparison plot …")
        _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = _PLOTS_DIR / "regime_comparison_BTC.png"
        plot_regime_comparison(
            shared_stream,
            per_asset_stream,
            "BTC_USDT",
            aligned_btc_df,
            plot_path,
        )
        mlflow.log_artifact(str(plot_path))

        # ── Step 7: Conditional context.py patch ─────────────────────────
        verdict = crash_result.get("verdict", "NOT_DETECTED")
        if verdict == "CRASH_DETECTED":
            logger.info("\n[Step 7] Crash detected — patching agents/context.py …")
            _patch_context_for_per_asset_regime()
            mlflow.log_param("context_patched", True)
        else:
            logger.info(
                "\n[Step 7] Crash NOT detected — agents/context.py left unchanged."
            )
            mlflow.log_param("context_patched", False)

        mlflow.log_param("btc_march2020_verdict", verdict)

    run_url = (
        f"http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run_id}"
    )

    _print_summary(
        per_asset_models, semantic_summaries, crash_result, comparison, run_url
    )

    console.print(
        "\n[dim]Start MLflow UI with:[/dim] "
        "[cyan]conda run -n marta mlflow ui[/cyan]"
    )
    console.print(
        "[dim]Comparison plot:[/dim] "
        f"[cyan]{plot_path}[/cyan]"
    )


if __name__ == "__main__":
    main()
