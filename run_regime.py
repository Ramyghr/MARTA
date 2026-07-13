"""
run_regime.py
Master script for MARTA Week 3 — Gaussian HMM regime clustering.

Usage:
    conda run -n marta python run_regime.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from regime.hmm import run_hmm_pipeline  # noqa: E402
from regime.validate import validate_regime_semantics  # noqa: E402
from regime.visualize import (  # noqa: E402
    plot_bic_curve,
    plot_regime_timeline,
    plot_regime_tsne,
)
from utils.logger import logger  # noqa: E402

console = Console()
_PLOTS_DIR = _ROOT / "regime" / "plots"


def _load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _print_summary(
    best_n: int,
    bic_results: dict,
    regime_df: pd.DataFrame,
    semantic_summary: dict,
    run_url: str,
) -> None:
    bic_n2 = bic_results.get(2, {}).get("bic", float("inf"))
    bic_best = bic_results.get(best_n, {}).get("bic", float("inf"))
    if bic_n2 not in (float("inf"), float("-inf")) and bic_n2 != 0:
        bic_improvement = (bic_n2 - bic_best) / abs(bic_n2) * 100
    else:
        bic_improvement = 0.0

    dist = regime_df["regime_label"].value_counts(normalize=True).sort_index()

    table = Table(
        title="MARTA Regime Clustering — Week 3 Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="cyan", min_width=36)
    table.add_column("Value", justify="right")

    table.add_row("n_states selected (BIC)", str(best_n))
    table.add_row("BIC improvement over n=2", f"{bic_improvement:+.2f}%")
    table.add_row(
        "Semantic validation",
        "[green]PASSED[/green]"
        if semantic_summary["semantic_validation_passed"]
        else "[red]FAILED[/red]",
    )
    table.add_row(
        "vol std across regimes",
        f"{semantic_summary['vol_std_across_regimes']:.6f}",
    )
    table.add_row("Total regime stream rows", f"{len(regime_df):,}")

    for state, pct in dist.items():
        table.add_row(f"  Regime {state} time share", f"{pct:.1%}")

    table.add_row("MLflow run URL", run_url)

    console.print()
    console.print(table)


def main() -> None:
    config_path = _ROOT / "configs" / "assets.yaml"
    config = _load_config(config_path)

    embeddings_path = _ROOT / "data" / "processed" / "embeddings.npy"
    if not embeddings_path.exists():
        logger.error(
            "Embeddings not found at data/processed/embeddings.npy. "
            "Run `python run_encoder.py` (Week 2) first."
        )
        sys.exit(1)

    mlflow.set_tracking_uri(str(_ROOT / "mlruns"))
    mlflow.set_experiment("marta_regime_hmm")

    with mlflow.start_run(run_name="week3_hmm_pipeline") as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        logger.info(f"MLflow run started: {run_id}")
        mlflow.log_param("config_path", str(config_path))

        # ── Step 1: HMM pipeline ──────────────────────────────────────────
        result = run_hmm_pipeline(config)
        regime_df: pd.DataFrame = result["regime_df"]
        bic_results: dict = result["bic_results"]
        best_n: int = result["best_n"]

        # ── Step 2: semantic validation ───────────────────────────────────
        semantic_summary = validate_regime_semantics(regime_df, config)

        # ── Step 3: plots ─────────────────────────────────────────────────
        _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        bic_plot = _PLOTS_DIR / "bic_curve.png"
        plot_bic_curve(bic_results, best_n, bic_plot)
        mlflow.log_artifact(str(bic_plot))
        logger.success(f"BIC curve → {bic_plot}")

        # t-SNE: use raw 128-dim embeddings for direct Week 2 comparison
        embeddings = np.load(embeddings_path)
        regime_labels = regime_df["regime_label"].values
        tsne_plot = _PLOTS_DIR / "regime_tsne.png"
        logger.info("Running t-SNE (this may take a few minutes) …")
        plot_regime_tsne(embeddings, regime_labels, tsne_plot)
        mlflow.log_artifact(str(tsne_plot))
        logger.success(f"t-SNE → {tsne_plot}")

        # per-asset regime timeline. Aligned parquets are named by histdata
        # instrument (XAUUSD) but the regime stream keys on the display
        # names (GOLD), so filter/label by the aliased stream name.
        processed_dir = _ROOT / config["data"]["processed_dir"]
        alias = {"XAUUSD": "GOLD"}
        for asset in config["data"]["assets"]:
            aligned_path = processed_dir / f"{asset}_aligned.parquet"
            if not aligned_path.exists():
                logger.warning(f"Aligned parquet missing for {asset}, skipping timeline.")
                continue
            price_df = pd.read_parquet(aligned_path, engine="pyarrow")
            stream_asset = alias.get(asset, asset)
            timeline_plot = _PLOTS_DIR / f"regime_timeline_{stream_asset}.png"
            plot_regime_timeline(regime_df, price_df, stream_asset, timeline_plot)
            mlflow.log_artifact(str(timeline_plot))
            logger.success(f"Timeline → {timeline_plot}")

    run_url = (
        f"http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run_id}"
    )

    _print_summary(best_n, bic_results, regime_df, semantic_summary, run_url)

    console.print(f"\n[bold green]MLflow run URL:[/bold green] {run_url}")
    console.print(
        "[dim]Start the UI with:[/dim] [cyan]conda run -n marta mlflow ui[/cyan]"
    )


if __name__ == "__main__":
    main()
