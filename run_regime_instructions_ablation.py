"""A/B test: does an active regime trading-instructions block change debate quality,
versus the current passive regime description, versus no regime context at all?

Three prompt variants, same (timestamp, asset) pairs, via
agents.context.serialize_context_for_prompt:
  - "none"             include_regime=False                          (existing ablation baseline)
  - "description_only" include_regime=True,  include_instructions=False (current production default)
  - "instructions"     include_regime=True,  include_instructions=True  (new — see agents/context.py
                        _REGIME_TRADING_INSTRUCTIONS)

Timestamps are stratified by classify_regime_volatility_level() bucket (normal/elevated/
extreme_vol) so the extreme-vol hypothesis (h1, h2 below) actually has enough extreme-vol
rows to be testable — a uniform random sample would be ~93% normal_vol per the crypto5
panel's actual distribution and starve the buckets that matter most for this test.
Restricted to timestamps < 2025-01-01 (train+val only) to keep the 2025 test split
untouched by an exploratory prompt-engineering pass, consistent with this project's
test-set discipline elsewhere (configs/splits.py).

Does NOT modify data/debates/debate_results.json or debate_results_no_regime.json.
Output: data/debates/regime_instructions_ablation.json (raw per-debate records) +
data/results/regime_instructions_ablation_summary.json (aggregated metrics) +
data/results/regime_instructions_ablation_hypotheses.json (pre-registration, written
before any debate runs).

Usage:
  python run_regime_instructions_ablation.py                      # full run, 10/bucket = 90 debates
  python run_regime_instructions_ablation.py --n-per-bucket 1      # smoke test, 9 debates
  python run_regime_instructions_ablation.py --n-per-bucket 5      # 45 debates
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

import agents.base as base_module
import agents.context as context_module
import agents.contrarian_agent as contrarian_module
import agents.fund_manager as fm_module
import agents.macro_agent as macro_module
import agents.risk_agent as risk_module
import agents.sentiment_agent as sentiment_module
import agents.technical_agent as technical_module
from agents.context import _load_regime_stats, classify_regime_volatility_level
from agents.contrarian_agent import ContrarianAgent
from agents.fund_manager import FundManagerAgent
from agents.graph import DebateGraph, run_debate as _run_debate_raw
from agents.macro_agent import MacroAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from functools import partial
from utils.freshness import CRYPTO5_DEBATE_ASSETS, CRYPTO5_PROCESSED, CRYPTO5_SEMANTICS_PATH, load_debate_data
from utils.logger import logger

_DATA_DIR = CRYPTO5_PROCESSED
_DEBATE_ASSETS = CRYPTO5_DEBATE_ASSETS
_SEMANTICS_PATH = CRYPTO5_SEMANTICS_PATH
_OUTPUT_RAW = Path("data/debates/regime_instructions_ablation.json")
_OUTPUT_SUMMARY = Path("data/results/regime_instructions_ablation_summary.json")
_OUTPUT_HYPOTHESES = Path("data/results/regime_instructions_ablation_hypotheses.json")
_MLFLOW_EXPERIMENT = "marta_agent_deliberation"
_FORWARD_HOURS = 24
_MIN_GAP_DAYS = 14  # minimum spacing between sampled timestamps within the same (asset, bucket)
_TEST_SPLIT_START = pd.Timestamp("2025-01-01", tz="UTC")

run_debate = partial(_run_debate_raw, semantics_path=_SEMANTICS_PATH)

_PATCH_TARGETS = [
    context_module,
    base_module,
    macro_module,
    technical_module,
    sentiment_module,
    contrarian_module,
    risk_module,
    fm_module,
]

_VARIANTS = ["none", "description_only", "instructions"]


@contextmanager
def _variant_patch(variant: str):
    """Force serialize_context_for_prompt to a fixed (include_regime, include_instructions)
    combo across every module that imported it directly (each holds its own module-level
    binding — patching agents.context alone would miss the rest, same issue
    run_clean_ablation_debates.py's _patch_no_regime already documents)."""
    original = context_module.serialize_context_for_prompt
    include_regime = variant != "none"
    include_instructions = variant == "instructions"

    def _forced(ctx, *args, **kwargs):
        return original(ctx, include_regime=include_regime, include_instructions=include_instructions)

    for mod in _PATCH_TARGETS:
        mod.serialize_context_for_prompt = _forced
    try:
        yield
    finally:
        for mod in _PATCH_TARGETS:
            mod.serialize_context_for_prompt = original


def _stratified_sample(
    regime_stream: pd.DataFrame,
    aligned_dfs: dict[str, pd.DataFrame],
    n_per_bucket: int,
    seed: int = 42,
) -> list[tuple[pd.Timestamp, str, str]]:
    """Returns (timestamp, asset, vol_level) tuples, n_per_bucket per vol_level bucket,
    round-robined across assets, spaced >= _MIN_GAP_DAYS apart within the same
    (asset, vol_level), restricted to rows where a full _FORWARD_HOURS-ahead window exists
    in that asset's aligned df (needed for the forward-return ground truth) and to the
    pre-2025 train+val split."""
    stats_by_label = _load_regime_stats(_SEMANTICS_PATH)
    label_to_vol = {label: classify_regime_volatility_level(s) for label, s in stats_by_label.items()}

    df = regime_stream.copy()
    df["vol_level"] = df["regime_label"].map(label_to_vol)
    df = df[df["timestamp"] < _TEST_SPLIT_START]
    df = df[df["timestamp"].dt.hour == 0]  # daily stride, matches existing debate cache cadence

    rng = np.random.default_rng(seed)
    picked: list[tuple[pd.Timestamp, str, str]] = []

    for bucket in ["normal_vol", "elevated_vol", "extreme_vol"]:
        bucket_df = df[df["vol_level"] == bucket]
        last_ts_per_asset: dict[str, pd.Timestamp] = {}
        candidates = bucket_df.sample(frac=1.0, random_state=rng.integers(0, 1 << 31)).itertuples()
        count = 0
        for row in candidates:
            if count >= n_per_bucket:
                break
            asset, ts = row.asset, row.timestamp
            aligned_df = aligned_dfs.get(asset)
            if aligned_df is None:
                continue
            if ts + pd.Timedelta(hours=_FORWARD_HOURS) > aligned_df.index.max():
                continue
            prior = last_ts_per_asset.get(asset)
            if prior is not None and abs((ts - prior).days) < _MIN_GAP_DAYS:
                continue
            last_ts_per_asset[asset] = ts
            picked.append((ts, asset, bucket))
            count += 1
        if count < n_per_bucket:
            logger.warning(f"Bucket {bucket}: only found {count}/{n_per_bucket} valid candidates")

    return picked


def _forward_return_sign(aligned_dfs: dict[str, pd.DataFrame], ts: pd.Timestamp, asset: str) -> int | None:
    """Sign of the summed close_return over (ts, ts + _FORWARD_HOURS] — the ground truth
    a debate's directional call is judged against. None if the window isn't available."""
    aligned_df = aligned_dfs.get(asset)
    if aligned_df is None:
        return None
    window = aligned_df[(aligned_df.index > ts) & (aligned_df.index <= ts + pd.Timedelta(hours=_FORWARD_HOURS))]
    if window.empty or "close_return" not in window.columns:
        return None
    total = float(window["close_return"].sum())
    return 1 if total > 0 else (-1 if total < 0 else 0)


def _write_hypotheses() -> None:
    _OUTPUT_HYPOTHESES.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_HYPOTHESES.write_text(json.dumps({
        "h1": "instructions variant has higher directional accuracy than description_only, "
              "specifically within the extreme_vol bucket",
        "h2": "instructions variant produces smaller mean position_size than description_only "
              "in the elevated_vol and extreme_vol buckets",
        "h3": "none variant is not meaningfully worse than description_only on directional "
              "accuracy (regime context, passively described, is not harmful even if not "
              "clearly helpful — consistent with docs/RESULTS_AND_METHODOLOGY.md Section 10's "
              "finding that real regime conditioning hasn't distinguished itself from a "
              "placebo/noise control in this project so far)",
        "primary_metric": "directional_accuracy_by_variant_and_vol_level",
        "note": "Sample sizes here (~10 debates/bucket/variant) are far below what this "
                "project's own block-bootstrap CIs elsewhere require for a significance claim "
                "(see docs/RESULTS_AND_METHODOLOGY.md Section 9/10) -- this run is a directional "
                "check on whether to invest further, not a defensible standalone result.",
    }, indent=2))
    logger.info(f"Pre-registered hypotheses written to {_OUTPUT_HYPOTHESES}")


def _run_all(samples: list[tuple[pd.Timestamp, str, str]], regime_stream, aligned_dfs, debate_graph) -> list[dict]:
    console = Console()
    results: list[dict] = []
    total = len(samples) * len(_VARIANTS)
    i = 0
    for variant in _VARIANTS:
        with _variant_patch(variant):
            for ts, asset, vol_level in samples:
                i += 1
                console.print(f"[bold]{i}/{total}[/bold] variant={variant} vol_level={vol_level} {asset} @ {ts.date()}")
                try:
                    result = run_debate(ts, asset, regime_stream, aligned_dfs, debate_graph)
                except Exception as exc:
                    logger.warning(f"Debate failed ({variant}, {asset}, {ts}): {exc} -- skipping")
                    continue
                fwd = _forward_return_sign(aligned_dfs, ts, asset)
                fm = result.fund_manager_decision
                results.append({
                    "variant": variant,
                    "vol_level": vol_level,
                    "timestamp": ts.isoformat(),
                    "asset": asset,
                    "final_action": fm.final_action,
                    "position_size": fm.position_size,
                    "conviction": fm.conviction,
                    "consensus_direction": fm.consensus_direction,
                    "dissenting_agents": fm.dissenting_agents,
                    "forward_return_sign_24h": fwd,
                    "round1_biases": [t.directional_bias for t in result.round1_theses],
                    "round2_flips": sum(1 for r in result.round2_rebuttals if not r.maintains_bias),
                    "round2_count": len(result.round2_rebuttals),
                })
    return results


def _agent_agreement(biases: list[str]) -> float:
    """Fraction of directional (non-neutral) R1 theses that agree with the majority
    directional bias -- a simple pairwise-agreement stand-in, not Krippendorff's alpha."""
    directional = [b for b in biases if b != "neutral"]
    if len(directional) < 2:
        return 1.0
    bulls = sum(1 for b in directional if b == "bullish")
    bears = len(directional) - bulls
    return max(bulls, bears) / len(directional)


def _summarize(results: list[dict]) -> dict:
    df = pd.DataFrame(results)
    summary: dict = {"n_debates": len(df), "by_variant": {}}

    for variant in _VARIANTS:
        sub = df[df["variant"] == variant]
        if sub.empty:
            continue
        directional = sub[sub["final_action"] != "hold"].copy()
        directional["correct"] = directional.apply(
            lambda r: (r["forward_return_sign_24h"] == 1 and r["final_action"] == "long")
            or (r["forward_return_sign_24h"] == -1 and r["final_action"] == "short"),
            axis=1,
        )
        directional = directional[directional["forward_return_sign_24h"].notna()]

        variant_summary = {
            "n": len(sub),
            "directional_accuracy_overall": float(directional["correct"].mean()) if len(directional) else None,
            "n_directional_calls": len(directional),
            "mean_position_size": float(sub["position_size"].mean()),
            "mean_conviction": float(sub["conviction"].mean()),
            "round2_flip_rate": float(sub["round2_flips"].sum() / max(1, sub["round2_count"].sum())),
            "mean_agent_agreement": float(sub["round1_biases"].apply(_agent_agreement).mean()),
            "by_vol_level": {},
        }
        for bucket in ["normal_vol", "elevated_vol", "extreme_vol"]:
            bsub = sub[sub["vol_level"] == bucket]
            bdir = directional[directional["vol_level"] == bucket] if len(directional) else directional
            variant_summary["by_vol_level"][bucket] = {
                "n": len(bsub),
                "directional_accuracy": float(bdir["correct"].mean()) if len(bdir) else None,
                "mean_position_size": float(bsub["position_size"].mean()) if len(bsub) else None,
            }
        summary["by_variant"][variant] = variant_summary

    return summary


def _print_summary(summary: dict) -> None:
    console = Console()
    table = Table(title="Regime Instructions Ablation — Summary")
    table.add_column("Variant")
    table.add_column("n")
    table.add_column("Dir. Acc. (overall)")
    table.add_column("Mean Position Size")
    table.add_column("Mean Conviction")
    table.add_column("R2 Flip Rate")
    table.add_column("Agent Agreement")
    for variant, s in summary["by_variant"].items():
        acc = f"{s['directional_accuracy_overall']:.3f}" if s["directional_accuracy_overall"] is not None else "n/a"
        table.add_row(
            variant, str(s["n"]), acc,
            f"{s['mean_position_size']:.4f}", f"{s['mean_conviction']:.3f}",
            f"{s['round2_flip_rate']:.3f}", f"{s['mean_agent_agreement']:.3f}",
        )
    console.print(table)

    ev_table = Table(title="Mean Position Size by Volatility Bucket")
    ev_table.add_column("Variant")
    ev_table.add_column("normal_vol")
    ev_table.add_column("elevated_vol")
    ev_table.add_column("extreme_vol")
    for variant, s in summary["by_variant"].items():
        row = [variant]
        for bucket in ["normal_vol", "elevated_vol", "extreme_vol"]:
            v = s["by_vol_level"][bucket]["mean_position_size"]
            row.append(f"{v:.4f}" if v is not None else "n/a")
        ev_table.add_row(*row)
    console.print(ev_table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-bucket", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    console = Console()
    console.rule("[bold magenta]MARTA Regime Trading-Instructions Ablation[/bold magenta]")

    _write_hypotheses()

    logger.info("Loading data...")
    regime_stream, aligned_dfs = load_debate_data(processed_dir=_DATA_DIR, assets=_DEBATE_ASSETS)

    samples = _stratified_sample(regime_stream, aligned_dfs, args.n_per_bucket, args.seed)
    logger.info(f"Sampled {len(samples)} (timestamp, asset, vol_level) pairs "
                f"-> {len(samples) * len(_VARIANTS)} total debates across {len(_VARIANTS)} variants")

    logger.info("Initializing agents (crypto-only panel)...")
    macro_agent = MacroAgent(assets=_DEBATE_ASSETS, data_dir=_DATA_DIR)
    tech_agent = TechnicalAgent()
    risk_agent = RiskAgent(data_dir=_DATA_DIR)
    fm_agent = FundManagerAgent(semantics_path=_SEMANTICS_PATH)
    sentiment_agent = SentimentAgent(macro_assets={"macro"}, fear_greed_assets=set(_DEBATE_ASSETS))
    contrarian_agent = ContrarianAgent(regime_stream_df=regime_stream)
    debate_graph = DebateGraph(
        macro_agent, tech_agent, risk_agent, fm_agent, sentiment_agent, contrarian_agent
    )

    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="regime_instructions_ablation"):
        mlflow.log_param("n_per_bucket", args.n_per_bucket)
        mlflow.log_param("n_samples", len(samples))
        mlflow.log_param("variants", _VARIANTS)

        results = _run_all(samples, regime_stream, aligned_dfs, debate_graph)

        _OUTPUT_RAW.parent.mkdir(parents=True, exist_ok=True)
        _OUTPUT_RAW.write_text(json.dumps(results, indent=2, default=str))
        mlflow.log_artifact(str(_OUTPUT_RAW))

        summary = _summarize(results)
        _OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        _OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
        mlflow.log_artifact(str(_OUTPUT_SUMMARY))
        for variant, s in summary["by_variant"].items():
            if s["directional_accuracy_overall"] is not None:
                mlflow.log_metric(f"{variant}_directional_accuracy", s["directional_accuracy_overall"])
            mlflow.log_metric(f"{variant}_mean_position_size", s["mean_position_size"])

    _print_summary(summary)
    console.print(f"[bold green]Done:[/bold green] {len(results)}/{len(samples) * len(_VARIANTS)} debates "
                   f"-> {_OUTPUT_RAW}, {_OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
