"""
regime/validate.py
Semantic validation of HMM-discovered regimes against financial features.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

_ROOT = Path(__file__).resolve().parent.parent
_console = Console()

from utils.logger import logger  # noqa: E402


def validate_regime_semantics(
    regime_df: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """
    Compute per-regime financial statistics and assert semantic coherence.

    For each regime state computes: realized_vol, close_return, rsi, adx
    statistics plus annualised return, vol, Sharpe, time share, and dominant asset.
    Logs all metrics to MLflow and saves a JSON summary.
    """
    processed_dir = _ROOT / config["data"]["processed_dir"]
    assets = regime_df["asset"].unique().tolist()
    n_states = int(regime_df["regime_label"].nunique())

    # merge aligned features with regime labels
    frames: list[pd.DataFrame] = []
    for asset in assets:
        from utils.assets import aligned_path
        asset_path = aligned_path(asset, processed_dir)
        if not asset_path.exists():
            logger.warning(f"Aligned parquet not found for {asset}, skipping.")
            continue

        adf = pd.read_parquet(asset_path, engine="pyarrow").reset_index()
        adf["asset"] = asset
        adf["timestamp"] = pd.to_datetime(adf["timestamp"], utc=True)

        asset_regime = regime_df.loc[
            regime_df["asset"] == asset, ["timestamp", "regime_label"]
        ].copy()
        asset_regime["timestamp"] = pd.to_datetime(
            asset_regime["timestamp"], utc=True
        )

        merged = adf.merge(asset_regime, on="timestamp", how="inner")
        frames.append(merged)

    if not frames:
        raise RuntimeError("No aligned data found — cannot validate regime semantics.")

    combined = pd.concat(frames, ignore_index=True)
    n_total = len(combined)
    logger.info(f"Feature+regime merge: {n_total:,} rows across {len(assets)} assets")

    regime_stats: dict[str, Any] = {}

    for state in sorted(combined["regime_label"].unique()):
        mask = combined["regime_label"] == state
        sub = combined[mask]

        ann_ret = float(sub["close_return"].mean() * 24 * 365)
        ann_vol = float(sub["close_return"].std() * np.sqrt(24 * 365))
        sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0
        dominant_asset = str(sub["asset"].value_counts().idxmax())

        stats: dict[str, Any] = {
            "realized_vol_mean": float(sub["realized_vol"].mean()),
            "realized_vol_std": float(sub["realized_vol"].std()),
            "close_return_mean": float(sub["close_return"].mean()),
            "close_return_std": float(sub["close_return"].std()),
            "rsi_mean": float(sub["rsi"].mean()),
            "rsi_std": float(sub["rsi"].std()),
            "adx_mean": float(sub["adx"].mean()),
            "adx_std": float(sub["adx"].std()),
            "annualized_return": ann_ret,
            "annualized_vol": ann_vol,
            "sharpe": sharpe,
            "time_pct": float(len(sub) / n_total),
            "dominant_asset": dominant_asset,
            "n_samples": int(len(sub)),
        }
        regime_stats[str(state)] = stats

        for metric, value in stats.items():
            if metric != "dominant_asset":
                mlflow.log_metric(f"regime_{state}_{metric}", float(value))

        logger.info(
            f"Regime {state}: rvol={stats['realized_vol_mean']:.4f}  "
            f"sharpe={sharpe:.2f}  time={stats['time_pct']:.1%}  "
            f"dom={dominant_asset}"
        )

    # rich summary table
    table = Table(
        title=f"Regime Semantic Validation — {n_states} states",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Regime", style="cyan", min_width=8)
    table.add_column("RVol μ", justify="right")
    table.add_column("Ret (ann.)", justify="right")
    table.add_column("Vol (ann.)", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("RSI μ", justify="right")
    table.add_column("ADX μ", justify="right")
    table.add_column("Time %", justify="right")
    table.add_column("Dom. Asset", justify="right")

    for sk, s in regime_stats.items():
        table.add_row(
            f"R{sk}",
            f"{s['realized_vol_mean']:.4f}",
            f"{s['annualized_return']:.2%}",
            f"{s['annualized_vol']:.4f}",
            f"{s['sharpe']:.2f}",
            f"{s['rsi_mean']:.1f}",
            f"{s['adx_mean']:.1f}",
            f"{s['time_pct']:.1%}",
            s["dominant_asset"],
        )

    _console.print()
    _console.print(table)

    # semantic distinctness check
    vol_means = [regime_stats[k]["realized_vol_mean"] for k in regime_stats]
    vol_std_across = float(np.std(vol_means))
    semantic_passed = vol_std_across > 0.001

    mlflow.log_metric("regime_vol_std_across", vol_std_across)
    mlflow.log_metric("semantic_validation_passed", int(semantic_passed))

    if not semantic_passed:
        logger.critical(
            f"Semantic validation FAILED: std(realized_vol μ) = {vol_std_across:.6f} ≤ 0.001. "
            "HMM found arbitrary clusters — regimes lack distinct volatility profiles."
        )
    else:
        logger.success(
            f"Semantic validation PASSED: std(realized_vol μ) = {vol_std_across:.6f} > 0.001"
        )

    # save JSON
    validated_dir = _ROOT / "data" / "validated"
    validated_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "n_states": n_states,
        "semantic_validation_passed": semantic_passed,
        "vol_std_across_regimes": vol_std_across,
        "regimes": regime_stats,
    }
    json_path = validated_dir / "regime_semantics.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.success(f"Saved regime semantics → {json_path}")
    mlflow.log_artifact(str(json_path))

    return summary
