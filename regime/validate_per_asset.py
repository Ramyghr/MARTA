"""
regime/validate_per_asset.py
Per-asset semantic validation of per-asset HMM regimes.

Mirrors the logic of regime/validate.py but evaluates each asset independently
rather than pooling all assets together. Also performs the critical BTC March 2020
crash detection check.
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


def validate_per_asset_semantics(
    per_asset_stream_df: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """
    Run semantic validation independently for each asset.

    For each asset computes per-regime statistics (realized_vol, close_return,
    annualized return/vol, Sharpe) and asserts std(realized_vol μ across regimes) > 0.001.

    Returns:
        dict mapping asset_name → validation summary with keys:
            n_states, semantic_validation_passed, vol_std_across_regimes, regimes
    """
    processed_dir = _ROOT / config["data"]["processed_dir"]
    assets = per_asset_stream_df["asset"].unique().tolist()
    all_summaries: dict[str, Any] = {}

    for asset in assets:
        asset_stream = per_asset_stream_df[
            per_asset_stream_df["asset"] == asset
        ].copy()
        asset_stream["timestamp"] = pd.to_datetime(asset_stream["timestamp"], utc=True)
        n_states = int(asset_stream["regime_label"].nunique())

        aligned_path = processed_dir / f"{asset}_aligned.parquet"
        if not aligned_path.exists():
            logger.warning(f"[{asset}] Aligned parquet not found — skipping validation")
            continue

        adf = pd.read_parquet(aligned_path, engine="pyarrow").reset_index()
        adf["asset"] = asset
        adf["timestamp"] = pd.to_datetime(adf["timestamp"], utc=True)

        merged = adf.merge(
            asset_stream[["timestamp", "regime_label"]], on="timestamp", how="inner"
        )
        n_merged = len(merged)
        if n_merged == 0:
            logger.warning(f"[{asset}] Merge produced 0 rows — skipping")
            continue

        logger.info(f"[{asset}] Merged {n_merged:,} rows for validation ({n_states} states)")

        regime_stats: dict[str, Any] = {}
        for state in sorted(merged["regime_label"].unique()):
            sub = merged[merged["regime_label"] == state]

            ann_ret = float(sub["close_return"].mean() * 24 * 365)
            ann_vol = float(sub["close_return"].std() * np.sqrt(24 * 365))
            sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0

            stats: dict[str, Any] = {
                "realized_vol_mean": float(sub["realized_vol"].mean()),
                "realized_vol_std": float(sub["realized_vol"].std()),
                "close_return_mean": float(sub["close_return"].mean()),
                "close_return_std": float(sub["close_return"].std()),
                "rsi_mean": float(sub["rsi"].mean()),
                "adx_mean": float(sub["adx"].mean()),
                "annualized_return": ann_ret,
                "annualized_vol": ann_vol,
                "sharpe": sharpe,
                "time_pct": float(len(sub) / n_merged),
                "n_samples": int(len(sub)),
            }
            regime_stats[str(state)] = stats

            mlflow.log_metric(
                f"{asset}_R{state}_rvol_mean", stats["realized_vol_mean"]
            )
            mlflow.log_metric(f"{asset}_R{state}_sharpe", sharpe)
            mlflow.log_metric(f"{asset}_R{state}_time_pct", stats["time_pct"])

        # Rich table per asset
        table = Table(
            title=f"{asset} — Per-Asset Regime Validation ({n_states} states)",
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
            )

        _console.print()
        _console.print(table)

        vol_means = [regime_stats[k]["realized_vol_mean"] for k in regime_stats]
        vol_std = float(np.std(vol_means))
        passed = vol_std > 0.001

        mlflow.log_metric(f"{asset}_vol_std_across_regimes", vol_std)
        mlflow.log_metric(f"{asset}_semantic_validation_passed", int(passed))

        if passed:
            logger.success(
                f"[{asset}] Semantic validation PASSED: std(rvol μ) = {vol_std:.6f}"
            )
        else:
            logger.critical(
                f"[{asset}] Semantic validation FAILED: std(rvol μ) = {vol_std:.6f} ≤ 0.001"
            )

        all_summaries[asset] = {
            "n_states": n_states,
            "semantic_validation_passed": passed,
            "vol_std_across_regimes": vol_std,
            "regimes": regime_stats,
        }

    validated_dir = _ROOT / "data" / "validated"
    validated_dir.mkdir(parents=True, exist_ok=True)
    json_path = validated_dir / "regime_semantics_per_asset.json"
    with open(json_path, "w") as fh:
        json.dump(all_summaries, fh, indent=2)
    logger.success(f"Saved per-asset semantics → {json_path}")
    mlflow.log_artifact(str(json_path))

    return all_summaries


def check_btc_march_2020(
    per_asset_stream_df: pd.DataFrame,
    aligned_btc_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Check whether the per-asset BTC HMM assigns a distinct regime to March 12-13 2020.

    A crash is considered detected if:
      - The dominant regime during March 12-13 differs from March 1-11
      - OR KL divergence between mean probability vectors > 0.5

    Returns:
        crash_regime_labels       : list of hourly regime labels during March 12-13
        dominant_crash_regime     : int
        dominant_pre_crash_regime : int
        crash_differs             : bool
        kl_divergence             : float (KL of crash distribution vs pre-crash)
        pre_crash_realized_vol    : float
        crash_realized_vol        : float
        verdict                   : 'CRASH_DETECTED' or 'NOT_DETECTED'
    """
    btc_stream = per_asset_stream_df[
        per_asset_stream_df["asset"] == "BTC_USDT"
    ].copy()
    btc_stream["timestamp"] = pd.to_datetime(btc_stream["timestamp"], utc=True)

    pre_start = pd.Timestamp("2020-03-01", tz="UTC")
    pre_end = pd.Timestamp("2020-03-12", tz="UTC")
    crash_start = pd.Timestamp("2020-03-12", tz="UTC")
    crash_end = pd.Timestamp("2020-03-14", tz="UTC")

    pre_window = btc_stream[
        (btc_stream["timestamp"] >= pre_start) & (btc_stream["timestamp"] < pre_end)
    ]
    crash_window = btc_stream[
        (btc_stream["timestamp"] >= crash_start) & (btc_stream["timestamp"] < crash_end)
    ]

    if pre_window.empty or crash_window.empty:
        logger.warning(
            "BTC March 2020 check: insufficient data in stream "
            f"(pre={len(pre_window)}, crash={len(crash_window)})"
        )
        return {
            "crash_regime_labels": [],
            "dominant_crash_regime": -1,
            "dominant_pre_crash_regime": -1,
            "crash_differs": False,
            "kl_divergence": 0.0,
            "pre_crash_realized_vol": 0.0,
            "crash_realized_vol": 0.0,
            "verdict": "NOT_DETECTED",
        }

    dominant_pre = int(pre_window["regime_label"].value_counts().idxmax())
    dominant_crash = int(crash_window["regime_label"].value_counts().idxmax())
    crash_differs = dominant_pre != dominant_crash

    # KL divergence between mean probability vectors
    prob_cols = sorted(
        [c for c in btc_stream.columns if c.startswith("regime_prob_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    eps = 1e-10
    p_pre = pre_window[prob_cols].mean().values.astype(float)
    p_crash = crash_window[prob_cols].mean().values.astype(float)
    kl_div = float(np.sum(p_crash * np.log((p_crash + eps) / (p_pre + eps))))

    # Realized vol from aligned BTC data
    aligned_btc_df.index = pd.to_datetime(aligned_btc_df.index, utc=True)
    pre_btc = aligned_btc_df[
        (aligned_btc_df.index >= pre_start) & (aligned_btc_df.index < pre_end)
    ]
    crash_btc = aligned_btc_df[
        (aligned_btc_df.index >= crash_start) & (aligned_btc_df.index < crash_end)
    ]
    pre_rvol = float(pre_btc["realized_vol"].mean()) if len(pre_btc) > 0 else 0.0
    crash_rvol = float(crash_btc["realized_vol"].mean()) if len(crash_btc) > 0 else 0.0

    detected = crash_differs or kl_div > 0.5
    verdict = "CRASH_DETECTED" if detected else "NOT_DETECTED"

    result: dict[str, Any] = {
        "crash_regime_labels": crash_window["regime_label"].tolist(),
        "dominant_crash_regime": dominant_crash,
        "dominant_pre_crash_regime": dominant_pre,
        "crash_differs": crash_differs,
        "kl_divergence": kl_div,
        "pre_crash_realized_vol": pre_rvol,
        "crash_realized_vol": crash_rvol,
        "verdict": verdict,
    }

    mlflow.log_metric("btc_march2020_crash_detected", int(detected))
    mlflow.log_metric("btc_march2020_kl_divergence", kl_div)
    mlflow.log_metric("btc_march2020_dominant_pre_crash_regime", dominant_pre)
    mlflow.log_metric("btc_march2020_dominant_crash_regime", dominant_crash)
    mlflow.log_metric("btc_march2020_pre_rvol", pre_rvol)
    mlflow.log_metric("btc_march2020_crash_rvol", crash_rvol)

    if detected:
        logger.success(
            f"BTC March 2020: CRASH_DETECTED  "
            f"pre_regime=R{dominant_pre} → crash_regime=R{dominant_crash}  "
            f"KL={kl_div:.3f}  rvol: {pre_rvol:.3f} → {crash_rvol:.3f}"
        )
    else:
        logger.warning(
            f"BTC March 2020: NOT_DETECTED  "
            f"pre_regime=R{dominant_pre}  crash_regime=R{dominant_crash}  "
            f"KL={kl_div:.3f}"
        )

    return result
