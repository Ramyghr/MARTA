"""
regime/compare.py
Comparison between shared HMM and per-asset HMM regime assignments.

Aligns regime labels across HMMs by vol-percentile rank so that HMMs with different
n_states can be compared on a common "low / mid / high volatility" scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

_PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
]

from utils.logger import logger  # noqa: E402

_CRASH_START = pd.Timestamp("2020-03-12", tz="UTC")
_CRASH_END = pd.Timestamp("2020-03-14", tz="UTC")


def _load_aligned(asset: str) -> pd.DataFrame:
    path = _ROOT / "data" / "processed" / f"{asset}_aligned.parquet"
    df = pd.read_parquet(path, engine="pyarrow")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _vol_categories(
    stream: pd.DataFrame,
    asset: str,
    aligned_df: pd.DataFrame,
) -> pd.Series:
    """
    Map each timestamp's regime label to a coarse vol category (low / mid / high).

    Uses the 33rd and 67th percentiles of per-regime mean realized_vol as thresholds,
    computed from the aligned features joined with the stream for this asset.
    """
    asset_stream = stream[stream["asset"] == asset].copy()
    asset_stream["timestamp"] = pd.to_datetime(asset_stream["timestamp"], utc=True)

    aligned_reset = aligned_df.reset_index().rename(columns={"index": "timestamp"})
    aligned_reset["timestamp"] = pd.to_datetime(aligned_reset["timestamp"], utc=True)

    merged = asset_stream.merge(
        aligned_reset[["timestamp", "realized_vol"]], on="timestamp", how="left"
    )

    regime_vol = (
        merged.groupby("regime_label")["realized_vol"].mean().to_dict()
    )
    vol_means = np.array(list(regime_vol.values()))
    p33 = float(np.percentile(vol_means, 33))
    p67 = float(np.percentile(vol_means, 67))

    def _cat(label: int) -> str:
        v = regime_vol.get(label, 0.0)
        if v < p33:
            return "low"
        if v < p67:
            return "mid"
        return "high"

    return merged["regime_label"].map(_cat)


def compare_regime_streams(
    shared_stream: pd.DataFrame,
    per_asset_stream: pd.DataFrame,
    asset: str,
) -> dict[str, Any]:
    """
    Compare shared HMM and per-asset HMM regime assignments for one asset.

    Computes:
      - n_states for each HMM (for this asset's regimes)
      - Agreement rate: % of timestamps where both HMMs assign the same vol category
        (low / mid / high, aligned by mean realized_vol of each regime)
      - March 2020 comparison for BTC: which regime each HMM assigns during the crash
      - Semantic distinctness: which HMM has higher std of realized_vol across regime means

    Returns dict with comparison metrics.
    """
    aligned_df = _load_aligned(asset)

    shared_asset = shared_stream[shared_stream["asset"] == asset].copy()
    per_asset = per_asset_stream[per_asset_stream["asset"] == asset].copy()
    shared_asset["timestamp"] = pd.to_datetime(shared_asset["timestamp"], utc=True)
    per_asset["timestamp"] = pd.to_datetime(per_asset["timestamp"], utc=True)

    shared_n_states = int(shared_asset["regime_label"].nunique())
    per_asset_n_states = int(per_asset["regime_label"].nunique())

    # Vol-category alignment
    shared_cats = _vol_categories(shared_stream, asset, aligned_df)
    per_asset_cats = _vol_categories(per_asset_stream, asset, aligned_df)

    # Align on timestamps
    shared_asset = shared_asset.reset_index(drop=True)
    per_asset = per_asset.reset_index(drop=True)
    merged = shared_asset[["timestamp", "regime_label"]].merge(
        per_asset[["timestamp", "regime_label"]].rename(
            columns={"regime_label": "pa_regime_label"}
        ),
        on="timestamp",
        how="inner",
    )
    n_common = len(merged)

    shared_cat_series = _vol_categories(
        shared_stream[shared_stream["asset"] == asset].assign(
            timestamp=lambda df: pd.to_datetime(df["timestamp"], utc=True)
        ).merge(merged[["timestamp"]], on="timestamp"),
        asset,
        aligned_df,
    ) if n_common > 0 else pd.Series(dtype=str)

    # Recompute per-merged timestamps for clean category vectors
    aligned_reset = aligned_df.reset_index().rename(columns={"index": "timestamp"})
    aligned_reset["timestamp"] = pd.to_datetime(aligned_reset["timestamp"], utc=True)

    def _regime_vol_map(stream_asset: pd.DataFrame) -> dict[int, float]:
        s = stream_asset.copy()
        s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True)
        m = s.merge(aligned_reset[["timestamp", "realized_vol"]], on="timestamp", how="left")
        return m.groupby("regime_label")["realized_vol"].mean().to_dict()

    shared_rvm = _regime_vol_map(shared_asset)
    pa_rvm = _regime_vol_map(per_asset)

    def _make_thresholds(rvm: dict[int, float]) -> tuple[float, float]:
        vals = np.array(list(rvm.values()))
        return float(np.percentile(vals, 33)), float(np.percentile(vals, 67))

    sh_p33, sh_p67 = _make_thresholds(shared_rvm)
    pa_p33, pa_p67 = _make_thresholds(pa_rvm)

    def _cat(v: float, p33: float, p67: float) -> str:
        if v < p33:
            return "low"
        if v < p67:
            return "mid"
        return "high"

    if n_common > 0:
        sh_cats_merged = merged["regime_label"].map(
            lambda lbl: _cat(shared_rvm.get(lbl, 0.0), sh_p33, sh_p67)
        )
        pa_cats_merged = merged["pa_regime_label"].map(
            lambda lbl: _cat(pa_rvm.get(lbl, 0.0), pa_p33, pa_p67)
        )
        agreement_rate = float((sh_cats_merged == pa_cats_merged).mean())
    else:
        agreement_rate = 0.0

    # Semantic distinctness
    shared_vol_std = float(np.std(list(shared_rvm.values()))) if shared_rvm else 0.0
    pa_vol_std = float(np.std(list(pa_rvm.values()))) if pa_rvm else 0.0

    result: dict[str, Any] = {
        "asset": asset,
        "shared_n_states": shared_n_states,
        "per_asset_n_states": per_asset_n_states,
        "agreement_rate": agreement_rate,
        "shared_vol_std_across_regimes": shared_vol_std,
        "per_asset_vol_std_across_regimes": pa_vol_std,
        "per_asset_more_distinct": pa_vol_std > shared_vol_std,
    }

    # BTC March 2020 comparison
    if asset == "BTC_USDT":
        shared_crash = shared_asset[
            (shared_asset["timestamp"] >= _CRASH_START)
            & (shared_asset["timestamp"] < _CRASH_END)
        ]
        pa_crash = per_asset[
            (per_asset["timestamp"] >= _CRASH_START)
            & (per_asset["timestamp"] < _CRASH_END)
        ]

        shared_crash_regime = (
            int(shared_crash["regime_label"].value_counts().idxmax())
            if not shared_crash.empty else -1
        )
        pa_crash_regime = (
            int(pa_crash["regime_label"].value_counts().idxmax())
            if not pa_crash.empty else -1
        )
        shared_crash_vol_cat = _cat(shared_rvm.get(shared_crash_regime, 0.0), sh_p33, sh_p67)
        pa_crash_vol_cat = _cat(pa_rvm.get(pa_crash_regime, 0.0), pa_p33, pa_p67)

        result["btc_march2020_shared_regime"] = shared_crash_regime
        result["btc_march2020_per_asset_regime"] = pa_crash_regime
        result["btc_march2020_shared_vol_cat"] = shared_crash_vol_cat
        result["btc_march2020_per_asset_vol_cat"] = pa_crash_vol_cat
        result["btc_march2020_per_asset_detected_high_vol"] = pa_crash_vol_cat == "high"

        logger.info(
            f"BTC March 2020 — Shared HMM: R{shared_crash_regime} ({shared_crash_vol_cat}) | "
            f"Per-asset HMM: R{pa_crash_regime} ({pa_crash_vol_cat})"
        )

    mlflow.log_metric(f"compare_{asset}_agreement_rate", agreement_rate)
    mlflow.log_metric(f"compare_{asset}_shared_vol_std", shared_vol_std)
    mlflow.log_metric(f"compare_{asset}_per_asset_vol_std", pa_vol_std)

    logger.info(
        f"[{asset}] Comparison: shared_n={shared_n_states}  pa_n={per_asset_n_states}  "
        f"agreement={agreement_rate:.1%}  "
        f"vol_std: shared={shared_vol_std:.4f}  per_asset={pa_vol_std:.4f}"
    )

    return result


def plot_regime_comparison(
    shared_stream: pd.DataFrame,
    per_asset_stream: pd.DataFrame,
    asset: str,
    price_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Two-panel publication-quality plot comparing shared vs per-asset HMM on BTC.

    Top panel:    BTC log price with shared HMM regime bands
    Bottom panel: BTC log price with per-asset HMM regime bands
    Both panels mark March 12-13 2020 with a vertical red dashed line.
    """
    shared_asset = (
        shared_stream[shared_stream["asset"] == asset]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    per_asset = (
        per_asset_stream[per_asset_stream["asset"] == asset]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    shared_asset["timestamp"] = pd.to_datetime(shared_asset["timestamp"], utc=True)
    per_asset["timestamp"] = pd.to_datetime(per_asset["timestamp"], utc=True)

    # Reconstruct price index from close_return
    ts_idx = pd.DatetimeIndex(shared_asset["timestamp"])
    price_df.index = pd.to_datetime(price_df.index, utc=True)
    close_ret = price_df["close_return"].reindex(ts_idx).fillna(0.0)
    price_index = np.exp(close_ret.cumsum().values)

    shared_regimes = shared_asset["regime_label"].values
    pa_regimes = per_asset["regime_label"].values
    shared_ts = shared_asset["timestamp"].values
    pa_ts = per_asset["timestamp"].values

    shared_n = int(shared_asset["regime_label"].nunique())
    pa_n = int(per_asset["regime_label"].nunique())
    shared_labels = sorted(shared_asset["regime_label"].unique())
    pa_labels = sorted(per_asset["regime_label"].unique())

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 10), sharex=False,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.suptitle(
        f"{asset} — Shared HMM vs Per-Asset HMM Regime Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    def _draw_panel(
        ax: plt.Axes,
        timestamps: np.ndarray,
        prices: np.ndarray,
        regimes: np.ndarray,
        labels: list[int],
        title: str,
        colors: list[str],
    ) -> None:
        ts_dt = pd.to_datetime(timestamps)
        p_low = prices.min() * 0.8
        p_high = prices.max() * 1.2

        for i, lbl in enumerate(labels):
            mask = regimes == lbl
            color = colors[i % len(colors)]
            ax.fill_between(
                ts_dt, p_low, p_high,
                where=mask,
                alpha=0.18,
                color=color,
                zorder=0,
            )

        ax.semilogy(ts_dt, prices, color="#212121", linewidth=0.7, alpha=0.9, zorder=2)

        # Mark March 12-13 2020
        crash_dt = _CRASH_START.to_pydatetime()
        ax.axvline(
            crash_dt, color="#D32F2F", linestyle="--", linewidth=1.5,
            alpha=0.85, zorder=3, label="March 12, 2020"
        )

        ax.set_title(title, fontsize=11, pad=6)
        ax.set_ylabel("Price Index (log scale)", fontsize=9)
        ax.grid(True, alpha=0.2, which="both")
        ax.set_xlim(ts_dt[0], ts_dt[-1])

        legend_patches = [
            mpatches.Patch(color=colors[i % len(colors)], alpha=0.6, label=f"R{lbl}")
            for i, lbl in enumerate(labels)
        ] + [
            plt.Line2D(
                [0], [0], color="#D32F2F", linestyle="--",
                linewidth=1.5, label="Mar 12, 2020"
            )
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            fontsize=8,
            ncol=min(len(labels) + 1, 8),
            framealpha=0.7,
        )

    _draw_panel(
        ax_top,
        shared_ts,
        price_index,
        shared_regimes,
        shared_labels,
        f"Shared HMM ({shared_n} states across all 5 assets)",
        _PALETTE,
    )

    # Per-asset may have different timestamps if stream length differs; realign price
    pa_ts_idx = pd.DatetimeIndex(pa_ts)
    pa_close_ret = price_df["close_return"].reindex(pa_ts_idx).fillna(0.0)
    pa_price_index = np.exp(pa_close_ret.cumsum().values)

    _draw_panel(
        ax_bot,
        pa_ts,
        pa_price_index,
        pa_regimes,
        pa_labels,
        f"Per-Asset BTC HMM ({pa_n} states, fitted on BTC embeddings only)",
        _PALETTE[1:] + _PALETTE[:1],  # offset palette to visually distinguish
    )

    ax_top.set_xlabel("")
    ax_bot.set_xlabel("Date", fontsize=9)

    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved comparison plot → {save_path}")
