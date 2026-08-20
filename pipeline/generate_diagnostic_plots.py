"""Data-pipeline diagnostic plots — regime-layer audit (2026-08-19), crypto-only
5-asset panel (BTC/ETH/BNB/LTC/XRP).

Generates, under pipeline/plots/:
  1. raw_data_availability_heatmap.png     — weekly row-coverage %, per asset, from RAW
     (pre-pipeline) parquet — shows XRP's later start (2018-05-04) and any exchange-
     downtime gaps.
  2. raw_price_volume_overlay.png          — close price (log) + volume, per asset,
     full raw history — first-look sanity check that the ingested data looks right.
  3. feature_distributions_before_normalization.png
     feature_distributions_after_normalization.png
                                            — histogram+KDE for each of the 12
     rolling-z-scored indicators, pooled across all 5 assets, raw vs after
     normalize_features().
  4. feature_correlation_matrix.png        — Pearson correlation heatmap of the 14
     final (aligned, normalized) feature columns, pooled across all 5 assets.
  5. rolling_zscore_stationarity_check.png — realized_vol z-scored at the production
     720h(30d) horizon vs the Phase 2 90d/180d extensions, BTC_USDT full history —
     visual check for whether any horizon shows residual long-run drift away from 0
     (raw hourly z-score, faint, + a 720h rolling-mean smoothing, bold).

Run: conda run -n marta python pipeline/generate_diagnostic_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.normalize import FEATURE_COLS, NORMALIZE_COLS, normalize_features
from utils.logger import logger

PLOTS_DIR = ROOT / "pipeline" / "plots"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]
_PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


def _load_raw(asset: str) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / f"{asset}_1h_raw.parquet", engine="pyarrow")


def _load_features(asset: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{asset}_features.parquet", engine="pyarrow")


def plot_availability_heatmap() -> None:
    # Bin edges must be IDENTICAL across assets — resample("7D") anchors each series'
    # own bins to its own first timestamp, so per-asset weekly boundaries land at
    # different hours and combining the resulting Series into one DataFrame (which
    # aligns by the union of indices) produces spurious near-miss NaN cells that read
    # as fake "gaps" striping every asset at once. A shared reference grid fixes it.
    bin_edges = pd.date_range("2018-01-01", "2026-02-01", freq="7D", tz="UTC")
    weekly = {}
    for asset in ASSETS:
        df = _load_raw(asset)
        counts = df["close"].groupby(pd.cut(df.index, bin_edges, right=False)).count()
        coverage = (counts / (7 * 24)).clip(upper=1.0)
        coverage.index = bin_edges[:-1]
        weekly[asset] = coverage
    mat = pd.DataFrame(weekly).T.fillna(0.0)  # assets x weeks; unfetched weeks -> 0 coverage

    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(mat, cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "weekly coverage"}, ax=ax)
    ax.set_title("Raw data availability — weekly candle coverage (1.0 = no gaps)")
    ax.set_xlabel("week")
    n = mat.shape[1]
    step = max(1, n // 20)
    ax.set_xticks(np.arange(0, n, step) + 0.5)
    ax.set_xticklabels([mat.columns[i].strftime("%Y-%m") for i in range(0, n, step)], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "raw_data_availability_heatmap.png", dpi=120)
    plt.close(fig)
    logger.success("Saved raw_data_availability_heatmap.png")


def plot_price_volume_overlay() -> None:
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(14, 3 * len(ASSETS)), sharex=False)
    for i, asset in enumerate(ASSETS):
        df = _load_raw(asset)
        # daily volume, not ~70K individual hourly bars — hourly bars over an 8-year
        # span are both unreadably dense and very slow for matplotlib to render
        daily_volume = df["volume"].resample("1D").sum()
        ax = axes[i]
        ax.plot(df.index, df["close"], color=_PALETTE[i], linewidth=0.6)
        ax.set_yscale("log")
        ax.set_ylabel(f"{asset}\nclose (log)", fontsize=9)
        ax2 = ax.twinx()
        ax2.fill_between(daily_volume.index, daily_volume.values, color="gray", alpha=0.25, step="mid")
        ax2.set_ylabel("volume (daily)", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle("Raw price (log) + volume overlay, per asset — full history")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "raw_price_volume_overlay.png", dpi=120)
    plt.close(fig)
    logger.success("Saved raw_price_volume_overlay.png")


def plot_feature_distributions() -> None:
    with open(ROOT / "configs" / "assets.yaml") as f:
        config = yaml.safe_load(f)

    raw_pool = {col: [] for col in NORMALIZE_COLS}
    norm_pool = {col: [] for col in NORMALIZE_COLS}
    for asset in ASSETS:
        feats = _load_features(asset)
        normed = normalize_features(feats, config)
        for col in NORMALIZE_COLS:
            raw_pool[col].append(feats[col].dropna().to_numpy())
            norm_pool[col].append(normed[col].dropna().to_numpy())

    for tag, pool in [("before_normalization", raw_pool), ("after_normalization", norm_pool)]:
        fig, axes = plt.subplots(4, 3, figsize=(15, 14))
        for ax, col in zip(axes.flat, NORMALIZE_COLS):
            data = np.concatenate(pool[col])
            # clip extreme outliers for a readable histogram range — raw indicators can
            # have heavy tails; this only affects the display range, not the underlying data
            lo, hi = np.percentile(data, [0.5, 99.5])
            data_clipped = data[(data >= lo) & (data <= hi)]
            sns.histplot(data_clipped, kde=True, ax=ax, color="#2196F3", stat="density", bins=80)
            ax.set_title(col, fontsize=10)
            ax.set_ylabel("")
        fig.suptitle(f"Feature distributions — {tag.replace('_', ' ')} (pooled across 5 assets)")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"feature_distributions_{tag}.png", dpi=120)
        plt.close(fig)
        logger.success(f"Saved feature_distributions_{tag}.png")


def plot_correlation_matrix() -> None:
    frames = []
    for asset in ASSETS:
        path = PROCESSED_DIR / "crypto5" / f"{asset}_aligned.parquet"
        frames.append(pd.read_parquet(path, engine="pyarrow"))
    pooled = pd.concat(frames, axis=0)
    corr = pooled[FEATURE_COLS].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature correlation matrix — 14 final columns, pooled across 5 crypto assets")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_correlation_matrix.png", dpi=120)
    plt.close(fig)
    logger.success("Saved feature_correlation_matrix.png")


def plot_rolling_zscore_stationarity() -> None:
    path = PROCESSED_DIR / "crypto5_multihorizon" / "BTC_USDT_aligned.parquet"
    df = pd.read_parquet(path, engine="pyarrow")

    horizons = [
        ("realized_vol", "720h / 30d (production)"),
        ("realized_vol_z90d", "90d"),
        ("realized_vol_z180d", "180d"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, (col, label) in zip(axes, horizons):
        series = df[col]
        rolling_mean = series.rolling(720, min_periods=168).mean()
        ax.plot(df.index, series, color="#90CAF9", linewidth=0.3, alpha=0.6, label="hourly z-score")
        ax.plot(df.index, rolling_mean, color="#0D47A1", linewidth=1.2, label="720h rolling mean")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle("realized_vol rolling z-score stationarity check — BTC_USDT, 3 normalization horizons")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "rolling_zscore_stationarity_check.png", dpi=120)
    plt.close(fig)
    logger.success("Saved rolling_zscore_stationarity_check.png")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_availability_heatmap()
    plot_price_volume_overlay()
    plot_feature_distributions()
    plot_correlation_matrix()
    plot_rolling_zscore_stationarity()
    logger.success(f"All plots saved -> {PLOTS_DIR}")


if __name__ == "__main__":
    main()
