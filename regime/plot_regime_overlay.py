"""Visualizes regime labels overlaid on price, per asset, for the crypto5_multihorizon
panel (n=6) — a direct visual check on whether the measured dwell times (14-23h at n=6)
look plausible, requested after the numeric dwell-time tables alone left that an open
question.

Two views, both under regime/plots/:
  1. regime_overlay_full_history.png — full 2018-2025 history, all 5 assets, price
     (log scale) colored by regime label (scatter — ~18K regime episodes across the
     panel makes per-episode shading too dense to be useful at this zoom; this view is
     for the big-picture regime/price correspondence, not for judging dwell time).
  2. regime_overlay_zoomed.png — a shared ~120-day window (2025-09-01 to 2025-12-31,
     inside the test split, all 5 assets have data here) at full hourly resolution,
     regime episodes shown as colored background bands — this is the one that actually
     answers "does 14-23h dwell look right."

Run: conda run -n marta python regime/plot_regime_overlay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logger import logger

PANEL_DIR = ROOT / "data" / "processed" / "crypto5_multihorizon"
RAW_DIR = ROOT / "data" / "raw"
PLOTS_DIR = ROOT / "regime" / "plots"
ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]

_PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#FF5722", "#607D8B"]

ZOOM_START = pd.Timestamp("2025-09-01", tz="UTC")
ZOOM_END = pd.Timestamp("2025-12-31", tz="UTC")


def _load(asset: str, regime_stream: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_parquet(RAW_DIR / f"{asset}_1h_raw.parquet", engine="pyarrow")[["close"]]
    rs_asset = regime_stream[regime_stream["asset"] == asset].set_index("timestamp")[["regime_label"]]
    rs_asset.index = pd.to_datetime(rs_asset.index, utc=True)
    return raw.join(rs_asset, how="inner")


def plot_full_history(data: dict[str, pd.DataFrame], n_states: int) -> None:
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(15, 3.2 * len(ASSETS)), sharex=False)
    cmap = mcolors.ListedColormap(_PALETTE[:n_states])
    for ax, asset in zip(axes, ASSETS):
        df = data[asset]
        ax.scatter(
            df.index, df["close"], c=df["regime_label"], cmap=cmap,
            vmin=-0.5, vmax=n_states - 0.5, s=2, linewidths=0,
        )
        ax.set_yscale("log")
        ax.set_ylabel(asset, fontsize=10)
    handles = [mpatches.Patch(color=_PALETTE[i], label=f"regime {i}") for i in range(n_states)]
    axes[0].legend(handles=handles, loc="upper left", ncol=n_states, fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("time")
    fig.suptitle("Price (log) colored by regime label — full history, crypto5_multihorizon (n=6)")
    fig.tight_layout()
    out = PLOTS_DIR / "regime_overlay_full_history.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.success(f"Saved {out}")


def plot_zoomed(data: dict[str, pd.DataFrame], n_states: int) -> None:
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(16, 3.4 * len(ASSETS)), sharex=True)
    for ax, asset in zip(axes, ASSETS):
        df = data[asset]
        window = df[(df.index >= ZOOM_START) & (df.index <= ZOOM_END)]
        labels = window["regime_label"].to_numpy()
        idx = window.index
        start_i = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start_i]:
                end_idx = idx[-1] if i == len(labels) else idx[i - 1]
                ax.axvspan(
                    idx[start_i], end_idx,
                    color=_PALETTE[int(labels[start_i]) % len(_PALETTE)], alpha=0.28, linewidth=0,
                )
                start_i = i
        ax.plot(window.index, window["close"], color="black", linewidth=0.9)
        ax.set_yscale("log")
        ax.set_ylabel(asset, fontsize=10)
        ax.set_xlim(ZOOM_START, ZOOM_END)
    handles = [mpatches.Patch(color=_PALETTE[i], alpha=0.5, label=f"regime {i}") for i in range(n_states)]
    axes[0].legend(handles=handles, loc="upper left", ncol=n_states, fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("time")
    fig.suptitle(
        f"Price + regime bands, hourly — {ZOOM_START.date()} to {ZOOM_END.date()}, "
        f"crypto5_multihorizon (n=6)"
    )
    fig.tight_layout()
    out = PLOTS_DIR / "regime_overlay_zoomed.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.success(f"Saved {out}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    regime_stream = pd.read_parquet(PANEL_DIR / "regime_stream.parquet", engine="pyarrow")
    data = {a: _load(a, regime_stream) for a in ASSETS}
    n_states = int(max(df["regime_label"].max() for df in data.values())) + 1
    logger.info(f"n_states={n_states}")
    plot_full_history(data, n_states)
    plot_zoomed(data, n_states)


if __name__ == "__main__":
    main()
