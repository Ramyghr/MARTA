"""
regime/visualize.py
Visualization functions for MARTA regime discovery (Week 3).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

_PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
]


def plot_bic_curve(
    bic_results: dict[int, dict[str, float]],
    best_n: int,
    save_path: Path,
) -> None:
    """Line plot of BIC, AIC, and log-likelihood vs n_states."""
    ns = sorted(n for n in bic_results if bic_results[n]["bic"] < float("inf"))
    bics = [bic_results[n]["bic"] for n in ns]
    aics = [bic_results[n]["aic"] for n in ns]
    lls = [bic_results[n]["log_likelihood"] for n in ns]

    fig, (ax_ic, ax_ll) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_ic.plot(ns, bics, "o-", color="#2196F3", label="BIC", linewidth=2, markersize=6)
    ax_ic.plot(ns, aics, "s--", color="#F44336", label="AIC", linewidth=2, markersize=6)
    ax_ic.axvline(best_n, color="#FF9800", linestyle="--", alpha=0.9, linewidth=1.5,
                  label=f"Selected n={best_n}")
    ax_ic.set_ylabel("Information Criterion")
    ax_ic.set_title("HMM Model Selection — BIC / AIC vs n_states")
    ax_ic.legend(loc="upper right")
    ax_ic.grid(True, alpha=0.3)

    ax_ll.plot(ns, lls, "^-", color="#4CAF50", label="Log-likelihood", linewidth=2, markersize=6)
    ax_ll.axvline(best_n, color="#FF9800", linestyle="--", alpha=0.9, linewidth=1.5,
                  label=f"Selected n={best_n}")
    ax_ll.set_xlabel("Number of HMM states")
    ax_ll.set_ylabel("Log-likelihood")
    ax_ll.legend(loc="lower right")
    ax_ll.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regime_timeline(
    regime_df: pd.DataFrame,
    price_df: pd.DataFrame,
    asset: str,
    save_path: Path,
) -> None:
    """
    Two-panel figure sharing x-axis:
      Top   — log-scale price index (reconstructed from close_return)
      Bottom — stacked area chart of regime probabilities over time

    price_df is the aligned parquet for the asset (13 features, DatetimeIndex).
    """
    asset_regime = (
        regime_df[regime_df["asset"] == asset]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    prob_cols = sorted(
        [c for c in asset_regime.columns if c.startswith("regime_prob_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    n_states = len(prob_cols)

    # reconstruct relative price index from close_return
    ts_index = pd.DatetimeIndex(pd.to_datetime(asset_regime["timestamp"], utc=True))
    price_close_return = (
        price_df["close_return"]
        .reindex(ts_index)
        .fillna(0.0)
    )
    price_index = np.exp(price_close_return.cumsum().values)

    timestamps = pd.to_datetime(asset_regime["timestamp"])

    fig, (ax_price, ax_prob) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_price.semilogy(timestamps, price_index, color="#263238", linewidth=0.7, alpha=0.9)
    ax_price.set_ylabel("Price Index (log scale)")
    ax_price.set_title(f"{asset} — Price & Regime Probability Timeline")
    ax_price.grid(True, alpha=0.2)

    colors = _PALETTE[:n_states]
    ax_prob.stackplot(
        timestamps,
        *[asset_regime[c].values for c in prob_cols],
        labels=[f"R{i}" for i in range(n_states)],
        colors=colors,
        alpha=0.85,
    )
    ax_prob.set_ylabel("Regime Probability")
    ax_prob.set_ylim(0, 1)
    ax_prob.legend(loc="upper left", ncol=n_states, fontsize=8)
    ax_prob.grid(True, alpha=0.2)

    fig.autofmt_xdate()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regime_tsne(
    embeddings: np.ndarray,
    regime_labels: np.ndarray,
    save_path: Path,
    n_samples: int = 3000,
) -> None:
    """
    t-SNE on a uniform subsample of embeddings, colored by HMM regime label.
    Uses perplexity=50 (matching Week 2 settings) so geometric clusters are comparable.
    """
    rng = np.random.default_rng(42)
    n = min(n_samples, len(embeddings))
    idx = rng.choice(len(embeddings), size=n, replace=False)
    idx.sort()

    sample_emb = embeddings[idx].astype(np.float32)
    sample_labels = regime_labels[idx]

    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    coords = tsne.fit_transform(sample_emb)

    n_states = int(sample_labels.max()) + 1
    cmap = mcolors.ListedColormap(_PALETTE[:n_states])

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=sample_labels, cmap=cmap,
        s=6, alpha=0.7, linewidths=0,
        vmin=-0.5, vmax=n_states - 0.5,
    )
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_states))
    cbar.set_label("Regime label")
    cbar.set_ticklabels([f"R{i}" for i in range(n_states)])

    ax.set_title(
        f"t-SNE of {n:,} embeddings colored by HMM regime ({n_states} states)"
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
