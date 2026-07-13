"""
Training visualisations: t-SNE embedding plot and training-curve dashboard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

_ROOT = Path(__file__).resolve().parent.parent
_ASSET_COLORS = {
    "BTC_USDT": "#F7931A",
    "ETH_USDT": "#627EEA",
    "EURUSD": "#2E86AB",
    "XAUUSD": "#FFD700",
}


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    timestamps: list,
    save_path: Path,
    n_samples: int = 3000,
) -> None:
    """
    Two-panel t-SNE plot:
      Left  — coloured by chronological order (viridis) to reveal temporal drift.
      Right — coloured by asset.
    Subsamples n_samples rows with fixed seed=42 for reproducibility.
    Saves PNG to save_path and logs it as an MLflow artifact.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Load asset map for per-asset colouring
    asset_map_path = _ROOT / "data" / "processed" / "embedding_asset_map.parquet"
    asset_map = pd.read_parquet(asset_map_path, engine="pyarrow")
    n_total = embeddings.shape[0]
    asset_labels = np.empty(n_total, dtype=object)
    for _, row in asset_map.iterrows():
        asset_labels[int(row["start_idx"]) : int(row["end_idx"])] = row["asset_name"]

    # Subsample
    rng = np.random.default_rng(42)
    idx = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    idx = np.sort(idx)
    sub_emb = embeddings[idx]
    sub_labels = asset_labels[idx]

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=50,
        learning_rate=200,
        max_iter=1000,
        random_state=42,
    )
    coords = tsne.fit_transform(sub_emb)  # (n_samples, 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("t-SNE of Regime Embeddings", fontsize=14, fontweight="bold")

    # Left: chronological order
    sc0 = axes[0].scatter(
        coords[:, 0], coords[:, 1],
        c=np.arange(len(coords)), cmap="viridis", s=4, alpha=0.7,
    )
    plt.colorbar(sc0, ax=axes[0], label="Window index (chronological)")
    axes[0].set_title("Temporal drift")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # Right: by asset
    unique_assets = [a for a in _ASSET_COLORS if a in set(sub_labels)]
    for asset in unique_assets:
        mask = sub_labels == asset
        axes[1].scatter(
            coords[mask, 0], coords[mask, 1],
            c=_ASSET_COLORS[asset], label=asset, s=4, alpha=0.7,
        )
    axes[1].legend(markerscale=3, loc="upper right")
    axes[1].set_title("Asset membership")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    mlflow.log_artifact(str(save_path))


def plot_training_curves(mlflow_run_id: str, save_path: Path) -> None:
    """
    4-subplot training dashboard loaded from MLflow metric history.
    Saves PNG to save_path and logs it as an MLflow artifact.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    client = mlflow.tracking.MlflowClient()

    def _get(metric_name: str) -> tuple[list[int], list[float]]:
        history = client.get_metric_history(mlflow_run_id, metric_name)
        steps = [m.step for m in history]
        values = [m.value for m in history]
        return steps, values

    train_steps, train_loss = _get("train_loss")
    val_steps, val_loss = _get("val_loss")
    _, train_acc = _get("train_top1_acc")
    _, val_acc = _get("val_top1_acc")
    temp_steps, temperature = _get("temperature")
    lr_steps, lr = _get("learning_rate")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("MARTA Encoder Training", fontsize=13, fontweight="bold")

    axes[0, 0].plot(train_steps, train_loss, label="train", color="#2196F3")
    axes[0, 0].plot(val_steps, val_loss, label="val", color="#F44336")
    axes[0, 0].set_title("InfoNCE Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(train_steps, train_acc, label="train", color="#2196F3")
    axes[0, 1].plot(val_steps, val_acc, label="val", color="#F44336")
    axes[0, 1].set_title("Top-1 Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(temp_steps, temperature, color="#9C27B0")
    axes[1, 0].axhline(0.07, linestyle="--", color="gray", linewidth=0.8, label="init=0.07")
    axes[1, 0].set_title("Learned Temperature")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(lr_steps, lr, color="#FF9800")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    mlflow.log_artifact(str(save_path))
