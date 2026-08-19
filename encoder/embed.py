"""
Post-training embedding extraction.

Slides a 60-candle window over each asset's full aligned dataset (no train/val split)
and runs model.encode() in batches to produce a unified embedding matrix.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from encoder.model import TFTRegimeEncoder

_ROOT = Path(__file__).resolve().parent.parent


def extract_embeddings(
    model: TFTRegimeEncoder,
    config: dict,
) -> tuple[np.ndarray, list]:
    """
    Embed every sliding window for all assets and persist results to data/processed/.

    Returns (embedding_matrix, timestamps) where:
      embedding_matrix: (n_total_windows, 128)
      timestamps:       list of end-of-window timestamps (length n_total_windows)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    window_size: int = config["training"]["window_size"]
    assets: list[str] = config["data"]["assets"]
    # Opt-in (2026-08-14) — "_aligned" preserves exact prior behavior. See
    # encoder/train.py's _load_asset_split docstring for why "_normalized" exists (the
    # pre-alignment, gap-free file); must match whatever file_suffix training itself used,
    # or the embeddings extracted here would silently come from different (and possibly
    # gap-corrupted) data than what the checkpoint was actually trained on.
    file_suffix: str = str(config["data"].get("file_suffix", "_aligned"))
    processed_dir = _ROOT / config["data"]["processed_dir"]
    # Read path (processed_dir) may be a read-only mount (e.g. /kaggle/input).
    # Generated artifacts go to output_dir, which defaults to processed_dir but
    # can be overridden to a writable location (e.g. /kaggle/working/data/processed).
    out_dir = Path(config["data"].get("output_dir") or processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 512

    all_embeddings: list[np.ndarray] = []
    all_timestamps: list = []
    asset_map_rows: list[dict] = []
    total_idx = 0

    for asset in assets:
        df = pd.read_parquet(
            processed_dir / f"{asset}{file_suffix}.parquet", engine="pyarrow"
        )
        data = df.values.astype(np.float32)           # (T, 14)
        timestamps = df.index.tolist()

        n_windows = len(df) - window_size
        if n_windows <= 0:
            continue

        # Collect raw windows and their end timestamps
        windows: list[np.ndarray] = []
        win_timestamps: list = []
        for start in range(n_windows):
            windows.append(data[start : start + window_size])
            win_timestamps.append(timestamps[start + window_size - 1])

        # Batch encode
        asset_embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = np.stack(windows[i : i + batch_size])
                tensor = torch.from_numpy(batch).to(device)
                emb = model.encode(tensor)             # (B, 128)
                asset_embeddings.append(emb.cpu().numpy())

        asset_emb_arr = np.concatenate(asset_embeddings, axis=0)  # (n_windows, 128)
        all_embeddings.append(asset_emb_arr)
        all_timestamps.extend(win_timestamps)

        asset_map_rows.append(
            {
                "start_idx": total_idx,
                "end_idx": total_idx + n_windows,
                "asset_name": asset,
            }
        )
        total_idx += n_windows

    embedding_matrix = np.concatenate(all_embeddings, axis=0)  # (N, 128)

    # Persist (to writable out_dir, not the possibly read-only read path)
    np.save(out_dir / "embeddings.npy", embedding_matrix)
    pd.DataFrame({"timestamp": all_timestamps}).to_parquet(
        out_dir / "embedding_timestamps.parquet", engine="pyarrow"
    )
    pd.DataFrame(asset_map_rows).to_parquet(
        out_dir / "embedding_asset_map.parquet", engine="pyarrow"
    )

    # MLflow logging
    mlflow.log_metric("embedding_rows", embedding_matrix.shape[0])
    mlflow.log_metric("embedding_dim", embedding_matrix.shape[1])
    for row in asset_map_rows:
        n = row["end_idx"] - row["start_idx"]
        mlflow.log_metric(f"windows_{row['asset_name']}", n)

    return embedding_matrix, all_timestamps
