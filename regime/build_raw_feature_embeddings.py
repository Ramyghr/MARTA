"""Builds an 'embeddings.npy' analog directly from raw (already normalized/aligned) 14
features, with NO encoder involved — used to feed regime/hmm.py:run_hmm_pipeline for the
raw-feature baseline (see project decision: contrastive encoder rejected, pivoting to
raw features + optional regime detection, 2026-08-12).

Writes the exact same 3-file schema encoder/embed.py's extract_embeddings() produces
(embeddings.npy, embedding_timestamps.parquet, embedding_asset_map.parquet) so
run_hmm_pipeline needs no further changes beyond the pca_n_components=None option already
added — it's agnostic to whether the N-dim input came from an encoder or raw features.

Unlike the encoder's embeddings (which lose window_size-1=59 rows/asset to LSTM warmup),
this uses every row of {asset}_aligned.parquet directly — no windowing, since there's
no sequence model to feed. 45,921 rows/asset (vs the encoder's 45,861).

Run: conda run -n marta python regime/build_raw_feature_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.assets import ASSET_FILE_STEM, aligned_path

OUT_DIR = ROOT / "data" / "processed" / "raw_baseline"
ASSETS = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]  # logical ids, same order as configs/assets.yaml


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    blocks: list[np.ndarray] = []
    ts_rows: list[pd.Timestamp] = []
    asset_map_rows: list[dict] = []
    cursor = 0

    for asset in ASSETS:
        path = aligned_path(asset, ROOT / "data" / "processed")
        df = pd.read_parquet(path, engine="pyarrow").sort_index()
        n = len(df)
        blocks.append(df.values.astype(np.float32))
        ts_rows.extend(df.index.tolist())
        # asset_map stores the file-stem name (e.g. "XAUUSD"), matching what
        # encoder/embed.py already writes and what regime/hmm.py's alias
        # dict ({"XAUUSD": "GOLD"}) expects to translate back.
        file_stem = ASSET_FILE_STEM.get(asset, asset)
        asset_map_rows.append({"asset_name": file_stem, "start_idx": cursor, "end_idx": cursor + n})
        cursor += n
        print(f"{asset} ({file_stem}): {n:,} rows  [{df.index.min()} -> {df.index.max()}]  "
              f"columns={list(df.columns)}")

    embeddings = np.concatenate(blocks, axis=0)
    assert embeddings.shape[1] == 14, f"expected 14 raw features, got {embeddings.shape[1]}"
    assert not np.isnan(embeddings).any(), "NaN in raw feature matrix"

    np.save(OUT_DIR / "embeddings.npy", embeddings)
    pd.DataFrame({"timestamp": ts_rows}).to_parquet(
        OUT_DIR / "embedding_timestamps.parquet", engine="pyarrow", index=False
    )
    pd.DataFrame(asset_map_rows).to_parquet(
        OUT_DIR / "embedding_asset_map.parquet", engine="pyarrow", index=False
    )

    print(f"\nDone: {embeddings.shape[0]:,} rows x {embeddings.shape[1]} raw features "
          f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
