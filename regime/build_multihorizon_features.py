"""Builds a 22-column ({14 production} + {4 vol-shape features x 2 extra horizons:
90d, 180d}) raw-feature embeddings set — Phase 2 of the regime-layer audit
(2026-08-19). No encoder, no PCA — raw features straight to the HMM, consistent with
the project's raw-feature pivot (2026-08-12), so the normalization-horizon variable
isn't confounded with encoder complexity.

Tests whether the production 720h(30d) rolling z-score is silently capping how long a
"regime" the HMM discovers can persist (measured production dwell times were 13-20h —
far shorter than a classical multi-week market regime). See
pipeline/normalize.py:normalize_features_multihorizon for the extra-horizon columns.

Pipeline (self-contained — does NOT call pipeline/align.py, which writes to the
sacred data/processed/{asset}_aligned.parquet):
  {stem}_features.parquet (18-col, already on disk from compute_features)
    -> normalize_features_multihorizon()  (22-col, per-asset)
    -> inner-join align across the 4 production assets (BTC backbone)
    -> data/processed/multihorizon/{asset}_aligned.parquet
    -> embeddings.npy / embedding_timestamps.parquet / embedding_asset_map.parquet

Run: conda run -n marta python regime/build_multihorizon_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.normalize import normalize_features_multihorizon
from utils.assets import ASSET_FILE_STEM
from utils.logger import logger

OUT_DIR = ROOT / "data" / "processed" / "multihorizon"
ASSETS = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]  # logical ids, matches production's 4-asset set
BTC_KEY = "BTC_USDT"


def main() -> None:
    with open(ROOT / "configs" / "assets.yaml") as f:
        config = yaml.safe_load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    normalized: dict[str, pd.DataFrame] = {}
    for asset in ASSETS:
        stem = ASSET_FILE_STEM.get(asset, asset)
        features_path = ROOT / "data" / "processed" / f"{stem}_features.parquet"
        df = pd.read_parquet(features_path, engine="pyarrow")
        out = normalize_features_multihorizon(df, config)
        normalized[asset] = out
        logger.info(
            f"[{asset}] {len(out):,} rows x {len(out.columns)} cols "
            f"[{out.index.min()} -> {out.index.max()}]"
        )

    ref_cols = list(normalized[BTC_KEY].columns)
    for asset, df in normalized.items():
        assert list(df.columns) == ref_cols, f"[{asset}] column mismatch vs {BTC_KEY}: {df.columns.tolist()}"

    # inner-join align across all 4 assets, BTC backbone (mirrors pipeline/align.py's
    # logic; reimplemented here rather than imported so this script never touches
    # pipeline.align's hardcoded PROCESSED_DIR="data/processed" module constant)
    btc_index = normalized[BTC_KEY].index
    common_index = btc_index
    for df in normalized.values():
        common_index = common_index.intersection(df.index)
    overlap_pct = len(common_index) / len(btc_index) * 100
    logger.info(f"Alignment: {len(common_index):,} common timestamps ({overlap_pct:.1f}% of BTC range)")

    blocks: list[np.ndarray] = []
    ts_rows: list[pd.Timestamp] = []
    asset_map_rows: list[dict] = []
    cursor = 0

    for asset in ASSETS:
        df_aligned = normalized[asset].reindex(common_index)
        assert not df_aligned.isnull().any().any(), f"[{asset}] NaN after alignment"

        out_path = OUT_DIR / f"{asset}_aligned.parquet"
        df_aligned.to_parquet(out_path, engine="pyarrow")

        n = len(df_aligned)
        blocks.append(df_aligned.values.astype(np.float32))
        ts_rows.extend(df_aligned.index.tolist())
        stem = ASSET_FILE_STEM.get(asset, asset)
        asset_map_rows.append({"asset_name": stem, "start_idx": cursor, "end_idx": cursor + n})
        cursor += n
        logger.success(f"[{asset}] aligned -> {out_path} ({n:,} rows)")

    embeddings = np.concatenate(blocks, axis=0)
    assert embeddings.shape[1] == len(ref_cols), f"expected {len(ref_cols)} cols, got {embeddings.shape[1]}"
    assert not np.isnan(embeddings).any(), "NaN in multi-horizon feature matrix"

    np.save(OUT_DIR / "embeddings.npy", embeddings)
    pd.DataFrame({"timestamp": ts_rows}).to_parquet(
        OUT_DIR / "embedding_timestamps.parquet", engine="pyarrow", index=False
    )
    pd.DataFrame(asset_map_rows).to_parquet(
        OUT_DIR / "embedding_asset_map.parquet", engine="pyarrow", index=False
    )

    logger.success(
        f"Done: {embeddings.shape[0]:,} rows x {embeddings.shape[1]} features "
        f"({len(ref_cols) - 14} extra multi-horizon cols) -> {OUT_DIR}"
    )
    print(f"columns: {ref_cols}")


if __name__ == "__main__":
    main()
