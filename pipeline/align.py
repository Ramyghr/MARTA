"""
pipeline/align.py
Aligns all assets to a common timestamp index (BTC backbone, inner join).
Produces per-asset aligned parquets + a combined multi-index parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.logger import logger

PROCESSED_DIR = Path("data/processed")

# BTC uses the slash convention in the config; after _safe_symbol_name it becomes BTC_USDT
BTC_KEY = "BTC_USDT"


def align_assets(
    normalized_assets: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """
    Align all normalized asset DataFrames to the intersection of their timestamp indices.

    - Uses BTC's index as the backbone reference.
    - Asserts the intersection covers >= min_timestamp_overlap_pct of BTC's range.
    - Saves per-asset aligned parquets and a combined multi-indexed parquet.

    Returns dict mapping symbol -> aligned DataFrame (14 feature columns each).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    min_overlap: float = config["validation"]["min_timestamp_overlap_pct"]

    if BTC_KEY not in normalized_assets:
        raise RuntimeError(
            f"BTC ({BTC_KEY}) not found in normalized_assets — cannot establish backbone index."
        )

    btc_index: pd.DatetimeIndex = normalized_assets[BTC_KEY].index
    btc_len = len(btc_index)
    logger.info(f"Alignment backbone: {BTC_KEY} with {btc_len:,} timestamps.")

    # Compute intersection across all assets
    common_index: pd.DatetimeIndex = btc_index
    for symbol, df in normalized_assets.items():
        common_index = common_index.intersection(df.index)

    overlap_pct = len(common_index) / btc_len if btc_len > 0 else 0.0

    if overlap_pct < min_overlap:
        logger.critical(
            f"Timestamp intersection covers only {overlap_pct * 100:.1f}% of BTC range "
            f"(threshold: {min_overlap * 100:.0f}%). Data quality issue!"
        )
    else:
        logger.success(
            f"Timestamp intersection: {len(common_index):,} rows "
            f"({overlap_pct * 100:.1f}% of BTC range). "
            f"Range: {common_index.min()} → {common_index.max()}"
        )

    aligned: dict[str, pd.DataFrame] = {}
    frames_for_combined: list[pd.DataFrame] = []

    for symbol, df in normalized_assets.items():
        original_len = len(df)
        df_aligned = df.reindex(common_index)

        retained_pct = len(df_aligned) / original_len * 100 if original_len > 0 else 0.0
        logger.info(
            f"[{symbol}] Aligned: {len(df_aligned):,} rows retained "
            f"({retained_pct:.1f}% of {original_len:,})"
        )

        out_path = PROCESSED_DIR / f"{symbol}_aligned.parquet"
        df_aligned.to_parquet(out_path, engine="pyarrow")
        logger.info(f"[{symbol}] Saved aligned → {out_path}")

        aligned[symbol] = df_aligned

        # Prepare for combined multi-index file
        df_named = df_aligned.copy()
        df_named.columns = pd.MultiIndex.from_product([[symbol], df_named.columns])
        frames_for_combined.append(df_named)

    # Combined file with (asset, feature) multi-index columns
    if frames_for_combined:
        combined = pd.concat(frames_for_combined, axis=1)
        combined_path = PROCESSED_DIR / "all_assets_aligned.parquet"
        combined.to_parquet(combined_path, engine="pyarrow")
        logger.success(f"Combined aligned dataset saved → {combined_path}")

    logger.success(
        f"Alignment complete: {len(aligned)} assets × {len(common_index):,} timestamps."
    )
    return aligned
