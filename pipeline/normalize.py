"""
pipeline/normalize.py
Rolling z-score normalization of indicator features.

Layer 1 v2 output (14 features per asset):
  - 12 z-scored indicators (atr … autocorr_lag1, including the now-single-normalized
    obv_raw)
  - realized_vol_raw  — passed through UNNORMALIZED (preserves absolute magnitude
    for crash detection)
  - close_return      — log return

The old is_gap_fill column and weekend volume-zeroing are gone: histdata gives
genuine hourly bars, and forex weekend gaps are simply absent rows (dropped by the
inner join in align.py), not forward-filled rows that need flagging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import logger

PROCESSED_DIR = Path("data/processed")

# Indicators that receive the standard 720h rolling z-score.
NORMALIZE_COLS = [
    "atr",
    "adx",
    "rsi",
    "bb_width",
    "macd_histogram",
    "obv_raw",
    "vol_zscore",
    "realized_vol",
    "vol_ratio",
    "return_skew",
    "return_kurtosis",
    "autocorr_lag1",
]

# Features excluded from z-score normalization — must preserve absolute scale.
NO_NORMALIZE = {"realized_vol_raw"}

# Full set of feature columns in the normalized output (12 + 1 passthrough + close_return).
FEATURE_COLS = NORMALIZE_COLS + sorted(NO_NORMALIZE) + ["close_return"]


def _rolling_zscore(
    series: pd.Series, window: int, min_periods: int
) -> pd.Series:
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    return (series - mean) / (std + 1e-8)


def normalize_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Apply rolling z-score normalization to the 12 normalizable indicator columns.
    `realized_vol_raw` passes through unchanged; `close_return` (log return) is added.

    Returns a 14-column DataFrame:
        [12 normalized indicators] + [realized_vol_raw] + [close_return]
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    window: int = config["normalization"]["window_hours"]
    min_periods: int = config["normalization"]["min_periods"]

    out = df.copy()

    # Compute log return from close before dropping price columns
    out["close_return"] = np.log(out["close"] / out["close"].shift(1))

    # Rolling z-score every normalizable indicator; realized_vol_raw is left as-is.
    for col in NORMALIZE_COLS:
        if col in NO_NORMALIZE:
            continue
        out[col] = _rolling_zscore(out[col], window=window, min_periods=min_periods)

    # Drop raw price and volume columns — not needed after feature/return computation
    out.drop(columns=["open", "high", "low", "close", "volume"], inplace=True)

    # Drop NaN rows from rolling-window warmup and the first log-return row
    rows_before_drop = len(out)
    out.dropna(subset=FEATURE_COLS, inplace=True)
    rows_after = len(out)
    dropped = rows_before_drop - rows_after
    logger.info(
        f"Normalization: dropped {dropped:,} NaN rows from rolling warmup "
        f"({dropped / rows_before_drop * 100:.1f}%). Remaining: {rows_after:,}"
    )

    expected_feature_cols = set(FEATURE_COLS)
    assert set(out.columns) == expected_feature_cols, (
        f"Column mismatch after normalization. Got: {sorted(out.columns)}, "
        f"expected: {sorted(expected_feature_cols)}"
    )
    assert len(out.columns) == 14, f"Expected 14 feature columns, got {len(out.columns)}"

    return out


def normalize_and_save(
    df: pd.DataFrame, symbol: str, config: dict[str, Any]
) -> pd.DataFrame:
    """Convenience wrapper that normalizes and saves to parquet."""
    out = normalize_features(df, config)
    out_path = PROCESSED_DIR / f"{symbol}_normalized.parquet"
    out.to_parquet(out_path, engine="pyarrow")
    logger.success(f"[{symbol}] Normalized → {out_path} ({len(out):,} rows, {len(out.columns)} cols)")
    return out
