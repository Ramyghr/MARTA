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

# Anchored to this file's location, not the caller's cwd — see pipeline/features.py's
# identical fix for why (a real PermissionError surfaced when called with a non-root cwd).
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

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


# ---------------------------------------------------------------------------
# Multi-horizon experiment (Phase 2, regime-layer audit, 2026-08-19) — additive only,
# normalize_features()/FEATURE_COLS/normalize_and_save() above are untouched.
#
# Tests whether the 720h(30d) rolling z-score is silently capping how long a "regime"
# the HMM discovers can persist (measured production dwell times were 13-20h — far
# shorter than a classical multi-week market regime). See regime/build_crypto5_features.py.
# ---------------------------------------------------------------------------

# Vol-shape features that get extra, longer normalization horizons — not applied to
# every column: RSI/MACD/skew/kurtosis/autocorr are oscillator-like and not primarily
# about how long a volatility regime has been running.
EXTRA_HORIZON_COLS = ["realized_vol", "atr", "bb_width", "vol_ratio"]
EXTRA_HORIZONS = [(2160, "90d"), (4320, "180d")]  # (window_hours, column-suffix)


def normalize_features_multihorizon(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """normalize_features()'s 14 columns, plus a 90d and 180d rolling z-score (in
    addition to the production 720h/30d one) of each column in EXTRA_HORIZON_COLS.

    Deliberately reuses the SAME min_periods as the production window
    (config["normalization"]["min_periods"], currently 168h) rather than scaling it up
    with the window — so these longer horizons don't drop any extra warmup rows beyond
    what normalize_features() already drops: every column becomes non-NaN at the same
    point, no extra row-alignment logic needed.

    Returns 14 + len(EXTRA_HORIZON_COLS)*len(EXTRA_HORIZONS) columns (22 with the
    current 4 cols x 2 horizons).
    """
    out = normalize_features(df, config)  # existing 14-column production path, unchanged
    min_periods: int = config["normalization"]["min_periods"]

    extra_cols: list[str] = []
    for col in EXTRA_HORIZON_COLS:
        raw = df[col]  # pre-normalization raw indicator value — df is untouched by normalize_features
        for window_hours, suffix in EXTRA_HORIZONS:
            new_col = f"{col}_z{suffix}"
            out[new_col] = _rolling_zscore(raw, window=window_hours, min_periods=min_periods).reindex(out.index)
            extra_cols.append(new_col)

    rows_before_drop = len(out)
    out.dropna(subset=extra_cols, inplace=True)
    dropped = rows_before_drop - len(out)
    if dropped:
        logger.info(f"Multi-horizon normalization: dropped {dropped:,} additional NaN rows.")

    return out
