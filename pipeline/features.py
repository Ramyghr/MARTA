"""
pipeline/features.py
Computes all 13 technical indicators for a single asset DataFrame.

Layer 1 v2 changes:
  - obv_raw: raw OBV cumsum only (NO z-score here — normalize.py owns the single
    720h z-score; the old 168h z-score was a double-normalization bug).
  - realized_vol_raw: 14th indicator slot, the UNNORMALIZED annualized realized
    vol. It is excluded from z-scoring in normalize.py so it preserves absolute
    magnitude — critical for crash detection (March 2020 reads far above normal,
    a signal the rolling z-score otherwise compresses away).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ta.momentum as ta_mom
import ta.trend as ta_trend
import ta.volatility as ta_vol

from utils.logger import logger

# Anchored to this file's location, not the caller's cwd (Path("data/processed") alone
# resolved wherever the process happened to be launched from — silently wrong, and
# occasionally a real PermissionError, when the caller's cwd isn't the project root,
# e.g. the dashboard backend's uvicorn process running from dashboard/backend/).
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

INDICATOR_COLS = [
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
    "realized_vol_raw",
]

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _rolling_zscore(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    return (series - mean) / (std + 1e-8)


def _compute_atr(df: pd.DataFrame) -> pd.Series:
    indicator = ta_vol.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14, fillna=False
    )
    return indicator.average_true_range()


def _compute_adx(df: pd.DataFrame) -> pd.Series:
    indicator = ta_trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14, fillna=False
    )
    return indicator.adx()


def _compute_rsi(df: pd.DataFrame) -> pd.Series:
    indicator = ta_mom.RSIIndicator(close=df["close"], window=14, fillna=False)
    return indicator.rsi()


def _compute_bb_width(df: pd.DataFrame) -> pd.Series:
    indicator = ta_vol.BollingerBands(
        close=df["close"], window=20, window_dev=2, fillna=False
    )
    upper = indicator.bollinger_hband()
    lower = indicator.bollinger_lband()
    middle = indicator.bollinger_mavg()
    return (upper - lower) / (middle + 1e-8)


def _compute_macd_histogram(df: pd.DataFrame) -> pd.Series:
    indicator = ta_trend.MACD(
        close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False
    )
    return indicator.macd_diff()


def _compute_obv_raw(df: pd.DataFrame) -> pd.Series:
    """
    OBV: cumulative volume signed by price direction. Raw cumsum ONLY — the
    single 720h z-score is applied later in normalize.py (no double-normalization).
    """
    direction = np.sign(df["close"].diff())
    direction.iloc[0] = 0.0
    return (df["volume"] * direction).cumsum()


def _compute_vol_zscore(df: pd.DataFrame, window: int = 24) -> pd.Series:
    return _rolling_zscore(df["volume"], window=window, min_periods=max(1, window // 4))


def _compute_realized_vol(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """Rolling std of log returns, annualized to hourly frequency."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    annualization = np.sqrt(24 * 365)
    return log_ret.rolling(window=window, min_periods=max(1, window // 4)).std() * annualization


def _compute_realized_vol_raw(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    Unnormalized annualized realized vol — NOT z-scored anywhere in the pipeline.
    Preserves absolute volatility magnitude (March 2020 crash >> normal), which the
    rolling z-score on `realized_vol` destroys. Full 24h window required (no warmup
    short-circuit) so the magnitude is comparable across the whole series.
    """
    log_returns = np.log(df["close"] / df["close"].shift(1))
    return log_returns.rolling(window).std() * np.sqrt(24 * 365)


def _compute_vol_ratio(df: pd.DataFrame) -> pd.Series:
    """realized_vol_24h / realized_vol_168h, clipped to [0.1, 10]."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    annualization = np.sqrt(24 * 365)
    rv24 = log_ret.rolling(window=24, min_periods=6).std() * annualization
    rv168 = log_ret.rolling(window=168, min_periods=42).std() * annualization
    ratio = rv24 / (rv168 + 1e-8)
    return ratio.clip(lower=0.1, upper=10.0)


def _compute_return_skew(df: pd.DataFrame, window: int = 48) -> pd.Series:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    return log_ret.rolling(window=window, min_periods=max(1, window // 4)).skew()


def _compute_return_kurtosis(df: pd.DataFrame, window: int = 48) -> pd.Series:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    return log_ret.rolling(window=window, min_periods=max(1, window // 4)).kurt()


def _compute_autocorr_lag1(df: pd.DataFrame, window: int = 24) -> pd.Series:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    return log_ret.rolling(window=window, min_periods=max(1, window // 2)).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )


def compute_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Compute all 13 technical indicators for the given OHLCV DataFrame.
    Returns a DataFrame with original OHLCV columns + 13 indicator columns (18 total).
    Drops NaN rows from rolling warmup.
    Saves to data/processed/{symbol}_features.parquet.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if df.empty:
        raise ValueError(f"[{symbol}] Empty DataFrame passed to compute_features")

    # Ensure float dtype for all OHLCV
    df = df[OHLCV_COLS].astype(float).copy()
    rows_before = len(df)

    logger.info(f"[{symbol}] Computing features on {rows_before:,} rows")

    out = df.copy()
    out["atr"] = _compute_atr(df)
    out["adx"] = _compute_adx(df)
    out["rsi"] = _compute_rsi(df)
    out["bb_width"] = _compute_bb_width(df)
    out["macd_histogram"] = _compute_macd_histogram(df)
    out["obv_raw"] = _compute_obv_raw(df)
    out["vol_zscore"] = _compute_vol_zscore(df)
    out["realized_vol"] = _compute_realized_vol(df)
    out["vol_ratio"] = _compute_vol_ratio(df)
    out["return_skew"] = _compute_return_skew(df)
    out["return_kurtosis"] = _compute_return_kurtosis(df)
    out["autocorr_lag1"] = _compute_autocorr_lag1(df)
    out["realized_vol_raw"] = _compute_realized_vol_raw(df)

    # Genuine 1-minute → hourly data (histdata) has no constant-return windows, so the
    # old daily-upsample fillna(0) special-case is gone. Standard dropna across all
    # indicators removes only the rolling-warmup rows at the start of each series.
    rows_before_drop = len(out)
    out.dropna(subset=INDICATOR_COLS + ["close"], inplace=True)
    rows_after = len(out)
    dropped = rows_before_drop - rows_after
    pct_dropped = dropped / rows_before_drop * 100 if rows_before_drop > 0 else 0

    logger.info(
        f"[{symbol}] Dropped {dropped:,} NaN rows ({pct_dropped:.1f}%) from rolling warmup. "
        f"Remaining: {rows_after:,}"
    )

    expected_cols = set(OHLCV_COLS + INDICATOR_COLS)
    actual_cols = set(out.columns)
    assert actual_cols == expected_cols, (
        f"[{symbol}] Column mismatch. Got: {sorted(actual_cols)}, "
        f"expected: {sorted(expected_cols)}"
    )
    assert len(out.columns) == 18, f"[{symbol}] Expected 18 columns, got {len(out.columns)}"

    out_path = PROCESSED_DIR / f"{symbol}_features.parquet"
    out.to_parquet(out_path, engine="pyarrow")
    logger.success(f"[{symbol}] Features saved → {out_path} ({rows_after:,} rows, 18 cols)")

    return out
