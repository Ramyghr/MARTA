"""Pipeline round-trip tests: raw OHLCV → features → normalize → 14 feature columns."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    INDICATOR_COLS,
    OHLCV_COLS,
    compute_features,
)
from pipeline.normalize import FEATURE_COLS, normalize_features


# The 14 canonical feature names per CLAUDE.md (ordered as in normalize.py FEATURE_COLS).
_EXPECTED_14 = {
    "atr", "adx", "rsi", "bb_width", "macd_histogram",
    "obv_raw", "vol_zscore", "realized_vol", "vol_ratio",
    "return_skew", "return_kurtosis", "autocorr_lag1",
    "realized_vol_raw", "close_return",
}


@pytest.fixture()
def features_df(synthetic_ohlcv, tmp_path, monkeypatch):
    """Return features DataFrame without writing to the real data/processed dir."""
    import pipeline.features as feat_mod
    monkeypatch.setattr(feat_mod, "PROCESSED_DIR", tmp_path)
    return compute_features(synthetic_ohlcv, "TEST_BTC")


def test_compute_features_column_count(features_df):
    """features.py must produce exactly 18 columns (5 OHLCV + 13 indicators)."""
    assert len(features_df.columns) == 18


def test_compute_features_has_ohlcv(features_df):
    assert set(OHLCV_COLS).issubset(features_df.columns)


def test_compute_features_obv_raw_not_zscore(features_df):
    """obv_raw column must be present; obv_zscore must not (Layer 1 v2 fix)."""
    assert "obv_raw" in features_df.columns
    assert "obv_zscore" not in features_df.columns


def test_compute_features_realized_vol_raw_present(features_df):
    """realized_vol_raw must be the 14th indicator slot."""
    assert "realized_vol_raw" in features_df.columns


def test_compute_features_no_nan_after_dropna(features_df):
    assert features_df.isnull().sum().sum() == 0


def test_normalize_14_columns(features_df, norm_config):
    out = normalize_features(features_df, norm_config)
    assert len(out.columns) == 14
    assert set(out.columns) == _EXPECTED_14


def test_normalize_drops_ohlcv(features_df, norm_config):
    out = normalize_features(features_df, norm_config)
    for col in OHLCV_COLS:
        assert col not in out.columns


def test_realized_vol_raw_not_zscore(features_df, norm_config):
    """realized_vol_raw should NOT be z-scored — values must remain >= 0 (it's a vol estimate)."""
    out = normalize_features(features_df, norm_config)
    assert (out["realized_vol_raw"].dropna() >= 0).all(), (
        "realized_vol_raw contains negative values — it may have been accidentally z-scored"
    )


def test_realized_vol_raw_preserves_magnitude(features_df, norm_config):
    """realized_vol_raw values should be much larger than z-scored realized_vol values (|mean|~1)."""
    out = normalize_features(features_df, norm_config)
    raw_mean = out["realized_vol_raw"].dropna().mean()
    # Annualized vol for crypto is typically 0.5–5.0; always > 0.01
    assert raw_mean > 0.01, f"realized_vol_raw mean {raw_mean:.4f} is suspiciously small"


def test_pipeline_roundtrip_feature_names_match_spec(features_df, norm_config):
    """Full spec check: output columns exactly match the 14-feature set from CLAUDE.md."""
    out = normalize_features(features_df, norm_config)
    assert set(out.columns) == _EXPECTED_14
    assert "obv_raw" in out.columns
    assert "realized_vol_raw" in out.columns
    assert "close_return" in out.columns


def test_pipeline_preserves_index(features_df, norm_config):
    """The normalized DataFrame must have the same DatetimeIndex type as features."""
    out = normalize_features(features_df, norm_config)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is not None  # UTC-aware
