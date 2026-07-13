"""Shared fixtures and sys.path setup for the MARTA test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Make the project root importable regardless of where pytest is invoked.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Synthetic OHLCV fixture (800 hourly rows — enough for all rolling windows)
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 800
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    close = 10_000.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
    noise = rng.uniform(0.999, 1.001, n)
    df = pd.DataFrame(
        {
            "open":   close * rng.uniform(0.998, 1.002, n),
            "high":   close * np.maximum(1.0, noise + rng.uniform(0.001, 0.003, n)),
            "low":    close * np.minimum(1.0, noise - rng.uniform(0.001, 0.003, n)),
            "close":  close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )
    # Ensure OHLCV consistency
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"]  = df[["open", "low",  "close"]].min(axis=1)
    return df


@pytest.fixture()
def norm_config() -> dict:
    return {
        "normalization": {
            "window_hours": 720,
            "min_periods": 168,
        }
    }


# ---------------------------------------------------------------------------
# Minimal regime stream fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_regime_stream(synthetic_ohlcv: pd.DataFrame) -> pd.DataFrame:
    idx = synthetic_ohlcv.index
    n = len(idx)
    rows = []
    for ts in idx:
        row = {
            "timestamp": ts,
            "asset": "BTC_USDT",
            "regime_label": 1,
        }
        for k in range(7):
            row[f"regime_prob_{k}"] = 1.0 / 7
        rows.append(row)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Minimal TFTRegimeEncoder fixture (untrained, random weights)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_encoder() -> "TFTRegimeEncoder":
    from encoder.model import TFTRegimeEncoder
    enc = TFTRegimeEncoder(input_dim=14, hidden_dim=128)
    enc.eval()
    return enc
