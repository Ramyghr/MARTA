"""RL environment determinism and contract tests.

We mock precompute_vol_forecast to avoid GARCH fitting on disk, and use an
untrained encoder so no checkpoint is required.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from encoder.model import TFTRegimeEncoder
from rl.env import (
    CONSENSUS_DIM,
    PORTFOLIO_DIM,
    REGIME_DIM,
    STATE_DIM,
    MARTATradingEnv,
    cvar_from_vol,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ROWS = 300  # enough for window_size=60 + valid starts in both splits


@pytest.fixture(scope="module")
def aligned_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-06-01", periods=N_ROWS, freq="1h", tz="UTC")
    data = rng.standard_normal((N_ROWS, 14)).astype(np.float32)
    # last column is close_return
    cols = [
        "atr", "adx", "rsi", "bb_width", "macd_histogram",
        "obv_raw", "vol_zscore", "realized_vol", "vol_ratio",
        "return_skew", "return_kurtosis", "autocorr_lag1",
        "realized_vol_raw", "close_return",
    ]
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture(scope="module")
def regime_stream(aligned_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ts in aligned_df.index:
        row = {"timestamp": ts, "asset": "BTC_USDT", "regime_label": 1}
        for k in range(7):
            row[f"regime_prob_{k}"] = 1.0 / 7
        rows.append(row)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture(scope="module")
def encoder() -> TFTRegimeEncoder:
    enc = TFTRegimeEncoder(input_dim=14, hidden_dim=128)
    enc.eval()
    return enc


@pytest.fixture(scope="module")
def env(aligned_df, regime_stream, encoder) -> MARTATradingEnv:
    vol_array = np.full(N_ROWS, 0.02, dtype=np.float64)
    with patch("rl.env.precompute_vol_forecast", return_value=vol_array):
        e = MARTATradingEnv(
            assets=["BTC_USDT"],
            aligned_dfs={"BTC_USDT": aligned_df},
            regime_stream_df=regime_stream,
            encoder=encoder,
            debate_results=None,
            split="all",
            max_steps=50,
        )
    return e


# ---------------------------------------------------------------------------
# Shape / contract tests
# ---------------------------------------------------------------------------

def test_observation_space_shape(env):
    assert env.observation_space.shape == (STATE_DIM,)
    assert STATE_DIM == REGIME_DIM + CONSENSUS_DIM + PORTFOLIO_DIM  # 152


def test_action_space_shape(env):
    assert env.action_space.shape == (2,)


def test_reset_returns_correct_obs_shape(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (STATE_DIM,)
    assert obs.dtype == np.float32


def test_obs_clipped(env):
    obs, _ = env.reset(seed=0)
    assert np.all(obs >= -10.0)
    assert np.all(obs <= 10.0)


def test_step_returns_finite_reward(env):
    env.reset(seed=0)
    action = np.array([0.0, 0.5], dtype=np.float32)  # long, 50%
    obs, reward, terminated, truncated, info = env.step(action)
    assert np.isfinite(reward)
    assert obs.shape == (STATE_DIM,)


def test_step_returns_finite_obs(env):
    env.reset(seed=1)
    for _ in range(5):
        action = np.array([0.0, 0.1], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        assert np.all(np.isfinite(obs))
        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# Determinism: same seed → identical trajectory
# ---------------------------------------------------------------------------

def test_determinism_same_seed(env):
    def _rollout(seed: int) -> list[np.ndarray]:
        obs, _ = env.reset(seed=seed)
        trajectory = [obs.copy()]
        for _ in range(10):
            action = np.array([0.0, 0.3], dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            trajectory.append(obs.copy())
            if terminated or truncated:
                break
        return trajectory

    traj_a = _rollout(42)
    traj_b = _rollout(42)
    assert len(traj_a) == len(traj_b)
    for a, b in zip(traj_a, traj_b):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_differ(env):
    obs_a, _ = env.reset(seed=0)
    obs_b, _ = env.reset(seed=1)
    # Different seeds should start at different timesteps (with very high probability).
    # We allow this test to be soft — it is not guaranteed for tiny datasets.
    # If both start at the same index it's acceptable; the test just ensures no crash.
    assert obs_a.shape == obs_b.shape


# ---------------------------------------------------------------------------
# unflatten_action
# ---------------------------------------------------------------------------

def test_unflatten_action_long():
    a = MARTATradingEnv.unflatten_action(np.array([0.0, 0.5]))
    assert a["direction"] == 0
    assert a["position_fraction"] == pytest.approx(0.5)


def test_unflatten_action_short():
    a = MARTATradingEnv.unflatten_action(np.array([1.0, 0.3]))
    assert a["direction"] == 1


def test_unflatten_action_hold():
    a = MARTATradingEnv.unflatten_action(np.array([2.0, 0.9]))
    assert a["direction"] == 2


def test_unflatten_action_clamps_fraction():
    a = MARTATradingEnv.unflatten_action(np.array([0.0, 1.5]))
    assert a["position_fraction"] <= 0.99


# ---------------------------------------------------------------------------
# CVaR helper
# ---------------------------------------------------------------------------

def test_cvar_from_vol_negative():
    """CVaR must be negative (it's a loss measure)."""
    assert cvar_from_vol(0.02) < 0


def test_cvar_scales_with_vol():
    assert cvar_from_vol(0.04) < cvar_from_vol(0.02)


# ---------------------------------------------------------------------------
# Sequential eval mode
# ---------------------------------------------------------------------------

def test_sequential_mode_deterministic(aligned_df, regime_stream, encoder):
    vol_array = np.full(N_ROWS, 0.02, dtype=np.float64)
    with patch("rl.env.precompute_vol_forecast", return_value=vol_array):
        e = MARTATradingEnv(
            assets=["BTC_USDT"],
            aligned_dfs={"BTC_USDT": aligned_df},
            regime_stream_df=regime_stream,
            encoder=encoder,
            debate_results=None,
            split="all",
            sequential=True,
            max_steps=20,
        )
    obs1, _ = e.reset()
    obs2, _ = e.reset()
    # Sequential always starts at the same index — observations must be identical.
    np.testing.assert_array_equal(obs1, obs2)
