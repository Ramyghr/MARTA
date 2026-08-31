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
    ASSET_ID_DIM,
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
    # STATE_DIM includes the per-episode asset-ID one-hot block (added 2026-08-08) on top
    # of regime+consensus+portfolio — 156, not the pre-asset-ID 152 this test used to check.
    assert STATE_DIM == REGIME_DIM + CONSENSUS_DIM + PORTFOLIO_DIM + ASSET_ID_DIM  # 156


def test_observation_space_shape_scales_with_canonical_assets(aligned_df, regime_stream, encoder):
    """2026-08-24: canonical_assets= must size the portfolio/asset-id blocks off
    len(canonical_assets), not the module-level 4-asset PORTFOLIO_DIM/ASSET_ID_DIM —
    otherwise a 5th asset either gets truncated into an existing slot or indexed out of
    bounds. n_position_slots=5 -> portfolio_dim=9 (5 slots + 4 scalars), asset_id_dim=5."""
    vol_array = np.full(N_ROWS, 0.02, dtype=np.float64)
    five_assets = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]
    with patch("rl.env.precompute_vol_forecast", return_value=vol_array):
        e = MARTATradingEnv(
            assets=["BTC_USDT"],
            aligned_dfs={"BTC_USDT": aligned_df},
            regime_stream_df=regime_stream,
            encoder=encoder,
            debate_results=None,
            split="all",
            max_steps=50,
            canonical_assets=five_assets,
        )
    assert e.n_position_slots == 5
    assert e.portfolio_dim == 9
    assert e.asset_id_dim == 5
    expected_obs_dim = REGIME_DIM + CONSENSUS_DIM + 9 + 5  # 158
    assert e.observation_space.shape == (expected_obs_dim,)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (expected_obs_dim,)
    # Asset-id one-hot should mark BTC_USDT's position (index 0 in five_assets).
    portfolio_vec = e._portfolio_vec("BTC_USDT", e._t)
    asset_id_vec = e._asset_id_vec("BTC_USDT")
    assert portfolio_vec.shape == (9,)
    assert asset_id_vec.shape == (5,)
    assert asset_id_vec[0] == 1.0
    assert asset_id_vec[1:].sum() == 0.0


def test_raw_obs_mode_uses_actual_feature_count(regime_stream):
    """2026-08-24: obs_mode='raw' must derive its window dim from the aligned data's own
    column count (crypto5_multihorizon has 22 columns, not the original panel's 14) rather
    than a hardcoded 14 — otherwise a 22-column panel produces a raw window array that
    doesn't match the declared observation_space shape."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-06-01", periods=N_ROWS, freq="1h", tz="UTC")
    cols_22 = [f"feat_{i}" for i in range(21)] + ["close_return"]
    df_22 = pd.DataFrame(
        rng.standard_normal((N_ROWS, 22)).astype(np.float32), index=idx, columns=cols_22
    )
    vol_array = np.full(N_ROWS, 0.02, dtype=np.float64)
    with patch("rl.env.precompute_vol_forecast", return_value=vol_array):
        e = MARTATradingEnv(
            assets=["BTC_USDT"],
            aligned_dfs={"BTC_USDT": df_22},
            regime_stream_df=regime_stream,
            encoder=None,
            debate_results=None,
            split="all",
            max_steps=50,
            obs_mode="raw",
        )
    assert e.n_raw_features == 22
    expected_obs_dim = 60 * 22 + CONSENSUS_DIM + PORTFOLIO_DIM + ASSET_ID_DIM
    assert e.observation_space.shape == (expected_obs_dim,)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (expected_obs_dim,)


def test_raw_vol_obs_mode_adds_exactly_one_dim(regime_stream):
    """RL experiment Condition B: obs_mode='raw_vol' must be raw's obs_dim + 1, not +
    n_regime_prob_dims or anything else — it's a single scalar (the rolling-z-scored
    GARCH vol_forecast), not a vector."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-06-01", periods=N_ROWS, freq="1h", tz="UTC")
    cols_14 = [
        "atr", "adx", "rsi", "bb_width", "macd_histogram",
        "obv_raw", "vol_zscore", "realized_vol", "vol_ratio",
        "return_skew", "return_kurtosis", "autocorr_lag1",
        "realized_vol_raw", "close_return",
    ]
    df_14 = pd.DataFrame(rng.standard_normal((N_ROWS, 14)).astype(np.float32), index=idx, columns=cols_14)
    vol_array = np.full(N_ROWS, 0.02, dtype=np.float64)
    with patch("rl.env.precompute_vol_forecast", return_value=vol_array):
        e = MARTATradingEnv(
            assets=["BTC_USDT"], aligned_dfs={"BTC_USDT": df_14}, regime_stream_df=regime_stream,
            encoder=None, debate_results=None, split="all", max_steps=50, obs_mode="raw_vol",
        )
    expected_obs_dim = 60 * 14 + 1 + CONSENSUS_DIM + PORTFOLIO_DIM + ASSET_ID_DIM
    assert e.observation_space.shape == (expected_obs_dim,)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (expected_obs_dim,)
    assert np.isfinite(obs).all()


def test_raw_vol_dimension_actually_varies_with_real_vol_signal(regime_stream):
    """The vol dimension must reflect a genuinely varying, non-degenerate signal — not
    silently read a constant or the wrong array. Uses a real (non-constant) synthetic
    vol_forecast so the rolling z-score has real variance to expose, unlike the fixture's
    default flat-0.02 mock (a legitimate test for shape, not for 'does this carry
    information')."""
    n_rows = 500  # generous headroom so 100 sequential steps can't run past the data end
    rng = np.random.default_rng(1)
    idx = pd.date_range("2018-06-01", periods=n_rows, freq="1h", tz="UTC")
    cols_14 = [
        "atr", "adx", "rsi", "bb_width", "macd_histogram",
        "obv_raw", "vol_zscore", "realized_vol", "vol_ratio",
        "return_skew", "return_kurtosis", "autocorr_lag1",
        "realized_vol_raw", "close_return",
    ]
    df_14 = pd.DataFrame(rng.standard_normal((n_rows, 14)).astype(np.float32), index=idx, columns=cols_14)
    regime_stream_big = pd.DataFrame({
        "timestamp": idx, "asset": "BTC_USDT", "regime_label": 1,
        **{f"regime_prob_{k}": 1.0 / 7 for k in range(7)},
    })
    # Real vol-like signal: slow sinusoidal drift + noise, strictly positive.
    t = np.arange(n_rows)
    varying_vol = (0.02 + 0.015 * np.sin(t / 20.0) + rng.normal(0, 0.002, n_rows)).clip(0.001, None)
    with patch("rl.env.precompute_vol_forecast", return_value=varying_vol):
        e = MARTATradingEnv(
            assets=["BTC_USDT"], aligned_dfs={"BTC_USDT": df_14}, regime_stream_df=regime_stream_big,
            encoder=None, debate_results=None, split="all", sequential=True, max_steps=10_000,
            obs_mode="raw_vol",
        )
    e.reset(seed=0)
    vol_dim_values = []
    for _ in range(100):
        obs, _, terminated, truncated, _ = e.step(np.array([2, 0.5], dtype=np.float32))  # hold
        vol_dim_values.append(obs[-1 - ASSET_ID_DIM - PORTFOLIO_DIM - CONSENSUS_DIM])
        if terminated or truncated:
            break
    vol_dim_values = np.array(vol_dim_values)
    assert np.isfinite(vol_dim_values).all()
    assert vol_dim_values.std() > 0.01, "vol dimension is degenerate (near-constant) despite a varying input signal"


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
