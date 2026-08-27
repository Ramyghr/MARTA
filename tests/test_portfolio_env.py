"""Tests for rl/portfolio_env.py — MultiAssetPortfolioEnv (Step 3 of the pre-RL audit's
remediation plan). Mirrors tests/test_rl_env.py's fixture conventions (synthetic aligned
data, untrained encoder, flat regime stream) but for a joint multi-asset panel.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from configs.splits import TestSetContaminationError
from encoder.model import TFTRegimeEncoder
from rl import obs_builder
from rl.portfolio_env import MultiAssetPortfolioEnv

N_ROWS = 300
ASSETS = ["ASSET_A", "ASSET_B", "ASSET_C"]
COLS = [
    "atr", "adx", "rsi", "bb_width", "macd_histogram",
    "obv_raw", "vol_zscore", "realized_vol", "vol_ratio",
    "return_skew", "return_kurtosis", "autocorr_lag1",
    "realized_vol_raw", "close_return",
]


@pytest.fixture(scope="module")
def aligned_dfs() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2018-06-01", periods=N_ROWS, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    out = {}
    for a in ASSETS:
        data = rng.standard_normal((N_ROWS, len(COLS))).astype(np.float32)
        out[a] = pd.DataFrame(data, index=idx, columns=COLS)
    return out


@pytest.fixture(scope="module")
def regime_stream(aligned_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = aligned_dfs[ASSETS[0]].index
    rows = []
    for a in ASSETS:
        for ts in idx:
            row = {"timestamp": ts, "asset": a, "regime_label": 1}
            for k in range(6):
                row[f"regime_prob_{k}"] = 1.0 / 6
            rows.append(row)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture(scope="module")
def encoder() -> TFTRegimeEncoder:
    enc = TFTRegimeEncoder(input_dim=len(COLS), hidden_dim=128)
    enc.eval()
    return enc


def _make_env(aligned_dfs, regime_stream, encoder=None, **kwargs) -> MultiAssetPortfolioEnv:
    vol_array = np.full(N_ROWS, 0.02, dtype=np.float64)
    defaults = dict(
        assets=ASSETS,
        aligned_dfs=aligned_dfs,
        regime_stream_df=regime_stream,
        encoder=encoder,
        debate_results=None,
        split="all",
        max_steps=50,
        obs_mode="raw",
    )
    defaults.update(kwargs)
    with patch("rl.portfolio_env.precompute_vol_forecast", return_value=vol_array):
        return MultiAssetPortfolioEnv(**defaults)


@pytest.fixture()
def env(aligned_dfs, regime_stream) -> MultiAssetPortfolioEnv:
    return _make_env(aligned_dfs, regime_stream)


# ---------------------------------------------------------------------------
# Construction / shape contracts
# ---------------------------------------------------------------------------

def test_observation_space_shape(env):
    n = len(ASSETS)
    per_asset = 60 * len(COLS) + obs_builder.CONSENSUS_DIM
    tail = (n + 1) + 2
    assert env.observation_space.shape == (n * per_asset + tail,)
    assert env.per_asset_block_dim == per_asset
    assert env.tail_dim == n + 1 + 2


def test_action_space_shape(env):
    assert env.action_space.shape == (len(ASSETS) + 1,)


def test_raw_regime_obs_mode_adds_regime_dims(aligned_dfs, regime_stream):
    e = _make_env(aligned_dfs, regime_stream, obs_mode="raw_regime")
    per_asset = 60 * len(COLS) + 6 + obs_builder.CONSENSUS_DIM  # +6 regime_prob dims
    assert e.per_asset_block_dim == per_asset


def test_requires_at_least_two_assets(aligned_dfs, regime_stream):
    with pytest.raises(ValueError):
        _make_env(aligned_dfs, regime_stream, assets=["ASSET_A"])


def test_mismatched_index_raises(aligned_dfs, regime_stream):
    bad = dict(aligned_dfs)
    shifted_idx = bad["ASSET_B"].index + pd.Timedelta(hours=1)
    bad["ASSET_B"] = bad["ASSET_B"].set_axis(shifted_idx)
    with pytest.raises(ValueError, match="index differs"):
        _make_env(bad, regime_stream)


def test_asset_missing_from_aligned_dfs_raises(aligned_dfs, regime_stream):
    with pytest.raises(ValueError, match="missing from aligned_dfs"):
        _make_env(aligned_dfs, regime_stream, assets=ASSETS + ["ASSET_D"])


# ---------------------------------------------------------------------------
# reset() / weights
# ---------------------------------------------------------------------------

def test_reset_starts_fully_in_cash(env):
    obs, info = env.reset(seed=0)
    assert info["weights"]["cash"] == pytest.approx(1.0)
    for a in ASSETS:
        assert info["weights"][a] == pytest.approx(0.0)
    assert obs.shape == env.observation_space.shape


def test_deterministic_reset_same_seed(env):
    _, info1 = env.reset(seed=42)
    t1 = env._t
    _, info2 = env.reset(seed=42)
    t2 = env._t
    assert t1 == t2


def test_weights_from_action_sums_to_one(env):
    rng = np.random.default_rng(1)
    for _ in range(10):
        logits = rng.standard_normal(len(ASSETS) + 1) * 5
        weights = env.weights_from_action(logits)
        assert weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert (weights >= 0).all()


def test_step_weights_sum_to_one(env):
    env.reset(seed=0)
    logits = np.array([2.0, -1.0, 0.5, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(logits)
    total = info["weights"]["cash"] + sum(info["weights"][a] for a in ASSETS)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_strong_cash_logit_gives_near_full_cash_allocation(env):
    env.reset(seed=0)
    logits = np.array([-10.0, -10.0, -10.0, 10.0], dtype=np.float32)
    _, _, _, _, info = env.step(logits)
    assert info["weights"]["cash"] > 0.999


# ---------------------------------------------------------------------------
# Transaction cost
# ---------------------------------------------------------------------------

def test_transaction_cost_scales_with_turnover(env):
    env.reset(seed=0)
    small_move = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, _, info_small = env.step(small_move)

    env.reset(seed=0)
    big_move = np.array([10.0, -10.0, -10.0, -10.0], dtype=np.float32)
    _, _, _, _, info_big = env.step(big_move)

    assert info_big["turnover_cost_fraction"] > info_small["turnover_cost_fraction"]


def test_zero_turnover_when_action_matches_current_weights():
    """Staying in cash (the reset default) with a strong cash logit should incur ~0 cost."""
    idx = pd.date_range("2018-06-01", periods=N_ROWS, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    dfs = {a: pd.DataFrame(rng.standard_normal((N_ROWS, len(COLS))).astype(np.float32), index=idx, columns=COLS) for a in ASSETS}
    regime = pd.DataFrame({"timestamp": [], "asset": [], "regime_label": []})
    e = _make_env(dfs, regime)
    e.reset(seed=0)
    logits = np.array([-10.0, -10.0, -10.0, 10.0], dtype=np.float32)
    _, _, _, _, info = e.step(logits)
    assert info["turnover_cost_fraction"] < 1e-3


# ---------------------------------------------------------------------------
# Observation includes previous weights (policy memory of its own allocation)
# ---------------------------------------------------------------------------

def test_observation_tail_reflects_previous_weights(env):
    env.reset(seed=0)
    logits = np.array([5.0, -5.0, -5.0, -5.0], dtype=np.float32)  # heavy ASSET_A
    obs, _, _, _, info = env.step(logits)

    per_asset_len = len(ASSETS) * env.per_asset_block_dim
    tail = obs[per_asset_len:]
    weight_tail = tail[: len(ASSETS) + 1]
    np.testing.assert_allclose(weight_tail[0], info["weights"]["ASSET_A"], atol=1e-5)
    assert weight_tail[0] > weight_tail[1]  # ASSET_A weight >> ASSET_B weight


# ---------------------------------------------------------------------------
# Episode control
# ---------------------------------------------------------------------------

def test_max_steps_truncates_training_episode(aligned_dfs, regime_stream):
    e = _make_env(aligned_dfs, regime_stream, max_steps=3, sequential=False)
    e.reset(seed=0)
    logits = np.zeros(len(ASSETS) + 1, dtype=np.float32)
    truncated = False
    for _ in range(10):
        _, _, terminated, truncated, _ = e.step(logits)
        if terminated or truncated:
            break
    assert truncated or terminated


def test_split_contamination_guard_raises(aligned_dfs):
    """This guard (identical to MARTATradingEnv's own _prepare_asset check) is a safety
    net for _split_mask ITSELF being broken — configs/splits.py's TestSetContamination
    Error docstring says it verifies "the resulting slice actually respects the boundary
    — not just that the slicing code was written correctly." Under a CORRECT
    _split_mask, split='train' can never include a post-TEST_START timestamp by
    construction (every valid train row already satisfies ts <= TRAIN_END < TEST_START),
    so this test simulates a broken mask (always-True) on an index that genuinely spans
    past TEST_START, proving the safety net actually fires rather than being dead code.
    """
    idx = pd.date_range("2024-12-25", periods=N_ROWS, freq="1h", tz="UTC")  # crosses TEST_START
    rng = np.random.default_rng(0)
    dfs = {a: pd.DataFrame(rng.standard_normal((N_ROWS, len(COLS))).astype(np.float32), index=idx, columns=COLS) for a in ASSETS}
    regime = pd.DataFrame({"timestamp": [], "asset": [], "regime_label": []})
    with patch("rl.portfolio_env.precompute_vol_forecast", return_value=np.full(N_ROWS, 0.02)), \
         patch.object(MultiAssetPortfolioEnv, "_split_mask", return_value=np.ones(N_ROWS, dtype=bool)):
        with pytest.raises(TestSetContaminationError):
            MultiAssetPortfolioEnv(
                assets=ASSETS, aligned_dfs=dfs, regime_stream_df=regime,
                encoder=None, split="train", obs_mode="raw",
            )


def test_sequential_mode_deterministic(aligned_dfs, regime_stream):
    e1 = _make_env(aligned_dfs, regime_stream, sequential=True)
    e2 = _make_env(aligned_dfs, regime_stream, sequential=True)
    e1.reset(seed=0)
    e2.reset(seed=0)
    logits = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32)
    for _ in range(5):
        o1, r1, t1, tr1, _ = e1.step(logits)
        o2, r2, t2, tr2, _ = e2.step(logits)
        np.testing.assert_allclose(o1, o2)
        assert r1 == pytest.approx(r2)
        if t1 or tr1:
            break
