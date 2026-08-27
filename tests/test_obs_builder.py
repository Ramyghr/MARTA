"""Direct unit tests for rl/obs_builder.py's pure per-asset observation-block functions
— extracted from rl/env.py's private methods so a future portfolio env can reuse them.
These test the functions in isolation (no MARTATradingEnv construction needed), which is
the whole point of the extraction: a portfolio env should be able to call market_block()
etc. directly against plain numpy/pandas inputs.

tests/test_rl_env.py's existing 17 tests (unchanged, all still passing after the
extraction) already prove MARTATradingEnv's externally-observable behavior didn't change;
these tests cover the extracted functions' own contract directly, including the one
subtle behavior that had to be preserved exactly: zero_regime zeroes the encoder embedding
or the regime-prob tail, but never the raw feature window itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl import obs_builder


# ---------------------------------------------------------------------------
# build_regime_label_series / build_regime_prob_series
# ---------------------------------------------------------------------------

def test_build_regime_label_series_forward_fill_only():
    index = pd.date_range("2020-01-01", periods=5, freq="1h", tz="UTC")
    stream = pd.DataFrame({
        "timestamp": [index[0], index[2]],
        "asset": ["BTC_USDT", "BTC_USDT"],
        "regime_label": [1, 3],
    })
    labels = obs_builder.build_regime_label_series(stream, "BTC_USDT", index)
    # row0=1 (exact), row1=1 (ffill from row0), row2=3 (exact), rows3-4=3 (ffill)
    np.testing.assert_array_equal(labels, [1, 1, 3, 3, 3])


def test_build_regime_label_series_empty_returns_minus_one():
    index = pd.date_range("2020-01-01", periods=3, freq="1h", tz="UTC")
    stream = pd.DataFrame({"timestamp": [], "asset": [], "regime_label": []})
    labels = obs_builder.build_regime_label_series(stream, "BTC_USDT", index)
    np.testing.assert_array_equal(labels, [-1, -1, -1])


def test_build_regime_prob_series_forward_fill_and_zero_fallback():
    index = pd.date_range("2020-01-01", periods=4, freq="1h", tz="UTC")
    stream = pd.DataFrame({
        "timestamp": [index[1]],
        "asset": ["BTC_USDT"],
        "regime_prob_0": [0.7],
        "regime_prob_1": [0.3],
    })
    probs = obs_builder.build_regime_prob_series(
        stream, ["regime_prob_0", "regime_prob_1"], "BTC_USDT", index
    )
    assert probs.shape == (4, 2)
    np.testing.assert_allclose(probs[0], [0.0, 0.0])  # before first known row -> 0, never future
    np.testing.assert_allclose(probs[1], [0.7, 0.3])
    np.testing.assert_allclose(probs[2], [0.7, 0.3])  # forward-filled
    np.testing.assert_allclose(probs[3], [0.7, 0.3])


def test_build_regime_prob_series_no_cols_returns_zero_width():
    index = pd.date_range("2020-01-01", periods=3, freq="1h", tz="UTC")
    stream = pd.DataFrame({"timestamp": [], "asset": []})
    probs = obs_builder.build_regime_prob_series(stream, [], "BTC_USDT", index)
    assert probs.shape[0] == 3


# ---------------------------------------------------------------------------
# build_consensus_lookup
# ---------------------------------------------------------------------------

def test_build_consensus_lookup_basic():
    debates = [{
        "asset": "BTC_USDT",
        "timestamp": "2024-01-01T00:00:00Z",
        "round1_theses": [
            {"directional_bias": "bullish", "confidence": 0.8},
            {"directional_bias": "bearish", "confidence": 0.4},
        ],
        "fund_manager_decision": {
            "final_action": "long", "conviction": 0.6, "position_size": 0.05,
            "regime_override": True, "dissenting_agents": ["ContrarianAgent"],
        },
    }]
    lookup = obs_builder.build_consensus_lookup(debates)
    key = ("BTC_USDT", pd.Timestamp("2024-01-01T00:00:00Z"))
    assert key in lookup
    vec = lookup[key]
    assert vec.shape == (obs_builder.CONSENSUS_DIM,)
    assert vec[0] == pytest.approx(0.8)   # bullish * 0.8
    assert vec[1] == pytest.approx(-0.4)  # bearish * 0.4
    assert vec[5] == pytest.approx(0.6)   # long * conviction
    assert vec[6] == pytest.approx(0.05)
    assert vec[7] == 1.0
    assert vec[8] == 1.0


def test_build_consensus_lookup_empty_input():
    assert obs_builder.build_consensus_lookup(None) == {}
    assert obs_builder.build_consensus_lookup([]) == {}


# ---------------------------------------------------------------------------
# raw_window_vec / regime_prob_vec / consensus_vec
# ---------------------------------------------------------------------------

def test_raw_window_vec_shape_and_values():
    arr = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    vec = obs_builder.raw_window_vec(arr, t=9, window_size=5)
    expected = arr[5:10].reshape(-1)
    np.testing.assert_array_equal(vec, expected)


def test_regime_prob_vec_clamps_past_end():
    arr = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=np.float32)
    np.testing.assert_allclose(obs_builder.regime_prob_vec(arr, t=1), [0.2, 0.8])
    np.testing.assert_allclose(obs_builder.regime_prob_vec(arr, t=100), [0.3, 0.7])


def test_consensus_vec_hit_and_miss():
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    lookup = {("BTC_USDT", ts): np.ones(obs_builder.CONSENSUS_DIM, dtype=np.float32)}
    hit = obs_builder.consensus_vec(lookup, "BTC_USDT", ts)
    miss = obs_builder.consensus_vec(lookup, "ETH_USDT", ts)
    np.testing.assert_array_equal(hit, np.ones(obs_builder.CONSENSUS_DIM, dtype=np.float32))
    np.testing.assert_array_equal(miss, np.zeros(obs_builder.CONSENSUS_DIM, dtype=np.float32))


# ---------------------------------------------------------------------------
# encode_window (needs a real, if untrained, encoder)
# ---------------------------------------------------------------------------

def test_encode_window_shape(mock_encoder):
    arr = np.random.default_rng(0).standard_normal((100, 14)).astype(np.float32)
    emb = obs_builder.encode_window(mock_encoder, arr, t=99, window_size=60, device="cpu")
    assert emb.shape == (obs_builder.REGIME_DIM,)
    assert emb.dtype == np.float32


# ---------------------------------------------------------------------------
# market_block — the critical zero_regime asymmetry
# ---------------------------------------------------------------------------

def test_market_block_encoder_mode_zero_regime_zeroes_embedding(mock_encoder):
    arr = np.random.default_rng(0).standard_normal((100, 14)).astype(np.float32)
    zeroed = obs_builder.market_block(
        "encoder", arr, t=99, window_size=60, encoder=mock_encoder, zero_regime=True
    )
    np.testing.assert_array_equal(zeroed, np.zeros(obs_builder.REGIME_DIM, dtype=np.float32))


def test_market_block_raw_mode_ignores_zero_regime():
    """The key preserved asymmetry: obs_mode='raw' never zeroes anything, regardless of
    zero_regime — the original rl/env.py._raw_window_vec never checked the flag."""
    arr = np.random.default_rng(1).standard_normal((30, 4)).astype(np.float32)
    normal = obs_builder.market_block("raw", arr, t=20, window_size=10, zero_regime=False)
    zeroed_flag = obs_builder.market_block("raw", arr, t=20, window_size=10, zero_regime=True)
    np.testing.assert_array_equal(normal, zeroed_flag)
    assert not np.allclose(normal, 0.0)


def test_market_block_raw_regime_mode_zeroes_only_the_regime_tail():
    arr = np.random.default_rng(2).standard_normal((30, 4)).astype(np.float32)
    regime_probs = np.random.default_rng(3).uniform(0, 1, size=(30, 3)).astype(np.float32)

    normal = obs_builder.market_block(
        "raw_regime", arr, t=20, window_size=10, regime_probs_arr=regime_probs, zero_regime=False
    )
    zeroed = obs_builder.market_block(
        "raw_regime", arr, t=20, window_size=10, regime_probs_arr=regime_probs, zero_regime=True
    )
    raw_dim = 10 * 4
    # Raw window portion identical...
    np.testing.assert_array_equal(normal[:raw_dim], zeroed[:raw_dim])
    assert not np.allclose(normal[:raw_dim], 0.0)
    # ...regime-prob tail zeroed only in the zero_regime=True case.
    np.testing.assert_array_equal(zeroed[raw_dim:], np.zeros(3, dtype=np.float32))
    assert not np.allclose(normal[raw_dim:], 0.0)


def test_market_block_encoder_mode_requires_encoder():
    arr = np.zeros((100, 14), dtype=np.float32)
    with pytest.raises(ValueError):
        obs_builder.market_block("encoder", arr, t=99, window_size=60, encoder=None)


def test_market_block_raw_regime_requires_probs_array():
    arr = np.zeros((30, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        obs_builder.market_block("raw_regime", arr, t=20, window_size=10, regime_probs_arr=None)


def test_market_block_unknown_mode_raises():
    arr = np.zeros((30, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        obs_builder.market_block("bogus_mode", arr, t=20, window_size=10)
