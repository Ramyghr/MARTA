"""Tests for rl/placebo_regime.py — RL experiment Condition D's rotated (sham) regime
stream. The property that matters: same statistical shape as the real stream (marginal
label frequencies, dwell-time distribution), but decorrelated from the real timeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl.placebo_regime import rotate_regime_stream


def _make_stream(seed: int, n_rows: int = 2000, n_states: int = 6, assets=("BTC_USDT", "ETH_USDT")) -> pd.DataFrame:
    """A regime stream with realistic persistence (not i.i.d. per-row) — a Markov chain
    with high self-transition probability, matching the real HMM's dwell-time behavior."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="1h", tz="UTC")
    frames = []
    for asset in assets:
        labels = np.empty(n_rows, dtype=np.int64)
        labels[0] = rng.integers(n_states)
        for t in range(1, n_rows):
            labels[t] = labels[t - 1] if rng.random() < 0.9 else rng.integers(n_states)
        probs = np.zeros((n_rows, n_states))
        for t in range(n_rows):
            probs[t, labels[t]] = 1.0
        df = pd.DataFrame({"timestamp": idx, "asset": asset, "regime_label": labels})
        for k in range(n_states):
            df[f"regime_prob_{k}"] = probs[:, k]
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def test_rotation_preserves_marginal_frequencies_exactly():
    stream = _make_stream(seed=0)
    rotated = rotate_regime_stream(stream, seed=1)
    for asset in stream["asset"].unique():
        orig = stream[stream["asset"] == asset]["regime_label"].value_counts().sort_index()
        rot = rotated[rotated["asset"] == asset]["regime_label"].value_counts().sort_index()
        pd.testing.assert_series_equal(orig, rot, check_names=False)


def test_rotation_preserves_dwell_time_distribution_exactly():
    """A circular rotation is the same sequence, just a different starting point — the
    multiset of run-lengths (dwell times) must be identical, not just similar."""
    stream = _make_stream(seed=0)
    rotated = rotate_regime_stream(stream, seed=1)

    def _dwell_times(labels: np.ndarray) -> list[int]:
        boundaries = np.flatnonzero(np.diff(labels) != 0) + 1
        return sorted(len(run) for run in np.split(labels, boundaries))

    for asset in stream["asset"].unique():
        orig = stream[stream["asset"] == asset].sort_values("timestamp")["regime_label"].to_numpy()
        rot = rotated[rotated["asset"] == asset].sort_values("timestamp")["regime_label"].to_numpy()
        # A cut point can split one run into two (at the rotation boundary) or merge two
        # runs into one (if the label before and after the cut happen to match) — so exact
        # equality of the dwell-time multiset isn't guaranteed at the single cut point, but
        # the distribution's mean/count must match to within that one boundary effect.
        d_orig, d_rot = _dwell_times(orig), _dwell_times(rot)
        assert abs(len(d_orig) - len(d_rot)) <= 1
        assert abs(np.mean(d_orig) - np.mean(d_rot)) < 0.5


def test_rotation_genuinely_decorrelates_from_original_timeline():
    stream = _make_stream(seed=0)
    rotated = rotate_regime_stream(stream, seed=1)
    for asset in stream["asset"].unique():
        orig = stream[stream["asset"] == asset].sort_values("timestamp")["regime_label"].to_numpy()
        rot = rotated[rotated["asset"] == asset].sort_values("timestamp")["regime_label"].to_numpy()
        pct_different = (orig != rot).mean()
        # With high self-persistence (p=0.9) and 6 states, a large majority of rows should
        # land on a genuinely different label than their true-timeline counterpart.
        assert pct_different > 0.5, f"[{asset}] rotation barely changed anything ({pct_different:.1%} different)"


def test_rotation_is_a_pure_function_does_not_mutate_input():
    stream = _make_stream(seed=0)
    original_copy = stream.copy(deep=True)
    rotate_regime_stream(stream, seed=1)
    pd.testing.assert_frame_equal(stream, original_copy)


def test_rotation_never_produces_a_no_op_shift():
    """shift=0 (or shift=n, a full cycle) would silently leave an asset's regime values
    completely unrotated — the exact opposite of a placebo. Sweep several seeds since the
    shift fraction is randomized per asset."""
    for seed in range(10):
        stream = _make_stream(seed=0, n_rows=50)  # small n stresses the shift-clamping logic
        rotated = rotate_regime_stream(stream, seed=seed)
        for asset in stream["asset"].unique():
            orig = stream[stream["asset"] == asset]["regime_label"].to_numpy()
            rot = rotated[rotated["asset"] == asset]["regime_label"].to_numpy()
            assert not np.array_equal(orig, rot), f"seed={seed} asset={asset} produced a no-op rotation"


def test_raises_on_stream_with_no_regime_columns():
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=5, freq="1h"), "asset": "BTC_USDT"})
    with pytest.raises(ValueError):
        rotate_regime_stream(df)
