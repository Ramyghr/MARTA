"""
rl/obs_builder.py

Per-asset observation-block construction, extracted from rl/env.py's private methods
(2026-08-26) so both MARTATradingEnv (single-asset-per-episode) and a future portfolio
env can share one tested implementation instead of two copies that could silently drift
apart — Step 2 of the pre-RL audit's remediation plan (Step 1: regime/causal_regime_stream.py).

Every function here is pure: no `self`, no hidden state, takes precomputed per-asset
arrays/lookups and an index, returns a numpy block. rl/env.py's MARTATradingEnv now calls
these directly from its private methods (_encode_window, _raw_window_vec, _regime_prob_vec,
_consensus_vec, _build_regime_label_series, _build_regime_prob_series,
_build_consensus_lookup — all now thin delegating wrappers); behavior is verified
byte-identical via tests/test_rl_env.py's existing 15 tests (unchanged, all still passing)
plus tests/test_obs_builder.py's new direct coverage of the functions below.

Deliberately NOT extracted here: _prepare_asset's GARCH/vol precompute
(precompute_vol_forecast/cvar_from_vol/risk_cap_from_vol are already free functions in
rl/env.py, directly importable without further extraction) and valid-starts/split-mask
logic (single-asset-episode-structural — a portfolio env needs a different, joint
valid-starts rule across all assets simultaneously, not a per-asset one, so there is
nothing generic to share there).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

REGIME_DIM = 128
CONSENSUS_DIM = 16


def build_regime_label_series(
    regime_stream_df: pd.DataFrame, asset: str, index: pd.DatetimeIndex
) -> np.ndarray:
    """Forward-fill-only regime label lookup — never a future label."""
    sub = regime_stream_df[regime_stream_df["asset"] == asset]
    if sub.empty:
        return np.full(len(index), -1, dtype=np.int64)
    sub = sub.set_index("timestamp")["regime_label"].sort_index()
    aligned = sub.reindex(index, method="pad")
    return aligned.fillna(-1).to_numpy(dtype=np.int64)


def build_regime_prob_series(
    regime_stream_df: pd.DataFrame,
    regime_prob_cols: list[str],
    asset: str,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    """Forward-fill-only regime probability-vector lookup. Missing rows -> all-zero
    (flat/uninformative prior), never a future value."""
    n = len(regime_prob_cols)
    sub = regime_stream_df[regime_stream_df["asset"] == asset]
    if sub.empty or n == 0:
        return np.zeros((len(index), max(n, 1)), dtype=np.float32)
    sub = sub.set_index("timestamp")[regime_prob_cols].sort_index()
    aligned = sub.reindex(index, method="pad")
    return aligned.fillna(0.0).to_numpy(dtype=np.float32)


def build_consensus_lookup(
    debate_results: Optional[list[dict]],
) -> dict[tuple[str, pd.Timestamp], np.ndarray]:
    """(asset, timestamp) -> 16-dim signed-consensus vector, from cached debate JSON."""
    lookup: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
    if not debate_results:
        return lookup
    bias_sign = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
    action_sign = {"long": 1.0, "short": -1.0, "hold": 0.0}
    for d in debate_results:
        asset = d.get("asset")
        ts_raw = d.get("timestamp")
        if asset is None or ts_raw is None:
            continue
        ts = pd.Timestamp(ts_raw)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        vec = np.zeros(CONSENSUS_DIM, dtype=np.float32)
        theses = d.get("round1_theses", []) or []
        for i, th in enumerate(theses[:5]):
            conf = float(th.get("confidence", 0.0))
            vec[i] = bias_sign.get(th.get("directional_bias", "neutral"), 0.0) * conf
        fmd = d.get("fund_manager_decision", {}) or {}
        vec[5] = action_sign.get(fmd.get("final_action", "hold"), 0.0) * float(
            fmd.get("conviction", 0.0)
        )
        vec[6] = float(fmd.get("position_size", 0.0))
        vec[7] = 1.0 if fmd.get("regime_override", False) else 0.0
        vec[8] = float(len(fmd.get("dissenting_agents", [])))
        lookup[(asset, ts)] = vec
    return lookup


def encode_window(
    encoder, arr: np.ndarray, t: int, window_size: int, device: str = "cpu"
) -> np.ndarray:
    """128-dim TFTRegimeEncoder embedding of arr[t-window_size+1 : t+1]."""
    window = arr[t - window_size + 1 : t + 1]
    tensor = torch.from_numpy(window).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = encoder.encode(tensor)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def raw_window_vec(arr: np.ndarray, t: int, window_size: int) -> np.ndarray:
    """Flattened (window_size, n_features) raw feature window — zero learned compression."""
    window = arr[t - window_size + 1 : t + 1]
    return window.reshape(-1).astype(np.float32)


def regime_prob_vec(regime_probs_arr: np.ndarray, t: int) -> np.ndarray:
    """Row t of a precomputed per-asset regime-prob array (clamped to the last row past
    the end of the array, same convention as rl/env.py's original _regime_prob_vec)."""
    idx = min(t, len(regime_probs_arr) - 1)
    return regime_probs_arr[idx]


def vol_scalar_vec(vol_zscore_arr: np.ndarray, t: int) -> np.ndarray:
    """1-dim (t,) slice of a precomputed rolling-z-scored vol_forecast array — obs_mode
    ='raw_vol' (RL experiment Condition B). Same clamp-to-last-row convention as
    regime_prob_vec, for the same reason (t can run past array bounds in edge cases)."""
    idx = min(t, len(vol_zscore_arr) - 1)
    return vol_zscore_arr[idx : idx + 1].astype(np.float32)


def consensus_vec(
    consensus_lookup: dict[tuple[str, pd.Timestamp], np.ndarray],
    asset: str,
    ts: pd.Timestamp,
) -> np.ndarray:
    return consensus_lookup.get((asset, ts), np.zeros(CONSENSUS_DIM, dtype=np.float32))


def market_block(
    obs_mode: str,
    arr: np.ndarray,
    t: int,
    window_size: int,
    encoder=None,
    device: str = "cpu",
    regime_probs_arr: Optional[np.ndarray] = None,
    zero_regime: bool = False,
    vol_zscore_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The per-asset 'what does the market look like' block that varies by obs_mode —
    encoder embedding, raw window, raw window + regime probs, or raw window + a
    volatility scalar. This is the single piece both MARTATradingEnv (one asset/episode)
    and a future portfolio env (all assets/episode, one market_block() call per asset)
    need identically per asset.

    zero_regime replicates an existing, deliberate asymmetry from rl/env.py's original
    per-mode methods: it zeroes the encoder embedding (obs_mode="encoder") or the
    regime-probability tail (obs_mode="raw_regime"), but the raw feature window itself is
    NEVER zeroed in any mode — the original _raw_window_vec never checked zero_regime at
    all. Preserved exactly, not "fixed", since this extraction must not change behavior.
    obs_mode="raw_vol" is NOT gated by zero_regime — see rl/env.py's _vol_zscore_vec for
    why (volatility and regime are orthogonal ablation axes).
    """
    if obs_mode == "encoder":
        if zero_regime:
            return np.zeros(REGIME_DIM, dtype=np.float32)
        if encoder is None:
            raise ValueError("obs_mode='encoder' requires a non-None encoder")
        return encode_window(encoder, arr, t, window_size, device)
    if obs_mode == "raw":
        return raw_window_vec(arr, t, window_size)
    if obs_mode == "raw_vol":
        raw = raw_window_vec(arr, t, window_size)
        if vol_zscore_arr is None:
            raise ValueError("obs_mode='raw_vol' requires vol_zscore_arr")
        return np.concatenate([raw, vol_scalar_vec(vol_zscore_arr, t)])
    if obs_mode == "raw_regime":
        raw = raw_window_vec(arr, t, window_size)
        if zero_regime:
            n = regime_probs_arr.shape[1] if regime_probs_arr is not None else 0
            probs = np.zeros(n, dtype=np.float32)
        else:
            if regime_probs_arr is None:
                raise ValueError("obs_mode='raw_regime' requires regime_probs_arr")
            probs = regime_prob_vec(regime_probs_arr, t)
        return np.concatenate([raw, probs])
    raise ValueError(f"Unknown obs_mode: {obs_mode!r}")
