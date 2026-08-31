"""rl/placebo_regime.py

RL experiment Condition D — a placebo regime stream for isolating whether the causal
regime signal itself matters, vs. an agent just learning to exploit *any* structured,
persistent categorical signal in its observation regardless of content.

Design: circular block-rotation, not a per-timestep shuffle. A naive per-row shuffle
would destroy the regime stream's dwell-time/transition structure (regimes normally
persist for many consecutive hours; a shuffled stream would flip state almost every
step), which is itself a detectable, exploitable difference an agent could learn to
distinguish from the real thing — a bad placebo. Rotating each asset's own label/prob
sequence by roughly half its length preserves every real statistical property (marginal
label frequencies, dwell-time distribution, transition matrix — it's the exact same
sequence) while decorrelating regime *timing* from the real market state at each
timestamp, since timestamp t now sees whatever regime genuinely occurred at a
different, unrelated point in that asset's history.

Run standalone to sanity-check a given regime stream:
    conda run -n marta python rl/placebo_regime.py --tag crypto5_multihorizon
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def rotate_regime_stream(
    regime_stream_df: pd.DataFrame,
    seed: int = 42,
    min_shift_fraction: float = 0.3,
    max_shift_fraction: float = 0.7,
) -> pd.DataFrame:
    """Return a copy of regime_stream_df with regime_label/regime_prob_* columns
    circularly shifted per-asset. `timestamp`/`asset` columns are untouched — only the
    regime values move, so a downstream consumer reading (asset, timestamp) -> regime
    still gets a value for every real row, just the wrong (but realistically-shaped) one.

    Shift amount is randomized per asset within [min_shift_fraction, max_shift_fraction]
    of that asset's own row count (not a hardcoded half) — a small enough shift range
    that it can never round down near 0 or up near the full length (which would leave
    rows unshifted or shift by a full cycle back to themselves), while still varying
    enough across assets that two assets' shifts are extremely unlikely to be identical.
    """
    rng = np.random.default_rng(seed)
    regime_cols = [c for c in regime_stream_df.columns if c.startswith("regime_")]
    if not regime_cols:
        raise ValueError("regime_stream_df has no regime_label/regime_prob_* columns to rotate")

    rotated_frames = []
    for asset, sub in regime_stream_df.groupby("asset", sort=False):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        n = len(sub)
        frac = rng.uniform(min_shift_fraction, max_shift_fraction)
        shift = int(n * frac)
        shift = max(1, min(shift, n - 1))  # never 0 (no-op) or n (full cycle, also a no-op)

        rotated = sub.copy()
        for col in regime_cols:
            rotated[col] = np.roll(sub[col].to_numpy(), shift)
        rotated_frames.append(rotated)

    return pd.concat(rotated_frames, axis=0, ignore_index=True)


def _sanity_check(original: pd.DataFrame, rotated: pd.DataFrame) -> None:
    """Prints the checks that matter: marginal frequencies preserved exactly, dwell-time
    distribution preserved exactly (same sequence, just a different starting point), and
    genuine decorrelation from the real timeline (rotated label != original label at most
    rows) — the three properties that make this a fair placebo rather than either a no-op
    or an unrealistically-shaped fake signal."""
    for asset in original["asset"].unique():
        o = original[original["asset"] == asset].sort_values("timestamp")["regime_label"].to_numpy()
        r = rotated[rotated["asset"] == asset].sort_values("timestamp")["regime_label"].to_numpy()

        o_freq = np.bincount(o[o >= 0]) / len(o)
        r_freq = np.bincount(r[r >= 0]) / len(r)
        freq_match = np.allclose(
            o_freq, r_freq[: len(o_freq)] if len(r_freq) >= len(o_freq) else np.pad(r_freq, (0, len(o_freq) - len(r_freq)))
        )

        def _dwell_times(labels: np.ndarray) -> np.ndarray:
            change = np.diff(labels) != 0
            boundaries = np.flatnonzero(change) + 1
            runs = np.split(labels, boundaries)
            return np.array([len(run) for run in runs])

        o_dwell_mean = _dwell_times(o).mean()
        r_dwell_mean = _dwell_times(r).mean()

        pct_different = float((o != r).mean())

        print(
            f"[{asset}] marginal_freq_preserved={freq_match}  "
            f"dwell_mean orig={o_dwell_mean:.2f} rotated={r_dwell_mean:.2f}  "
            f"pct_rows_different={pct_different:.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="crypto5_multihorizon")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_dir = Path("data/processed") / args.tag
    from utils.freshness import resolve_regime_stream_path

    stream_path = resolve_regime_stream_path(processed_dir)
    original = pd.read_parquet(stream_path, engine="pyarrow")
    rotated = rotate_regime_stream(original, seed=args.seed)

    print(f"Rotated {stream_path} ({len(original):,} rows, seed={args.seed})")
    _sanity_check(original, rotated)


if __name__ == "__main__":
    main()
