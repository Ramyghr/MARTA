"""Single, pre-registered long-run pass for the crypto5 weekly-cadence pilot checkpoints
(rl/train_weekly_crypto5_grid.py --pilot, 20 cells, 5 assets x 4 conditions, seed 0).

Combines both things asked for in one pass over each checkpoint (not two separate re-runs
against overlapping data, which would blur the "single test-set look" discipline this
project holds elsewhere -- see docs/RESULTS_AND_METHODOLOGY.md Section 10a):

  1. A continuous walk-forward from 2024-01-01 through the last available bar. The
     crypto5_multihorizon panel's aligned data ends 2025-12-31 23:00 UTC for all 5
     assets (verified directly, not assumed) -- there is no 2026 data in this pipeline,
     so "through today" (2026-09-01) is not reachable here; this covers everything that
     IS available, val+test as one continuous 104-week stretch, reported as a long-run
     behavior diagnostic (equity curve, max drawdown, whole-period Sharpe) -- more weekly
     observations than either year alone, but not itself the formal significance claim.
  2. The 2025-only sub-slice of that same trajectory (no separate re-run), block-
     bootstrapped against buy-hold exactly like the earlier BTC-only 2025 test pass --
     this is the one directly comparable to that prior result and to the reconciled
     2024-validation numbers already reported for this pilot.

No checkpoint selection is needed here (unlike the earlier BTC-only test pass): the
crypto5 pilot only has one seed per (asset, condition) cell, so every checkpoint is used
-- there is nothing to cherry-pick between.

This is read-only analysis of already-trained checkpoints -- no new training, no network.
Run exactly once; do not re-run and compare against a more flattering pass.

Run:
  /home/ramy/miniconda3/envs/marta/bin/python rl/weekly_grid_crypto5_longrun.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ramy/Documents/MARTA/marta")
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from run_rl import _policy_class_for
from rl.env import MARTATradingEnv
from rl.evaluate import run_walk_forward
from rl.train import load_data, build_env_factory, load_debate_results
from rl.train_weekly_crypto5_grid import CRYPTO5_ASSETS, build_weekly_config
from rl.train_weekly_smoketest import CONDITIONS, DECISION_INTERVAL
from configs.splits import VAL_START, TEST_START, TEST_END_EXCLUSIVE

PPY = (24 * 365) / DECISION_INTERVAL


def _sharpe(r: np.ndarray, ppy: float) -> float:
    r = np.asarray(r, dtype=np.float64)
    return float(r.mean() / (r.std() + 1e-8) * np.sqrt(ppy))


def _max_drawdown(returns: np.ndarray) -> float:
    pv = np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
    peak = np.maximum.accumulate(pv)
    dd = (peak - pv) / peak
    return float(dd.max())


def block_bootstrap_delta(returns_a, returns_b, ppy, block_size=4, n_bootstrap=1000, seed=42):
    a = np.asarray(returns_a, dtype=np.float64)
    b = np.asarray(returns_b, dtype=np.float64)
    n = min(len(a), len(b))
    n_blocks = n // block_size
    if n_blocks < 2:
        d = _sharpe(a[:n], ppy) - _sharpe(b[:n], ppy)
        return d, d, d, n_blocks
    a = a[: n_blocks * block_size].reshape(n_blocks, block_size)
    b = b[: n_blocks * block_size].reshape(n_blocks, block_size)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        deltas.append(_sharpe(a[idx].ravel(), ppy) - _sharpe(b[idx].ravel(), ppy))
    deltas = np.array(deltas)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), n_blocks


def restrict_env_to_window(env: MARTATradingEnv, asset: str, start: pd.Timestamp, end: pd.Timestamp) -> MARTATradingEnv:
    idx = env._index[asset]
    in_window = np.flatnonzero((idx >= start) & (idx <= end))
    valid = np.intersect1d(env._valid_starts[asset], in_window)
    if len(valid) < 2:
        raise RuntimeError(f"Window {start}..{end} has <2 valid bars for {asset}")
    env._valid_starts[asset] = valid
    return env


def weekly_buy_hold(env: MARTATradingEnv, asset: str, timestamps: np.ndarray) -> np.ndarray:
    raw_returns = env._returns[asset]
    index = env._index[asset]
    positions = np.searchsorted(index, timestamps)
    bh = []
    for i in range(1, len(positions)):
        lo, hi = positions[i - 1], positions[i]
        seg = np.nan_to_num(raw_returns[lo + 1 : hi + 1])
        bh.append(float(np.prod(1.0 + seg) - 1.0))
    return np.asarray(bh, dtype=np.float64)


def main() -> None:
    long_start = pd.Timestamp(VAL_START, tz="UTC")
    test_start = pd.Timestamp(TEST_START, tz="UTC")
    long_end_nominal = pd.Timestamp(TEST_END_EXCLUSIVE, tz="UTC") - pd.Timedelta(seconds=1)

    results = {}
    for asset in CRYPTO5_ASSETS:
        buy_hold_ref = None
        timestamps_ref = None
        for label in CONDITIONS:
            cfg = CONDITIONS[label]
            file_label = f"{asset}_{label}_seed0"
            config = build_weekly_config(asset, cfg["obs_mode"], 0, cfg["regime_path"], cfg["extra_tag"])
            config["seed"] = 0

            aligned_dfs, regime_stream_df, encoder = load_data(config)
            debate_results = load_debate_results()
            env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config)

            env = env_factory(assets=[asset], split="all", sequential=True, zero_regime=False, debate=False)
            long_end = min(long_end_nominal, env._index[asset][-1])
            restrict_env_to_window(env, asset, long_start, long_end)

            best_path = ROOT / "rl" / "checkpoints" / f"ppo_best{config['checkpoint_suffix']}.zip"
            model = PPO.load(str(best_path), device="cpu", custom_objects={"policy_class": _policy_class_for(config)})
            traj = run_walk_forward(model, env, deterministic=True)
            marta_returns = traj["returns"]
            timestamps = traj["timestamps"]

            timestamps_match_ref = (
                timestamps_ref is not None
                and len(timestamps) == len(timestamps_ref)
                and (timestamps == timestamps_ref).all()
            )
            if buy_hold_ref is None or not timestamps_match_ref:
                bh = weekly_buy_hold(env, asset, timestamps)
                if buy_hold_ref is None:
                    buy_hold_ref = bh
                    timestamps_ref = timestamps
                    print(f"\n=== {asset}: long window {timestamps[0]} -> {timestamps[-1]} "
                          f"({len(bh)} weekly steps) ===")
                bh_for_cell = bh
            else:
                bh_for_cell = buy_hold_ref

            # Whole-period (2024-2025) view.
            full_delta, full_lo, full_hi, full_blocks = block_bootstrap_delta(marta_returns, bh_for_cell, PPY)
            full_marta_sharpe = _sharpe(marta_returns, PPY)
            full_bh_sharpe = _sharpe(bh_for_cell, PPY)
            full_dd = _max_drawdown(marta_returns)

            # 2025-only sub-slice of the SAME trajectory -- no separate re-run.
            # returns[k] corresponds to the transition ENDING at timestamps[k+1] (timestamps
            # has one extra leading entry for the initial reset state, same reason
            # weekly_buy_hold() loops from i=1) -- so mask against timestamps[1:], not
            # timestamps itself, to keep the boolean mask the same length as returns.
            mask_2025 = timestamps[1:] >= test_start
            m25 = marta_returns[mask_2025]
            bh25 = bh_for_cell[mask_2025]
            if mask_2025.sum() >= 2:
                t_delta, t_lo, t_hi, t_blocks = block_bootstrap_delta(m25, bh25, PPY)
                t_marta_sharpe = _sharpe(m25, PPY)
                t_bh_sharpe = _sharpe(bh25, PPY)
            else:
                t_delta = t_lo = t_hi = t_marta_sharpe = t_bh_sharpe = float("nan")
                t_blocks = 0

            results[file_label] = {
                "asset": asset, "label": label, "obs_mode": cfg["obs_mode"], "extra_tag": cfg["extra_tag"],
                "full_period": {
                    "window": [str(timestamps[0]), str(timestamps[-1])],
                    "n_weekly_steps": int(len(marta_returns)),
                    "marta_sharpe": full_marta_sharpe, "buy_hold_sharpe": full_bh_sharpe,
                    "delta_bootstrap_mean": full_delta, "ci_95": [full_lo, full_hi],
                    "n_blocks": full_blocks, "significant": full_lo > 0 or full_hi < 0,
                    "marta_max_drawdown": full_dd,
                    "marta_total_return": float(np.prod(1.0 + marta_returns) - 1.0),
                },
                "test2025_subslice": {
                    "n_weekly_steps": int(mask_2025.sum()),
                    "marta_sharpe": t_marta_sharpe, "buy_hold_sharpe": t_bh_sharpe,
                    "delta_bootstrap_mean": t_delta, "ci_95": [t_lo, t_hi],
                    "n_blocks": t_blocks,
                    "significant": (t_lo > 0 or t_hi < 0) if t_blocks >= 2 else None,
                },
            }
            print(f"  {label}: FULL 24-25 marta={full_marta_sharpe:+.3f} buy_hold={full_bh_sharpe:+.3f} "
                  f"delta={full_delta:+.3f} CI=[{full_lo:+.3f},{full_hi:+.3f}] sig={results[file_label]['full_period']['significant']} "
                  f"maxDD={full_dd:.3f}  ||  2025-only marta={t_marta_sharpe:+.3f} delta={t_delta:+.3f} "
                  f"CI=[{t_lo:+.3f},{t_hi:+.3f}] sig={results[file_label]['test2025_subslice']['significant']}")

    out_path = ROOT / "data" / "results" / "weekly_grid_crypto5_longrun.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
