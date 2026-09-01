"""Per-asset block-bootstrap reconciliation for the crypto5 weekly-cadence pilot.

The pilot (rl/train_weekly_crypto5_grid.py --pilot) reports each cell's raw Sharpe and
a naive buy-hold delta from evaluate_ppo()'s own baseline. That baseline is a legitimate
number (continuous hourly buy-hold, correctly asset-scoped since the evaluate_ppo fix)
but it is NOT statistically tested -- no block-bootstrap CI, unlike every other
comparison in this project (rl/ablation.py's bootstrap_sharpe_ci, the BTC-only weekly
grid's reconciliation, docs/RESULTS_AND_METHODOLOGY.md Section 10/10a). This script
closes that gap: for every (asset, condition) pilot cell, it reloads the trained
checkpoint, re-runs the walk-forward on 2024 validation, reconstructs buy-hold at the
IDENTICAL weekly-decision cadence (not the continuous-hourly one evaluate_ppo uses --
that mismatch is what makes a block bootstrap invalid, since the two series must share
a frequency to be resampled in aligned blocks), and reports delta-Sharpe with a 95% CI.

This is read-only analysis of already-trained checkpoints -- no new training, safe to
re-run any time, fast (a handful of seconds per cell).

Run:
  /home/ramy/miniconda3/envs/marta/bin/python rl/reconcile_weekly_grid_crypto5.py
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
from rl.evaluate import run_walk_forward
from rl.train import load_data, build_env_factory, load_debate_results
from rl.train_weekly_crypto5_grid import CRYPTO5_ASSETS, build_weekly_config
from rl.train_weekly_smoketest import CONDITIONS, DECISION_INTERVAL

PPY = (24 * 365) / DECISION_INTERVAL


def _sharpe(r: np.ndarray, ppy: float) -> float:
    r = np.asarray(r, dtype=np.float64)
    return float(r.mean() / (r.std() + 1e-8) * np.sqrt(ppy))


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


def main() -> None:
    results = {}
    for asset in CRYPTO5_ASSETS:
        buy_hold_weekly = None
        timestamps_ref = None
        for label, seed in [(l, 0) for l in CONDITIONS]:
            cfg = CONDITIONS[label]
            file_label = f"{asset}_{label}_seed{seed}"
            config = build_weekly_config(asset, cfg["obs_mode"], seed, cfg["regime_path"], cfg["extra_tag"])
            config["seed"] = seed

            aligned_dfs, regime_stream_df, encoder = load_data(config)
            debate_results = load_debate_results()
            env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config)

            env = env_factory(assets=[asset], split="val", sequential=True, zero_regime=False, debate=False)

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
            if buy_hold_weekly is None or not timestamps_match_ref:
                raw_returns = env._returns[asset]
                index = env._index[asset]
                positions = np.searchsorted(index, timestamps)
                bh = []
                for i in range(1, len(positions)):
                    lo, hi = positions[i - 1], positions[i]
                    seg = np.nan_to_num(raw_returns[lo + 1 : hi + 1])
                    bh.append(float(np.prod(1.0 + seg) - 1.0))
                bh_for_cell = np.asarray(bh, dtype=np.float64)
                if buy_hold_weekly is None:
                    buy_hold_weekly = bh_for_cell
                    timestamps_ref = timestamps
                    print(f"\n=== {asset}: val window {timestamps[0]} -> {timestamps[-1]} "
                          f"({len(buy_hold_weekly)} weekly steps), buy_hold_sharpe={_sharpe(buy_hold_weekly, PPY):.3f} ===")
                else:
                    print(f"  WARNING [{file_label}]: timestamps differ from this asset's reference run "
                          f"({len(timestamps)} vs {len(timestamps_ref)} steps) -- rebuilt buy-hold for this cell")
            else:
                bh_for_cell = buy_hold_weekly

            delta, lo_ci, hi_ci, n_blocks = block_bootstrap_delta(marta_returns, bh_for_cell, PPY, block_size=4)
            marta_sharpe = _sharpe(marta_returns, PPY)
            bh_sharpe = _sharpe(bh_for_cell, PPY)
            sig = lo_ci > 0 or hi_ci < 0

            results[file_label] = {
                "asset": asset, "label": label, "obs_mode": cfg["obs_mode"], "extra_tag": cfg["extra_tag"],
                "n_weekly_steps": int(len(marta_returns)),
                "marta_sharpe": marta_sharpe,
                "buy_hold_sharpe": bh_sharpe,
                "delta_sharpe_bootstrap_mean": delta,
                "ci_95_lower": lo_ci,
                "ci_95_upper": hi_ci,
                "n_blocks": n_blocks,
                "significant": sig,
            }
            print(f"  {label}: marta={marta_sharpe:+.3f}  buy_hold={bh_sharpe:+.3f}  "
                  f"delta={delta:+.3f}  CI=[{lo_ci:+.3f},{hi_ci:+.3f}]  sig={sig}")

    out_path = ROOT / "data" / "results" / "weekly_grid_crypto5_reconciled.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {out_path}")

    n_sig_positive = sum(1 for r in results.values() if r["significant"] and r["delta_sharpe_bootstrap_mean"] > 0)
    n_sig_negative = sum(1 for r in results.values() if r["significant"] and r["delta_sharpe_bootstrap_mean"] < 0)
    n_positive_point = sum(1 for r in results.values() if r["delta_sharpe_bootstrap_mean"] > 0)
    print(f"\n{len(results)} cells: {n_positive_point} positive point estimate, "
          f"{n_sig_positive} significant positive, {n_sig_negative} significant negative")


if __name__ == "__main__":
    main()
