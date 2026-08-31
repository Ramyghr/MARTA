"""Single, pre-registered 2025 test-set pass for the native weekly-cadence grid.

Follow-up to rl/train_weekly_smoketest.py (2026-08-31 grid: 4 conditions x 3 seeds,
validated on 2024, see docs/RESULTS_AND_METHODOLOGY.md Section 10 and
data/results/weekly_grid_reconciled.json). That validation pass found 12/12 runs beat
buy-hold at the point estimate but 0/12 individually significant, all sharing the
single 2024 market window -- the open question this script resolves is whether the
effect survives a market year excluded from both training and that bootstrap analysis.

CHECKPOINT SELECTION IS PRE-REGISTERED, NOT CHOSEN AFTER SEEING TEST RESULTS: one seed
per condition (A/B/C/D), selected by MEDIAN validation Sharpe among that condition's 3
seeds (data/results/weekly_grid_reconciled.json's *_recomputed field) -- not best-of-3,
which would already be a mild form of test-set-adjacent cherry-picking on the val set.
This selection was fixed before this script was ever run against 2025 data:
    A -> seed 1 (val Sharpe 1.688, median of [1.677, 1.688, 1.736])
    B -> seed 1 (val Sharpe 1.883, median of [1.686, 1.883, 2.706])
    C -> seed 0 (val Sharpe 1.755, median of [1.665, 1.755, 2.173])
    D -> seed 1 (val Sharpe 1.764, median of [1.670, 1.764, 2.723])

This is meant to run ONCE. Per this project's established test-set discipline (sacred
2025 split, see configs/splits.py), do not re-run this repeatedly while tweaking config
and keep only the flattering pass -- that would silently defeat the point of holding
2025 out at all.

Run (no network required -- local checkpoints + local parquet only):
    /home/ramy/miniconda3/envs/marta/bin/python rl/weekly_grid_test2025.py
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

from run_rl import build_config, _policy_class_for
from rl.env import MARTATradingEnv
from rl.evaluate import run_walk_forward
from rl.train import (
    CRYPTO5_CANONICAL_ASSETS,
    CRYPTO5_PROCESSED,
    CRYPTO5_REGIME_STREAM_CAUSAL_PATH,
    build_env_factory,
    load_data,
    load_debate_results,
)
from rl.train_weekly_smoketest import CONDITIONS, DECISION_INTERVAL, MAX_STEPS_PER_EPISODE
from configs.splits import TEST_END_EXCLUSIVE, TEST_START

ASSET = "BTC_USDT"
PPY = (24 * 365) / DECISION_INTERVAL

# Pre-registered selection -- see module docstring. Do not change after seeing results.
SELECTED = {"A": 1, "B": 1, "C": 0, "D": 1}


def build_weekly_config(obs_mode, seed, regime_path=None, extra_tag=None):
    config = build_config(quick=False)
    config["processed_dir"] = str(CRYPTO5_PROCESSED)
    config["regime_stream_path"] = regime_path or str(CRYPTO5_REGIME_STREAM_CAUSAL_PATH)
    config["canonical_assets"] = CRYPTO5_CANONICAL_ASSETS
    config["all_assets"] = [ASSET]
    config["obs_mode"] = obs_mode
    config["decision_interval"] = DECISION_INTERVAL
    config["device"] = "cpu"
    config["phases"] = [{"assets": [ASSET], "sharpe_threshold": -999, "max_steps": 1, "idle_penalty": 0.0}]
    config["max_episode_steps"] = MAX_STEPS_PER_EPISODE
    tag_suffix = f"_{extra_tag}" if extra_tag else ""
    config["checkpoint_suffix"] = f"_crypto5_weekly_{obs_mode}{tag_suffix}_seed{seed}"
    return config


def restrict_env_to_window(env: MARTATradingEnv, start: pd.Timestamp, end: pd.Timestamp, asset: str = ASSET) -> MARTATradingEnv:
    idx = env._index[asset]
    in_window = np.flatnonzero((idx >= start) & (idx <= end))
    valid = np.intersect1d(env._valid_starts[asset], in_window)
    if len(valid) < 2:
        raise RuntimeError(f"Test window {start}..{end} has <2 valid bars for {asset}")
    env._valid_starts[asset] = valid
    return env


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
    buy_hold_weekly = None
    timestamps_ref = None
    test_start = pd.Timestamp(TEST_START, tz="UTC")
    test_end = pd.Timestamp(TEST_END_EXCLUSIVE, tz="UTC") - pd.Timedelta(seconds=1)

    for label, seed in SELECTED.items():
        cfg = CONDITIONS[label]
        file_label = f"{label}_seed{seed}"
        config = build_weekly_config(cfg["obs_mode"], seed, cfg["regime_path"], cfg["extra_tag"])
        config["seed"] = seed

        aligned_dfs, regime_stream_df, encoder = load_data(config)
        debate_results = load_debate_results()
        env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config)

        env = env_factory(assets=[ASSET], split="all", sequential=True, debate=False)
        test_end_clamped = min(test_end, env._index[ASSET][-1])
        restrict_env_to_window(env, test_start, test_end_clamped)

        best_path = ROOT / "rl" / "checkpoints" / f"ppo_best{config['checkpoint_suffix']}.zip"
        model = PPO.load(str(best_path), device="cpu", custom_objects={"policy_class": _policy_class_for(config)})
        traj = run_walk_forward(model, env, deterministic=True)
        marta_returns = traj["returns"]
        timestamps = traj["timestamps"]

        if buy_hold_weekly is None:
            raw_returns = env._returns[ASSET]
            index = env._index[ASSET]
            positions = np.searchsorted(index, timestamps)
            bh = []
            for i in range(1, len(positions)):
                lo, hi = positions[i - 1], positions[i]
                seg = np.nan_to_num(raw_returns[lo + 1 : hi + 1])
                bh.append(float(np.prod(1.0 + seg) - 1.0))
            buy_hold_weekly = np.asarray(bh, dtype=np.float64)
            timestamps_ref = timestamps
            print(f"Test window resolved: {timestamps[0]} -> {timestamps[-1]}  ({len(buy_hold_weekly)} weekly steps)")
        else:
            assert len(timestamps) == len(timestamps_ref), f"{file_label}: timestamp length mismatch vs reference"
            assert (timestamps == timestamps_ref).all(), f"{file_label}: timestamps differ from reference run"

        delta, lo, hi, n_blocks = block_bootstrap_delta(marta_returns, buy_hold_weekly, PPY, block_size=4)
        marta_sharpe = _sharpe(marta_returns, PPY)
        bh_sharpe = _sharpe(buy_hold_weekly, PPY)

        results[file_label] = {
            "label": label, "seed": seed, "obs_mode": cfg["obs_mode"], "extra_tag": cfg["extra_tag"],
            "val_sharpe_that_selected_this_seed": None,  # filled by caller/report if desired
            "n_weekly_steps": int(len(marta_returns)),
            "marta_sharpe_2025": marta_sharpe,
            "buy_hold_sharpe_2025": bh_sharpe,
            "delta_sharpe_bootstrap_mean": delta,
            "ci_95_lower": lo,
            "ci_95_upper": hi,
            "n_blocks": n_blocks,
            "significant": lo > 0 or hi < 0,
        }
        print(f"{file_label}: marta={marta_sharpe:.3f} buy_hold={bh_sharpe:.3f} "
              f"delta={delta:.3f} CI=[{lo:.3f},{hi:.3f}] sig={results[file_label]['significant']}")

    out_path = ROOT / "data" / "results" / "weekly_grid_test2025.json"
    out_path.write_text(json.dumps({
        "note": "Single, pre-registered 2025 test-set pass -- one seed per condition, "
                "selected by median 2024 validation Sharpe BEFORE this script was run "
                "(see module docstring). Do not re-run and cherry-pick.",
        "test_window": {"start": str(test_start), "end": str(test_end)},
        "buy_hold_sharpe_2025_shared": _sharpe(buy_hold_weekly, PPY),
        "buy_hold_total_return_2025": float(np.prod(1.0 + buy_hold_weekly) - 1.0),
        "n_weekly_steps": int(len(buy_hold_weekly)),
        "runs": results,
    }, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
