"""Majority-vote ensemble of the 3 seeds per condition, BTC-only native-weekly grid.

Idea taken from the FinRL Contest 2025 (AlphaSeek Crypto) starter kit's
`task2_eval.py::_ensemble_action` -- train N agents independently, then at each
decision step combine their actions by majority vote instead of trusting one
checkpoint. Applied here to the 12 already-trained checkpoints from
`rl/train_weekly_smoketest.py --all` (conditions A/B/C/D x seeds 0/1/2, see
docs/RESULTS_AND_METHODOLOGY.md Section 10). No new training -- read-only
analysis of existing checkpoints, same spirit as rl/reconcile_weekly_grid_crypto5.py.

Ensemble rule per weekly decision step (all 3 seeds share the same obs_mode/obs
space within a condition, so voting within-condition is well-defined; the 4
conditions are NOT pooled together since B/C have different obs dims than A):
  1. Each of the 3 models predicts its own {direction, position_fraction} on the
     SAME observation (env.unflatten_action decodes each raw Box(2) action).
  2. Majority direction wins (>=2/3 agreement); position_fraction is the mean of
     the fraction proposed by the agreeing models only.
  3. On a genuine 3-way tie (all three models pick a different direction --
     possible since there are exactly 3 discrete choices and 3 voters), the
     ensemble defaults to hold. This is a deliberate conservative tie-break, not
     an artifact -- "no consensus" should not force a position.

Reports, per condition: each seed's own Sharpe (recomputed on this run's exact
env instance), the ensemble's Sharpe, and a block-bootstrap delta-Sharpe of the
ensemble vs the SAME weekly-aligned buy-hold series reconcile_weekly_grid_crypto5.py
and docs Section 10 use (block_size=4 ~ one month, 1000 resamples, seed=42) --
so this number is directly comparable to the already-published per-seed deltas
in data/results/weekly_grid_reconciled.json.

Run:
  /home/ramy/miniconda3/envs/marta/bin/python rl/ensemble_weekly_backtest.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ramy/Documents/MARTA/marta")
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from run_rl import _policy_class_for
from rl.env import MARTATradingEnv
from rl.evaluate import _EVAL_SEED, compute_metrics, run_walk_forward
from rl.train import build_env_factory, load_data, load_debate_results
from rl.train_weekly_smoketest import CONDITIONS, DECISION_INTERVAL, build_weekly_config
from utils.reproducibility import set_seed

PPY = (24 * 365) / DECISION_INTERVAL
SEEDS = (0, 1, 2)


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


def run_walk_forward_ensemble(models: list[PPO], env: MARTATradingEnv, deterministic: bool = True) -> dict:
    """Like rl.evaluate.run_walk_forward, but combines len(models) models' actions
    by majority vote (see module docstring) before stepping the shared env."""
    set_seed(_EVAL_SEED)
    obs, info = env.reset(seed=_EVAL_SEED)
    portfolio_values = [info["portfolio_value"]]
    positions = [info["signed_position"]]
    timestamps = [env._index[env._asset][env._t]]
    returns: list[float] = []
    n_no_consensus = 0

    done = False
    while not done:
        votes = []
        for m in models:
            action, _ = m.predict(obs, deterministic=deterministic)
            votes.append(MARTATradingEnv.unflatten_action(action))

        directions = [v["direction"] for v in votes]
        counts = Counter(directions)
        top_count = max(counts.values())
        top_dirs = [d for d, c in counts.items() if c == top_count]
        if len(top_dirs) == 1 and top_count >= 2:
            majority_dir = top_dirs[0]
            frac = float(np.mean([v["position_fraction"] for v in votes if v["direction"] == majority_dir]))
        else:
            majority_dir, frac = 2, 0.01  # 3-way tie -> conservative hold, not an artifact
            n_no_consensus += 1

        combined_action = np.array([majority_dir, frac], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(combined_action)
        portfolio_values.append(info["portfolio_value"])
        positions.append(info["signed_position"])
        timestamps.append(env._index[env._asset][env._t])
        returns.append(info["pnl_return"])
        done = terminated or truncated

    return {
        "portfolio_values": np.asarray(portfolio_values, dtype=np.float64),
        "positions": np.asarray(positions, dtype=np.float64),
        "timestamps": pd.DatetimeIndex(timestamps),
        "returns": np.asarray(returns, dtype=np.float64),
        "n_no_consensus_steps": n_no_consensus,
    }


def main() -> None:
    results = {}

    for label, cfg in CONDITIONS.items():
        config0 = build_weekly_config(cfg["obs_mode"], 0, cfg["regime_path"], cfg["extra_tag"])
        aligned_dfs, regime_stream_df, encoder = load_data(config0)
        debate_results = load_debate_results()
        env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config0)
        # debate=True (not False) deliberately -- matches evaluate_ppo()'s own default
        # env-building convention (rl/evaluate.py:585), which is what produced the
        # per-seed numbers in data/results/weekly_grid_reconciled.json this script's
        # seed_sharpes are meant to be comparable to. An earlier draft of this script
        # copied rl/reconcile_weekly_grid_crypto5.py's debate=False, which is correct for
        # THAT (crypto5 pilot) script's own comparisons but silently produced a 1.545 vs
        # 1.677 Sharpe mismatch on A_seed0 here -- exactly the "match the debate-cache
        # configuration when comparing eval runs" trap CLAUDE.md's eval-reliability audit
        # warns about. Note debate_results.json has grown since these checkpoints were
        # trained (dense BTC coverage added 2026-09-02), so this still won't be bit-exact
        # against the original numbers -- see this run's printed seed_sharpes for the
        # current, correctly-configured values.
        env = env_factory(assets=["BTC_USDT"], split="val", sequential=True, zero_regime=False, debate=True)

        models = []
        for seed in SEEDS:
            config = build_weekly_config(cfg["obs_mode"], seed, cfg["regime_path"], cfg["extra_tag"])
            best_path = ROOT / "rl" / "checkpoints" / f"ppo_best{config['checkpoint_suffix']}.zip"
            models.append(PPO.load(str(best_path), device="cpu", custom_objects={"policy_class": _policy_class_for(config)}))

        seed_sharpes = []
        for seed, model in zip(SEEDS, models):
            traj = run_walk_forward(model, env, deterministic=True)
            m = compute_metrics(traj["returns"], traj["portfolio_values"], traj["positions"], periods_per_year=PPY)
            seed_sharpes.append(m["sharpe"])

        ens_traj = run_walk_forward_ensemble(models, env, deterministic=True)
        ens_metrics = compute_metrics(
            ens_traj["returns"], ens_traj["portfolio_values"], ens_traj["positions"], periods_per_year=PPY
        )

        raw_returns = env._returns["BTC_USDT"]
        index = env._index["BTC_USDT"]
        idx = np.searchsorted(index, ens_traj["timestamps"])
        bh = []
        for i in range(1, len(idx)):
            lo, hi = idx[i - 1], idx[i]
            seg = np.nan_to_num(raw_returns[lo + 1 : hi + 1])
            bh.append(float(np.prod(1.0 + seg) - 1.0))
        bh = np.asarray(bh, dtype=np.float64)
        bh_sharpe = _sharpe(bh, PPY)

        delta, lo_ci, hi_ci, n_blocks = block_bootstrap_delta(ens_traj["returns"], bh, PPY, block_size=4)
        sig = lo_ci > 0 or hi_ci < 0

        result = {
            "label": label,
            "obs_mode": cfg["obs_mode"],
            "extra_tag": cfg["extra_tag"],
            "n_weekly_steps": int(len(ens_traj["returns"])),
            "seed_sharpes": {str(s): sh for s, sh in zip(SEEDS, seed_sharpes)},
            "seed_sharpe_mean": float(np.mean(seed_sharpes)),
            "seed_sharpe_std": float(np.std(seed_sharpes)),
            "ensemble_sharpe": ens_metrics["sharpe"],
            "ensemble_total_return": ens_metrics["total_return"],
            "ensemble_max_drawdown": ens_metrics["max_drawdown"],
            "ensemble_n_meaningful_trades": ens_metrics["n_meaningful_trades"],
            "ensemble_n_no_consensus_steps": ens_traj["n_no_consensus_steps"],
            "buy_hold_sharpe": bh_sharpe,
            "delta_ensemble_vs_buyhold_bootstrap_mean": delta,
            "ci_95_lower": lo_ci,
            "ci_95_upper": hi_ci,
            "n_blocks": n_blocks,
            "significant": sig,
        }
        results[label] = result

        print(
            f"{label} ({cfg['obs_mode']}{'+' + cfg['extra_tag'] if cfg['extra_tag'] else ''}): "
            f"seeds={[f'{s:.3f}' for s in seed_sharpes]} mean={result['seed_sharpe_mean']:.3f} "
            f"std={result['seed_sharpe_std']:.3f}  ensemble={ens_metrics['sharpe']:.3f}  "
            f"buy_hold={bh_sharpe:.3f}  delta_vs_bh={delta:+.3f} CI=[{lo_ci:+.3f},{hi_ci:+.3f}] sig={sig}  "
            f"no_consensus_steps={ens_traj['n_no_consensus_steps']}/{result['n_weekly_steps']}"
        )

    out_path = ROOT / "data" / "results" / "weekly_ensemble_backtest.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
