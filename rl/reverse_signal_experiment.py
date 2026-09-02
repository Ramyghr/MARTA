"""Reverse-the-signal diagnostic: long<->short every decision, hold stays hold.

Applied to the same 12 already-trained BTC-only native-weekly checkpoints as
rl/ensemble_weekly_backtest.py (conditions A/B/C/D x seeds 0/1/2, see
docs/RESULTS_AND_METHODOLOGY.md Section 10). No new training -- read-only
re-evaluation of existing checkpoints.

Rationale: if a policy's Sharpe is negative or below buy-hold, inverting every
call and re-measuring answers "does this policy have anti-correlated skill
(systematically wrong direction) or is it just noise?" -- anti-correlated skill
would show up as reversal turning a loser into a winner; noise would show up as
reversal landing close to 0 or symmetric around the original in magnitude.

Reversal rule (env.py:121's _DIRECTION_SIGN = {0: long/+1, 1: short/-1, 2: hold/0}):
  direction 0 (long)  -> 1 (short)
  direction 1 (short) -> 0 (long)
  direction 2 (hold)  -> 2 (hold), unchanged
position_fraction is left untouched in all cases (irrelevant when direction=2
anyway, since target_signed = _DIRECTION_SIGN[direction] * frac * risk_cap = 0).

Uses debate=True (matching evaluate_ppo()'s default), the same config this
project's Eval Reliability Audit requires matching across comparable eval runs
-- see rl/ensemble_weekly_backtest.py's docstring for the bug this caught last
time it was gotten wrong.

Run:
  /home/ramy/miniconda3/envs/marta/bin/python rl/reverse_signal_experiment.py
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
from rl.evaluate import _EVAL_SEED, compute_metrics, run_walk_forward
from rl.train import build_env_factory, load_data, load_debate_results
from rl.train_weekly_smoketest import CONDITIONS, DECISION_INTERVAL, build_weekly_config
from utils.reproducibility import set_seed

PPY = (24 * 365) / DECISION_INTERVAL
SEEDS = (0, 1, 2)
_REVERSE_DIRECTION = {0: 1, 1: 0, 2: 2}  # long<->short, hold unchanged


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


def run_walk_forward_reversed(model: PPO, env: MARTATradingEnv, deterministic: bool = True) -> dict:
    """Like rl.evaluate.run_walk_forward, but inverts the model's chosen direction
    (long<->short, hold unchanged) before stepping the env -- see module docstring."""
    set_seed(_EVAL_SEED)
    obs, info = env.reset(seed=_EVAL_SEED)
    portfolio_values = [info["portfolio_value"]]
    positions = [info["signed_position"]]
    timestamps = [env._index[env._asset][env._t]]
    returns: list[float] = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        decoded = MARTATradingEnv.unflatten_action(action)
        reversed_action = np.array(
            [_REVERSE_DIRECTION[decoded["direction"]], decoded["position_fraction"]], dtype=np.float32
        )
        obs, reward, terminated, truncated, info = env.step(reversed_action)
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
    }


def main() -> None:
    results = {}

    for label, cfg in CONDITIONS.items():
        config0 = build_weekly_config(cfg["obs_mode"], 0, cfg["regime_path"], cfg["extra_tag"])
        aligned_dfs, regime_stream_df, encoder = load_data(config0)
        debate_results = load_debate_results()
        env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config0)
        env = env_factory(assets=["BTC_USDT"], split="val", sequential=True, zero_regime=False, debate=True)

        raw_returns = env._returns["BTC_USDT"]
        index = env._index["BTC_USDT"]
        buy_hold_weekly = None

        for seed in SEEDS:
            config = build_weekly_config(cfg["obs_mode"], seed, cfg["regime_path"], cfg["extra_tag"])
            best_path = ROOT / "rl" / "checkpoints" / f"ppo_best{config['checkpoint_suffix']}.zip"
            model = PPO.load(str(best_path), device="cpu", custom_objects={"policy_class": _policy_class_for(config)})

            orig_traj = run_walk_forward(model, env, deterministic=True)
            orig_metrics = compute_metrics(
                orig_traj["returns"], orig_traj["portfolio_values"], orig_traj["positions"], periods_per_year=PPY
            )

            rev_traj = run_walk_forward_reversed(model, env, deterministic=True)
            rev_metrics = compute_metrics(
                rev_traj["returns"], rev_traj["portfolio_values"], rev_traj["positions"], periods_per_year=PPY
            )

            if buy_hold_weekly is None:
                idx = np.searchsorted(index, rev_traj["timestamps"])
                bh = []
                for i in range(1, len(idx)):
                    lo, hi = idx[i - 1], idx[i]
                    seg = np.nan_to_num(raw_returns[lo + 1 : hi + 1])
                    bh.append(float(np.prod(1.0 + seg) - 1.0))
                buy_hold_weekly = np.asarray(bh, dtype=np.float64)
            bh_sharpe = _sharpe(buy_hold_weekly, PPY)

            delta_vs_bh, lo_bh, hi_bh, n_blocks = block_bootstrap_delta(rev_traj["returns"], buy_hold_weekly, PPY)
            delta_vs_orig, lo_orig, hi_orig, _ = block_bootstrap_delta(rev_traj["returns"], orig_traj["returns"], PPY)

            file_label = f"{label}_seed{seed}"
            result = {
                "label": label, "obs_mode": cfg["obs_mode"], "extra_tag": cfg["extra_tag"], "seed": seed,
                "n_weekly_steps": int(len(rev_traj["returns"])),
                "original_sharpe": orig_metrics["sharpe"],
                "original_n_meaningful_trades": orig_metrics["n_meaningful_trades"],
                "reversed_sharpe": rev_metrics["sharpe"],
                "reversed_total_return": rev_metrics["total_return"],
                "reversed_max_drawdown": rev_metrics["max_drawdown"],
                "reversed_n_meaningful_trades": rev_metrics["n_meaningful_trades"],
                "buy_hold_sharpe": bh_sharpe,
                "delta_reversed_vs_buyhold_bootstrap_mean": delta_vs_bh,
                "ci_95_lower_vs_buyhold": lo_bh,
                "ci_95_upper_vs_buyhold": hi_bh,
                "significant_vs_buyhold": lo_bh > 0 or hi_bh < 0,
                "delta_reversed_vs_original_bootstrap_mean": delta_vs_orig,
                "ci_95_lower_vs_original": lo_orig,
                "ci_95_upper_vs_original": hi_orig,
                "significant_vs_original": lo_orig > 0 or hi_orig < 0,
                "n_blocks": n_blocks,
            }
            results[file_label] = result

            print(
                f"{file_label} ({cfg['obs_mode']}{'+' + cfg['extra_tag'] if cfg['extra_tag'] else ''}): "
                f"original={orig_metrics['sharpe']:+.3f}  reversed={rev_metrics['sharpe']:+.3f}  "
                f"buy_hold={bh_sharpe:+.3f}  "
                f"rev_vs_bh={delta_vs_bh:+.3f} CI=[{lo_bh:+.3f},{hi_bh:+.3f}] sig={result['significant_vs_buyhold']}  "
                f"rev_vs_orig={delta_vs_orig:+.3f} CI=[{lo_orig:+.3f},{hi_orig:+.3f}] sig={result['significant_vs_original']}"
            )

    out_path = ROOT / "data" / "results" / "weekly_reversed_signal_experiment.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {out_path}")

    n_flipped_positive = sum(
        1 for r in results.values() if r["original_sharpe"] < 0 and r["reversed_sharpe"] > 0
    )
    n_orig_negative = sum(1 for r in results.values() if r["original_sharpe"] < 0)
    n_reversed_beats_original = sum(1 for r in results.values() if r["reversed_sharpe"] > r["original_sharpe"])
    print(
        f"\n{len(results)} runs: {n_orig_negative} had negative original Sharpe "
        f"({n_flipped_positive} of those flipped positive on reversal); "
        f"{n_reversed_beats_original}/{len(results)} reversed Sharpe > original Sharpe"
    )


if __name__ == "__main__":
    main()
