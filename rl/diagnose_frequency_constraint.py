"""rl/diagnose_frequency_constraint.py

RL health diagnostic, Step 2 / Test C — does forcing an already-trained checkpoint to
trade at a weekly cadence (instead of its learned ~9-18h cadence) recover the profitability
its natural trading frequency destroys?

Eval-only. No retraining, no changes to any checkpoint's weights. Confirmed on 2 of 12
crypto5 grid checkpoints already (C-seed0: Sharpe -2.304 -> -0.134; A-seed0: -1.609 ->
+1.444, ~matching agents_no_regime's own 51-trade/year cadence and Sharpe). This script
scales that same test across the remaining checkpoints and logs each result to MLflow
(experiment "marta_frequency_constraint_diagnostic") so progress is visible in
`mlflow ui` while it runs, rather than only in a log file.

Baseline Sharpe/return per checkpoint is NOT recomputed here — it's already known exactly
from the recovered GRID_ppo_evaluation_*.json files (verified against checkpoint
provenance earlier in this audit) and passed in as a reference value alongside the fresh
Test C result, so this script only pays for the one new walk-forward pass it actually needs.

Run:
  conda run -n marta python rl/diagnose_frequency_constraint.py --label "A_seed1"
  conda run -n marta python rl/diagnose_frequency_constraint.py --all   # loops every remaining checkpoint sequentially
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_rl import build_config, _policy_class_for  # noqa: E402
from rl.env import cvar_from_vol, risk_cap_from_vol  # noqa: E402
from rl.evaluate import compute_metrics  # noqa: E402
from rl.train import (  # noqa: E402
    CRYPTO5_CANONICAL_ASSETS,
    CRYPTO5_CURRICULUM_PHASES,
    CRYPTO5_PROCESSED,
    CRYPTO5_REGIME_STREAM_CAUSAL_PATH,
    build_env_factory,
    load_data,
    load_debate_results,
)

PLACEBO_STREAM_PATH = CRYPTO5_PROCESSED / "regime_stream_placebo.parquet"
FREQ_CAP_STEPS = 168  # ~1 week of hourly bars

# label -> (obs_mode, checkpoint path, regime_stream_path override or None, known baseline sharpe/return)
# Baseline numbers pulled from data/results/GRID_ppo_evaluation_{cond}_seed{seed}.json,
# already verified against checkpoint provenance earlier in this audit.
_ALREADY_DONE = {"A_seed0", "C_seed0"}
RUNS: dict[str, dict] = {
    "A_seed1": dict(obs_mode="raw", ckpt="ppo_best_crypto5_raw_seed1.zip", regime_path=None,
                     baseline_sharpe=-1.271, baseline_return=-0.209),
    "A_seed2": dict(obs_mode="raw", ckpt="ppo_best_crypto5_raw_seed2.zip", regime_path=None,
                     baseline_sharpe=-2.826, baseline_return=-0.459),
    "B_seed0": dict(obs_mode="raw_vol", ckpt="ppo_best_crypto5_raw_vol_seed0.zip", regime_path=None,
                     baseline_sharpe=-1.441, baseline_return=-0.203),
    "B_seed1": dict(obs_mode="raw_vol", ckpt="ppo_best_crypto5_raw_vol_seed1.zip", regime_path=None,
                     baseline_sharpe=-2.036, baseline_return=-0.329),
    "B_seed2": dict(obs_mode="raw_vol", ckpt="ppo_best_crypto5_raw_vol_seed2.zip", regime_path=None,
                     baseline_sharpe=-2.323, baseline_return=-0.379),
    "C_seed1": dict(obs_mode="raw_regime", ckpt="ppo_best_crypto5_raw_regime_seed1.zip", regime_path=None,
                     baseline_sharpe=-1.207, baseline_return=-0.225),
    "C_seed2": dict(obs_mode="raw_regime", ckpt="ppo_best_crypto5_raw_regime_seed2.zip", regime_path=None,
                     baseline_sharpe=-2.182, baseline_return=-0.337),
    "D_seed0": dict(obs_mode="raw_regime", ckpt="ppo_best_crypto5_raw_regime_placebo_seed0.zip",
                     regime_path=str(PLACEBO_STREAM_PATH), baseline_sharpe=-2.655, baseline_return=-0.429),
    "D_seed1": dict(obs_mode="raw_regime", ckpt="ppo_best_crypto5_raw_regime_placebo_seed1.zip",
                     regime_path=str(PLACEBO_STREAM_PATH), baseline_sharpe=-1.828, baseline_return=-0.303),
    "D_seed2": dict(obs_mode="raw_regime", ckpt="ppo_best_crypto5_raw_regime_placebo_seed2.zip",
                     regime_path=str(PLACEBO_STREAM_PATH), baseline_sharpe=-2.152, baseline_return=-0.354),
}


def build_env_and_model(obs_mode: str, checkpoint: str, regime_path: str | None):
    config = build_config(quick=False)
    config["phases"] = copy.deepcopy(CRYPTO5_CURRICULUM_PHASES)
    config["processed_dir"] = str(CRYPTO5_PROCESSED)
    config["regime_stream_path"] = regime_path or str(CRYPTO5_REGIME_STREAM_CAUSAL_PATH)
    config["checkpoint_suffix"] = "_crypto5"
    config["obs_mode"] = obs_mode
    config["canonical_assets"] = CRYPTO5_CANONICAL_ASSETS
    config["all_assets"] = ["BTC_USDT"]
    config["device"] = "cpu"
    aligned_dfs, regime_stream_df, encoder = load_data(config)
    debate_results = load_debate_results()
    env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config)
    model = PPO.load(checkpoint, device="cpu", custom_objects={"policy_class": _policy_class_for(config)})
    env = env_factory(assets=["BTC_USDT"], split="val", sequential=True, idle_penalty=0.0,
                       device="cpu", min_hold_bars=4)
    return model, env


def run_freq_capped(model, env, freq_cap_steps: int = FREQ_CAP_STEPS) -> dict:
    obs, info = env.reset(seed=42)
    positions, portfolio_values, returns = [info["signed_position"]], [info["portfolio_value"]], []
    steps_since_trade = 10_000
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action, dtype=np.float64).copy()
        if steps_since_trade < freq_cap_steps:
            prev_signed = env._signed_position
            asset = env._asset
            vol = float(env._vol[asset][env._t])
            risk_cap = risk_cap_from_vol(vol) * env.max_leverage
            if abs(prev_signed) < 1e-9:
                action[0], action[1] = 2, 0.01
            else:
                action[0] = 0 if prev_signed > 0 else 1
                action[1] = float(np.clip(abs(prev_signed) / max(risk_cap, 1e-8), 0.01, 0.99))
        prev_signed = env._signed_position
        obs, reward, terminated, truncated, info = env.step(action)
        traded = abs(info["signed_position"] - prev_signed) > 0.005
        steps_since_trade = 0 if traded else steps_since_trade + 1
        positions.append(info["signed_position"])
        portfolio_values.append(info["portfolio_value"])
        returns.append(info["pnl_return"])
        done = terminated or truncated

    positions, portfolio_values, returns = np.array(positions), np.array(portfolio_values), np.array(returns)
    m = compute_metrics(returns, portfolio_values, positions)
    m["total_return"] = float(portfolio_values[-1] / portfolio_values[0] - 1.0)
    return m


def run_one(label: str) -> dict:
    cfg = RUNS[label]
    ckpt_path = f"rl/checkpoints/{cfg['ckpt']}"
    t0 = time.time()
    model, env = build_env_and_model(cfg["obs_mode"], ckpt_path, cfg["regime_path"])
    m = run_freq_capped(model, env)
    elapsed = time.time() - t0

    result = {
        "label": label, "obs_mode": cfg["obs_mode"], "checkpoint": ckpt_path,
        "baseline_sharpe": cfg["baseline_sharpe"], "baseline_return": cfg["baseline_return"],
        "testC_sharpe": m["sharpe"], "testC_return": m["total_return"],
        "testC_max_drawdown": m["max_drawdown"], "testC_n_meaningful_trades": m["n_meaningful_trades"],
        "sharpe_delta": m["sharpe"] - cfg["baseline_sharpe"],
        "elapsed_seconds": elapsed,
    }

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("marta_frequency_constraint_diagnostic")
    with mlflow.start_run(run_name=f"freqcap_{label}"):
        mlflow.log_params({"label": label, "obs_mode": cfg["obs_mode"], "checkpoint": ckpt_path,
                            "freq_cap_steps": FREQ_CAP_STEPS})
        mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, (int, float))})

    out_path = Path(f"data/results/freqcap_{label}.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[{label}] baseline sharpe={cfg['baseline_sharpe']:.3f} return={cfg['baseline_return']:.3f}  |  "
          f"TestC sharpe={m['sharpe']:.3f} return={m['total_return']:.3f} n_trades={m['n_meaningful_trades']}  "
          f"({elapsed:.0f}s)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default=None, choices=list(RUNS.keys()))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for label in RUNS:
            run_one(label)
    elif args.label:
        run_one(args.label)
    else:
        parser.error("pass --label <name> or --all")
