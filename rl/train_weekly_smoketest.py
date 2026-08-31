"""rl/train_weekly_smoketest.py

Native weekly-decision-cadence smoke test — the follow-up to the RL health diagnostic's
eval-time patch (Test C: forcing an already-trained hourly checkpoint to trade at most
once/week turned a -1.99 mean Sharpe into +0.28 across all 12 crypto5 grid checkpoints).
That was a patch on a policy trained for hourly credit assignment; this trains a policy
FOR weekly credit assignment from scratch, using MARTATradingEnv's new
decision_interval=168 mode (rl/env.py). Two runs only, per the decision-tree this is
meant to resolve:

  A raw, seed 0    — strongest baseline signal from the eval-patch grid
  C raw+regime, seed 0 — the condition the regime question is actually about

If native training matches or beats the eval-patch numbers for these two (A~+1.44,
C~-0.13), commit to the full A/B/C/D x 3-seed grid at native weekly cadence. If it
underperforms substantially, the eval patch was likely exploiting something about the
already-trained hourly policy rather than reflecting a cleanly learnable weekly signal.

Config changes from hourly training, all per the smoke-test design:
  - decision_interval=168 (one step = one week of hourly bars, held and aggregated)
  - idle_penalty=0.0 (flat is a legitimate strategic choice at this frequency, not a
    collapse to avoid)
  - Single phase, BTC-only, max_steps=52 (~1 trading year per episode)
  - total_timesteps and n_steps unchanged in kind from hourly training (n_steps=512,
    n_envs=8) — timed empirically at ~216 steps/s, the same order as hourly mode, since
    wall-clock cost is dominated by the policy forward/backward pass, not env bookkeeping
    (see this file's own dev log for the timing check that established this)

Run:
  conda run -n marta python rl/train_weekly_smoketest.py --obs-mode raw --seed 0
  conda run -n marta python rl/train_weekly_smoketest.py --obs-mode raw_regime --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow

from run_rl import build_config, _policy_class_for, sanity_check_env
from rl.evaluate import evaluate_ppo
from rl.train import (
    CRYPTO5_CANONICAL_ASSETS,
    CRYPTO5_PROCESSED,
    CRYPTO5_REGIME_STREAM_CAUSAL_PATH,
    train_ppo,
)
from utils.logger import logger

TOTAL_TIMESTEPS = 150_000  # ~36 policy updates at n_steps=512 x n_envs=8 (4096/update)
MAX_STEPS_PER_EPISODE = 52  # ~1 trading year of weekly decisions
DECISION_INTERVAL = 168


PLACEBO_STREAM_PATH = CRYPTO5_PROCESSED / "regime_stream_placebo.parquet"

# Full-grid label -> (obs_mode, regime_stream_path override or None, extra_tag or None).
# Mirrors the hourly grid's A/B/C/D exactly (rl/diagnose_frequency_constraint.py's RUNS
# dict) so results are directly comparable condition-for-condition.
CONDITIONS: dict[str, dict] = {
    "A": dict(obs_mode="raw", regime_path=None, extra_tag=None),
    "B": dict(obs_mode="raw_vol", regime_path=None, extra_tag=None),
    "C": dict(obs_mode="raw_regime", regime_path=None, extra_tag=None),
    "D": dict(obs_mode="raw_regime", regime_path=str(PLACEBO_STREAM_PATH), extra_tag="placebo"),
}


def build_weekly_config(obs_mode: str, seed: int, regime_path: str | None = None, extra_tag: str | None = None) -> dict:
    config = build_config(quick=False)
    config["processed_dir"] = str(CRYPTO5_PROCESSED)
    config["regime_stream_path"] = regime_path or str(CRYPTO5_REGIME_STREAM_CAUSAL_PATH)
    config["canonical_assets"] = CRYPTO5_CANONICAL_ASSETS
    config["all_assets"] = ["BTC_USDT"]
    config["obs_mode"] = obs_mode
    config["decision_interval"] = DECISION_INTERVAL
    config["device"] = "cpu"  # weekly mode never touches the encoder either; CPU is free and avoids VRAM contention
    config["phases"] = [{
        "assets": ["BTC_USDT"], "sharpe_threshold": -999, "max_steps": TOTAL_TIMESTEPS,
        "idle_penalty": 0.0,
    }]
    config["max_episode_steps"] = MAX_STEPS_PER_EPISODE
    # extra_tag disambiguates C (real causal regime) from D (placebo regime) — both use
    # obs_mode="raw_regime", so without this their checkpoints/results would collide
    # exactly like the hourly grid's original run did before --extra-tag was added there.
    tag_suffix = f"_{extra_tag}" if extra_tag else ""
    config["checkpoint_suffix"] = f"_crypto5_weekly_{obs_mode}{tag_suffix}_seed{seed}"
    return config


def run_one(label: str, seed: int) -> dict:
    cfg = CONDITIONS[label]
    file_label = f"{label}_seed{seed}"  # e.g. "D_seed1" -- matches the hourly grid's naming
    config = build_weekly_config(cfg["obs_mode"], seed, cfg["regime_path"], cfg["extra_tag"])
    config["seed"] = seed

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("marta_weekly_grid")
    with mlflow.start_run(run_name=f"weekly_{file_label}"):
        mlflow.log_params({
            "label": label, "obs_mode": cfg["obs_mode"], "seed": seed,
            "extra_tag": cfg["extra_tag"] or "", "decision_interval": DECISION_INTERVAL,
            "total_timesteps": TOTAL_TIMESTEPS, "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            "idle_penalty": 0.0,
        })

        t0 = time.time()
        train_out = train_ppo(config, pre_train_check=sanity_check_env)
        train_elapsed = time.time() - t0
        mlflow.log_metric("train_wall_seconds", train_elapsed)
        logger.success(f"[{file_label}] training done in {train_elapsed:.0f}s")

        # Reload best-by-validation-Sharpe before evaluating (run_rl.py's convention) —
        # train_out["model"] is just the live end-of-.learn() state (FINAL), which the
        # first weekly smoke test mistakenly evaluated directly and, for raw_regime,
        # materially understated true performance (Sharpe 0.53 on final vs 2.21 on
        # best) — final and best are genuinely different checkpoints here.
        from stable_baselines3 import PPO as _PPO
        best_path = train_out["best_path"]
        if best_path.exists():
            model = _PPO.load(str(best_path), device="cpu", custom_objects={"policy_class": _policy_class_for(config)})
        else:
            model = train_out["model"]
        env_factory = train_out["env_factory"]
        eval_path = Path(f"data/results/weekly_grid_{file_label}.json")
        eval_res = evaluate_ppo(model, env_factory, config, output_path=eval_path)

        m = eval_res["marta_ppo"]
        mlflow.log_metrics({
            "val_sharpe": m["sharpe"], "val_total_return": m["total_return"],
            "val_max_drawdown": m["max_drawdown"], "val_n_meaningful_trades": m["n_meaningful_trades"],
        })

        summary = {
            "label": label, "obs_mode": cfg["obs_mode"], "seed": seed,
            "sharpe": m["sharpe"], "total_return": m["total_return"],
            "max_drawdown": m["max_drawdown"], "n_meaningful_trades": m["n_meaningful_trades"],
            "train_wall_seconds": train_elapsed,
        }
        print(json.dumps(summary, indent=2))
        Path(f"data/results/weekly_grid_summary_{file_label}.json").write_text(json.dumps(summary, indent=2))
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, choices=list(CONDITIONS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="run all 4 conditions x 3 seeds sequentially")
    args = parser.parse_args()

    if args.all:
        for label in CONDITIONS:
            for seed in (0, 1, 2):
                run_one(label, seed)
    elif args.label is not None:
        run_one(args.label, args.seed)
    else:
        parser.error("pass --label {A,B,C,D} --seed N, or --all")


if __name__ == "__main__":
    main()
