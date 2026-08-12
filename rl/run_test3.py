"""TEST 3A / 3B — does PPO learn anything from real BTC data, and does swapping in the
MARTA observation (frozen encoder embedding) help or hurt, relative to simple raw
features? Uses ``rl.minimal_btc_env.MinimalBTCEnv`` — same class, same reward, same
action space for both; ONLY ``observation_mode`` differs.

SB3 PPO's library defaults are used throughout (no hyperparameter tuning) with SB3's
stock "MlpPolicy" (not the custom MARTAActorCriticPolicy — Beta position sizing is
explicitly out of scope for TEST 3, see minimal_btc_env.py's docstring).

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.run_test3                      # both 3A and 3B
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.run_test3 --mode simple         # 3A only
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.run_test3 --mode marta          # 3B only
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.run_test3 --steps 30000 --eval-freq 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from encoder.model import TFTRegimeEncoder
from rl.evaluate import compute_metrics
from rl.minimal_btc_env import MinimalBTCEnv, build_env
from utils.assets import aligned_path
from utils.logger import logger

_console = Console()
_PROCESSED = Path("data/processed")
_ENCODER_PATH = Path("encoder/checkpoints/best_encoder.pt")
_RESULTS_DIR = Path("data/results")
_SEED = 42

_PPO_DIAGNOSTIC_KEYS = [
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
]


def load_encoder(device: str = "cpu") -> TFTRegimeEncoder:
    encoder = TFTRegimeEncoder(input_dim=14, hidden_dim=128)
    ckpt = torch.load(_ENCODER_PATH, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["model_state"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


# --------------------------------------------------------------------- trajectory runner
def run_episode(env: MinimalBTCEnv, action_fn: Callable[[np.ndarray, int], int]) -> dict[str, np.ndarray]:
    """Walk env sequentially (env must have sequential=True) with a given action policy."""
    obs, info = env.reset(seed=_SEED)
    portfolio_values = [info["portfolio_value"]]
    positions = [info["signed_position"]]
    pnl_returns: list[float] = []
    next_returns: list[float] = []
    directions: list[int] = []

    step_idx = 0
    done = False
    while not done:
        action = action_fn(obs, step_idx)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        positions.append(info["signed_position"])
        pnl_returns.append(info["pnl_return"])
        next_returns.append(info["next_return"])
        directions.append(info["direction"])
        step_idx += 1
        done = terminated or truncated

    return {
        "portfolio_values": np.asarray(portfolio_values, dtype=np.float64),
        "positions": np.asarray(positions, dtype=np.float64),
        "pnl_returns": np.asarray(pnl_returns, dtype=np.float64),
        "next_returns": np.asarray(next_returns, dtype=np.float64),
        "directions": np.asarray(directions, dtype=np.int64),
    }


def summarize_trajectory(traj: dict[str, np.ndarray]) -> dict[str, Any]:
    metrics = compute_metrics(traj["pnl_returns"], traj["portfolio_values"], traj["positions"])
    pos = traj["positions"][1:]
    nret = traj["next_returns"]
    # Genuine forecast accuracy: did the DIRECTION the agent chose (sign of position) match
    # the sign of the return that actually happened next? (NOT sign(position*return) vs
    # sign(pnl_return) — that pair is tautologically ~100% whenever cost << gross P&L, since
    # pnl_return = position*next_return - cost, and only tests env sign-consistency, which
    # rl.diagnose_env.py already verified separately. This is a real market-direction check.)
    mask = (np.abs(pos) > 1e-9) & (nret != 0)
    direction_accuracy = (
        float((np.sign(pos)[mask] == np.sign(nret)[mask]).mean())
        if mask.sum() > 0 else None
    )
    dirs = traj["directions"]
    n = len(dirs)
    action_dist = {
        "long_frac": float((dirs == 0).mean()) if n else 0.0,
        "short_frac": float((dirs == 1).mean()) if n else 0.0,
        "hold_frac": float((dirs == 2).mean()) if n else 0.0,
    }
    return {
        "mean_reward": float(traj["pnl_returns"].mean()),
        "std_reward": float(traj["pnl_returns"].std()),
        "total_return": metrics["total_return"],
        "sharpe": metrics["sharpe"],
        "n_meaningful_trades": metrics["n_meaningful_trades"],
        "mean_trade_duration_hours": metrics["mean_trade_duration_hours"],
        "hold_ratio": metrics["hold_ratio"],
        "direction_accuracy": direction_accuracy,
        "action_distribution": action_dist,
        "n_samples": n,
    }


def make_fixed_policy(direction: int) -> Callable[[np.ndarray, int], int]:
    return lambda obs, step_idx: direction


def make_random_policy(seed: int) -> Callable[[np.ndarray, int], int]:
    rng = np.random.default_rng(seed)
    return lambda obs, step_idx: int(rng.integers(0, 3))


def make_model_policy(model: PPO) -> Callable[[np.ndarray, int], int]:
    def _policy(obs: np.ndarray, step_idx: int) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return _policy


# --------------------------------------------------------------------- training callback
class LearningCurveCallback(BaseCallback):
    """Periodic val-set eval + PPO training diagnostics, for the TEST 3 learning curve."""

    def __init__(self, val_env_factory: Callable[[], MinimalBTCEnv], eval_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.val_env_factory = val_env_factory
        self.eval_freq = eval_freq
        self.history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        val_env = self.val_env_factory()
        traj = run_episode(val_env, make_model_policy(self.model))
        summary = summarize_trajectory(traj)
        diagnostics = {
            k.split("/")[-1]: self.model.logger.name_to_value.get(k) for k in _PPO_DIAGNOSTIC_KEYS
        }
        record = {"timesteps": self.num_timesteps, **summary, **diagnostics}
        self.history.append(record)
        logger.info(
            f"[t={self.num_timesteps}] val_mean_reward={summary['mean_reward']:.6f} "
            f"total_return={summary['total_return']:.4%} sharpe={summary['sharpe']:.3f} "
            f"n_trades={summary['n_meaningful_trades']} "
            f"entropy_loss={diagnostics.get('entropy_loss')} "
            f"approx_kl={diagnostics.get('approx_kl')} "
            f"explained_variance={diagnostics.get('explained_variance')}"
        )
        return True


# --------------------------------------------------------------------- one full mode run
def run_mode(
    aligned_df: pd.DataFrame,
    encoder: TFTRegimeEncoder,
    mode: str,
    total_steps: int,
    eval_freq: int,
) -> dict[str, Any]:
    _console.rule(f"[bold]TEST 3{'A' if mode == 'simple' else 'B'} — observation_mode={mode}")

    def _train_env() -> MinimalBTCEnv:
        return build_env(aligned_df, encoder if mode == "marta" else None, mode, split="train", sequential=False)

    def _val_env() -> MinimalBTCEnv:
        return build_env(aligned_df, encoder if mode == "marta" else None, mode, split="val", sequential=True)

    train_env = Monitor(_train_env())

    # SB3 PPO library defaults throughout — no hyperparameter tuning (per audit instructions).
    model = PPO("MlpPolicy", train_env, verbose=0, seed=_SEED)

    curve_cb = LearningCurveCallback(_val_env, eval_freq=eval_freq)
    # Record pre-training (random-init policy) baseline as timestep 0 of the learning curve.
    pre_traj = run_episode(_val_env(), make_model_policy(model))
    pre_summary = summarize_trajectory(pre_traj)
    curve_cb.history.append({"timesteps": 0, **pre_summary,
                              **{k.split("/")[-1]: None for k in _PPO_DIAGNOSTIC_KEYS}})
    logger.info(f"[t=0, pre-training] val_mean_reward={pre_summary['mean_reward']:.6f}")

    model.learn(total_timesteps=total_steps, callback=curve_cb, progress_bar=False)

    # ---------------------------------------------------------------- final evaluation
    policies = {
        "PPO_TRAINED": make_model_policy(model),
        "RANDOM": make_random_policy(_SEED + 1),
        "ALWAYS_LONG": make_fixed_policy(0),
        "ALWAYS_SHORT": make_fixed_policy(1),
        "ALWAYS_HOLD": make_fixed_policy(2),
    }
    final_results: dict[str, Any] = {}
    for name, policy in policies.items():
        traj = run_episode(_val_env(), policy)
        final_results[name] = summarize_trajectory(traj)
        logger.success(f"[final] {name}: {final_results[name]}")

    out = {
        "mode": mode,
        "total_steps": total_steps,
        "eval_freq": eval_freq,
        "learning_curve": curve_cb.history,
        "final_evaluation": final_results,
    }
    out_path = _RESULTS_DIR / f"test3_{mode}.json"
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    logger.success(f"TEST 3{'A' if mode == 'simple' else 'B'} results written to {out_path}")
    return out


def print_comparison(results: dict[str, dict[str, Any]]) -> None:
    table = Table(title="TEST 3A vs 3B — final evaluation on BTC val split")
    table.add_column("Mode")
    table.add_column("Policy")
    table.add_column("Mean Reward", justify="right")
    table.add_column("Total Return", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("N Trades", justify="right")
    table.add_column("Avg Hold (h)", justify="right")
    table.add_column("Dir. Acc.", justify="right")
    table.add_column("L/S/H %", justify="right")
    for mode, res in results.items():
        for policy_name, s in res["final_evaluation"].items():
            dist = s["action_distribution"]
            da = "n/a" if s["direction_accuracy"] is None else f"{s['direction_accuracy']:.1%}"
            table.add_row(
                mode, policy_name,
                f"{s['mean_reward']:.6f}", f"{s['total_return']:.4%}", f"{s['sharpe']:.3f}",
                str(s["n_meaningful_trades"]), f"{s['mean_trade_duration_hours']:.1f}", da,
                f"{dist['long_frac']:.0%}/{dist['short_frac']:.0%}/{dist['hold_frac']:.0%}",
            )
    _console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="TEST 3A/3B minimal BTC PPO experiment")
    parser.add_argument("--mode", choices=["simple", "marta", "both"], default="both")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    args = parser.parse_args()

    aligned_df = pd.read_parquet(aligned_path("BTC_USDT", _PROCESSED), engine="pyarrow")
    encoder = load_encoder()

    modes = ["simple", "marta"] if args.mode == "both" else [args.mode]
    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        results[mode] = run_mode(aligned_df, encoder, mode, args.steps, args.eval_freq)

    if len(results) > 1:
        print_comparison(results)


if __name__ == "__main__":
    main()
