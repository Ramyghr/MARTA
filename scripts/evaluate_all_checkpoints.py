"""Re-evaluate every saved PPO checkpoint on a clean val_env (idle_penalty=0.0).

Standalone — every checkpoint under rl/checkpoints/ was potentially saved using an
inflated validation Sharpe (val_env leaked the phase's idle_penalty; see CLAUDE.md
"idle_penalty is a training signal only"). This script re-scores each checkpoint on a
freshly built val_env with idle_penalty forced to 0.0, so the comparison table reflects
true out-of-sample abstention-penalty-free performance.

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python scripts/evaluate_all_checkpoints.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO

from rl.env import MARTATradingEnv
from rl.evaluate import compute_metrics, run_walk_forward
from rl.train import build_env_factory, load_data, load_debate_results
from utils.logger import logger

_CHECKPOINT_DIR = Path("rl/checkpoints")
_RESULTS_PATH = Path("data/results/checkpoint_evaluation.json")

CONFIG: dict[str, Any] = {
    "all_assets": ["BTC_USDT", "ETH_USDT"],
    "device": "cuda",
    "initial_capital": 10_000.0,
    "window_size": 60,
    "transaction_cost": 0.001,
    "lambda_drawdown": 0.01,
    "max_leverage": 1.0,
    "max_episode_steps": 2000,
}


def evaluate_checkpoint(
    checkpoint_path: Path,
    env_factory: Any,
    device: str,
) -> dict[str, float]:
    """Load one checkpoint and score it on a fresh BTC val_env with idle_penalty=0.0."""
    model = PPO.load(str(checkpoint_path), device=device)
    val_env: MARTATradingEnv = env_factory(
        assets=["BTC_USDT"],
        split="val",
        sequential=True,
        debate=False,
        idle_penalty=0.0,
    )
    traj = run_walk_forward(model, val_env, deterministic=True)
    return compute_metrics(traj["returns"], traj["portfolio_values"], traj.get("positions"))


def main() -> None:
    checkpoints = sorted(_CHECKPOINT_DIR.glob("*.zip"))
    if not checkpoints:
        logger.warning(f"No checkpoints found under {_CHECKPOINT_DIR}")
        return

    aligned_dfs, regime_stream_df, encoder = load_data(CONFIG)
    debate_results = load_debate_results()
    env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, CONFIG)

    results: dict[str, dict[str, float]] = {}
    for ckpt in checkpoints:
        try:
            m = evaluate_checkpoint(ckpt, env_factory, CONFIG["device"])
            results[ckpt.name] = m
            print(
                f"{ckpt.name:<30} Sharpe={m['sharpe']:.4f} MaxDD={m['max_drawdown']:.1%} "
                f"Trades={m.get('n_meaningful_trades', 0)}"
            )
        except Exception as e:
            logger.warning(f"{ckpt.name}: evaluation failed — {e}")
            print(f"{ckpt.name:<30} ERROR: {e}")

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float))

    if results:
        best_name = max(results, key=lambda k: results[k]["sharpe"])
        print(f"\nBest checkpoint: {best_name}")
        print(f"Best Sharpe: {results[best_name]['sharpe']:.4f}")


if __name__ == "__main__":
    main()
