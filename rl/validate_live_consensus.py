"""Pre-production validation gate for the live "nearest cached debate" consensus lookup.

dashboard/backend/ppo_inference.py's live consensus vector (Part B of the portfolio+
consensus wiring work) snaps every prediction to the nearest debate within 7 days
(rl.ablation._make_nearest_consensus_fn's algorithm) rather than the exact-timestamp match
rl/env.py's MARTATradingEnv used during training. Training only ever saw consensus
populated at ~exact-match timesteps (a small fraction of validation steps) — switching to
nearest-within-7-days will populate it far more often live than it ever was in training,
which risks pushing ppo_best.zip's input distribution somewhere it wasn't trained for.

This script re-runs the val-set walk-forward with the SAME model under BOTH consensus
regimes — the exact-match lookup rl/env.py uses by default (identical to the existing
`marta_full` ablation variant), and the nearest-within-7-days lookup the live system will
actually use — and reports a block-bootstrap Sharpe delta between them. It changes nothing
in rl/env.py or rl/ablation.py; it only monkeypatches one env INSTANCE's bound
`_consensus_vec` method for the nearest-lookup run.

Usage:
    conda run -n marta python rl/validate_live_consensus.py
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np

from rl.ablation import _make_nearest_consensus_fn, bootstrap_sharpe_ci
from rl.env import CONSENSUS_DIM
from rl.evaluate import compute_metrics, run_walk_forward
from utils.logger import logger

_RESULTS_PATH = Path("data/results/live_consensus_validation.json")


def _patch_nearest_consensus(env, asset: str, max_days: int = 7) -> None:
    """Monkeypatch one env instance's `_consensus_vec` to use the nearest-within-`max_days`
    lookup instead of the default exact-timestamp match — mirrors exactly what
    dashboard/backend/ppo_inference.py::_nearest_consensus_vec does live."""
    nearest_fn = _make_nearest_consensus_fn(env, asset, max_days=max_days)

    def _patched_consensus_vec(self, a: str, t: int) -> np.ndarray:
        if self.zero_consensus:
            return np.zeros(CONSENSUS_DIM, dtype=np.float32)
        ts = self._index[a][t]
        vec = nearest_fn(ts)
        return vec if vec is not None else np.zeros(CONSENSUS_DIM, dtype=np.float32)

    env._consensus_vec = types.MethodType(_patched_consensus_vec, env)


def main() -> None:
    import torch
    from run_rl import _load_for_eval  # reuses the exact eval-checkpoint-loading path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "device": device,
        "initial_capital": 10_000.0,
        "window_size": 60,
        "transaction_cost": 0.001,
        "lambda_drawdown": 0.01,
        "max_leverage": 1.0,
        "max_episode_steps": 2000,
        "all_assets": ["BTC_USDT"],
    }

    checkpoint = "rl/checkpoints/ppo_best.zip"
    logger.info(f"Loading {checkpoint} for live-consensus validation")
    model, env_factory = _load_for_eval(config, checkpoint)

    env_kwargs = dict(
        assets=["BTC_USDT"], split="val", sequential=True,
        zero_regime=False, zero_consensus=False, debate=True,
    )

    logger.info("Running exact-timestamp consensus (today's marta_full baseline)")
    env_exact = env_factory(**env_kwargs)
    traj_exact = run_walk_forward(model, env_exact, deterministic=True)
    metrics_exact = compute_metrics(
        traj_exact["returns"], traj_exact["portfolio_values"], traj_exact["positions"]
    )

    logger.info("Running nearest-within-7-days consensus (what live predict() will use)")
    env_nearest = env_factory(**env_kwargs)
    _patch_nearest_consensus(env_nearest, env_nearest._asset, max_days=7)
    traj_nearest = run_walk_forward(model, env_nearest, deterministic=True)
    metrics_nearest = compute_metrics(
        traj_nearest["returns"], traj_nearest["portfolio_values"], traj_nearest["positions"]
    )

    delta, ci_lo, ci_hi = bootstrap_sharpe_ci(traj_nearest["returns"], traj_exact["returns"])
    # significantly WORSE means the nearest-lookup live scheme's Sharpe is below the
    # exact-match baseline with the whole 95% CI on the negative side.
    significantly_worse = ci_hi < 0
    significantly_better = ci_lo > 0

    if significantly_worse:
        verdict = "DO NOT ENABLE — nearest-within-7-days consensus significantly degrades Sharpe"
    elif significantly_better:
        verdict = "SAFE TO ENABLE — nearest-within-7-days consensus significantly improves Sharpe"
    else:
        verdict = "SAFE TO ENABLE — no significant Sharpe difference (CI straddles zero)"

    result = {
        "checkpoint": checkpoint,
        "exact_timestamp_match": metrics_exact,
        "nearest_within_7_days": metrics_nearest,
        "bootstrap_sharpe_delta_nearest_minus_exact": {
            "delta": delta, "ci_lower_95": ci_lo, "ci_upper_95": ci_hi,
        },
        "verdict": verdict,
    }

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(result, indent=2, default=float))

    logger.success(f"Results written to {_RESULTS_PATH}")
    logger.info(f"Exact-match Sharpe:   {metrics_exact['sharpe']:.4f}")
    logger.info(f"Nearest-lookup Sharpe: {metrics_nearest['sharpe']:.4f}")
    logger.info(f"Bootstrap delta (nearest - exact): {delta:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    logger.info(f"GATE VERDICT: {verdict}")


if __name__ == "__main__":
    main()
