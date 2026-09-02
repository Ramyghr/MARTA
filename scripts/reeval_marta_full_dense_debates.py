"""Cheap re-eval: does the trained ppo_best.zip ("marta_full") behave differently now
that data/debates/debate_results.json has dense real BTC 2024 coverage (384 debates,
up from ~33-39)?

No retraining — same checkpoint, same env config, only the debate cache on disk changed.
rl/obs_builder.py's consensus_vec() is an EXACT timestamp match (unlike rl/ablation.py's
own 7-day nearest-interpolation helper used for the agents_no_regime baseline), so the
PPO's observation now carries real non-zero consensus on ~384/5928 val steps (~6.5%)
instead of ~33/5928 (~0.6%) — still mostly zero, but a real ~10x density increase.

Run: /home/ramy/miniconda3/envs/marta/bin/python scripts/reeval_marta_full_dense_debates.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rl.ablation import ABLATION_VARIANTS, _run_buy_hold, bootstrap_sharpe_ci, run_variant
from rl.evaluate import _periods_per_year_for_env, compute_metrics
from run_rl import _load_for_eval

_OUTPUT_PATH = Path("data/results/marta_full_dense_debate_reeval.json")
_CHECKPOINT = "rl/checkpoints/ppo_best.zip"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "all_assets": ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"],
        "device": device,
        "initial_capital": 10_000.0,
        "window_size": 60,
        "transaction_cost": 0.001,
        "lambda_drawdown": 0.01,
        "max_leverage": 1.0,
        "max_episode_steps": 2000,
        "min_hold_bars": 4,
        "min_hold_stop_loss_dd": 0.05,
        "max_drawdown_termination": 0.20,
        "benchmark_relative": False,
        "action_mode": "hybrid",
        # ppo_best.zip requires an 868-dim observation (confirmed via the shape-mismatch
        # error against the encoder-mode default 156-dim obs) — it's a raw-features
        # checkpoint from the project's raw-feature pivot, not the encoder architecture
        # ABLATION_VARIANTS["marta_full"]'s {"regime": True} label implies. No regime
        # block in this obs_mode at all (docs/RESULTS_AND_METHODOLOGY.md: "ppo_best.zip
        # (current default) | raw, no regime prob").
        "obs_mode": "raw",
    }

    model, env_factory = _load_for_eval(config, _CHECKPOINT)

    res = run_variant("marta_full", ABLATION_VARIANTS["marta_full"], env_factory, model)
    metrics = res["metrics"]

    bh_env = env_factory(assets=["BTC_USDT"], split="val", sequential=True, debate=False)
    bh_traj = _run_buy_hold(bh_env)
    bh_metrics = compute_metrics(
        bh_traj["returns"], bh_traj["portfolio_values"], bh_traj["positions"],
        periods_per_year=_periods_per_year_for_env(bh_env),
    )
    delta, lo, hi = bootstrap_sharpe_ci(res["returns"], bh_traj["returns"], n_bootstrap=1000)

    result = {
        "checkpoint": _CHECKPOINT,
        "debate_cache_size": len(json.loads(Path("data/debates/debate_results.json").read_text())),
        "marta_full_metrics": metrics,
        "buy_hold_metrics": bh_metrics,
        "bootstrap_delta_sharpe_marta_full_vs_buy_hold": {
            "delta": delta, "ci_lower_95": lo, "ci_upper_95": hi,
            "significant": bool(lo > 0 or hi < 0),
        },
    }
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
