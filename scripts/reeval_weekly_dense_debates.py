"""Cheap re-eval: does BTC_USDT's native-weekly-cadence checkpoint (decision_interval=168,
docs/RESULTS_AND_METHODOLOGY.md Section 10b) behave differently with the dense debate
cache (384 real debates, up from ~33-39)?

Unlike the hourly marta_full re-eval, this one is a priori more likely to show a real
effect: the debate cache was generated at 24h stride, every day of 2024 (all timestamps
at 00:00 UTC). Weekly env steps land on 168h boundaries from a 00:00 UTC anchor, so
(168 % 24 == 0) every weekly step's representative timestamp likely exact-matches a real
cached debate now — near-100% consensus coverage per weekly decision, vs the hourly case's
~6.5%. Reruns evaluate_ppo() exactly as rl/train_weekly_crypto5_grid.py's run_one() does
(same config recipe, same checkpoint), against the on-disk debate cache as it is now.

Original (sparse-cache) stored numbers for comparison, from
data/results/weekly_grid_crypto5/BTC_USDT_{A,C}_seed0.json:
  A (raw, no regime):        sharpe=2.0419, total_return=0.5726, n_meaningful_trades=37
  C (raw_regime, real HMM):  sharpe=1.7384, total_return=0.6590, n_meaningful_trades=37

Run: /home/ramy/miniconda3/envs/marta/bin/python scripts/reeval_weekly_dense_debates.py
"""

from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3 import PPO

from rl.evaluate import evaluate_ppo
from rl.train import build_env_factory, load_data, load_debate_results
from rl.train_weekly_crypto5_grid import CONDITIONS, build_weekly_config
from run_rl import _policy_class_for

_ASSET = "BTC_USDT"
_SEED = 0
_ORIGINAL = {
    "A": {"sharpe": 2.041851409706358, "total_return": 0.5726088539735112, "n_meaningful_trades": 37},
    "C": {"sharpe": 1.738416985413198, "total_return": 0.6589748836257556, "n_meaningful_trades": 37},
}
_OUTPUT_PATH = Path("data/results/weekly_dense_debate_reeval.json")


def reeval_condition(label: str) -> dict:
    cfg = CONDITIONS[label]
    config = build_weekly_config(_ASSET, cfg["obs_mode"], _SEED, cfg["regime_path"], cfg["extra_tag"])
    config["seed"] = _SEED

    checkpoint = f"rl/checkpoints/ppo_best_crypto5_weekly_{_ASSET}_{cfg['obs_mode']}"
    if cfg["extra_tag"]:
        checkpoint += f"_{cfg['extra_tag']}"
    checkpoint += f"_seed{_SEED}.zip"

    aligned_dfs, regime_stream_df, encoder = load_data(config)
    debate_results = load_debate_results()
    env_factory = build_env_factory(aligned_dfs, regime_stream_df, encoder, debate_results, config)

    model = PPO.load(checkpoint, device="cpu", custom_objects={"policy_class": _policy_class_for(config)})
    eval_res = evaluate_ppo(model, env_factory, config, output_path=Path(f"/tmp/reeval_{label}.json"))
    m = eval_res["marta_ppo"]

    return {
        "label": label,
        "obs_mode": cfg["obs_mode"],
        "checkpoint": checkpoint,
        "n_debates_in_cache": len(debate_results),
        "reeval": {
            "sharpe": m["sharpe"],
            "total_return": m["total_return"],
            "max_drawdown": m["max_drawdown"],
            "n_meaningful_trades": m["n_meaningful_trades"],
        },
        "original_sparse_cache": _ORIGINAL[label],
        "sharpe_delta": m["sharpe"] - _ORIGINAL[label]["sharpe"],
    }


def main() -> None:
    results = {label: reeval_condition(label) for label in ("A", "C")}
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
