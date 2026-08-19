"""Phase 3 of the downstream pipeline validation (2026-08-14): Agents -> RL.

First establishes a clean no-regime baseline (obs_mode="raw"), then trains the exact same
setup with BTC-only-v3's regime info added (obs_mode="raw_regime", the validated Phase 1
regime stream), then compares. Only the regime variable changes between the two runs —
same seed, same BTC-only single-phase curriculum, same steps budget, same reward config,
same early-stop override.

Single-phase (BTC only), not the usual BTC+ETH 2-phase cap — this investigation has been
BTC-only throughout (encoder, HMM), so this keeps Phase 3 consistent with what was
actually validated in Phases 1-2, not a different asset scope.

phase_steps_multiplier=3 + early_stop_sharpe=-10 (disabled): the same fix validated
2026-08-14 for the BTC-only-v3 encoder retrain and confirmed necessary for the earlier
raw/raw_regime RL runs too (CurriculumCallback's early_stop_sharpe=-0.5/patience=3 cut
both prior attempts off at ~step 30k, well before the curriculum's target step count,
producing byte-identical, undertrained results). Applying it here from the start instead
of discovering the same problem again.

Run: conda run -n marta python run_phase3_btc_only_v3_rl.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import mlflow
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from run_rl import CURRICULUM_PHASES, build_config, check_regime_stream_compatibility
from rl.evaluate import evaluate_ppo
from rl.train import train_ppo
from utils.reproducibility import set_seed

console = Console()

_REGIME_STREAM_PATH = "data/processed/encoder_experiments/e1_predictive_btc_only_v3/regime_stream.parquet"
_RESULTS_DIR = Path("data/results")


def _base_config() -> dict:
    config = build_config(quick=False)
    config["phases"] = copy.deepcopy(CURRICULUM_PHASES[:1])  # BTC only, single phase
    for p in config["phases"]:
        p["max_steps"] = int(p["max_steps"] * 3)  # 200k -> 600k
    config["early_stop_sharpe"] = -10  # effectively disabled — see module docstring
    config["all_assets"] = ["BTC_USDT"]
    return config


def _run_variant(label: str, obs_mode: str, regime_stream_path: str | None) -> dict:
    console.rule(f"[bold]Phase 3 — {label} (obs_mode={obs_mode})[/bold]")
    config = _base_config()
    config["obs_mode"] = obs_mode
    config["checkpoint_suffix"] = f"_phase3_{label}"
    if regime_stream_path:
        config["regime_stream_path"] = regime_stream_path

    if obs_mode != "raw":
        check_regime_stream_compatibility(config)

    set_seed(config["seed"])
    mlflow.set_experiment("marta_ppo_training")
    with mlflow.start_run(run_name=f"phase3_{label}"):
        for key in ["learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
                    "gae_lambda", "clip_range", "ent_coef", "vf_coef", "lambda_drawdown",
                    "transaction_cost"]:
            mlflow.log_param(key, config[key])
        mlflow.log_param("obs_mode", obs_mode)
        mlflow.log_param("phase_steps", config["phases"][0]["max_steps"])

        train_out = train_ppo(config)
        env_factory = train_out["env_factory"]

        best_ckpt = train_out["best_path"]
        if best_ckpt.exists():
            from stable_baselines3 import PPO
            from rl.policy import MARTAActorCriticPolicy
            model = PPO.load(str(best_ckpt), device="cpu",
                              custom_objects={"policy_class": MARTAActorCriticPolicy})
            console.print(f"[green]Evaluating best checkpoint: {best_ckpt}[/green]")
        else:
            model = train_out["model"]
            console.print("[yellow]best checkpoint not found — evaluating final model[/yellow]")

        eval_out_path = _RESULTS_DIR / f"ppo_evaluation_phase3_{label}.json"
        eval_res = evaluate_ppo(model, env_factory, config, output_path=eval_out_path)
        m = eval_res["marta_ppo"]
        for k, v in m.items():
            mlflow.log_metric(f"val/{k}", float(v))

    return m


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    baseline = _run_variant("baseline_raw", "raw", None)
    with_regime = _run_variant("with_regime", "raw_regime", _REGIME_STREAM_PATH)

    table = Table(title="Phase 3 — Agents/Regime -> RL: Clean Baseline vs +Regime (BTC-only-v3)")
    table.add_column("Variant")
    table.add_column("Sharpe", justify="right")
    table.add_column("Total Return", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Meaningful Trades", justify="right")
    table.add_column("Hold Ratio", justify="right")
    for label, m in [("baseline (raw only)", baseline), ("+ regime (raw_regime)", with_regime)]:
        table.add_row(
            label, f"{m['sharpe']:.3f}", f"{m['total_return']:.2%}", f"{m['max_drawdown']:.2%}",
            f"{m['n_meaningful_trades']:.0f}", f"{m['hold_ratio']:.2%}",
        )
    console.print(table)

    summary = {"baseline_raw": baseline, "with_regime": with_regime}
    (_RESULTS_DIR / "phase3_btc_only_v3_comparison.json").write_text(json.dumps(summary, indent=2, default=float))
    console.print(f"[green]Saved -> {_RESULTS_DIR / 'phase3_btc_only_v3_comparison.json'}[/green]")


if __name__ == "__main__":
    main()
