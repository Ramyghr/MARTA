"""Month 3 master script — PPO execution agent: train (curriculum) → evaluate → ablation.

Usage:
    python run_rl.py                       # full curriculum (700k steps)
    python run_rl.py --quick               # tiny smoke run (fast end-to-end validation)
    python run_rl.py --seeds 0 1 2         # multi-seed run (3 seeds) for variance estimate
    python run_rl.py --eval-only           # skip training, evaluate ppo_best.zip
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from rl.ablation import run_ablation
from rl.evaluate import (
    analyze_regime_behavior,
    compute_metrics,
    evaluate_multi_asset,
    evaluate_ppo,
    run_walk_forward,
)
from rl.env import CANONICAL_ASSETS, TRAIN_END, VAL_END_EXTENDED
from rl.train import CURRICULUM_PHASES, train_ppo
from utils.logger import logger

_console = Console()
_EXPERIMENT = "marta_ppo_training"


def build_config(quick: bool) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config: dict[str, Any] = {
        "all_assets": ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"],
        "device": device,
        "initial_capital": 10_000.0,
        "window_size": 60,
        "transaction_cost": 0.001,
        "lambda_drawdown": 0.01,
        "max_leverage": 1.0,
        "max_episode_steps": 2000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.05,  # higher entropy prevents collapse to always-hold during early training
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "eval_freq": 10_000,
        "early_stop_sharpe": -0.5,
        "early_stop_patience": 3,
        "seed": 0,
        "verbose": 1,
        "progress_bar": False,
        "n_bootstrap": 1000,
        "phases": copy.deepcopy(CURRICULUM_PHASES),
    }

    if quick:
        config.update(
            {
                "n_steps": 256,
                "batch_size": 64,
                "n_epochs": 3,
                "eval_freq": 512,
                "max_episode_steps": 256,
                "n_bootstrap": 200,
                "phases": [
                    {"assets": ["BTC_USDT"], "sharpe_threshold": 0.5, "max_steps": 1024},
                    {"assets": ["BTC_USDT", "ETH_USDT"], "sharpe_threshold": 0.6, "max_steps": 1024},
                ],
            }
        )
    return config


def sanity_check_env(env, n_steps: int = 200) -> tuple[float, float]:
    """Run a random policy for ``n_steps`` before training starts.

    If mean reward is exactly 0.0 or all positions are near-zero even under random
    actions, the environment itself is broken (data loading, action→position mapping,
    or reward computation) — raise immediately instead of wasting hours of PPO training
    on a degenerate env.
    """
    obs, _ = env.reset()
    rewards: list[float] = []
    positions: list[float] = []

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(float(reward))
        positions.append(abs(float(info.get("position_size", 0.0))))
        if term or trunc:
            obs, _ = env.reset()

    mean_reward = float(np.mean(rewards))
    mean_position = float(np.mean(positions))
    logger.info(f"Sanity check: mean_reward={mean_reward:.4f}, mean_position={mean_position:.4f}")

    if mean_position < 0.001:
        raise RuntimeError(
            f"Environment sanity check FAILED: mean_position={mean_position:.6f}. "
            f"All positions are near-zero even with a random policy. "
            f"Check position sizing in rl/env.py step() (risk_cap / direction mapping) — "
            f"something is suppressing all actions before training even starts."
        )

    if abs(mean_reward) < 1e-8:
        raise RuntimeError(
            f"Environment sanity check FAILED: mean_reward={mean_reward:.8f}. "
            f"Rewards are exactly zero — check price data loading and close_return computation."
        )

    logger.info("✓ Environment sanity check PASSED")
    return mean_reward, mean_position


def check_regime_stream_compatibility(config: dict) -> int:
    """Verify regime_stream.parquet covers the configured assets and training window."""
    rs = pd.read_parquet("data/processed/regime_stream.parquet", engine="pyarrow")
    n_states = int(rs["regime_label"].nunique())

    expected_assets = set(config["all_assets"])
    actual_assets = set(rs["asset"].unique())
    missing = expected_assets - actual_assets
    if missing:
        raise RuntimeError(
            f"regime_stream.parquet missing assets: {sorted(missing)}. Re-run run_regime.py"
        )

    stream_end = rs["timestamp"].max()
    stream_start = rs["timestamp"].min()
    if stream_end.tzinfo is None:
        stream_end = stream_end.tz_localize("UTC")
    if stream_end < TRAIN_END:
        raise RuntimeError(
            f"regime_stream.parquet ends at {stream_end} but training needs data through "
            f"{TRAIN_END}. Re-run run_regime.py"
        )

    logger.info(
        f"✓ Regime stream: {n_states} states, {len(rs):,} rows, "
        f"{stream_start.date()} → {stream_end.date()}"
    )
    return n_states


def _print_curriculum(history: list[dict]) -> None:
    table = Table(title="Curriculum Phases")
    table.add_column("Phase", justify="right")
    table.add_column("Assets")
    table.add_column("Steps", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Met?", justify="center")
    if not history:
        table.add_row("—", "—", "—", "—", "—", "no phase completed")
    for h in history:
        table.add_row(
            str(h["phase"]),
            "+".join(h["assets"]),
            f"{h['steps']:,}",
            f"{h['sharpe']:.3f}",
            f"{h['threshold']:.2f}",
            "[green]yes[/green]" if h["threshold_met"] else "[yellow]no[/yellow]",
        )
    _console.print(table)


def _print_eval(eval_res: dict) -> None:
    m = eval_res["marta_ppo"]
    table = Table(title="Validation Metrics (MARTA PPO — BTC)")
    for k in ["sharpe", "sortino", "max_drawdown", "calmar", "total_return",
              "n_meaningful_trades", "mean_turnover"]:
        table.add_column(k, justify="right")
    table.add_row(
        f"{m['sharpe']:.3f}", f"{m['sortino']:.3f}", f"{m['max_drawdown']:.3f}",
        f"{m['calmar']:.3f}", f"{m['total_return']:.3%}",
        str(m.get("n_meaningful_trades", m["n_trades"])), f"{m.get('mean_turnover', 0.0):.5f}",
    )
    _console.print(table)

    rt = Table(title="Per-Regime Breakdown")
    rt.add_column("Regime", justify="right")
    rt.add_column("Steps", justify="right")
    rt.add_column("Sharpe", justify="right")
    rt.add_column("Avg Position", justify="right")
    for regime, stats in eval_res["per_regime"].items():
        rt.add_row(
            regime.replace("regime_", "R"),
            str(stats["n_steps"]),
            f"{stats['sharpe']:.3f}",
            f"{stats['avg_position_size']:.3f}",
        )
    _console.print(rt)


def _print_ablation(abl: dict) -> None:
    table = Table(title="Ablation Study (validation set)")
    table.add_column("Variant")
    table.add_column("Sharpe", justify="right")
    table.add_column("Sortino", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Total Ret", justify="right")
    for name, m in abl["variants"].items():
        table.add_row(
            name,
            f"{m['sharpe']:.3f}",
            f"{m['sortino']:.3f}",
            f"{m['max_drawdown']:.3f}",
            f"{m['total_return']:.3%}",
        )
    _console.print(table)

    bt = Table(title="Bootstrap CI — MARTA full vs baseline (Sharpe Δ, 95% CI)")
    bt.add_column("Comparison")
    bt.add_column("Δ Sharpe", justify="right")
    bt.add_column("CI lower", justify="right")
    bt.add_column("CI upper", justify="right")
    bt.add_column("Significant", justify="center")
    for name, ci in abl["bootstrap_ci"].items():
        bt.add_row(
            name,
            f"{ci['delta_sharpe']:.3f}",
            f"{ci['ci_lower_95']:.3f}",
            f"{ci['ci_upper_95']:.3f}",
            "[green]yes[/green]" if ci["significant"] else "[dim]no[/dim]",
        )
    _console.print(bt)
    _console.print(
        "[dim]Bootstrap CIs use 168-hour blocks (preserves autocorrelation). "
        "Standard bootstrap CIs available in ablation_results.json.[/dim]"
    )
    _console.print(
        "[dim]* ppo_no_regime/ppo_agents_no_regime use momentum baseline (PPO collapses\n"
        "  on out-of-distribution zero-regime input, confirming regime embedding dependency)[/dim]"
    )


def _print_regime_behavior(rb: dict) -> None:
    _console.print(
        "\n[bold yellow]IN-SAMPLE Regime Behavior[/bold yellow] "
        "[dim](training-split windows — diagnostic only, NOT a performance claim)[/dim]"
    )
    for label, w in rb.get("windows", {}).items():
        if "error" in w:
            _console.print(f"  [dim]{label}: {w['error']}[/dim]")
            continue
        table = Table(title=f"{label}  [{w['start']} → {w['end']}]  (IN-SAMPLE)")
        table.add_column("Regime", justify="right")
        table.add_column("Steps", justify="right")
        table.add_column("Avg Position", justify="right")
        for regime, stats in w["avg_position_by_regime"].items():
            table.add_row(
                regime.replace("regime_", "R"),
                str(stats["n_steps"]),
                f"{stats['avg_position_size']:.4f}",
            )
        _console.print(table)
        _console.print(
            f"  [dim]regime transitions: {w['regime_transitions']} | "
            f"mean |Δposition| at boundaries: {w['mean_abs_position_change_at_boundaries']:.4f}[/dim]"
        )


def _evaluate_per_asset(model: Any, env_factory: Any, config: dict) -> dict[str, Any]:
    """Evaluate the trained model separately on each canonical asset's validation split.

    Uses the same env_factory/model as the main BTC evaluation but a plain sequential
    val env per asset (no baselines/ablation — those stay BTC-only).
    """
    per_asset: dict[str, Any] = {}
    for asset in config.get("all_assets", CANONICAL_ASSETS):
        try:
            asset_env = env_factory(assets=[asset], split="val", sequential=True)
            if len(asset_env._valid_starts.get(asset, [])) == 0:
                logger.warning(f"{asset}: no valid val-split rows — skipping")
                per_asset[asset] = {"error": "no valid val-split rows"}
                continue
            traj = run_walk_forward(model, asset_env, deterministic=True)
            metrics = compute_metrics(traj["returns"], traj["portfolio_values"], traj["positions"])
            per_asset[asset] = metrics
            logger.info(
                f"{asset}: Sharpe={metrics['sharpe']:.3f} MaxDD={metrics['max_drawdown']:.1%} "
                f"Trades={metrics['n_meaningful_trades']} HoldRatio={metrics['hold_ratio']:.1%}"
            )
        except Exception as e:
            logger.warning(f"{asset}: evaluation failed — {e}")
            per_asset[asset] = {"error": str(e)}
    return per_asset


def _print_per_asset(per_asset: dict[str, Any]) -> None:
    table = Table(title="Per-Asset Validation Results")
    table.add_column("Asset")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Hold %", justify="right")
    table.add_column("Avg Trade (h)", justify="right")
    for asset, m in per_asset.items():
        if "error" in m:
            table.add_row(asset, "—", "—", "—", "—", f"[dim]{m['error']}[/dim]")
            continue
        table.add_row(
            asset,
            f"{m['sharpe']:.3f}",
            f"{m['max_drawdown']:.1%}",
            str(m["n_meaningful_trades"]),
            f"{m['hold_ratio']:.1%}",
            f"{m['mean_trade_duration_hours']:.1f}",
        )
    _console.print(table)


def _load_for_eval(config: dict, checkpoint: str) -> tuple[Any, Any]:
    """Build an env factory and load a saved PPO checkpoint (no training)."""
    from stable_baselines3 import PPO

    from rl.policy import MARTAActorCriticPolicy
    from rl.train import build_env_factory, load_data, load_debate_results

    aligned_dfs, regime_stream_df, encoder = load_data(config)
    debate_results = load_debate_results()
    env_factory = build_env_factory(
        aligned_dfs, regime_stream_df, encoder, debate_results, config
    )
    model = PPO.load(
        checkpoint,
        device=config["device"],
        custom_objects={"policy_class": MARTAActorCriticPolicy},
    )
    logger.success(f"Loaded checkpoint {checkpoint} for evaluation")
    return model, env_factory


def _run_single_seed(config: dict, seed: int) -> dict:
    """Train and evaluate one PPO seed. Returns eval_res dict."""
    import copy as _copy
    from rl.ablation import run_ablation
    from rl.evaluate import analyze_regime_behavior, evaluate_ppo
    from rl.train import train_ppo
    from stable_baselines3 import PPO
    from rl.policy import MARTAActorCriticPolicy

    cfg = _copy.deepcopy(config)
    cfg["seed"] = seed

    train_out = train_ppo(cfg, pre_train_check=sanity_check_env)
    env_factory = train_out["env_factory"]

    best_ckpt = Path(f"rl/checkpoints/ppo_best_seed{seed}.zip")
    default_best = Path("rl/checkpoints/ppo_best.zip")
    ckpt_path = best_ckpt if best_ckpt.exists() else (default_best if default_best.exists() else None)
    if ckpt_path:
        model = PPO.load(str(ckpt_path), device=cfg["device"],
                         custom_objects={"policy_class": MARTAActorCriticPolicy})
    else:
        model = train_out["model"]

    eval_res = evaluate_ppo(model, env_factory, cfg)
    abl = run_ablation(cfg, model, env_factory)
    return {"eval": eval_res, "ablation": abl, "curriculum": train_out["curriculum_history"],
            "best_sharpe": train_out["best_sharpe"], "seed": seed}


def _aggregate_seeds(seed_results: list[dict]) -> None:
    """Log and save cross-seed Sharpe statistics."""
    sharpes = [r["eval"]["marta_ppo"]["sharpe"] for r in seed_results]
    seeds = [r["seed"] for r in seed_results]
    summary = {
        "seeds": seeds,
        "sharpes": sharpes,
        "mean_sharpe": statistics.mean(sharpes),
        "std_sharpe": statistics.stdev(sharpes) if len(sharpes) > 1 else 0.0,
        "min_sharpe": min(sharpes),
        "max_sharpe": max(sharpes),
    }
    out = Path("data/results/multi_seed_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    _console.rule("[bold]Multi-Seed Summary")
    _console.print(f"Seeds: {seeds}")
    _console.print(f"Sharpe per seed: {[f'{s:.3f}' for s in sharpes]}")
    _console.print(f"Mean: {summary['mean_sharpe']:.3f}  Std: {summary['std_sharpe']:.3f}  "
                   f"Range: [{summary['min_sharpe']:.3f}, {summary['max_sharpe']:.3f}]")
    _console.print(f"Results saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MARTA Month 3 — PPO execution agent")
    parser.add_argument("--quick", action="store_true", help="tiny smoke run")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help="run multiple independent seeds (e.g. --seeds 0 1 2) for variance estimation",
    )
    parser.add_argument(
        "--eval-only",
        nargs="?",
        const="rl/checkpoints/ppo_best.zip",
        default=None,
        help="skip training; evaluate + ablate a saved checkpoint (default ppo_best.zip)",
    )
    parser.add_argument(
        "--extended-val",
        action="store_true",
        help="expand validation window to 2021-01-01 → 2022-12-31 (two full years)",
    )
    parser.add_argument(
        "--multi-asset-eval",
        action="store_true",
        help="evaluate on BTC+ETH as independent test paths and report mean±std Sharpe",
    )
    args = parser.parse_args()

    config = build_config(args.quick)
    # Eval-only is BTC-focused (val metrics, ablation, baselines are all BTC) → keep it light.
    if args.eval_only:
        config["all_assets"] = ["BTC_USDT"]
    logger.info(
        f"Device: {config['device']} | quick={args.quick} | eval_only={args.eval_only or False}"
    )

    # Pre-training compatibility check: regime stream must cover all configured assets
    # and the full training window (fails fast on a stale regime_stream.parquet).
    if not args.eval_only:
        check_regime_stream_compatibility(config)

    # Multi-seed path: run each seed independently, aggregate, then exit.
    if args.seeds and not args.eval_only:
        mlflow.set_experiment(_EXPERIMENT)
        seed_results = []
        for seed in args.seeds:
            logger.info(f"=== Seed {seed} ===")
            with mlflow.start_run(run_name=f"month3_ppo_seed{seed}"):
                mlflow.log_param("seed", seed)
                res = _run_single_seed(config, seed)
                mlflow.log_metric("val/sharpe", res["eval"]["marta_ppo"]["sharpe"])
                seed_results.append(res)
        _aggregate_seeds(seed_results)
        # Print summary from the median-Sharpe seed.
        sharpes = [r["eval"]["marta_ppo"]["sharpe"] for r in seed_results]
        best_idx = int(np.argmax(sharpes))
        _print_eval(seed_results[best_idx]["eval"])
        _print_ablation(seed_results[best_idx]["ablation"])
        return

    mlflow.set_experiment(_EXPERIMENT)
    run_name = "month3_ppo_eval" if args.eval_only else "month3_ppo"
    with mlflow.start_run(run_name=run_name) as run:
        for key in [
            "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
            "clip_range", "ent_coef", "vf_coef", "lambda_drawdown", "transaction_cost",
        ]:
            mlflow.log_param(key, config[key])
        mlflow.log_param("n_phases", len(config["phases"]))

        # 1. Train (or load a checkpoint for eval-only).
        if args.eval_only:
            model, env_factory = _load_for_eval(config, args.eval_only)
            train_out = {
                "model": model,
                "env_factory": env_factory,
                "curriculum_history": [],
                "best_sharpe": float("nan"),
            }
        else:
            if len(config["phases"]) > 2:
                # Cap curriculum at Phase 2 (BTC+ETH only).
                # Phases 3-4 add EURUSD/GOLD which have low intraday variance
                # causing PPO to collapse to always-hold. Documented in CLAUDE.md.
                config["phases"] = CURRICULUM_PHASES[:2]
            train_out = train_ppo(config, pre_train_check=sanity_check_env)
            env_factory = train_out["env_factory"]
            # Evaluate the best-by-val-Sharpe checkpoint, NOT the final model. The final
            # model can be a degraded policy — e.g. the curriculum collapses to always-flat
            # once daily-upsampled forex/commodity assets (zero intraday returns) are added
            # in phases 3-4. ppo_best.zip holds the best policy seen during training.
            from pathlib import Path

            from stable_baselines3 import PPO

            from rl.policy import MARTAActorCriticPolicy

            best_ckpt = Path("rl/checkpoints/ppo_best.zip")
            if best_ckpt.exists():
                model = PPO.load(
                    str(best_ckpt),
                    device=config["device"],
                    custom_objects={"policy_class": MARTAActorCriticPolicy},
                )
                logger.info("Evaluating ppo_best.zip (best val Sharpe), not the final model")
            else:
                logger.warning("ppo_best.zip not found — falling back to final model for eval")
                model = train_out["model"]

        # 2. Evaluate on validation set (optionally extended window / multi-asset).
        val_end_override = VAL_END_EXTENDED if args.extended_val else None
        eval_res = evaluate_ppo(model, env_factory, config, val_end=val_end_override)
        m = eval_res["marta_ppo"]
        for k, v in m.items():
            mlflow.log_metric(f"val/{k}", float(v))

        # 3. Ablation study.
        abl = run_ablation(config, model, env_factory)
        for name, vm in abl["variants"].items():
            mlflow.log_metric(f"ablation/{name}_sharpe", float(vm["sharpe"]))

        # 3a. Per-asset validation (each configured asset evaluated independently).
        per_asset_results = _evaluate_per_asset(model, env_factory, config)
        for asset, vm in per_asset_results.items():
            if "error" not in vm:
                mlflow.log_metric(f"per_asset/{asset}_sharpe", float(vm["sharpe"]))

        # 3b. Multi-asset / multi-seed evaluation (optional).
        if args.multi_asset_eval:
            all_models = [model]
            # Load any extra seed checkpoints if present
            for seed in range(1, 5):
                ckpt = Path(f"rl/checkpoints/ppo_best_seed{seed}.zip")
                if ckpt.exists():
                    from stable_baselines3 import PPO
                    from rl.policy import MARTAActorCriticPolicy
                    extra = PPO.load(
                        str(ckpt), device=config["device"],
                        custom_objects={"policy_class": MARTAActorCriticPolicy},
                    )
                    all_models.append(extra)
            multi_res = evaluate_multi_asset(
                all_models, env_factory, config,
                assets=["BTC_USDT", "ETH_USDT"],
                val_end=val_end_override,
            )
            agg = multi_res["aggregate"]
            _console.print(
                f"\n[bold]Multi-asset Sharpe:[/bold] "
                f"{agg.get('sharpe_mean', 0):.3f} ± {agg.get('sharpe_std', 0):.3f} "
                f"({len(all_models)} seeds × {len(['BTC_USDT','ETH_USDT'])} assets, "
                f"{multi_res['n_paths']} paths)"
            )
            for path in multi_res["paths"]:
                mlflow.log_metric(
                    f"multi/{path['asset']}_s{path['seed']}_sharpe",
                    float(path["metrics"]["sharpe"]),
                )

        # 4. IN-SAMPLE regime behavior analysis (diagnostic only — never a performance claim).
        regime_behavior = analyze_regime_behavior(model, env_factory, config)
        mlflow.log_artifact("data/results/regime_behavior_analysis.json")

        run_id = run.info.run_id

    # 5. Summary.
    _console.rule("[bold]MARTA Month 3 — Summary")
    _print_curriculum(train_out["curriculum_history"])
    _print_eval(eval_res)
    _print_ablation(abl)
    _print_per_asset(per_asset_results)
    _print_regime_behavior(regime_behavior)

    phase1_sharpe = m["sharpe"]
    bh_ci = abl["bootstrap_ci"].get("marta_full_vs_buy_hold", {})
    _console.print(f"\n[bold]Phase 1 (BTC) validation Sharpe:[/bold] {phase1_sharpe:.3f}")
    _console.print(
        f"[bold]MARTA full vs Buy&Hold:[/bold] ΔSharpe={bh_ci.get('delta_sharpe', 0):.3f} "
        f"CI=[{bh_ci.get('ci_lower_95', 0):.3f}, {bh_ci.get('ci_upper_95', 0):.3f}] "
        f"significant={bh_ci.get('significant', False)}"
    )
    _console.print(f"[bold]Best val Sharpe during training:[/bold] {train_out['best_sharpe']:.3f}")
    _console.print(f"\n[bold]MLflow run:[/bold] {run_id}")
    _console.print("[bold]MLflow UI:[/bold] http://127.0.0.1:5000  (run `mlflow ui`)")


if __name__ == "__main__":
    main()
