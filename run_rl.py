"""Month 3 master script — PPO execution agent: train (curriculum) → evaluate → ablation.

Usage:
    python run_rl.py                       # full curriculum (700k steps)
    python run_rl.py --quick               # tiny smoke run (fast end-to-end validation)
    python run_rl.py --seeds 0 1 2         # multi-seed run (3 seeds) for variance estimate
    python run_rl.py --eval-only           # skip training, evaluate ppo_best.zip
    python run_rl.py --benchmark-relative  # A/B: reward excess return over buy-and-hold
                                            # (saves to ppo_best_benchrel.zip — run once
                                            # with, once without, then diff the two
                                            # ppo_evaluation results explicitly)
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
from rl.train import (
    CRYPTO5_CANONICAL_ASSETS,
    CRYPTO5_CURRICULUM_PHASES,
    CRYPTO5_PROCESSED,
    CRYPTO5_REGIME_STREAM_CAUSAL_PATH,
    CURRICULUM_PHASES,
    train_ppo,
)
from utils.logger import logger
from utils.reproducibility import set_seed


def _policy_class_for(config: dict):
    """Return the policy class a checkpoint was trained with, based on
    config["action_mode"] — every PPO.load(..., custom_objects={"policy_class": ...})
    call site must use this instead of hardcoding MARTAActorCriticPolicy, or loading a
    checkpoint trained with --action-mode discrete crashes with a state_dict key
    mismatch (direction_head/position_head vs action_head — found running the first
    discrete-action smoke test, 2026-08-24)."""
    from rl.policy import MARTAActorCriticPolicy, MARTADiscretePolicy
    if config.get("action_mode", "hybrid") == "discrete":
        return MARTADiscretePolicy
    return MARTAActorCriticPolicy

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
        # Minimum bars a newly-opened position must be held before flattening/reversing
        # is allowed (escape hatch: min_hold_stop_loss_dd overrides the lock). Added
        # 2026-08 after a multi-seed audit found the trained policy satisfying
        # idle_penalty with single-bar open-then-immediately-close "pokes" — every
        # completed position in the eval trade log lasted exactly 1 hour. Applies at
        # train AND eval time (see rl/env.py's __init__ docstring) — it's a
        # market-structure rule, not reward shaping like idle_penalty.
        "min_hold_bars": 4,
        "min_hold_stop_loss_dd": 0.05,
        # Hard training-episode-terminating drawdown from peak (never applied during
        # sequential/eval walk-forwards — see rl/env.py's __init__ docstring). Replaces
        # the old "lose 80% of INITIAL capital" rule (2026-08 audit: literature review
        # found that far too loose to give useful risk feedback during training).
        "max_drawdown_termination": 0.20,
        # Parallel training environments (SB3 DummyVecEnv, see rl/train.py's train_ppo).
        # A 2026-08 multi-seed audit found single-env training gave high seed-to-seed
        # Sharpe variance (std=0.861 on mean=0.798 across 3 seeds) — one long
        # autocorrelated trajectory per rollout buffer. n_envs decorrelates the buffer
        # across n_envs independent random start points instead — the standard fix in
        # the RL-trading literature for this exact instability.
        "n_envs": 8,
        # n_steps lowered 4096→512 alongside n_envs=8: the rollout buffer size is
        # n_steps × n_envs, and this keeps it at 4096 (same as the prior single-env
        # config) — same amount of data per policy update, same number of updates per
        # total_timesteps, but now sourced from 8 decorrelated trajectories instead of
        # one. Do not raise n_steps back up without lowering n_envs correspondingly, or
        # the buffer (and PPO update cadence) will silently balloon.
        # Minimum meaningful trades an eval needs before it's eligible to become the new
        # best_sharpe / overwrite ppo_best.zip (rl/callbacks.py:CurriculumCallback). Added
        # 2026-08 after a checkpoint got crowned "best" off 3 trades and Sortino=4728 — a
        # statistical fluke, not a validated result. Raised 5->30 after a second 2026-08
        # audit found 5 still let a 9-trade eval (Sharpe 1.893, one seed's reported
        # "headline" result) through the guard — 9 trade outcomes over a ~247-day val
        # window has no real statistical power. 30 is still thin but is the practical
        # floor below which a Sharpe estimate shouldn't be trusted at all.
        "min_trades_for_best": 30,
        # Reward excess return over buy-and-hold instead of an absolute risk-adjusted
        # return (rl/env.py's benchmark_relative — see its docstring). Default False
        # preserves every prior result exactly; flip to True as a distinct experiment
        # aimed specifically at "beat buy-and-hold" rather than the original absolute
        # risk-adjusted-return research claim. NOT enabled by default — a 2026-08 audit
        # proposed this as the most likely lever to actually move that specific
        # comparison, but it changes what the policy is trained to do and should be
        # compared against the default explicitly, not silently substituted for it.
        "benchmark_relative": False,
        "learning_rate": 3e-4,
        # n_steps is PER-ENV steps per rollout; buffer size = n_steps × n_envs.
        # Originally raised 2048→4096 (2026-08, single env) to smooth the advantage
        # estimate against per-step reward noise (explained_variance was bouncing
        # -0.44..+0.4). Now lowered 4096→512 to pair with n_envs=8 above, keeping the
        # same 4096 buffer size — see the n_envs comment for why.
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        # gamma lowered 0.99→0.95 (2026-08): 0.99 gives an effective ~100-step horizon,
        # but actual completed trades were lasting ~1 hour — asking the critic to
        # correctly discount 100 steps of noisy hourly reward is a harder fitting
        # problem than the signal supports. 0.95 (~20-step horizon) is a smaller-variance
        # target that better matches realistic holding periods (now >= min_hold_bars).
        "gamma": 0.95,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.05,  # higher entropy prevents collapse to always-hold during early training
        # Decayed by EntropyDecayCallback (rl/callbacks.py) from "ent_coef" down to
        # ent_coef_final over ent_coef_decay_steps. Holding entropy at 0.05 for the whole
        # run was found (2026-08 audit) to keep the policy too noisy to ever commit to a
        # sustained directional stance — decaying it lets it explore early, commit later.
        "ent_coef_final": 0.005,
        "ent_coef_decay_steps": 100_000,
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
                "n_envs": 2,  # small fleet — enough to smoke-test the VecEnv path
                "n_steps": 128,  # buffer = 128 × 2 = 256, matches the pre-n_envs quick buffer
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


def _apply_phase_steps_multiplier(config: dict) -> None:
    """Scale config["phases"]' max_steps by config["phase_steps_multiplier"] (default 1.0,
    no-op). MUST be called AFTER any phase-capping logic (the "cap at 2 phases" blocks
    reset config["phases"] to a fresh, unscaled CURRICULUM_PHASES[:2] slice) — calling
    this before capping silently discards the scaling, which is exactly the bug this
    function fixes (found 2026-08-13: a --phase-steps-multiplier 3 run trained for the
    unscaled 500k steps, not 1.5M, because it was applied too early)."""
    mult = config.get("phase_steps_multiplier", 1.0)
    if mult == 1.0:
        return
    for p in config["phases"]:
        p["max_steps"] = int(p["max_steps"] * mult)
    logger.info(
        f"phase_steps_multiplier={mult}x applied — total steps now "
        f"{sum(p['max_steps'] for p in config['phases']):,}"
    )


def check_regime_stream_compatibility(config: dict) -> int:
    """Verify regime_stream.parquet covers the configured assets and training window.

    obs_mode="raw" never reads a regime stream at all — nothing to check, skipped by
    the caller. obs_mode="raw_regime" reads config["regime_stream_path"] (e.g. the
    raw-feature-baseline HMM stream) instead of the hardcoded production path.
    """
    regime_stream_path = config.get("regime_stream_path", "data/processed/regime_stream.parquet")
    rs = pd.read_parquet(regime_stream_path, engine="pyarrow")
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
    # n_meaningful_trades sits next to every Sharpe so a thin-trade-count number (no
    # statistical power) can't be read at face value the way the 9-trade "1.893" was.
    table.add_column("N Trades", justify="right")
    for name, m in abl["variants"].items():
        table.add_row(
            name,
            f"{m['sharpe']:.3f}",
            f"{m['sortino']:.3f}",
            f"{m['max_drawdown']:.3f}",
            f"{m['total_return']:.3%}",
            str(m.get("n_meaningful_trades", m.get("n_trades", "?"))),
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

    from rl.train import build_env_factory, load_data, load_debate_results

    aligned_dfs, regime_stream_df, encoder = load_data(config)
    debate_results = load_debate_results()
    env_factory = build_env_factory(
        aligned_dfs, regime_stream_df, encoder, debate_results, config
    )
    # CPU for eval — deterministic float ops, removes GPU/cuDNN nondeterminism from the
    # policy forward pass (the network is tiny: 156->256->128->64, cost is negligible).
    model = PPO.load(
        checkpoint,
        device="cpu",
        custom_objects={"policy_class": _policy_class_for(config)},
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

    cfg = _copy.deepcopy(config)
    cfg["seed"] = seed
    # P0.3: seed Python random/NumPy/torch globally before training starts. SB3's own
    # PPO(seed=...) constructor argument (rl/train.py) only covers randomness inside SB3's
    # own scope (policy init, its internal RNG) — it does not touch this repo's own
    # env-level or data-prep randomness that can run before .learn() is called.
    set_seed(seed)
    # Isolate this seed's checkpoints from every other seed's — see CurriculumCallback's
    # checkpoint_suffix docstring. Before this fix, every seed saved to the same bare
    # "ppo_best.zip"/"ppo_phase_N.zip", so only the LAST seed to run in a --seeds sweep
    # left recoverable weights on disk; the lookup below already expected this filename,
    # nothing ever wrote it. Appended (not overwritten) so combining --seeds with
    # --benchmark-relative still isolates both dimensions (e.g. "ppo_best_benchrel_seed0.zip").
    cfg["checkpoint_suffix"] = cfg.get("checkpoint_suffix", "") + f"_seed{seed}"

    train_out = train_ppo(cfg, pre_train_check=sanity_check_env)
    env_factory = train_out["env_factory"]

    # train_out["best_path"] already carries the full checkpoint_suffix (seed + any
    # benchmark_relative marker) — see the comment above.
    best_ckpt = train_out["best_path"]
    default_best = Path("rl/checkpoints/ppo_best.zip")
    ckpt_path = best_ckpt if best_ckpt.exists() else (default_best if default_best.exists() else None)
    if ckpt_path:
        # CPU for eval — see _load_for_eval for why.
        model = PPO.load(str(ckpt_path), device="cpu",
                         custom_objects={"policy_class": _policy_class_for(cfg)})
    else:
        model = train_out["model"]

    # Seed-specific paths — see evaluate_ppo/run_ablation docstrings. Every seed's full
    # result is preserved on disk (useful for an appendix / reproducibility check); the
    # canonical fixed-path files are written explicitly by _aggregate_seeds' caller once
    # the median seed is chosen, not by whichever seed happens to run last.
    eval_path = Path(f"data/results/ppo_evaluation_seed{seed}.json")
    abl_path = Path(f"data/results/ablation_results_seed{seed}.json")
    eval_res = evaluate_ppo(model, env_factory, cfg, output_path=eval_path, checkpoint_path=ckpt_path)
    abl = run_ablation(cfg, model, env_factory, output_path=abl_path)
    return {"eval": eval_res, "ablation": abl, "curriculum": train_out["curriculum_history"],
            "best_sharpe": train_out["best_sharpe"], "seed": seed}


def _aggregate_seeds(seed_results: list[dict], median_seed: int) -> None:
    """Log and save cross-seed Sharpe statistics.

    Includes n_meaningful_trades per seed alongside each Sharpe — a Sharpe from a
    handful of trades over a ~247-day val window carries far less weight than one from
    hundreds, and the bare ratio alone hid exactly that in the 2026-08 audit (a 9-trade
    eval read as a headline 1.893 with nothing in the summary to flag it as thin).
    """
    sharpes = [r["eval"]["marta_ppo"]["sharpe"] for r in seed_results]
    n_trades = [r["eval"]["marta_ppo"]["n_meaningful_trades"] for r in seed_results]
    seeds = [r["seed"] for r in seed_results]
    summary = {
        "seeds": seeds,
        "sharpes": sharpes,
        "n_meaningful_trades": n_trades,
        "mean_sharpe": statistics.mean(sharpes),
        "std_sharpe": statistics.stdev(sharpes) if len(sharpes) > 1 else 0.0,
        "min_sharpe": min(sharpes),
        "max_sharpe": max(sharpes),
        "median_seed": median_seed,
    }
    out = Path("data/results/multi_seed_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    _console.rule("[bold]Multi-Seed Summary")
    _console.print(f"Seeds: {seeds}")
    _console.print(f"Sharpe per seed: {[f'{s:.3f}' for s in sharpes]}  "
                    f"(n_meaningful_trades: {n_trades})")
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
    parser.add_argument(
        "--benchmark-relative",
        action="store_true",
        help=(
            "reward excess return over buy-and-hold instead of absolute risk-adjusted "
            "return (rl/env.py's benchmark_relative). A distinct experiment, not a "
            "default — checkpoints save to a '_benchrel' suffix so they never overwrite "
            "or get confused with the default-reward-mode checkpoints, letting both be "
            "compared explicitly (e.g. run once with and once without this flag, then "
            "diff the two ppo_evaluation results)."
        ),
    )
    parser.add_argument(
        "--obs-mode",
        choices=["encoder", "raw", "raw_regime", "raw_vol"],
        default="encoder",
        help=(
            "observation source (rl/env.py's obs_mode, added 2026-08-12 raw-feature-"
            "baseline pivot). 'encoder' (default) is unchanged prior behavior. 'raw' "
            "replaces the 128-dim encoder embedding with the flattened 60x14 raw "
            "feature window (840-dim), no encoder loaded. 'raw_regime' adds the regime "
            "probability vector on top (see --regime-stream-path). 'raw_vol' (RL "
            "experiment Condition B, added 2026-08-30) adds one dim instead: a "
            "rolling-z-scored GARCH vol_forecast, isolating continuous volatility from "
            "regime discretization. Checkpoints get a '_<obs_mode>' suffix so they "
            "never collide with the default encoder-mode run."
        ),
    )
    parser.add_argument(
        "--regime-stream-path",
        default=None,
        metavar="PATH",
        help=(
            "override regime_stream.parquet path (rl/train.py's load_data). Only "
            "meaningful with --obs-mode raw_regime — e.g. "
            "data/processed/raw_baseline/regime_stream.parquet for the raw-feature "
            "HMM (n_states=6) validated in Step 1 of the raw-feature-baseline pivot."
        ),
    )
    parser.add_argument(
        "--extra-tag",
        default=None,
        metavar="TAG",
        help=(
            "append an arbitrary '_<TAG>' marker to checkpoint_suffix, on top of "
            "whatever obs_mode/crypto5/seed markers already apply. Needed when two runs "
            "would otherwise share an identical auto-generated suffix and silently "
            "overwrite each other's checkpoint/results — e.g. RL experiment Condition C "
            "(--obs-mode raw_regime, real causal stream) vs Condition D (--obs-mode "
            "raw_regime --regime-stream-path <placebo file>) both produce "
            "'_raw_regime_seed{N}' without this, since --regime-stream-path itself "
            "carries no suffix marker."
        ),
    )
    parser.add_argument(
        "--crypto5",
        action="store_true",
        help=(
            "train on the crypto-only 5-asset panel (BTC/ETH/BNB/LTC/XRP_USDT, "
            "data/processed/crypto5_multihorizon) instead of the old 4-asset production "
            "panel (2026-08-24 migration). Sets all_assets/canonical_assets/processed_dir "
            "to the crypto5 panel and phases to CRYPTO5_CURRICULUM_PHASES (BTC, then "
            "BTC+ETH — same step budgets as CURRICULUM_PHASES[:2] for a controlled "
            "comparison). Combine with --obs-mode raw_regime and this flag also sets "
            "--regime-stream-path to the crypto5 panel's own regime_stream.parquet "
            "unless you override it explicitly. obs_mode='encoder' is NOT meaningful "
            "here — no encoder was ever trained on this panel."
        ),
    )
    parser.add_argument(
        "--phase-steps-multiplier",
        type=float,
        default=1.0,
        metavar="X",
        help=(
            "scale every curriculum phase's max_steps by X (default 1.0, exact prior "
            "behavior). Added for the raw-feature-baseline agents (2026-08-12): both "
            "obs_mode='raw'/'raw_regime' agents took large (mean ~0.43), ~50/50-win-rate "
            "directional bets at the default 500k steps — steady bleed, not overtrading "
            "or transaction-cost drag (both were checked and ruled out). Hypothesis: an "
            "840-dim raw window needs more steps than the pre-compressed 128-dim encoder "
            "embedding to learn well-calibrated position sizing, not just direction. "
            "Adds a '_Nx' checkpoint_suffix marker so a scaled run never collides with "
            "the default-length run's checkpoints/eval output."
        ),
    )
    parser.add_argument(
        "--early-stop-sharpe",
        type=float,
        default=None,
        metavar="S",
        help=(
            "override CurriculumCallback's early_stop_sharpe (default -0.5, patience 3 "
            "evals — rl/callbacks.py). Found 2026-08-13: a --phase-steps-multiplier 3 run "
            "still finished in 7 minutes because early stopping triggered at step 30,000 "
            "(3 consecutive evals below -0.5), before the multiplier could ever be tested "
            "— both the 1x and 3x runs are seeded identically and diverge only after step "
            "30k, so they produced byte-identical results. Pass e.g. -10 to effectively "
            "disable early stopping for one diagnostic run."
        ),
    )
    parser.add_argument(
        "--min-hold-bars",
        type=int,
        default=None,
        metavar="N",
        help=(
            "override the minimum bars a newly-opened position must be held before "
            "flattening/reversing (default 4 — config['min_hold_bars'], rl/env.py). Raise "
            "this if the trained policy is churning (opening/closing every few bars) more "
            "than a real directional edge would justify."
        ),
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        metavar="C",
        help="override PPO's starting entropy coefficient (default 0.05 — config['ent_coef']).",
    )
    parser.add_argument(
        "--ent-coef-final",
        type=float,
        default=None,
        metavar="C",
        help="override the entropy coefficient EntropyDecayCallback decays to (default 0.005).",
    )
    parser.add_argument(
        "--ent-coef-decay-steps",
        type=int,
        default=None,
        metavar="N",
        help="override how many steps EntropyDecayCallback takes to decay ent_coef (default 100_000).",
    )
    parser.add_argument(
        "--bc-warmstart",
        action="store_true",
        help=(
            "behavior-cloning pretrain the policy on a buy_hold teacher (rl/behavior_cloning.py) "
            "before PPO training starts. See that module's docstring for why buy_hold, not "
            "agents_no_regime, is the teacher — the debate cache has essentially zero coverage "
            "over the training split, so agents_no_regime's rule would just clone 'hold'."
        ),
    )
    parser.add_argument(
        "--bc-epochs", type=int, default=None, metavar="N",
        help="BC pretraining epochs over the teacher dataset (default 20).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, metavar="LR",
        help="override PPO's learning rate (default 3e-4). Lower (e.g. 1e-5) after --bc-warmstart "
             "so PPO fine-tunes near the cloned prior instead of immediately overwriting it.",
    )
    parser.add_argument(
        "--clip-range", type=float, default=None, metavar="C",
        help="override PPO's clip_range (default 0.2). Tighter (e.g. 0.1) keeps fine-tuning "
             "close to a BC-warm-started prior.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["hybrid", "discrete"],
        default="hybrid",
        help=(
            "'hybrid' (default): Categorical direction + Beta position-fraction "
            "(MARTAActorCriticPolicy, exact prior behavior). 'discrete': single "
            "Categorical over rl.policy.DISCRETE_POSITION_MAP's fixed position levels "
            "(MARTADiscretePolicy, 2026-08-24) — removes the Beta head entirely, added "
            "after rl/crypto5_feature_predictability.py found ~zero R2 on next-bar "
            "return magnitude (only a small directional edge), so the Beta head has no "
            "real magnitude signal to fit and micro-jitters instead."
        ),
    )
    parser.add_argument(
        "--single-phase",
        action="store_true",
        help="train only phases[0], ignoring every later curriculum phase. For "
             "standalone single-asset experiments (e.g. --crypto5 --action-mode discrete) "
             "that shouldn't auto-advance to a multi-asset phase.",
    )
    args = parser.parse_args()

    config = build_config(args.quick)
    # NOTE: multiplier is APPLIED later (_apply_phase_steps_multiplier), right before each
    # train_ppo() call — config["phases"] gets reset to a fresh CURRICULUM_PHASES[:2] slice
    # by the "cap at 2 phases" logic further down in both the --seeds and default paths, so
    # scaling here would be silently discarded. Only the checkpoint_suffix marker is set now.
    if args.phase_steps_multiplier != 1.0:
        config["phase_steps_multiplier"] = args.phase_steps_multiplier
        config["checkpoint_suffix"] = (
            config.get("checkpoint_suffix", "") + f"_{args.phase_steps_multiplier:g}x"
        )
    if args.early_stop_sharpe is not None:
        config["early_stop_sharpe"] = args.early_stop_sharpe
        config["checkpoint_suffix"] = (
            config.get("checkpoint_suffix", "") + f"_es{args.early_stop_sharpe:g}"
        )
    if args.min_hold_bars is not None:
        config["min_hold_bars"] = args.min_hold_bars
        config["checkpoint_suffix"] = config.get("checkpoint_suffix", "") + f"_hold{args.min_hold_bars}"
    if args.ent_coef is not None:
        config["ent_coef"] = args.ent_coef
    if args.ent_coef_final is not None:
        config["ent_coef_final"] = args.ent_coef_final
    if args.ent_coef_decay_steps is not None:
        config["ent_coef_decay_steps"] = args.ent_coef_decay_steps
    if args.bc_warmstart:
        config["bc_warmstart"] = True
        config["checkpoint_suffix"] = config.get("checkpoint_suffix", "") + "_bc"
    if args.bc_epochs is not None:
        config["bc_epochs"] = args.bc_epochs
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.clip_range is not None:
        config["clip_range"] = args.clip_range
    if args.action_mode != "hybrid":
        config["action_mode"] = args.action_mode
        config["checkpoint_suffix"] = config.get("checkpoint_suffix", "") + f"_{args.action_mode}"
    # P0.3: seed Python random/NumPy/torch globally before anything else runs. The
    # --seeds sweep re-seeds per-seed inside _run_single_seed; this covers the
    # single-seed (default) path, which previously relied solely on SB3's internal
    # PPO(seed=...) handling and was never seeded at the script level.
    set_seed(config["seed"])
    # Crypto-only panel (2026-08-24 migration) — set before obs_mode/regime_stream_path
    # handling below so --crypto5's own regime_stream_path default is in place before an
    # explicit --regime-stream-path (checked after) can still override it.
    if args.crypto5:
        if args.obs_mode == "encoder":
            raise SystemExit(
                "--crypto5 requires --obs-mode raw or raw_regime — no encoder was ever "
                "trained on the crypto5 panel, so obs_mode='encoder' would silently feed "
                "out-of-distribution data through it."
            )
        config["all_assets"] = CRYPTO5_CANONICAL_ASSETS
        config["canonical_assets"] = CRYPTO5_CANONICAL_ASSETS
        # str(), not a bare Path — rl/evaluate.py's _build_provenance() serializes
        # config["regime_stream_path"] straight into ppo_evaluation.json's JSON output via
        # json.dumps(..., default=float); a raw PosixPath there crashes json.dumps AFTER
        # training completes and a real checkpoint is already saved (found running the
        # first real crypto5 training run — the CLI's own --regime-stream-path flag never
        # hit this because argparse stores it as a plain str already).
        config["processed_dir"] = str(CRYPTO5_PROCESSED)
        # --quick already set its own tiny (1024-step) BTC/BTC+ETH phases in build_config —
        # both asset names are valid crypto5 assets too, so keep those for a fast smoke
        # test instead of clobbering them with the full 500k-step curriculum.
        if not args.quick:
            config["phases"] = copy.deepcopy(CRYPTO5_CURRICULUM_PHASES)
        # Causal stream (regime/causal_regime_stream.py), not the leaky
        # regime_stream.parquet — see rl/train.py's CRYPTO5_REGIME_STREAM_PATH docstring
        # for why the leaky one must never be a default again. --regime-stream-path below
        # can still override this explicitly (e.g. to reproduce the old leaky numbers).
        config["regime_stream_path"] = str(CRYPTO5_REGIME_STREAM_CAUSAL_PATH)
        config["checkpoint_suffix"] = config.get("checkpoint_suffix", "") + "_crypto5"
    # --single-phase must run AFTER --crypto5 (which sets config["phases"] to the full
    # 2-phase CRYPTO5_CURRICULUM_PHASES) and before the "cap at N phases" length checks
    # further below (both only trigger on len(phases) > 2, so a length-1 list from here
    # passes through untouched either way) — placed here so it reliably wins regardless
    # of what any earlier flag set config["phases"] to.
    if args.single_phase:
        config["phases"] = config["phases"][:1]
        config["checkpoint_suffix"] = config.get("checkpoint_suffix", "") + "_1phase"
    # Eval-only is BTC-focused (val metrics, ablation, baselines are all BTC) → keep it light.
    if args.eval_only:
        config["all_assets"] = ["BTC_USDT"]
    if args.benchmark_relative:
        config["benchmark_relative"] = True
        config["checkpoint_suffix"] = config.get("checkpoint_suffix", "") + "_benchrel"
    if args.obs_mode != "encoder":
        config["obs_mode"] = args.obs_mode
        config["checkpoint_suffix"] = config.get("checkpoint_suffix", "") + f"_{args.obs_mode}"
        if args.regime_stream_path:
            config["regime_stream_path"] = args.regime_stream_path
    if args.extra_tag:
        config["checkpoint_suffix"] = config.get("checkpoint_suffix", "") + f"_{args.extra_tag}"
    logger.info(
        f"Device: {config['device']} | quick={args.quick} | eval_only={args.eval_only or False} "
        f"| benchmark_relative={config['benchmark_relative']} | obs_mode={args.obs_mode}"
    )

    # Pre-training compatibility check: regime stream must cover all configured assets
    # and the full training window (fails fast on a stale regime_stream.parquet).
    # Skipped entirely for obs_mode="raw" — no regime stream is read in that mode.
    if not args.eval_only and config.get("obs_mode", "encoder") != "raw":
        check_regime_stream_compatibility(config)

    # Multi-seed path: run each seed independently, aggregate, then exit.
    if args.seeds and not args.eval_only:
        if len(config["phases"]) > 2 and not args.crypto5:
            # Same 2-phase (BTC, BTC+ETH) cap the non-seeds path applies below — a 2026-08
            # gap audit found this branch never had it, so every --seeds sweep ran the
            # full 4-phase curriculum even though Phases 3-4 (EURUSD/GOLD) are documented
            # to collapse on daily-upsampled data. Early stopping usually catches a bad
            # Phase 2 before Phase 3 is reached, but not always — and letting it run
            # further just burns wall-clock and log volume on a known-bad regime.
            # `not args.crypto5` guard added 2026-10-04: this check only knows about the
            # OLD 4-asset panel's CURRICULUM_PHASES; crypto5's 5-asset curriculum (BTC/ETH/
            # BNB/LTC/XRP, extended 2026-09-30) has no forex/commodity daily-upsampling
            # problem, so capping it at 2 phases here would silently fall back to
            # CURRICULUM_PHASES[:2] (the wrong panel's phase configs) and Phases 3-5 would
            # never run at all.
            config["phases"] = CURRICULUM_PHASES[:2]
        _apply_phase_steps_multiplier(config)
        mlflow.set_experiment(_EXPERIMENT)
        seed_results = []
        for seed in args.seeds:
            logger.info(f"=== Seed {seed} ===")
            with mlflow.start_run(run_name=f"month3_ppo_seed{seed}"):
                mlflow.log_param("seed", seed)
                res = _run_single_seed(config, seed)
                mlflow.log_metric("val/sharpe", res["eval"]["marta_ppo"]["sharpe"])
                seed_results.append(res)
        # Select the median-Sharpe seed — NOT the best one. A 2026-08 audit found this
        # used to compute argmax (the single best-performing seed) despite the comment
        # already claiming "median": with 3 seeds landing at [1.902, 1.201, 0.000] (mean
        # 1.034, std 0.962 — std nearly as large as the mean), printing the best seed's
        # full eval/ablation/significance tables gave a badly misleading picture of "the"
        # result. argsort + middle index is the actual median for an odd seed count; for
        # an even count this picks the lower of the two middle values (a concrete seed's
        # full result dict is needed to print, so there's no seed to interpolate to).
        sharpes = [r["eval"]["marta_ppo"]["sharpe"] for r in seed_results]
        order = np.argsort(sharpes)
        median_idx = int(order[len(order) // 2])
        median_seed = seed_results[median_idx]["seed"]

        _aggregate_seeds(seed_results, median_seed)

        # Explicitly persist the chosen (median) seed's full eval + ablation to the
        # canonical fixed paths that every downstream consumer (dashboard, ablation
        # table, writeup) reads. A 2026-08 audit found this step never existed —
        # evaluate_ppo/run_ablation wrote to the same fixed path once per seed inside
        # _run_single_seed, so whichever seed ran LAST silently overwrote the other two.
        # In that run seed 2 (the worst of three, Sharpe 0.227) ran last and its result
        # sat in data/results/ablation_results.json — described in CLAUDE.md as the
        # project's "central research result" — instead of the median seed the code's
        # own comments already said it intended to report.
        canonical_eval_path = Path("data/results/ppo_evaluation.json")
        canonical_abl_path = Path("data/results/ablation_results.json")
        eval_out = dict(seed_results[median_idx]["eval"])
        eval_out["_source_seed"] = median_seed
        eval_out["_selection"] = "median of multi-seed sweep"
        abl_out = dict(seed_results[median_idx]["ablation"])
        abl_out["_source_seed"] = median_seed
        abl_out["_selection"] = "median of multi-seed sweep"
        canonical_eval_path.write_text(json.dumps(eval_out, indent=2, default=float))
        canonical_abl_path.write_text(json.dumps(abl_out, indent=2, default=float))
        logger.success(
            f"Canonical results (median seed={median_seed}) written to "
            f"{canonical_eval_path} and {canonical_abl_path}"
        )

        logger.info(
            f"Printing detailed eval/ablation for the MEDIAN seed: "
            f"seed={median_seed} sharpe={sharpes[median_idx]:.3f} "
            f"(all seeds: {[round(s, 3) for s in sharpes]})"
        )
        _print_eval(seed_results[median_idx]["eval"])
        _print_ablation(seed_results[median_idx]["ablation"])
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
            if len(config["phases"]) > 2 and not args.crypto5:
                # Cap curriculum at Phase 2 (BTC+ETH only).
                # Phases 3-4 add EURUSD/GOLD which have low intraday variance
                # causing PPO to collapse to always-hold. Documented in CLAUDE.md.
                # `not args.crypto5` guard added 2026-10-04 — see the matching comment
                # in the --seeds branch above for why this cap must not apply to the
                # crypto5 5-phase curriculum.
                config["phases"] = CURRICULUM_PHASES[:2]
            _apply_phase_steps_multiplier(config)
            train_out = train_ppo(config, pre_train_check=sanity_check_env)
            env_factory = train_out["env_factory"]
            # Evaluate the best-by-val-Sharpe checkpoint, NOT the final model. The final
            # model can be a degraded policy — e.g. the curriculum collapses to always-flat
            # once daily-upsampled forex/commodity assets (zero intraday returns) are added
            # in phases 3-4. ppo_best.zip holds the best policy seen during training.
            from stable_baselines3 import PPO

            # train_out["best_path"] already carries checkpoint_suffix (e.g. "_benchrel"
            # for --benchmark-relative runs) — reading it here instead of hardcoding
            # "ppo_best.zip" is what keeps a benchmark_relative run's checkpoint from
            # being confused with (or silently overwriting the read of) the default
            # reward mode's checkpoint.
            best_ckpt = train_out["best_path"]
            if best_ckpt.exists():
                # CPU for eval — see _load_for_eval for why.
                model = PPO.load(
                    str(best_ckpt),
                    device="cpu",
                    custom_objects={"policy_class": _policy_class_for(config)},
                )
                logger.info("Evaluating ppo_best.zip (best val Sharpe), not the final model")
            else:
                logger.warning("ppo_best.zip not found — falling back to final model for eval")
                model = train_out["model"]

        # 2. Evaluate on validation set (optionally extended window / multi-asset).
        # output_path carries checkpoint_suffix (e.g. "_raw", "_raw_regime", "_benchrel")
        # so a non-default run never silently overwrites the canonical encoder-mode
        # ppo_evaluation.json/ablation_results.json — found 2026-08-12 when a raw-mode
        # run's eval clobbered the previous run's file (gitignored, no MLflow artifact
        # backup, so it's an unrecoverable-but-non-production loss when this happens).
        suffix = config.get("checkpoint_suffix", "")
        eval_out_path = Path(f"data/results/ppo_evaluation{suffix}.json") if suffix else None
        abl_out_path = Path(f"data/results/ablation_results{suffix}.json") if suffix else None
        val_end_override = VAL_END_EXTENDED if args.extended_val else None
        eval_checkpoint_path = args.eval_only if args.eval_only else best_ckpt
        eval_res = evaluate_ppo(
            model, env_factory, config, val_end=val_end_override, output_path=eval_out_path,
            checkpoint_path=eval_checkpoint_path,
        )
        m = eval_res["marta_ppo"]
        for k, v in m.items():
            mlflow.log_metric(f"val/{k}", float(v))

        # 3. Ablation study.
        abl = run_ablation(config, model, env_factory, output_path=abl_out_path)
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
                    # CPU for eval — see _load_for_eval for why.
                    extra = PPO.load(
                        str(ckpt), device="cpu",
                        custom_objects={"policy_class": _policy_class_for(config)},
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
