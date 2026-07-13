"""Final test-set evaluation — MARTA PPO vs trading baselines on genuinely unseen 2024 data.

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python run_test_set.py

Test window: 2024-01-01 → 2024-12-31 (clamped to the aligned data's actual end).

Provenance under the 2024-extension configuration (see configs/assets.yaml):
    - TFT encoder:   trained < 2023-01-01, early-stopped on 2023; the Kaggle training
                     dataset is trimmed to ≤ 2023-12-31, so 2024 never reaches the
                     encoder in any form (not even model selection)
    - HMM + PCA:     fitted on timestamps < 2023-01-01
    - RL PPO:        train ≤ 2022-12-31, val 2023-01-01 → 2023-12-31 (rl/env.py)

    ⇒ 2024 is untouched by every layer, including checkpoint selection.

If the PPO holds a zero position for the whole test window (the failure mode observed
on the pre-extension checkpoints), a diagnostic pass over the 2023 validation window
separates "policy abstains everywhere out-of-sample" from "2024-specific shift".

Baselines share the environment's transaction cost (0.001 per unit turnover) and are
computed on exactly the same bars as the PPO walk-forward.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from rl.ablation import (
    _run_buy_hold,
    _run_momentum_rule,
    _run_rule_positions,
    bootstrap_sharpe_ci,
    ma_crossover_strategy,
    rsi_strategy,
)
from rl.env import MARTATradingEnv, VAL_END, VAL_START
from rl.evaluate import compute_metrics, run_walk_forward
from utils.logger import logger

ALIGNED_PATH = Path("data/processed/BTC_USDT_aligned.parquet")
RESULTS_PATH = Path("data/results/test_set_results.json")
CHECKPOINT = "rl/checkpoints/ppo_best.zip"
ASSET = "BTC_USDT"

TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

STRATEGY_LABELS: dict[str, str] = {
    "marta_ppo": "MARTA PPO",
    "buy_hold": "Buy & Hold",
    "rsi_strategy": "RSI Strategy",
    "ma_crossover": "MA Crossover",
    "momentum": "Momentum",
}


def resolve_test_window(
    index: pd.DatetimeIndex,
) -> tuple[pd.Timestamp, pd.Timestamp, str, list[str]]:
    """Return (test_start, test_end, warning, caveats) for the 2024 test window,
    clamped to the aligned data's actual coverage."""
    start = pd.Timestamp(TEST_START, tz="UTC")
    end = min(pd.Timestamp(TEST_END + " 23:59:59", tz="UTC"), index[-1])
    if index[-1] < start:
        raise RuntimeError(
            f"Aligned data ends {index[-1]} — no 2024 test data. "
            "Run pipeline/fetch_2024.py and run_pipeline.py first."
        )
    warning = (
        "Single-asset (BTC), single-path test window — treat bootstrap CIs as "
        "sampling noise on one path, not as generalisation evidence"
    )
    caveats = [
        "2024 is untouched by every layer (encoder trained <2023 with the Kaggle "
        "dataset trimmed to ≤2023-12-31; HMM/PCA fit <2023; PPO train ≤2022, val 2023) "
        "— provided the trimmed-upload instruction was followed for the encoder",
        "BTC 2024 was a strong bull year (April halving, ~$73k ATH in March, ~$100k in "
        "November) — long-biased baselines like buy-and-hold are flattered",
        "No cached debates exist in 2024, so the PPO's 16-dim consensus slot is zeros "
        "(its usual state during training)",
        "One asset, one path: results should be replicated across assets before any "
        "generalisation claim",
    ]
    return start, end, warning, caveats


def restrict_env_to_window(
    env: MARTATradingEnv,
    start: pd.Timestamp,
    end: pd.Timestamp,
    asset: str = ASSET,
) -> MARTATradingEnv:
    """Intersect the env's valid start indices with [start, end] so a sequential pass
    traverses exactly the test window (same pattern as ``analyze_regime_behavior``)."""
    idx = env._index[asset]
    in_window = np.flatnonzero((idx >= start) & (idx <= end))
    valid = np.intersect1d(env._valid_starts[asset], in_window)
    if len(valid) < 2:
        raise RuntimeError(f"Test window {start}..{end} has <2 valid bars for {asset}")
    env._valid_starts[asset] = valid
    return env


def run_diagnostic_window(
    model: Any,
    env_factory: Callable[..., MARTATradingEnv],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Run the PPO over a secondary window and return its metrics (diagnostic only)."""
    env = env_factory(assets=[ASSET], split="all", sequential=True, debate=False)
    restrict_env_to_window(env, start, end)
    traj = run_walk_forward(model, env, deterministic=True)
    metrics = compute_metrics(traj["returns"], traj["portfolio_values"], traj["positions"])
    metrics["n_steps"] = int(len(traj["returns"]))
    return metrics


def evaluate_test_set() -> dict[str, Any]:
    """Run PPO + all baselines on the held-out 2024 test window and save the results."""
    t_start = time.time()

    # ── Step 1: verify what test data actually exists ────────────────────────
    df = pd.read_parquet(ALIGNED_PATH, engine="pyarrow")
    print(f"Full data range: {df.index[0]} → {df.index[-1]}  ({len(df)} rows)")
    test_start, test_end, warning, caveats = resolve_test_window(df.index)
    n_weeks = (test_end - test_start).days / 7.0
    logger.info(f"Test window: {test_start.date()} → {test_end.date()} (~{n_weeks:.0f} weeks)")

    # ── Step 2: load checkpoint + build window-restricted envs ───────────────
    from run_rl import _load_for_eval, build_config

    config = build_config(quick=False)
    config["all_assets"] = [ASSET]
    model, env_factory = _load_for_eval(config, CHECKPOINT)

    def make_test_env(**kwargs: Any) -> MARTATradingEnv:
        env = env_factory(assets=[ASSET], split="all", sequential=True, **kwargs)
        return restrict_env_to_window(env, test_start, test_end)

    logger.info("Running MARTA PPO walk-forward on the test window")
    ppo_env = make_test_env(debate=False)  # no cached debates in window → consensus zeros
    traj = run_walk_forward(model, ppo_env, deterministic=True)
    returns_by_strategy: dict[str, np.ndarray] = {"marta_ppo": traj["returns"]}
    results: dict[str, dict[str, float]] = {
        "marta_ppo": compute_metrics(traj["returns"], traj["portfolio_values"], traj["positions"])
    }
    results["marta_ppo"]["method"] = "trained PPO (ppo_best.zip), frozen encoder, consensus zeros"

    # ── Step 3: baselines on exactly the same bars ───────────────────────────
    baseline_runs: dict[str, tuple[Callable[[], dict[str, np.ndarray]], str]] = {
        "buy_hold": (
            lambda: _run_buy_hold(make_test_env(debate=False)),
            "buy-and-hold BTC over the test window",
        ),
        "rsi_strategy": (
            lambda: _run_rule_positions(make_test_env(debate=False), rsi_strategy),
            "Classic RSI mean-reversion (buy <30, sell >70, fixed 0.3 size), raw Wilder RSI(14)",
        ),
        "ma_crossover": (
            lambda: _run_rule_positions(make_test_env(debate=False), ma_crossover_strategy),
            "MA crossover (12h/26h, long above / short below, fixed 0.3 size)",
        ),
        "momentum": (
            lambda: _run_momentum_rule(
                make_test_env(zero_regime=True, zero_consensus=True, debate=False)
            ),
            "momentum rule (3-step unanimous close_return sign, fixed 0.3 size, "
            "CVaR risk cap via env) — same rule as ppo_no_regime in the ablation",
        ),
    }
    for name, (runner, method) in baseline_runs.items():
        logger.info(f"Running baseline '{name}' on the test window")
        b_traj = runner()
        returns_by_strategy[name] = b_traj["returns"]
        results[name] = compute_metrics(
            b_traj["returns"], b_traj["portfolio_values"], b_traj["positions"]
        )
        results[name]["method"] = method

    # ── Bootstrap CIs on Sharpe deltas (block bootstrap, 168h blocks) ────────
    bootstrap: dict[str, dict[str, float]] = {}
    for name in baseline_runs:
        delta, lo, hi = bootstrap_sharpe_ci(
            returns_by_strategy["marta_ppo"], returns_by_strategy[name]
        )
        bootstrap[f"marta_ppo_vs_{name}"] = {
            "delta_sharpe": delta,
            "ci_lower_95": lo,
            "ci_upper_95": hi,
            "significant": bool(lo > 0 or hi < 0),
        }

    # ── Abstention diagnostic ────────────────────────────────────────────────
    # If the policy is fully flat on 2024, probe the 2023 validation window: trading
    # there but not in 2024 ⇒ 2024-specific distribution shift; flat in both ⇒ the
    # same out-of-sample abstention pathology seen on the pre-extension checkpoints.
    key_finding: str | None = None
    diagnostics: dict[str, Any] = {}
    ppo_flat = results["marta_ppo"]["n_trades"] == 0
    if ppo_flat:
        logger.info("PPO fully flat on test window — probing the validation window")
        diag = run_diagnostic_window(model, env_factory, VAL_START, VAL_END)
        diagnostics[f"ppo_val_{VAL_START.year}"] = {
            **diag,
            "note": "validation window (policy was selected on this period)",
        }
        if diag["n_trades"] == 0:
            key_finding = (
                "MARTA PPO holds a zero position on EVERY step of the 2024 test window "
                "AND on its own validation window — the policy abstains everywhere "
                "out-of-sample, reproducing the pre-extension pathology. Check training "
                "logs before interpreting any other number in this file."
            )
        else:
            key_finding = (
                "MARTA PPO trades on its 2023 validation window but holds a zero "
                "position on every step of 2024 — consistent with a 2024-specific "
                "distribution shift rather than a degenerate policy. It preserves "
                "capital by abstaining, but produces no 2024 alpha."
            )

    # ── Step 4: save ─────────────────────────────────────────────────────────
    out: dict[str, Any] = {
        "test_period": {"start": str(test_start.date()), "end": str(test_end.date())},
        "n_steps": int(len(traj["returns"])),
        "warning": warning,
        "caveats": caveats,
        "key_finding": key_finding,
        "diagnostics": diagnostics,
        "checkpoint": CHECKPOINT,
        "asset": ASSET,
        "transaction_cost": config["transaction_cost"],
        "bootstrap_ci": bootstrap,
        **results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=float))
    logger.success(f"Test set results written to {RESULTS_PATH}")

    # ── Step 5: honest summary ───────────────────────────────────────────────
    bar = "═" * 55
    print(f"\n{bar}")
    print("MARTA — TEST SET EVALUATION (GENUINELY UNSEEN DATA)")
    print(
        f"Test period: {test_start.date()} → {test_end.date()} "
        f"(~{n_weeks:.0f} weeks, {out['n_steps']} hourly bars)"
    )
    print(f"WARNING: {warning}")
    print(bar)
    print(f"\n{'Strategy':<18}{'Sharpe':>8}{'Max DD':>10}{'Total Ret':>12}")
    print("─" * 48)
    for name, label in STRATEGY_LABELS.items():
        m = results[name]
        print(
            f"{label:<18}{m['sharpe']:>8.3f}{m['max_drawdown']:>9.1%}"
            f"{m['total_return']:>11.1%}"
        )
    ppo_sharpe = results["marta_ppo"]["sharpe"]
    print()
    for name in ("buy_hold", "rsi_strategy", "ma_crossover", "momentum"):
        ci = bootstrap[f"marta_ppo_vs_{name}"]
        print(
            f"MARTA vs {STRATEGY_LABELS[name]:<13} ΔSharpe = "
            f"{ppo_sharpe - results[name]['sharpe']:>7.3f}   "
            f"bootstrap 95% CI [{ci['ci_lower_95']:.3f}, {ci['ci_upper_95']:.3f}]"
            f"{'  *significant*' if ci['significant'] else ''}"
        )
    if key_finding:
        print(f"\nKEY FINDING:\n{key_finding}")
    print("\nHONEST CAVEATS:")
    for c in caveats:
        print(f"- {c}")
    print(bar)
    logger.success(f"run_test_set.py completed in {time.time() - t_start:.0f}s")
    return out


if __name__ == "__main__":
    evaluate_test_set()
