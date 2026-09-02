"""Multi-year strategy comparison — 9 variants x N calendar years, one metrics table per year.

Variants (all evaluated on BTC_USDT, transaction cost 0.001 matching rl/env.py's default):
    b&h                      buy-and-hold
    agent_no_regime          LLM agents (nearest cached debate, <=7 days), regime-blind prompts
    agent_with_regime        LLM agents, regime-informed prompts (main debate cache)
    ppo_only                 trained PPO, obs_mode="raw" (no regime, no agent consensus)
    ppo_regime                trained PPO, obs_mode="raw_regime" (regime prob vector, no agents)
    ppo_regime_agents        same raw_regime PPO + live agent consensus in the observation
    rsi                      classic Wilder RSI(14) mean-reversion
    ma_crossover              12h/26h MA crossover
    momentum                  3-bar unanimous-sign momentum

Checkpoints (reused, not retrained — see docs/RESULTS_AND_METHODOLOGY.md's 2026-08-24
reconciliation table, the project's own controlled obs_mode ablation):
    ppo_only        -> rl/checkpoints/ppo_best.zip                    (obs_mode="raw")
    ppo_regime       -> rl/checkpoints/ppo_best_3x_es-10_raw_regime.zip (obs_mode="raw_regime",
                        regime_stream_path=data/processed/raw_baseline/regime_stream.parquet)
Both predate the 2026-08-24 crypto-only migration and were trained on the 4-asset panel
(BTC_USDT, ETH_USDT, EURUSD, GOLD) with BTC_USDT as one of the canonical slots.

Agent-debate coverage caveat (checked empirically before writing this script): both debate
caches (`data/debates/debate_results.json`, `data/debates/debate_results_no_regime.json`)
have essentially all of their real density in 2024 (383/384 and 73/81 entries respectively;
the handful of stray 2018/2019 entries are demo-scenario leftovers from run_debate.py, not
real coverage). Outside 2024 the nearest-debate lookup (7-day window) returns nothing on
virtually every step, which would silently render as "always hold, Sharpe 0.0" rather than
an honest N/A. So `agent_no_regime`, `agent_with_regime`, and `ppo_regime_agents` are all
restricted to 2024 only, not just the marta_full-style variant the task asked to limit —
same root cause, applied consistently.

PPO train/val/test split (configs/splits.py): train < 2024, val = 2024, test = 2025. Years
2018-2023 are IN-SAMPLE for both PPO checkpoints (their numbers describe memorization
capacity, not generalization); 2024 is the validation window both checkpoints were selected
against; 2025 is the sacred held-out test window. Every year's split status is reported
alongside its metrics for exactly this reason.

Run:
    /home/ramy/miniconda3/envs/marta/bin/python scripts/multi_year_strategy_comparison.py
    /home/ramy/miniconda3/envs/marta/bin/python scripts/multi_year_strategy_comparison.py --years 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

# Running `python scripts/multi_year_strategy_comparison.py` puts scripts/ (not the repo
# root) on sys.path[0], so `import configs`/`import rl` fail — same gotcha CLAUDE.md notes
# for scripts/kaggle_debate_coverage.py. Fix it the same way: insert the repo root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.splits import TEST_START, TRAIN_END_EXCLUSIVE, VAL_START
from rl.ablation import (
    _load_no_regime_debates,
    _run_agents_rule,
    _run_buy_hold,
    _run_rule_positions,
    bootstrap_sharpe_ci,
    ma_crossover_strategy,
    momentum_strategy,
    rsi_strategy,
)
from rl.env import MARTATradingEnv
from rl.evaluate import _periods_per_year_for_env, compute_metrics, run_walk_forward
from utils.logger import logger

_console = Console()

ASSET = "BTC_USDT"
DEFAULT_YEARS = list(range(2019, 2026))  # 2019-2025; 2018 is a partial year (data starts 01-12)
AGENT_DEBATE_YEAR = 2024  # the only year either debate cache has real coverage — see module docstring

PPO_ONLY_CHECKPOINT = "rl/checkpoints/ppo_best.zip"
PPO_REGIME_CHECKPOINT = "rl/checkpoints/ppo_best_3x_es-10_raw_regime.zip"
PPO_REGIME_STREAM_PATH = "data/processed/raw_baseline/regime_stream.parquet"

RESULTS_PATH = Path("data/results/multi_year_strategy_comparison.json")

VARIANT_ORDER = [
    "buy_hold", "agent_no_regime", "agent_with_regime",
    "ppo_only", "ppo_regime", "ppo_regime_agents",
    "rsi", "ma_crossover", "momentum",
]
VARIANT_LABELS = {
    "buy_hold": "Buy & Hold",
    "agent_no_regime": "Agents (no regime)",
    "agent_with_regime": "Agents (with regime)",
    "ppo_only": "PPO only",
    "ppo_regime": "PPO + regime",
    "ppo_regime_agents": "PPO + regime + agents",
    "rsi": "RSI",
    "ma_crossover": "MA Crossover",
    "momentum": "Momentum",
}
# Variants gated on real debate-cache coverage — see module docstring.
_AGENT_GATED_VARIANTS = {"agent_no_regime", "agent_with_regime", "ppo_regime_agents"}


def _split_label(year: int) -> str:
    if year < pd.Timestamp(TRAIN_END_EXCLUSIVE).year:
        return "train (in-sample)"
    if year == pd.Timestamp(VAL_START).year:
        return "val"
    if year == pd.Timestamp(TEST_START).year:
        return "test (sacred)"
    return "out-of-window"


def _year_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
    return start, end


def _restrict_to_window(
    env: MARTATradingEnv, asset: str, start: pd.Timestamp, end: pd.Timestamp
) -> int:
    """Intersect env._valid_starts[asset] with [start, end] in place; returns the count
    of valid bars (same pattern as run_test_set.py's restrict_env_to_window /
    analyze_regime_behavior's window restriction)."""
    idx = env._index[asset]
    in_window = np.flatnonzero((idx >= start) & (idx <= end))
    valid = np.intersect1d(env._valid_starts[asset], in_window)
    env._valid_starts[asset] = valid
    return len(valid)


def _metrics_from_traj(traj: dict[str, np.ndarray], env: MARTATradingEnv) -> dict[str, float]:
    return compute_metrics(
        traj["returns"], traj["portfolio_values"], traj["positions"],
        periods_per_year=_periods_per_year_for_env(env),
    )


def _build_ppo_only_config() -> dict[str, Any]:
    from run_rl import build_config
    config = build_config(quick=False)
    config["all_assets"] = [ASSET]
    config["obs_mode"] = "raw"
    return config


def _build_ppo_regime_config() -> dict[str, Any]:
    from run_rl import build_config
    config = build_config(quick=False)
    config["all_assets"] = [ASSET]
    config["obs_mode"] = "raw_regime"
    config["regime_stream_path"] = PPO_REGIME_STREAM_PATH
    return config


def run_variant(
    name: str,
    year: int,
    factory_raw: Callable[..., MARTATradingEnv],
    factory_regime: Callable[..., MARTATradingEnv],
    model_ppo_only: Any,
    model_ppo_regime: Any,
) -> dict[str, Any]:
    """Run one (variant, year) cell. Returns a dict with either 'metrics'+'returns'+'method'
    or 'status' describing why it was skipped."""
    if name in _AGENT_GATED_VARIANTS and year != AGENT_DEBATE_YEAR:
        return {"status": "n/a", "note": f"real debate coverage exists only for {AGENT_DEBATE_YEAR}"}

    start, end = _year_window(year)

    if name == "buy_hold":
        env = factory_raw(assets=[ASSET], split="all", sequential=True, debate=False)
        n = _restrict_to_window(env, ASSET, start, end)
        if n < 2:
            return {"status": "insufficient_data"}
        traj = _run_buy_hold(env)
        method = "Buy-and-hold BTC over the calendar year"

    elif name == "rsi":
        env = factory_raw(assets=[ASSET], split="all", sequential=True, debate=False)
        n = _restrict_to_window(env, ASSET, start, end)
        if n < 2:
            return {"status": "insufficient_data"}
        traj = _run_rule_positions(env, rsi_strategy)
        method = "Wilder RSI(14) mean-reversion (buy <30, sell >70, fixed 0.3 size)"

    elif name == "ma_crossover":
        env = factory_raw(assets=[ASSET], split="all", sequential=True, debate=False)
        n = _restrict_to_window(env, ASSET, start, end)
        if n < 2:
            return {"status": "insufficient_data"}
        traj = _run_rule_positions(env, ma_crossover_strategy)
        method = "MA crossover (12h/26h, fixed 0.3 size)"

    elif name == "momentum":
        env = factory_raw(assets=[ASSET], split="all", sequential=True, debate=False)
        n = _restrict_to_window(env, ASSET, start, end)
        if n < 2:
            return {"status": "insufficient_data"}
        traj = _run_rule_positions(env, momentum_strategy)
        method = "3-bar unanimous-sign momentum (fixed 0.3 size, no CVaR cap)"

    elif name == "agent_no_regime":
        env = factory_raw(
            assets=[ASSET], split="all", sequential=True,
            zero_regime=True, debate=_load_no_regime_debates(),
        )
        n = _restrict_to_window(env, ASSET, start, end)
        if n < 2:
            return {"status": "insufficient_data"}
        traj = _run_agents_rule(env)
        method = "LLM agents debated regime-blind; nearest cached debate (<=7d) drives direction/size"

    elif name == "agent_with_regime":
        env = factory_raw(
            assets=[ASSET], split="all", sequential=True,
            zero_regime=False, debate=True,
        )
        n = _restrict_to_window(env, ASSET, start, end)
        if n < 2:
            return {"status": "insufficient_data"}
        traj = _run_agents_rule(env)
        method = "LLM agents debated with regime context; nearest cached debate (<=7d) drives direction/size"

    elif name == "ppo_only":
        env = factory_raw(
            assets=[ASSET], split="all", sequential=True,
            zero_consensus=True, debate=False,
        )
        n = _restrict_to_window(env, ASSET, start, end)
        if n < 2:
            return {"status": "insufficient_data"}
        traj = run_walk_forward(model_ppo_only, env, deterministic=True)
        method = f"trained PPO ({PPO_ONLY_CHECKPOINT}), obs_mode=raw, consensus zeroed"

    elif name == "ppo_regime":
        env = factory_regime(
            assets=[ASSET], split="all", sequential=True,
            zero_regime=False, zero_consensus=True, debate=False,
        )
        n = _restrict_to_window(env, ASSET, start, end)
        if n < 2:
            return {"status": "insufficient_data"}
        traj = run_walk_forward(model_ppo_regime, env, deterministic=True)
        method = f"trained PPO ({PPO_REGIME_CHECKPOINT}), obs_mode=raw_regime, consensus zeroed"

    else:  # ppo_regime_agents
        env = factory_regime(
            assets=[ASSET], split="all", sequential=True,
            zero_regime=False, zero_consensus=False, debate=True,
        )
        n = _restrict_to_window(env, ASSET, start, end)
        if n < 2:
            return {"status": "insufficient_data"}
        traj = run_walk_forward(model_ppo_regime, env, deterministic=True)
        method = f"trained PPO ({PPO_REGIME_CHECKPOINT}), obs_mode=raw_regime, live agent consensus"

    metrics = _metrics_from_traj(traj, env)
    metrics["method"] = method
    return {"status": "ok", "metrics": metrics, "returns": traj["returns"], "n_steps": n}


def run_all(years: list[int]) -> dict[str, Any]:
    from run_rl import _load_for_eval

    logger.info(f"Loading ppo_only checkpoint: {PPO_ONLY_CHECKPOINT}")
    model_ppo_only, factory_raw = _load_for_eval(_build_ppo_only_config(), PPO_ONLY_CHECKPOINT)
    logger.info(f"Loading ppo_regime checkpoint: {PPO_REGIME_CHECKPOINT}")
    model_ppo_regime, factory_regime = _load_for_eval(_build_ppo_regime_config(), PPO_REGIME_CHECKPOINT)

    all_results: dict[str, dict[str, Any]] = {}

    for year in years:
        logger.info(f"=== Year {year} ({_split_label(year)}) ===")
        year_results: dict[str, Any] = {}
        for name in VARIANT_ORDER:
            cell = run_variant(name, year, factory_raw, factory_regime, model_ppo_only, model_ppo_regime)
            year_results[name] = cell
            if cell["status"] == "ok":
                m = cell["metrics"]
                logger.info(
                    f"  {name:<20} sharpe={m['sharpe']:+.3f}  return={m['total_return']:+.1%}  "
                    f"maxDD={m['max_drawdown']:.1%}  trades={m['n_meaningful_trades']}"
                )
            else:
                logger.info(f"  {name:<20} skipped ({cell['status']}: {cell.get('note', '')})")
        all_results[str(year)] = year_results

    # Bootstrap delta-Sharpe of every variant vs buy_hold, within each year, when both have
    # real returns (same block-bootstrap machinery the rest of this project uses).
    for year in years:
        yr = all_results[str(year)]
        bh_cell = yr.get("buy_hold", {})
        if bh_cell.get("status") != "ok":
            continue
        bh_returns = bh_cell["returns"]
        for name in VARIANT_ORDER:
            if name == "buy_hold":
                continue
            cell = yr[name]
            if cell.get("status") != "ok":
                continue
            delta, lo, hi = bootstrap_sharpe_ci(cell["returns"], bh_returns)
            cell["vs_buy_hold"] = {
                "delta_sharpe": delta, "ci_lower_95": lo, "ci_upper_95": hi,
                "significant": bool(lo > 0 or hi < 0),
            }

    out = {
        "asset": ASSET,
        "years": years,
        "checkpoints": {"ppo_only": PPO_ONLY_CHECKPOINT, "ppo_regime": PPO_REGIME_CHECKPOINT},
        "agent_debate_only_year": AGENT_DEBATE_YEAR,
        "split_labels": {str(y): _split_label(y) for y in years},
        "results": {
            year: {name: _strip_returns(cell) for name, cell in yr.items()}
            for year, yr in all_results.items()
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=float))
    logger.success(f"Results written to {RESULTS_PATH}")
    return out


def _strip_returns(cell: dict[str, Any]) -> dict[str, Any]:
    """Drop the raw returns array (used only for the in-process bootstrap) before saving."""
    return {k: v for k, v in cell.items() if k != "returns"}


def print_tables(out: dict[str, Any]) -> None:
    for year_str, yr in out["results"].items():
        year = int(year_str)
        table = Table(title=f"{year} ({_split_label(year)}) — {ASSET}")
        table.add_column("Strategy")
        table.add_column("Sharpe", justify="right")
        table.add_column("Total Return", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Calmar", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("ΔSharpe vs B&H", justify="right")
        table.add_column("Note")

        for name in VARIANT_ORDER:
            cell = yr[name]
            label = VARIANT_LABELS[name]
            if cell["status"] != "ok":
                table.add_row(label, "—", "—", "—", "—", "—", "—", cell.get("note", cell["status"]))
                continue
            m = cell["metrics"]
            vs = cell.get("vs_buy_hold")
            if vs is not None:
                sig = "*" if vs["significant"] else ""
                delta_str = f"{vs['delta_sharpe']:+.3f}{sig}"
            else:
                delta_str = "—"
            table.add_row(
                label,
                f"{m['sharpe']:+.3f}",
                f"{m['total_return']:+.1%}",
                f"{m['max_drawdown']:.1%}",
                f"{m['calmar']:+.2f}",
                str(m["n_meaningful_trades"]),
                delta_str,
                "",
            )
        _console.print(table)
        _console.print()

    _console.print("* = 95% block-bootstrap CI excludes 0 (statistically significant vs buy-and-hold)")
    _console.print(f"Agent/consensus variants only run for {out['agent_debate_only_year']} — see module docstring")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    args = parser.parse_args()

    out = run_all(sorted(args.years))
    print_tables(out)


if __name__ == "__main__":
    main()
