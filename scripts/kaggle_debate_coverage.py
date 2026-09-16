"""
Standalone, Kaggle-runnable adaptation of run_weekly_debates.py for dense debate
coverage generation (the LLM-debate-layer trading-performance audit's blocking
dependency — see docs/RESULTS_AND_METHODOLOGY.md Section 3/6).

Differences from run_weekly_debates.py (the canonical local script — keep both in
sync if the debate-selection logic changes):
  - No `from rl.env import VAL_START, VAL_END` — avoids pulling in gymnasium/
    stable-baselines3 as Kaggle deps for a debate-only run. VAL_START/VAL_END are
    hardcoded below instead, sourced from rl/env.py's current values (2024 walk-
    forward validation window) — verify these still match rl/env.py if the split
    ever moves again (see CLAUDE.md "Walk-forward split").
  - No MLflow — this is a throwaway Kaggle run; the only artifact that matters is
    debate_results.json, downloaded back and merged locally. Matches this
    project's existing Kaggle pattern (encoder retrains download only the .pt
    checkpoint, no local MLflow run for the Kaggle-side training).
  - Defaults tuned for dense coverage: --stride-hours 24 --all-regimes, not the
    weekly/dominant-regime defaults of the local script.

Expects to run from a working directory containing:
  agents/, utils/, data/processed/crypto5_multihorizon/*, data/validated/*, data/news_cache.json
(see kaggle_debate_coverage_instructions.md for the exact notebook cells).

Run: python kaggle_debate_coverage.py --stride-hours 24 --all-regimes --asset BTC_USDT
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

# `python scripts/kaggle_debate_coverage.py` sets sys.path[0] to this file's own
# directory (scripts/), not the repo root — so sibling packages agents/ and utils/
# are not importable without this. Must run before the agents/utils imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from agents.contrarian_agent import ContrarianAgent
from agents.fund_manager import FundManagerAgent
from agents.graph import DebateGraph, run_debate as _run_debate_raw
from agents.macro_agent import MacroAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from utils.freshness import CRYPTO5_DEBATE_ASSETS, CRYPTO5_PROCESSED, CRYPTO5_SEMANTICS_PATH, load_debate_data
from utils.logger import logger

# Sourced from rl/env.py as of 2026-09-01 (CLAUDE.md "Walk-forward split (strict, current)").
# Re-check against rl/env.py's VAL_START/VAL_END before trusting this on a future run.
VAL_START = pd.Timestamp("2024-01-01", tz="UTC")
VAL_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")

_DATA_DIR = CRYPTO5_PROCESSED
_DEBATE_ASSETS = CRYPTO5_DEBATE_ASSETS
_SEMANTICS_PATH = CRYPTO5_SEMANTICS_PATH
_OUTPUT_DIR = Path("data/debates")
_OUTPUT_PATH = _OUTPUT_DIR / "debate_results.json"

_MAX_MATCH_HOURS = 24.0
_N_DOMINANT_REGIMES = 2

run_debate = partial(_run_debate_raw, semantics_path=_SEMANTICS_PATH)


def _load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    return load_debate_data(processed_dir=_DATA_DIR, assets=_DEBATE_ASSETS)


def _select_timestamps(
    regime_stream: pd.DataFrame,
    target_asset: str,
    stride_hours: int,
    all_regimes: bool,
) -> list[tuple[pd.Timestamp, int]]:
    stream = regime_stream.copy()
    stream["timestamp"] = pd.to_datetime(stream["timestamp"], utc=True)

    asset_all = stream[
        (stream["asset"] == target_asset)
        & (stream["timestamp"] >= VAL_START)
        & (stream["timestamp"] <= VAL_END)
    ]

    if all_regimes:
        candidates = asset_all.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Val window {VAL_START.date()} → {VAL_END.date()}: sampling all regimes")
    else:
        dominant = set(
            asset_all["regime_label"].value_counts().head(_N_DOMINANT_REGIMES).index.astype(int)
        )
        logger.info(f"Val window {VAL_START.date()} → {VAL_END.date()}: dominant regimes {sorted(dominant)}")
        candidates = asset_all[asset_all["regime_label"].isin(dominant)].sort_values(
            "timestamp"
        ).reset_index(drop=True)

    anchors = pd.date_range(start=VAL_START, end=VAL_END, freq=f"{stride_hours}h", tz="UTC")

    picks: list[tuple[pd.Timestamp, int]] = []
    seen: set[pd.Timestamp] = set()
    for anchor in anchors:
        delta = (candidates["timestamp"] - anchor).abs()
        if delta.empty:
            continue
        idx = delta.idxmin()
        if delta.loc[idx] > pd.Timedelta(hours=_MAX_MATCH_HOURS):
            continue
        ts = candidates.loc[idx, "timestamp"]
        label = int(candidates.loc[idx, "regime_label"])
        if ts in seen:
            continue
        seen.add(ts)
        picks.append((ts, label))

    return picks


def _load_existing() -> tuple[list[dict], set[tuple[str, str]]]:
    if not _OUTPUT_PATH.exists():
        return [], set()
    with open(_OUTPUT_PATH) as f:
        records = json.load(f)
    # Keyed on (timestamp, asset), not timestamp alone — every asset here uses the same
    # daily-stride picks over the same 2024-01-01..2024-12-31 window, so timestamps
    # collide across assets by construction. A timestamp-only key silently treated
    # e.g. ETH's 2024-01-01 debate as "already covered" by BTC's, dropping it. Found
    # 2026-09-15 after it had already destroyed 363/366 of a completed ETH run and
    # 365/366 of a completed XRP run in data/debates/debate_results.json.
    existing_keys = {
        (_normalize_ts(r["timestamp"]), r.get("asset"))
        for r in records if r.get("timestamp") is not None
    }
    return records, existing_keys


def _normalize_ts(ts: str | pd.Timestamp) -> str:
    return str(pd.Timestamp(ts, tz="UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts))


def _save(records: list[dict]) -> None:
    """Atomic write after every debate — a Kaggle session can be interrupted
    (idle timeout, quota, manual stop) with no warning; this makes sure whatever
    ran before that point is never lost."""
    tmp_path = _OUTPUT_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    tmp_path.replace(_OUTPUT_PATH)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle dense debate coverage runner")
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--all-regimes", action="store_true", default=True)
    parser.add_argument("--asset", type=str, default="BTC_USDT", choices=_DEBATE_ASSETS)
    args = parser.parse_args()

    console = Console()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console.rule(
        f"[bold magenta]MARTA Kaggle Debate Coverage — {args.asset} "
        f"{VAL_START.date()} → {VAL_END.date()} (stride={args.stride_hours}h)[/bold magenta]"
    )

    logger.info("Loading data...")
    regime_stream, aligned_dfs = _load_data()
    logger.info(f"Loaded regime stream: {len(regime_stream)} rows | {len(aligned_dfs)} aligned assets")

    logger.info("Initializing agents (crypto-only panel)...")
    macro_agent = MacroAgent(assets=_DEBATE_ASSETS, data_dir=_DATA_DIR)
    tech_agent = TechnicalAgent()
    risk_agent = RiskAgent(data_dir=_DATA_DIR)
    fm_agent = FundManagerAgent(semantics_path=_SEMANTICS_PATH)
    sentiment_agent = SentimentAgent(macro_assets={"macro"}, fear_greed_assets=set(_DEBATE_ASSETS))
    contrarian_agent = ContrarianAgent(regime_stream_df=regime_stream)
    debate_graph = DebateGraph(
        macro_agent, tech_agent, risk_agent, fm_agent, sentiment_agent, contrarian_agent
    )
    logger.info("All 5 agents + RiskAgent initialized")

    targets = _select_timestamps(regime_stream, args.asset, args.stride_hours, args.all_regimes)
    logger.info(f"Selected {len(targets)} timestamps in the validation window")

    existing_records, existing_keys = _load_existing()
    logger.info(f"Existing debate cache: {len(existing_records)} debates")

    new_count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running debates", total=len(targets))

        for i, (ts, label) in enumerate(targets):
            progress.update(task, description=f"[cyan]{ts.date()} R{label}[/cyan]")

            if (_normalize_ts(ts), args.asset) in existing_keys:
                progress.advance(task)
                continue

            try:
                result = run_debate(ts, args.asset, regime_stream, aligned_dfs, debate_graph)
            except Exception as e:
                logger.error(f"Debate failed for {ts}: {e}")
                progress.advance(task)
                continue

            existing_keys.add((_normalize_ts(ts), args.asset))
            d = result.model_dump()
            d["scenario_description"] = f"kaggle_{ts.year} {ts.date()} R{result.regime_label}"
            existing_records.append(d)
            _save(existing_records)
            new_count += 1

            fm = result.fund_manager_decision
            logger.info(
                f"{i+1}/{len(targets)} @ {ts}: R{result.regime_label} → "
                f"{fm.final_action} (conviction={fm.conviction:.3f})"
            )
            progress.advance(task)

    console.print(
        f"\n[bold green]Done:[/bold green] {new_count} new debates "
        f"({len(targets) - new_count} skipped: duplicate or failed)"
    )
    console.print(f"[bold]{_OUTPUT_PATH} now contains {len(existing_records)} total debates.[/bold]")


if __name__ == "__main__":
    main()
