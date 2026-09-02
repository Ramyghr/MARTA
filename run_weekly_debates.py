"""
Generate one debate per anchor date across the RL validation window (rl.env.VAL_START →
VAL_END) for BTC_USDT and append to data/debates/debate_results.json.

Default (no flags): one debate per week, candidates restricted to the window's two
most frequent ("dominant") BTC regimes — regime label semantics change whenever the
HMM is refit, so hard-coding labels would silently skip most weeks. Anchors with no
matching timestamp within 24h are skipped. Results are deduplicated by timestamp so
the script is idempotent, and the debate cache is written to disk after every single
debate (not just at the end), so a crash/timeout partway through a long run loses at
most the in-flight debate.

--stride-hours N   sample every N hours instead of weekly (e.g. 24 = daily)
--all-regimes      sample every regime, not just the top-2 dominant ones — needed
                    for dense per-decision coverage (e.g. the LLM debate-layer
                    trading-performance audit), where restricting to dominant
                    regimes would leave the same coverage gaps this sweep exists
                    to close.

Run: python run_weekly_debates.py
     python run_weekly_debates.py --stride-hours 24 --all-regimes
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import mlflow
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
from agents.schemas import DebateResult
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from utils.freshness import CRYPTO5_DEBATE_ASSETS, CRYPTO5_PROCESSED, CRYPTO5_SEMANTICS_PATH, load_debate_data
from utils.logger import logger

_DATA_DIR = CRYPTO5_PROCESSED
_DEBATE_ASSETS = CRYPTO5_DEBATE_ASSETS
_SEMANTICS_PATH = CRYPTO5_SEMANTICS_PATH
_OUTPUT_DIR = Path("data/debates")
_OUTPUT_PATH = _OUTPUT_DIR / "debate_results.json"
_MLFLOW_EXPERIMENT = "marta_agent_deliberation"

_TARGET_ASSET = "BTC_USDT"
_MAX_MATCH_HOURS = 24.0
_N_DOMINANT_REGIMES = 2  # debate in the window's two most frequent BTC regimes

# Every run_debate(...) call must pass the same semantics_path used to build
# FundManagerAgent below, or the displayed regime description and the
# conviction/override logic will disagree (agents/graph.py's run_debate docstring).
run_debate = partial(_run_debate_raw, semantics_path=_SEMANTICS_PATH)


def _load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Regime stream + per-asset aligned parquets, via the shared loader (P0.4),
    explicitly requesting the crypto-only 5-asset panel (data/processed/
    crypto5_multihorizon) — load_debate_data()'s own default is still the old 4-asset
    production panel (see run_debate.py for why)."""
    return load_debate_data(processed_dir=_DATA_DIR, assets=_DEBATE_ASSETS)


def _select_weekly_timestamps(
    regime_stream: pd.DataFrame,
    stride_hours: int = 168,
    all_regimes: bool = False,
) -> list[tuple[pd.Timestamp, int]]:
    """
    For each anchor date (every `stride_hours`, default 168 = weekly), return
    (timestamp, regime_label) of the closest BTC_USDT row within 24h.

    Default behaviour (stride_hours=168, all_regimes=False) is unchanged from the
    original weekly-only design: anchors land on Mondays and candidates are
    restricted to the window's two most frequent ("dominant") BTC regimes.

    `all_regimes=True` drops the dominant-regime restriction — needed for a dense,
    per-decision coverage sweep (e.g. the LLM-debate-layer trading-performance
    audit), where restricting to only the top-2 regimes would leave the same kind
    of coverage gaps on minority-regime periods that this whole sweep exists to
    close.
    """
    from rl.env import VAL_END, VAL_START

    stream = regime_stream.copy()
    stream["timestamp"] = pd.to_datetime(stream["timestamp"], utc=True)

    btc_all = stream[
        (stream["asset"] == _TARGET_ASSET)
        & (stream["timestamp"] >= VAL_START)
        & (stream["timestamp"] <= VAL_END)
    ]

    if all_regimes:
        btc = btc_all.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Val window {VAL_START.date()} → {VAL_END.date()}: sampling all regimes")
    else:
        dominant = set(
            btc_all["regime_label"].value_counts().head(_N_DOMINANT_REGIMES).index.astype(int)
        )
        logger.info(
            f"Val window {VAL_START.date()} → {VAL_END.date()}: dominant BTC regimes {sorted(dominant)}"
        )
        btc = btc_all[btc_all["regime_label"].isin(dominant)].sort_values(
            "timestamp"
        ).reset_index(drop=True)

    freq = "W-MON" if stride_hours == 168 else f"{stride_hours}h"
    anchors = pd.date_range(start=VAL_START, end=VAL_END, freq=freq, tz="UTC")

    picks: list[tuple[pd.Timestamp, int]] = []
    seen: set[pd.Timestamp] = set()

    for anchor in anchors:
        delta = (btc["timestamp"] - anchor).abs()
        if delta.empty:
            continue
        idx = delta.idxmin()
        if delta.loc[idx] > pd.Timedelta(hours=_MAX_MATCH_HOURS):
            logger.warning(f"No BTC timestamp within 24h of {anchor} — skipping")
            continue
        ts = btc.loc[idx, "timestamp"]
        label = int(btc.loc[idx, "regime_label"])
        if ts in seen:
            logger.info(f"Anchor {anchor} maps to already-selected {ts} — skipping duplicate")
            continue
        seen.add(ts)
        picks.append((ts, label))

    return picks


def _load_existing() -> tuple[list[dict], set[str]]:
    """Return (existing_records, set of normalized timestamp strings already present)."""
    if not _OUTPUT_PATH.exists():
        return [], set()
    with open(_OUTPUT_PATH) as f:
        records = json.load(f)
    existing_ts: set[str] = set()
    for rec in records:
        ts = rec.get("timestamp")
        if ts is not None:
            existing_ts.add(_normalize_ts(ts))
    return records, existing_ts


def _normalize_ts(ts: str | pd.Timestamp) -> str:
    """Normalize a timestamp to a canonical UTC string for dedup comparison."""
    return str(pd.Timestamp(ts, tz="UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts))


def _save(records: list[dict]) -> None:
    """Write the full debate cache to disk after every debate, not just at the end
    of the run. A 38+ week real-LLM sweep can take hours; without this, a crash,
    timeout, or kill partway through loses 100% of the completed work instead of
    just the in-flight debate. Temp file + atomic rename so a kill mid-write can't
    leave debate_results.json truncated."""
    tmp_path = _OUTPUT_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    tmp_path.replace(_OUTPUT_PATH)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MARTA weekly/dense debate coverage runner")
    parser.add_argument(
        "--stride-hours", type=int, default=168,
        help="hours between sampled anchors (default: 168 = weekly, matches original behaviour)",
    )
    parser.add_argument(
        "--all-regimes", action="store_true",
        help="sample all regimes, not just the window's top-2 dominant ones "
             "(needed for dense per-decision coverage, e.g. the debate-layer trading audit)",
    )
    args = parser.parse_args()

    console = Console()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from rl.env import VAL_END, VAL_START

    console.rule(
        f"[bold magenta]MARTA Debate Coverage — Validation Window "
        f"{VAL_START.date()} → {VAL_END.date()} (stride={args.stride_hours}h, "
        f"all_regimes={args.all_regimes})[/bold magenta]"
    )

    logger.info("Loading data...")
    regime_stream, aligned_dfs = _load_data()
    logger.info(f"Loaded regime stream: {len(regime_stream)} rows | {len(aligned_dfs)} aligned assets")

    logger.info("Initializing agents (crypto-only panel)...")
    macro_agent = MacroAgent(assets=_DEBATE_ASSETS, data_dir=_DATA_DIR)
    tech_agent = TechnicalAgent()
    risk_agent = RiskAgent(data_dir=_DATA_DIR)
    fm_agent = FundManagerAgent(semantics_path=_SEMANTICS_PATH)
    sentiment_agent = SentimentAgent(
        macro_assets={"macro"}, fear_greed_assets=set(_DEBATE_ASSETS)
    )
    contrarian_agent = ContrarianAgent(regime_stream_df=regime_stream)
    debate_graph = DebateGraph(
        macro_agent, tech_agent, risk_agent, fm_agent, sentiment_agent, contrarian_agent
    )
    logger.info("All 5 agents + RiskAgent initialized")

    weekly = _select_weekly_timestamps(regime_stream, args.stride_hours, args.all_regimes)
    logger.info(f"Selected {len(weekly)} weekly timestamps in the validation window")

    existing_records, existing_ts = _load_existing()
    logger.info(f"Existing debate cache: {len(existing_records)} debates")

    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    new_results: list[tuple[pd.Timestamp, DebateResult]] = []

    with mlflow.start_run(run_name=f"weekly_debates_{VAL_START.year}") as run:
        mlflow.log_param("n_weeks", len(weekly))
        mlflow.log_param("model", "mistral:7b-instruct")
        mlflow.log_param("asset", _TARGET_ASSET)
        mlflow.log_param("window", f"{VAL_START.date()}..{VAL_END.date()}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("week {task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running weekly debates", total=len(weekly))

            for i, (ts, label) in enumerate(weekly):
                progress.update(
                    task,
                    description=f"[cyan]{ts.date()} R{label}[/cyan]",
                )

                if _normalize_ts(ts) in existing_ts:
                    logger.info(f"Week {i+1}: {ts} already in cache — skipping")
                    progress.advance(task)
                    continue

                try:
                    result = run_debate(ts, _TARGET_ASSET, regime_stream, aligned_dfs, debate_graph)
                except Exception as e:
                    logger.error(f"Debate failed for {ts}: {e}")
                    progress.advance(task)
                    continue

                new_results.append((ts, result))
                existing_ts.add(_normalize_ts(ts))

                d = result.model_dump()
                d["scenario_description"] = f"weekly_{ts.year} {ts.date()} R{result.regime_label}"
                existing_records.append(d)
                _save(existing_records)

                fm = result.fund_manager_decision
                prefix = f"week_{i+1}"
                contrarian_theses = [t for t in result.round1_theses if t.agent_name == "ContrarianAgent"]
                contrarian_active = bool(
                    contrarian_theses and contrarian_theses[0].directional_bias != "neutral"
                )
                try:
                    mlflow.log_metrics({
                        f"{prefix}_regime_label": result.regime_label,
                        f"{prefix}_conviction": fm.conviction,
                        f"{prefix}_position_size": fm.position_size,
                        f"{prefix}_regime_override": int(fm.regime_override),
                        f"{prefix}_tokens": result.total_tokens_used,
                        f"{prefix}_duration_s": result.debate_duration_seconds,
                        f"{prefix}_vol_forecast": result.risk_limits.volatility_forecast,
                        f"{prefix}_n_rebuttals": len(result.round2_rebuttals),
                        f"{prefix}_contrarian_active": int(contrarian_active),
                    })
                    mlflow.log_param(f"{prefix}_action", fm.final_action)
                except Exception as e:
                    logger.error(f"MLflow logging failed for {ts}: {e}")

                logger.info(
                    f"Week {i+1}/{len(weekly)} @ {ts}: R{result.regime_label} → "
                    f"{fm.final_action} (conviction={fm.conviction:.3f})"
                )
                progress.advance(task)

        # existing_records was written to disk after every debate inside the loop
        # above (_save) — it's already the complete, current cache here.
        combined = existing_records

        mlflow.log_metric("new_debates_added", len(new_results))
        mlflow.log_metric("total_debates", len(combined))
        if new_results:
            mlflow.log_metric(
                "avg_conviction",
                sum(r.fund_manager_decision.conviction for _, r in new_results) / len(new_results),
            )
        mlflow.log_artifact(str(_OUTPUT_PATH))
        logger.success(f"Saved {len(combined)} total debates to {_OUTPUT_PATH}")

        # Summary table
        table = Table(title="MARTA Weekly Debates — Summary", show_lines=False)
        table.add_column("Week", style="dim")
        table.add_column("Regime", style="magenta")
        table.add_column("Action", style="bold")
        table.add_column("Conviction", justify="right")
        for ts, r in new_results:
            fm = r.fund_manager_decision
            action_color = {"long": "green", "short": "red", "hold": "yellow"}.get(
                fm.final_action, "white"
            )
            table.add_row(
                str(ts.date()),
                f"R{r.regime_label}",
                f"[{action_color}]{fm.final_action}[/{action_color}]",
                f"{fm.conviction:.3f}",
            )
        console.print()
        console.print(table)

        mlflow_url = f"http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        console.print(f"\n[bold green]MLflow run:[/bold green] {mlflow_url}")
        console.print(
            f"[bold green]New debates added:[/bold green] {len(new_results)} "
            f"(skipped {len(weekly) - len(new_results)} weeks: duplicate or failed)"
        )
        console.print(
            f"\n[bold]data/debates/debate_results.json now contains {len(combined)} total debates.[/bold]"
        )


if __name__ == "__main__":
    main()
