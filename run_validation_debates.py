"""
Run 6 additional BTC debates on specific 2021 validation-period timestamps and
append the results to ``data/debates/debate_results.json``.

These debates target the walk-forward validation window (2021-01-01 → 2021-10-12)
so the cached debate consensus used by the RL ablation has coverage inside the
validation split. Existing demo debates are preserved — the 6 new results are
appended, not overwritten.

Run: python run_validation_debates.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from rich.console import Console
from rich.table import Table

from agents.contrarian_agent import ContrarianAgent
from agents.fund_manager import FundManagerAgent
from agents.graph import DebateGraph, run_debate
from agents.macro_agent import MacroAgent
from agents.risk_agent import RiskAgent
from agents.schemas import DebateResult
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from utils.logger import logger

_DATA_DIR = Path("data/processed")
_OUTPUT_DIR = Path("data/debates")
_OUTPUT_PATH = _OUTPUT_DIR / "debate_results.json"
_ASSETS = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]
_MLFLOW_EXPERIMENT = "marta_agent_deliberation"

VALIDATION_DEBATES: list[tuple[str, str, str]] = [
    ("BTC_USDT", "2021-01-25 06:00:00+00:00", "BTC R1 Jan2021"),
    ("BTC_USDT", "2021-03-25 04:00:00+00:00", "BTC R1 Mar2021"),
    ("BTC_USDT", "2021-06-24 15:00:00+00:00", "BTC R1 Jun2021"),
    ("BTC_USDT", "2021-01-25 07:00:00+00:00", "BTC R4 Jan2021"),
    ("BTC_USDT", "2021-06-24 18:00:00+00:00", "BTC R4 Jun2021"),
    ("BTC_USDT", "2021-10-08 15:00:00+00:00", "BTC R4 Oct2021"),
]


def _load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    regime_stream = pd.read_parquet(_DATA_DIR / "regime_stream.parquet", engine="pyarrow")
    aligned_dfs: dict[str, pd.DataFrame] = {}
    for asset in _ASSETS:
        path = _DATA_DIR / f"{asset}_aligned.parquet"
        if path.exists():
            aligned_dfs[asset] = pd.read_parquet(path, engine="pyarrow")
    return regime_stream, aligned_dfs


def _load_existing_results() -> list[dict]:
    if _OUTPUT_PATH.exists():
        with open(_OUTPUT_PATH) as f:
            existing = json.load(f)
        if isinstance(existing, list):
            return existing
        logger.warning(f"{_OUTPUT_PATH} did not contain a list — starting from empty.")
    return []


def _print_summary(results: list[tuple[str, DebateResult]], console: Console) -> None:
    table = Table(title="MARTA Validation Debates — 2021 Walk-Forward Window", show_lines=True)
    table.add_column("Scenario", style="cyan", min_width=16)
    table.add_column("Timestamp", style="dim")
    table.add_column("Regime", style="magenta")
    table.add_column("Action", style="bold")
    table.add_column("Conviction", justify="right")
    table.add_column("Dur(s)", justify="right")

    for desc, r in results:
        fm = r.fund_manager_decision
        action_color = {"long": "green", "short": "red", "hold": "yellow"}.get(fm.final_action, "white")
        table.add_row(
            desc,
            r.timestamp[:19],
            f"R{r.regime_label}",
            f"[{action_color}]{fm.final_action}[/{action_color}]",
            f"{fm.conviction:.4f}",
            str(r.debate_duration_seconds),
        )

    console.print(table)


def main() -> None:
    console = Console()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console.rule("[bold magenta]MARTA Validation Debates — 2021 Walk-Forward Window[/bold magenta]")

    logger.info("Loading data...")
    regime_stream, aligned_dfs = _load_data()
    logger.info(f"Loaded regime stream: {len(regime_stream)} rows | {len(aligned_dfs)} aligned assets")

    logger.info("Initializing agents...")
    macro_agent = MacroAgent()
    tech_agent = TechnicalAgent()
    risk_agent = RiskAgent()
    fm_agent = FundManagerAgent()
    sentiment_agent = SentimentAgent()
    contrarian_agent = ContrarianAgent()
    debate_graph = DebateGraph(
        macro_agent, tech_agent, risk_agent, fm_agent, sentiment_agent, contrarian_agent
    )
    logger.info("All 5 agents + RiskAgent initialized")

    existing_results = _load_existing_results()
    logger.info(f"Loaded {len(existing_results)} existing debate(s) from {_OUTPUT_PATH}")

    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    new_results: list[tuple[str, DebateResult]] = []

    with mlflow.start_run(run_name="validation_debates_2021") as run:
        mlflow.log_param("n_debates", len(VALIDATION_DEBATES))
        mlflow.log_param("model", "mistral:7b-instruct")
        mlflow.log_param("agents", "MacroAgent,TechnicalAgent,SentimentAgent,ContrarianAgent,RiskAgent,FundManager")
        mlflow.log_param("window", "2021_validation")

        for i, (asset, ts_str, desc) in enumerate(VALIDATION_DEBATES):
            console.print(f"\n[bold]Running debate {i+1}/{len(VALIDATION_DEBATES)}: {desc}[/bold]")
            logger.info(f"Debate {i+1}: {desc} @ {ts_str} ({asset})")

            try:
                ts = pd.Timestamp(ts_str, tz="UTC")
                result = run_debate(ts, asset, regime_stream, aligned_dfs, debate_graph)
            except Exception as e:
                logger.error(f"Debate {i+1} ({desc}) failed: {e}")
                continue

            new_results.append((desc, result))

            prefix = f"debate_{i+1}"
            fm = result.fund_manager_decision
            contrarian_theses = [t for t in result.round1_theses if t.agent_name == "ContrarianAgent"]
            contrarian_active = bool(contrarian_theses and contrarian_theses[0].directional_bias != "neutral")

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
            mlflow.log_param(f"{prefix}_asset", result.asset)
            mlflow.log_param(f"{prefix}_action", fm.final_action)
            mlflow.log_param(f"{prefix}_consensus", fm.consensus_direction)
            mlflow.log_param(f"{prefix}_scenario", desc)

        if new_results:
            total_tokens = sum(r.total_tokens_used for _, r in new_results)
            total_duration = sum(r.debate_duration_seconds for _, r in new_results)
            mlflow.log_metrics({
                "total_tokens": total_tokens,
                "total_duration_s": total_duration,
                "avg_conviction": sum(r.fund_manager_decision.conviction for _, r in new_results) / len(new_results),
                "n_debates_succeeded": len(new_results),
            })

        # Append new results to existing — preserve prior debates.
        serialized_new: list[dict] = []
        for desc, r in new_results:
            d = r.model_dump()
            d["scenario_description"] = desc
            serialized_new.append(d)

        all_serialized = existing_results + serialized_new
        with open(_OUTPUT_PATH, "w") as f:
            json.dump(all_serialized, f, indent=2, default=str)

        mlflow.log_artifact(str(_OUTPUT_PATH))
        logger.success(f"Appended {len(serialized_new)} debate(s); saved to {_OUTPUT_PATH}")

        console.print()
        _print_summary(new_results, console)

        mlflow_url = f"http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        console.print(f"\n[bold green]MLflow run:[/bold green] {mlflow_url}")

    total_count = len(all_serialized)
    console.print(
        f"\n[bold green]✓ data/debates/debate_results.json now contains "
        f"{total_count} total debates[/bold green] "
        f"({len(existing_results)} existing + {len(serialized_new)} new)"
    )
    logger.success(f"debate_results.json total debate count: {total_count}")


if __name__ == "__main__":
    main()
