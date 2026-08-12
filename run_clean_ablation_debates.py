"""Clean no-regime-context ablation debates.

Re-runs the exact (timestamp, asset) pairs already present in
``data/debates/debate_results.json`` with ``serialize_context_for_prompt`` forced to
``include_regime=False`` for every agent — so the LLM agents deliberate without ever
seeing the regime label, regime description, or probability vector. Used as a clean
no-regime baseline for the ``agents_no_regime`` ablation variant (rl/ablation.py),
distinct from the regime-informed debate cache used elsewhere.

Does NOT modify data/debates/debate_results.json. Output: data/debates/debate_results_no_regime.json.

Usage:
    python run_clean_ablation_debates.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from rich.console import Console

import agents.base as base_module
import agents.context as context_module
import agents.contrarian_agent as contrarian_module
import agents.fund_manager as fm_module
import agents.macro_agent as macro_module
import agents.risk_agent as risk_module
import agents.sentiment_agent as sentiment_module
import agents.technical_agent as technical_module
from agents.contrarian_agent import ContrarianAgent
from agents.fund_manager import FundManagerAgent
from agents.graph import DebateGraph, run_debate
from agents.macro_agent import MacroAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from utils.freshness import load_debate_data
from utils.logger import logger

_DATA_DIR = Path("data/processed")
_SOURCE_PATH = Path("data/debates/debate_results.json")
_OUTPUT_PATH = Path("data/debates/debate_results_no_regime.json")
_MLFLOW_EXPERIMENT = "marta_agent_deliberation"

# Every module that did `from agents.context import serialize_context_for_prompt` —
# each holds its own module-level binding, so all of them need patching.
_PATCH_TARGETS = [
    context_module,
    base_module,
    macro_module,
    technical_module,
    sentiment_module,
    contrarian_module,
    risk_module,
    fm_module,
]


def _patch_no_regime() -> None:
    original = context_module.serialize_context_for_prompt

    def _forced_no_regime(ctx, *args, **kwargs):
        return original(ctx, include_regime=False)

    for mod in _PATCH_TARGETS:
        mod.serialize_context_for_prompt = _forced_no_regime
    logger.info(f"Patched serialize_context_for_prompt(include_regime=False) in {len(_PATCH_TARGETS)} modules")


def _load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Regime stream + per-asset aligned parquets, via the shared loader (P0.4) —
    raises StaleRegimeStreamError if regime_stream.parquet predates the deployed
    encoder checkpoint."""
    return load_debate_data()


def _load_source_timestamps() -> list[tuple[pd.Timestamp, str]]:
    """(timestamp, asset) pairs from the existing debate cache — reused as-is."""
    if not _SOURCE_PATH.exists():
        raise FileNotFoundError(f"{_SOURCE_PATH} not found — cannot mirror its timestamps")
    entries = json.loads(_SOURCE_PATH.read_text())
    return [(pd.Timestamp(e["timestamp"]), e["asset"]) for e in entries]


def main() -> None:
    console = Console()
    console.rule("[bold magenta]MARTA Clean No-Regime Ablation Debates[/bold magenta]")

    _patch_no_regime()

    logger.info("Loading data...")
    regime_stream, aligned_dfs = _load_data()
    pairs = _load_source_timestamps()
    logger.info(f"Mirroring {len(pairs)} (timestamp, asset) pairs from {_SOURCE_PATH}")

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
    logger.info("All 5 agents + RiskAgent initialized (no-regime context)")

    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    serialized: list[dict] = []

    with mlflow.start_run(run_name="clean_ablation_no_regime_debates") as run:
        mlflow.log_param("n_timestamps", len(pairs))
        mlflow.log_param("include_regime", False)
        mlflow.log_param("source", str(_SOURCE_PATH))

        for i, (ts, asset) in enumerate(pairs):
            console.print(f"[bold]Debate {i + 1}/{len(pairs)}:[/bold] {asset} @ {ts}")
            try:
                result = run_debate(ts, asset, regime_stream, aligned_dfs, debate_graph)
                d = result.model_dump()
                d["scenario_description"] = f"no-regime clean baseline {asset} @ {ts}"
                serialized.append(d)
                fm = result.fund_manager_decision
                mlflow.log_metrics({
                    f"debate_{i + 1}_conviction": fm.conviction,
                }, step=i)
            except Exception as exc:
                logger.warning(f"Debate {i + 1} failed ({exc}) — skipping")
                continue

        _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _OUTPUT_PATH.write_text(json.dumps(serialized, indent=2, default=str))
        mlflow.log_artifact(str(_OUTPUT_PATH))

    console.print(
        f"[bold green]Done:[/bold green] {len(serialized)}/{len(pairs)} debates saved → {_OUTPUT_PATH}"
    )
    logger.success(f"Clean no-regime debate results saved to {_OUTPUT_PATH} ({_SOURCE_PATH} untouched)")


if __name__ == "__main__":
    main()
