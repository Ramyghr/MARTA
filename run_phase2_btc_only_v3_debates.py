"""Phase 2 of the downstream pipeline validation (2026-08-14): HMM -> Agents.

Runs the 5-agent debate system on the BTC-only-v3 regime stream at two representative
timestamps (one high-confidence sample of each discovered regime: R0=calm/range-bound,
R1=volatile/trending), verifies the debate outputs are structurally sane, and compares
each timestamp WITH the new regime info vs WITHOUT (regime info stripped from every
agent's prompt, same monkey-patch mechanism as run_clean_ablation_debates.py).

Does NOT touch data/debates/debate_results.json or any production artifact. Output:
data/debates/phase2_btc_only_v3_debates.json.

Run: conda run -n marta python run_phase2_btc_only_v3_debates.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from rich.console import Console
from rich.table import Table

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
from utils.assets import aligned_path
from utils.logger import logger

console = Console()

_REGIME_STREAM_PATH = Path("data/processed/encoder_experiments/e1_predictive_btc_only_v3/regime_stream.parquet")
_SEMANTICS_PATH = Path("data/validated/btc_only_v3_regime_semantics.json")
_OUTPUT_PATH = Path("data/debates/phase2_btc_only_v3_debates.json")
_PROCESSED = Path("data/processed")

# Picked from the BTC-only-v3 regime stream: high-confidence (>0.95 posterior) samples
# of each discovered state, well within history (>60 bars of lookback available).
_SCENARIOS = [
    ("2021-10-12 15:00:00", "R0_calm"),
    ("2021-11-05 03:00:00", "R1_volatile"),
]

_PATCH_TARGETS = [
    context_module, base_module, macro_module, technical_module,
    sentiment_module, contrarian_module, risk_module, fm_module,
]


def _patch_no_regime() -> None:
    original = context_module.serialize_context_for_prompt

    def _forced_no_regime(ctx, *args, **kwargs):
        return original(ctx, include_regime=False)

    for mod in _PATCH_TARGETS:
        mod.serialize_context_for_prompt = _forced_no_regime


def _unpatch() -> None:
    """Restore the original (include_regime=True-capable) serialize_context_for_prompt
    in every module patched by _patch_no_regime()."""
    import importlib
    importlib.reload(context_module)
    for mod in _PATCH_TARGETS[1:]:
        mod.serialize_context_for_prompt = context_module.serialize_context_for_prompt


def _summarize(result) -> dict:
    fm = result.fund_manager_decision
    theses = {t.agent_name: {"bias": t.directional_bias, "conf": round(t.confidence, 3)}
              for t in result.round1_theses}
    return {
        "round1_theses": theses,
        "n_rebuttals": len(result.round2_rebuttals),
        "fund_manager": {
            "final_action": fm.final_action,
            "conviction": round(fm.conviction, 3),
            "position_size": round(fm.position_size, 4),
            "regime_override": fm.regime_override,
            "dissenting_agents": fm.dissenting_agents,
        },
    }


def main() -> None:
    console.rule("[bold magenta]Phase 2: HMM -> Agents (BTC-only-v3 regime stream)[/bold magenta]")

    regime_stream = pd.read_parquet(_REGIME_STREAM_PATH, engine="pyarrow")
    aligned_dfs = {"BTC_USDT": pd.read_parquet(aligned_path("BTC_USDT", _PROCESSED), engine="pyarrow")}
    logger.info(f"Loaded BTC-only-v3 regime stream: {len(regime_stream):,} rows, "
                f"n_states={regime_stream['regime_label'].nunique()}")

    def build_agents():
        return DebateGraph(
            MacroAgent(), TechnicalAgent(), RiskAgent(),
            FundManagerAgent(semantics_path=_SEMANTICS_PATH),
            SentimentAgent(), ContrarianAgent(),
        )

    mlflow.set_experiment("marta_agent_deliberation")
    results: dict[str, dict] = {}

    with mlflow.start_run(run_name="phase2_btc_only_v3_debates"):
        mlflow.log_param("n_scenarios", len(_SCENARIOS))
        mlflow.log_param("regime_stream", str(_REGIME_STREAM_PATH))

        for ts_str, label in _SCENARIOS:
            ts = pd.Timestamp(ts_str, tz="UTC")

            console.print(f"\n[bold]{label} @ {ts} — WITH regime[/bold]")
            debate_graph = build_agents()
            result_with = run_debate(ts, "BTC_USDT", regime_stream, aligned_dfs, debate_graph, semantics_path=_SEMANTICS_PATH)
            results[f"{label}_with_regime"] = _summarize(result_with)
            results[f"{label}_with_regime"]["regime_label"] = int(
                regime_stream[(regime_stream["timestamp"] == ts)]["regime_label"].iloc[0]
            ) if (regime_stream["timestamp"] == ts).any() else None

            console.print(f"[bold]{label} @ {ts} — WITHOUT regime (ablation)[/bold]")
            _patch_no_regime()
            debate_graph2 = build_agents()
            result_without = run_debate(ts, "BTC_USDT", regime_stream, aligned_dfs, debate_graph2, semantics_path=_SEMANTICS_PATH)
            results[f"{label}_without_regime"] = _summarize(result_without)
            _unpatch()

        _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _OUTPUT_PATH.write_text(json.dumps(results, indent=2, default=str))
        mlflow.log_artifact(str(_OUTPUT_PATH))

    table = Table(title="Phase 2 — With vs Without Regime Comparison")
    table.add_column("Scenario")
    table.add_column("Regime")
    table.add_column("Action")
    table.add_column("Conviction", justify="right")
    table.add_column("Pos. Size", justify="right")
    table.add_column("Override")
    table.add_column("Dissent")
    for key, r in results.items():
        fm = r["fund_manager"]
        table.add_row(
            key, str(r.get("regime_label", "-")), fm["final_action"], f"{fm['conviction']:.3f}",
            f"{fm['position_size']:.4f}", str(fm["regime_override"]), str(len(fm["dissenting_agents"])),
        )
    console.print(table)
    logger.success(f"Saved -> {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
