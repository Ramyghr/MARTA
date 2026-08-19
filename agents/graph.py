"""LangGraph debate state machine — 5-agent system with Contrarian and Round 2 rebuttals."""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.context import RegimeContext
from agents.contrarian_agent import ContrarianAgent
from agents.fund_manager import FundManagerAgent
from agents.macro_agent import MacroAgent
from agents.risk_agent import RiskAgent
from agents.schemas import (
    AgentRebuttal,
    AgentThesis,
    DebateResult,
    FundManagerDecision,
    RiskLimits,
)
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from utils.logger import logger

import pandas as pd


class DebateState(TypedDict):
    ctx: RegimeContext
    aligned_dfs: dict[str, pd.DataFrame]
    round1_theses: list[AgentThesis]
    contrarian_thesis: AgentThesis | None
    consensus_confidence: float
    rebuttal_triggered: bool
    round2_rebuttals: list[AgentRebuttal]
    risk_limits: RiskLimits | None
    fund_manager_decision: FundManagerDecision | None
    error: str | None
    start_time: float
    total_tokens: int


class DebateGraph:
    def __init__(
        self,
        macro_agent: MacroAgent,
        technical_agent: TechnicalAgent,
        risk_agent: RiskAgent,
        fund_manager: FundManagerAgent,
        sentiment_agent: SentimentAgent,
        contrarian_agent: ContrarianAgent,
    ) -> None:
        self._macro = macro_agent
        self._tech = technical_agent
        self._risk = risk_agent
        self._fm = fund_manager
        self._sentiment = sentiment_agent
        self._contrarian = contrarian_agent
        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(DebateState)
        g.add_node("round1", self._node_round1)
        g.add_node("contrarian", self._node_contrarian)
        g.add_node("round2", self._node_round2)
        g.add_node("fund_manager", self._node_fund_manager)
        g.add_node("ppo_stub", self._node_ppo_stub)
        g.add_node("error_handler", self._node_error_handler)

        g.add_edge(START, "round1")
        g.add_conditional_edges("round1", _route_or_error("contrarian"))
        g.add_conditional_edges("contrarian", _route_or_error("round2"))
        g.add_conditional_edges("round2", _route_or_error("fund_manager"))
        g.add_conditional_edges("fund_manager", _route_or_error("ppo_stub"))
        g.add_edge("ppo_stub", END)
        g.add_edge("error_handler", END)

        return g.compile()

    def _node_round1(self, state: DebateState) -> dict:
        """Run Macro + Technical + Sentiment agents sequentially; Risk agent computes GARCH limits."""
        ctx = state["ctx"]
        aligned_dfs = state["aligned_dfs"]
        theses: list[AgentThesis] = []
        tokens = state.get("total_tokens", 0)

        try:
            macro_thesis = self._macro.post_thesis(ctx)
            theses.append(macro_thesis)
            tokens += self._macro.tokens_used
            logger.info(f"MacroAgent R1: {macro_thesis.directional_bias} ({macro_thesis.confidence:.2f})")
        except Exception as e:
            logger.error(f"MacroAgent R1 failed: {e}")

        try:
            asset_df = aligned_dfs.get(ctx.asset)
            tech_thesis = self._tech.post_thesis(ctx, aligned_df=asset_df)
            theses.append(tech_thesis)
            tokens += self._tech.tokens_used
            logger.info(f"TechnicalAgent R1: {tech_thesis.directional_bias} ({tech_thesis.confidence:.2f})")
        except Exception as e:
            logger.error(f"TechnicalAgent R1 failed: {e}")

        try:
            sentiment_thesis = self._sentiment.post_thesis(ctx)
            theses.append(sentiment_thesis)
            tokens += self._sentiment.tokens_used
            logger.info(f"SentimentAgent R1: {sentiment_thesis.directional_bias} ({sentiment_thesis.confidence:.2f})")
        except Exception as e:
            logger.error(f"SentimentAgent R1 failed: {e}")

        risk_limits: RiskLimits | None = None
        try:
            risk_limits = self._risk.get_risk_limits(ctx)
            tokens += self._risk.tokens_used
            logger.info(f"RiskAgent: {risk_limits.risk_regime} regime, vol={risk_limits.volatility_forecast:.4f}")
        except Exception as e:
            logger.error(f"RiskAgent failed: {e}")

        return {
            "round1_theses": theses,
            "risk_limits": risk_limits,
            "total_tokens": tokens,
            "error": None,
        }

    def _node_contrarian(self, state: DebateState) -> dict:
        """ContrarianAgent runs after all R1 theses are in — needs to see consensus first."""
        ctx = state["ctx"]
        theses = state["round1_theses"]
        aligned_df = state["aligned_dfs"].get(ctx.asset)
        tokens = state.get("total_tokens", 0)

        contrarian_thesis: AgentThesis | None = None
        try:
            contrarian_thesis = self._contrarian.post_contrarian_thesis(ctx, theses, aligned_df)
            tokens += self._contrarian.tokens_used
            active = contrarian_thesis.directional_bias != "neutral"
            logger.info(
                f"ContrarianAgent: {contrarian_thesis.directional_bias} "
                f"({contrarian_thesis.confidence:.2f}) — {'ACTIVE' if active else 'inactive'}"
            )
        except Exception as e:
            logger.error(f"ContrarianAgent failed: {e}")

        avg_conf = float(sum(t.confidence for t in theses) / len(theses)) if theses else 0.0

        # Rebuttal fires when: Contrarian is active, OR R1 agents disagree, OR groupthink threshold exceeded
        contrarian_active = contrarian_thesis is not None and contrarian_thesis.directional_bias != "neutral"
        r1_biases = [t.directional_bias for t in theses]
        r1_opposition = "bullish" in r1_biases and "bearish" in r1_biases
        groupthink = avg_conf > 0.75 and not r1_opposition
        rebuttal_triggered = contrarian_active or r1_opposition or groupthink

        return {
            "contrarian_thesis": contrarian_thesis,
            "consensus_confidence": avg_conf,
            "rebuttal_triggered": rebuttal_triggered,
            "total_tokens": tokens,
            "error": None,
        }

    def _node_round2(self, state: DebateState) -> dict:
        """Round 2 rebuttals — fires when Contrarian is active or R1 agents have opposing views."""
        ctx = state["ctx"]
        theses = state["round1_theses"]
        contrarian_thesis = state.get("contrarian_thesis")
        rebuttal_triggered = state.get("rebuttal_triggered", False)
        rebuttals: list[AgentRebuttal] = []
        tokens = state.get("total_tokens", 0)

        if not rebuttal_triggered:
            logger.info("Round 2: no opposition detected — skipping rebuttals")
            return {"round2_rebuttals": rebuttals, "total_tokens": tokens, "error": None}

        # Full thesis pool that agents can target (includes Contrarian)
        all_theses = theses + ([contrarian_thesis] if contrarian_thesis else [])

        # Main agents rebut the strongest opposing thesis (Contrarian's if active, else each other)
        for agent in [self._macro, self._tech, self._sentiment]:
            try:
                my_r1 = next((t for t in theses if t.agent_name == agent.name), None)
                others = [t for t in all_theses if t.agent_name != agent.name]
                rebuttal = agent.post_rebuttal(ctx, others, my_r1_thesis=my_r1)
                if rebuttal is not None:
                    rebuttals.append(rebuttal)
                    tokens += agent.tokens_used
                    logger.info(
                        f"{agent.name} R2: rebuts {rebuttal.target_agent} | "
                        f"maintains={rebuttal.maintains_bias} conf={rebuttal.updated_confidence:.2f}"
                    )
            except Exception as e:
                logger.error(f"{agent.name} R2 failed: {e}")

        # Contrarian always fires a rebuttal when active
        if contrarian_thesis and contrarian_thesis.directional_bias != "neutral":
            try:
                others_for_contrarian = [t for t in all_theses if t.agent_name != "ContrarianAgent"]
                contrarian_rebuttal = self._contrarian.post_rebuttal(
                    ctx, others_for_contrarian, my_r1_thesis=contrarian_thesis
                )
                if contrarian_rebuttal is not None:
                    rebuttals.append(contrarian_rebuttal)
                    tokens += self._contrarian.tokens_used
                    logger.info(
                        f"ContrarianAgent R2: rebuts {contrarian_rebuttal.target_agent} | "
                        f"maintains={contrarian_rebuttal.maintains_bias}"
                    )
            except Exception as e:
                logger.error(f"ContrarianAgent R2 failed: {e}")

        return {"round2_rebuttals": rebuttals, "total_tokens": tokens, "error": None}

    def _node_fund_manager(self, state: DebateState) -> dict:
        ctx = state["ctx"]
        risk_limits = state["risk_limits"]
        contrarian_thesis = state.get("contrarian_thesis")
        tokens = state.get("total_tokens", 0)

        if risk_limits is None:
            from agents.risk_agent import _default_risk_limits
            risk_limits = _default_risk_limits()

        try:
            decision = self._fm.synthesize(
                ctx,
                state["round1_theses"],
                state["round2_rebuttals"],
                risk_limits,
                contrarian_thesis=contrarian_thesis,
            )
            tokens += self._fm.tokens_used
            logger.info(
                f"FundManager: {decision.final_action} | conviction={decision.conviction:.2f} | "
                f"override={decision.regime_override} | dissenting={decision.dissenting_agents}"
            )
            return {"fund_manager_decision": decision, "total_tokens": tokens, "error": None}
        except Exception as e:
            logger.error(f"FundManager failed: {e}")
            return {"error": str(e), "total_tokens": tokens}

    def _node_ppo_stub(self, state: DebateState) -> dict:
        """Month 3 placeholder — passes fund manager decision through unchanged."""
        return state

    def _node_error_handler(self, state: DebateState) -> dict:
        err = state.get("error", "unknown error")
        logger.error(f"Debate error handler triggered: {err}")
        return {
            "fund_manager_decision": FundManagerDecision(
                final_action="hold",
                position_size=0.0,
                conviction=0.0,
                consensus_direction="neutral",
                dissenting_agents=[],
                rationale=f"Error during debate: {err[:200]}",
                regime_override=False,
            ),
            "error": err,
        }

    def run(
        self,
        ctx: RegimeContext,
        aligned_dfs: dict[str, pd.DataFrame],
    ) -> DebateResult:
        start_time = time.time()
        initial_state: DebateState = {
            "ctx": ctx,
            "aligned_dfs": aligned_dfs,
            "round1_theses": [],
            "contrarian_thesis": None,
            "consensus_confidence": 0.0,
            "rebuttal_triggered": False,
            "round2_rebuttals": [],
            "risk_limits": None,
            "fund_manager_decision": None,
            "error": None,
            "start_time": start_time,
            "total_tokens": 0,
        }

        final_state = self._graph.invoke(initial_state)
        duration = time.time() - start_time

        decision = final_state.get("fund_manager_decision") or FundManagerDecision(
            final_action="hold",
            position_size=0.0,
            conviction=0.0,
            consensus_direction="neutral",
            dissenting_agents=[],
            rationale="No decision produced.",
            regime_override=False,
        )
        risk_limits = final_state.get("risk_limits") or _default_risk_limits_inline()

        # Include Contrarian thesis in round1_theses for the full DebateResult record
        all_r1 = list(final_state.get("round1_theses", []))
        contrarian_thesis = final_state.get("contrarian_thesis")
        if contrarian_thesis is not None:
            all_r1.append(contrarian_thesis)

        return DebateResult(
            timestamp=str(ctx.timestamp),
            asset=ctx.asset,
            regime_label=ctx.regime_label,
            round1_theses=all_r1,
            round2_rebuttals=final_state.get("round2_rebuttals", []),
            risk_limits=risk_limits,
            fund_manager_decision=decision,
            total_tokens_used=final_state.get("total_tokens", 0),
            debate_duration_seconds=round(duration, 2),
        )


def _route_or_error(next_node: str):
    def _route(state: DebateState) -> str:
        if state.get("error"):
            return "error_handler"
        return next_node
    return _route


def _default_risk_limits_inline():
    from agents.risk_agent import _default_risk_limits
    return _default_risk_limits()


def run_debate(
    timestamp: pd.Timestamp,
    asset: str,
    regime_stream_df: pd.DataFrame,
    aligned_dfs: dict[str, pd.DataFrame],
    debate_graph: DebateGraph,
    semantics_path=None,
) -> DebateResult:
    """Main entry point: build context, run the debate graph, return DebateResult.

    semantics_path is opt-in (2026-08-14) — None preserves exact prior behavior
    (production's regime_semantics.json). Pass an explicit path when regime_stream_df
    comes from a different HMM, so ctx.regime_description matches it — see
    agents/context.py's build_regime_context docstring. Must be paired with passing the
    SAME path to FundManagerAgent(semantics_path=...) when building debate_graph, or the
    displayed description and the conviction/override logic will disagree about which
    regime stream's semantics apply."""
    from agents.context import build_regime_context

    aligned_df = aligned_dfs.get(asset)
    if aligned_df is None:
        raise ValueError(f"No aligned data for asset {asset}")

    ctx = build_regime_context(timestamp, asset, regime_stream_df, aligned_df, semantics_path=semantics_path)
    logger.info(f"Starting debate: {asset} @ {timestamp} | Regime R{ctx.regime_label}: {ctx.regime_description}")

    return debate_graph.run(ctx, aligned_dfs)
