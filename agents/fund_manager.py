"""FundManagerAgent — synthesizes 5-agent debate into a final actionable decision."""

from __future__ import annotations

import json
import re

from agents.base import BaseAgent
from agents.context import RegimeContext, serialize_context_for_prompt
from agents.schemas import AgentRebuttal, AgentThesis, FundManagerDecision, RiskLimits
from utils.logger import logger

# Regime-direction conviction multipliers: (bullish_mult, bearish_mult)
_REGIME_CONVICTION_MULTIPLIERS: dict[int, tuple[float, float]] = {
    0: (0.85, 0.85),  # mixed/transitional — reduce conviction either direction
    1: (1.00, 0.70),  # calm bull — full bullish conviction, penalise bearish
    2: (0.40, 1.00),  # stress/crisis — heavy penalty on bullish
    3: (0.50, 1.00),  # bear trend — heavy penalty on bullish
    4: (0.65, 0.90),  # calm bear — moderate penalty on bullish (contradiction)
    5: (0.90, 0.85),  # trending forex A
    6: (0.90, 0.85),  # trending forex B
}

# Regimes where bullish consensus is overridden to hold
_BULLISH_PENALTY_REGIMES = {2, 3}
# Regimes where bullish consensus keeps long but is flagged as contradiction
_BULLISH_CAUTION_REGIMES = {4}
# Regimes where bearish consensus is flagged (calm bull regime)
_BEARISH_PENALTY_REGIMES = {1}

_CONSENSUS_MARGIN = 0.15
_CONTRARIAN_WEIGHT = 0.5  # Contrarian votes count at half weight


class FundManagerAgent(BaseAgent):
    name = "FundManager"

    def _build_system_prompt(self, ctx: RegimeContext) -> str:
        return (
            "You are the Fund Manager synthesizing a multi-agent debate into a trading decision.\n"
            "Your role is to write a concise rationale explaining the final action.\n"
            "You MUST respond with a single JSON object — no preamble, no markdown.\n\n"
            + serialize_context_for_prompt(ctx)
        )

    def _compute_weighted_consensus(
        self,
        round1_theses: list[AgentThesis],
        round2_rebuttals: list[AgentRebuttal],
        contrarian_thesis: AgentThesis | None = None,
    ) -> tuple[str, float, list[str]]:
        """Weighted vote: ContrarianAgent counts at 0.5x. Returns (consensus, conviction, dissenting)."""
        final_conf: dict[str, tuple[str, float]] = {}

        for t in round1_theses:
            w = _CONTRARIAN_WEIGHT if t.agent_name == "ContrarianAgent" else 1.0
            final_conf[t.agent_name] = (t.directional_bias, t.confidence * w)

        # Include contrarian if passed separately and not already in round1_theses
        if contrarian_thesis and contrarian_thesis.agent_name not in final_conf:
            final_conf[contrarian_thesis.agent_name] = (
                contrarian_thesis.directional_bias,
                contrarian_thesis.confidence * _CONTRARIAN_WEIGHT,
            )

        for r in round2_rebuttals:
            if r.agent_name in final_conf:
                bias = final_conf[r.agent_name][0]
                if not r.maintains_bias:
                    bias = "neutral"
                w = _CONTRARIAN_WEIGHT if r.agent_name == "ContrarianAgent" else 1.0
                final_conf[r.agent_name] = (bias, r.updated_confidence * w)

        bullish_score = sum(conf for _, (bias, conf) in final_conf.items() if bias == "bullish")
        bearish_score = sum(conf for _, (bias, conf) in final_conf.items() if bias == "bearish")
        total = bullish_score + bearish_score + 1e-9

        if bullish_score > bearish_score + _CONSENSUS_MARGIN:
            consensus = "bullish"
            conviction = bullish_score / total
        elif bearish_score > bullish_score + _CONSENSUS_MARGIN:
            consensus = "bearish"
            conviction = bearish_score / total
        else:
            consensus = "split"
            conviction = 0.5 - abs(bullish_score - bearish_score) / (2 * total)

        dissenting: list[str] = []
        if consensus in ("bullish", "bearish"):
            opposite = "bearish" if consensus == "bullish" else "bullish"
            dissenting = [name for name, (bias, _) in final_conf.items() if bias == opposite]

        return consensus, float(conviction), dissenting

    def synthesize(
        self,
        ctx: RegimeContext,
        round1_theses: list[AgentThesis],
        round2_rebuttals: list[AgentRebuttal],
        risk_limits: RiskLimits,
        contrarian_thesis: AgentThesis | None = None,
    ) -> FundManagerDecision:
        consensus, conviction, dissenting = self._compute_weighted_consensus(
            round1_theses, round2_rebuttals, contrarian_thesis
        )

        # Apply regime-aware conviction multiplier
        mult_pair = _REGIME_CONVICTION_MULTIPLIERS.get(ctx.regime_label, (1.0, 1.0))
        if consensus == "bullish":
            regime_multiplier = mult_pair[0]
        elif consensus == "bearish":
            regime_multiplier = mult_pair[1]
        else:
            # split: use the more conservative (smaller) multiplier
            regime_multiplier = min(mult_pair)
        conviction = max(0.1, min(1.0, conviction * regime_multiplier))

        # Detect regime contradictions (direction contradicts regime character)
        regime_contradiction = (
            ctx.regime_label in (_BULLISH_PENALTY_REGIMES | _BULLISH_CAUTION_REGIMES)
            and consensus == "bullish"
        ) or (
            ctx.regime_label in _BEARISH_PENALTY_REGIMES
            and consensus == "bearish"
        )

        # Determine final action and regime_override
        contrarian_active = contrarian_thesis is not None and contrarian_thesis.directional_bias != "neutral"
        any_agent_flipped = any(not r.maintains_bias for r in round2_rebuttals)
        regime_override = False

        if ctx.regime_label in _BULLISH_PENALTY_REGIMES and consensus == "bullish":
            # Stress/crisis or bear trend + bullish consensus → override to hold
            final_action = "hold"
            regime_override = True
        elif ctx.regime_label in _BULLISH_CAUTION_REGIMES and consensus == "bullish":
            # Calm bear + bullish consensus → keep long but flag contradiction
            final_action = "long"
            regime_override = True
        elif ctx.regime_label in _BEARISH_PENALTY_REGIMES and consensus == "bearish":
            # Calm bull + bearish consensus → keep short but flag
            final_action = "short"
            regime_override = True
        elif consensus == "bullish":
            final_action = "long"
        elif consensus == "bearish":
            final_action = "short"
        else:
            final_action = "hold"

        # Secondary override: Contrarian challenged and an agent flipped
        if contrarian_active and any_agent_flipped:
            regime_override = True

        position_size = round(min(conviction * 0.8, risk_limits.max_position_size), 4)
        if final_action == "hold":
            position_size = 0.0

        # Build compact prompt for rationale LLM call
        theses_str = "; ".join(
            f"{t.agent_name}={t.directional_bias}({t.confidence:.1f})"
            for t in round1_theses
        )
        if contrarian_thesis and contrarian_thesis.directional_bias != "neutral":
            theses_str += f"; ContrarianAgent={contrarian_thesis.directional_bias}({contrarian_thesis.confidence:.1f})[0.5x]"

        contrarian_note = ""
        if contrarian_active:
            flipped_str = "flipped after rebuttal" if any_agent_flipped else "held positions"
            contrarian_note = f"Contrarian challenged {consensus} consensus — agents {flipped_str}. "

        contradiction_note = ""
        if regime_contradiction:
            contradiction_note = (
                f"REGIME CONTRADICTION: R{ctx.regime_label} contradicts {consensus} consensus "
                f"(conviction penalty {regime_multiplier:.2f}x applied). "
            )

        system_prompt = self._build_system_prompt(ctx)
        user_prompt = (
            f"Action: {final_action} | Consensus: {consensus} | Regime: R{ctx.regime_label} | Risk: {risk_limits.risk_regime}\n"
            f"Theses: {theses_str}\n"
            f"{contrarian_note}{contradiction_note}"
            "Write one sentence rationale. Reply ONLY:\n"
            '{"rationale":"<one sentence explaining the decision>"}'
        )

        raw = self._call_llm(system_prompt, user_prompt)
        rationale = "Decision based on weighted agent consensus and regime context."
        try:
            json_match = re.search(r'\{[^{}]*"rationale"[^{}]*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                rationale = str(parsed.get("rationale", rationale))[:600]
            elif raw.strip():
                rationale = raw.strip()[:600]
        except Exception:
            pass

        return FundManagerDecision(
            final_action=final_action,
            position_size=position_size,
            conviction=round(conviction, 4),
            consensus_direction=consensus,
            dissenting_agents=dissenting,
            rationale=rationale,
            regime_override=regime_override,
            regime_penalty_applied=round(regime_multiplier, 4),
            regime_contradiction=regime_contradiction,
        )
