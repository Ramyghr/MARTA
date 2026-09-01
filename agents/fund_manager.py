"""FundManagerAgent — synthesizes 5-agent debate into a final actionable decision."""

from __future__ import annotations

import json
import re
from pathlib import Path

from agents.base import BaseAgent
from agents.context import (
    RegimeContext,
    _load_regime_stats,
    classify_regime_volatility_level,
    serialize_context_for_prompt,
)
from agents.schemas import AgentRebuttal, AgentThesis, FundManagerDecision, RiskLimits
from utils.logger import logger

# Volatility-based conviction multiplier, SYMMETRIC across bullish/bearish (2026-09-01,
# replaces the former _REGIME_KIND_CONVICTION_MULTIPLIERS + directional override sets).
# The prior scheme scaled bullish/bearish conviction asymmetrically — and outright
# overrode a bullish consensus to "hold" — based on a regime's historical sharpe SIGN,
# treating negative-sharpe regimes as reliably bearish and positive-sharpe ones as
# reliably bullish. An objective zigzag trend-leg test
# (regime/validate_regime_trend_alignment.py, 111 legs, all 5 crypto5 assets, 2021)
# found this direction assumption wrong at the per-instance level 51% of the time —
# worse than a coin flip for the regime nominally labeled "strong bull" (crypto5's R4,
# 40.6% up-legs) — so forcing a real trading action (hold, or "contradiction") off that
# assumption was not defensible and has been removed, not just re-labeled. What DOES
# hold up under validation is realized_vol_mean as a magnitude/uncertainty signal
# (classify_regime_volatility_level), so conviction is now scaled down when volatility
# is elevated — a risk-management principle that doesn't require knowing which
# direction the regime favors, applied equally regardless of consensus direction.
_REGIME_VOL_CONVICTION_MULTIPLIER: dict[str, float] = {
    "normal_vol": 1.00,
    "elevated_vol": 0.80,
    "extreme_vol": 0.60,
}

_CONSENSUS_MARGIN = 0.15
_CONTRARIAN_WEIGHT = 0.5  # Contrarian votes count at half weight


class FundManagerAgent(BaseAgent):
    name = "FundManager"

    def __init__(self, semantics_path: Path | None = None, **kwargs) -> None:
        """semantics_path is opt-in (2026-08-14) — None resolves to production's
        data/validated/regime_semantics.json (exact prior behavior). Pass an explicit
        path to derive the volatility-based conviction multiplier from a different
        regime stream's actual semantics (e.g.
        data/validated/btc_only_v3_regime_semantics.json) instead of production's —
        see classify_regime_volatility_level for why this must track the regime stream
        actually in use rather than a fixed table."""
        super().__init__(**kwargs)
        self._semantics_path = semantics_path

    def _regime_vol_level(self, regime_label: int) -> str:
        stats_by_label = _load_regime_stats(self._semantics_path)
        return classify_regime_volatility_level(stats_by_label.get(regime_label, {}))

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

        # Apply volatility-based conviction multiplier — symmetric across bullish/
        # bearish, since realized_vol_mean carries no direction claim (see module
        # docstring on _REGIME_VOL_CONVICTION_MULTIPLIER for why this replaced the
        # former sharpe-sign-based, asymmetric scheme).
        vol_level = self._regime_vol_level(ctx.regime_label)
        regime_multiplier = _REGIME_VOL_CONVICTION_MULTIPLIER.get(vol_level, 1.0)
        conviction = max(0.1, min(1.0, conviction * regime_multiplier))

        # Elevated-risk flag: a decisive (non-split) directional call made during an
        # elevated/extreme-volatility regime. This is NOT a claim that the regime
        # "disagrees" with the direction (that claim isn't validated — see above) — it's
        # a caution flag that a confident bet is being placed under conditions this
        # project's own regime-distinctness testing confirms carry more uncertainty.
        regime_contradiction = (
            vol_level in ("elevated_vol", "extreme_vol") and consensus in ("bullish", "bearish")
        )

        # Determine final action — reflects the debate's actual consensus directly.
        # Regime character no longer forces or overrides an action (see above); its
        # only effect on the decision is the conviction/position-size scaling above.
        contrarian_active = contrarian_thesis is not None and contrarian_thesis.directional_bias != "neutral"
        any_agent_flipped = any(not r.maintains_bias for r in round2_rebuttals)
        regime_override = False

        if consensus == "bullish":
            final_action = "long"
        elif consensus == "bearish":
            final_action = "short"
        else:
            final_action = "hold"

        # Override: Contrarian challenged and an agent flipped (unrelated to regime
        # character — this is about the debate's own internal dynamics).
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
                f"ELEVATED-RISK REGIME: R{ctx.regime_label} is {vol_level.replace('_', ' ')} "
                f"(conviction scaled {regime_multiplier:.2f}x for this {consensus} call). "
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
