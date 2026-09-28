"""TechnicalAgent — technical analysis over price/momentum indicators.

Previously also loaded a standalone TFT contrastive-encoder checkpoint
(encoder/checkpoints/best_encoder.pt) to compute a supplementary "attention map" of
which indicators the encoder weighted most heavily. Removed 2026-09-16: that encoder
architecture was found not to add value and was abandoned project-wide in favor of raw
features (see docs/RESULTS_AND_METHODOLOGY.md) — this agent was the last caller still
loading it, on a checkpoint that predated the crypto5 panel migration and was never
trained on BNB/LTC/XRP. The regime label/probabilities this agent reasons over still
come from the current regime stream via RegimeContext — only the extra encoder-attention
side-signal is gone. Analysis now runs on the indicator values alone.
"""

from __future__ import annotations

import dataclasses

from agents.base import BaseAgent
from agents.context import RegimeContext, serialize_context_for_prompt
from agents.schemas import AgentThesis


class TechnicalAgent(BaseAgent):
    name = "TechnicalAgent"

    def __init__(self, model: str = "mistral:7b-instruct", temperature: float = 0.4) -> None:
        super().__init__(model=model, temperature=temperature)

    def _build_system_prompt(self, ctx: RegimeContext) -> str:
        # Strip the regime's qualitative description ("High-volatility bull regime", etc.)
        # before serializing — keep regime_label/regime_probs (the actual numeric signal,
        # confirmed informative by the null-regime test) but drop the loaded "bull"/"bear"
        # wording. A 2026-08-23 ablation found this text alone drove the model to call
        # "bullish" on an unambiguous bearish extreme (RSI=12, ADX=45, MACD strongly
        # negative, close_return=-6%) in 3/3 runs; stripping just the description text
        # (numbers intact) corrected this to 4/5. Every other agent still gets the full
        # description — this override is scoped to TechnicalAgent only.
        ctx_for_prompt = dataclasses.replace(
            ctx, regime_description=f"Regime {ctx.regime_label} (see probability vector below)"
        )
        return (
            "You are the Technical Pattern Agent for a quantitative trading system.\n"
            "Ground your technical analysis in the indicator values below.\n\n"
            "Interpretation rules for RSI/momentum extremes — apply these before concluding:\n"
            "- Check ADX first. ADX > 25 means the market is trending strongly. In a trending "
            "market, an extreme RSI (>70 or <30) confirms the prevailing move is strong — read it "
            "as momentum continuation, NOT as an imminent reversal or bounce. Only read extreme "
            "RSI as a mean-reversion/reversal signal when ADX < 20 (a genuinely ranging market).\n"
            "- MACD histogram sign should agree with your call. A large negative MACD histogram "
            "alongside a large negative recent return and ADX > 25 is bearish momentum, not an "
            "oversold setup about to bounce.\n"
            "- Base your call on RSI/MACD/ADX/recent return, not on the regime label — use the "
            "regime probability vector only as secondary corroboration.\n\n"
            "You MUST respond with a single JSON object — no preamble, no markdown, no explanation outside the JSON.\n\n"
            + serialize_context_for_prompt(ctx_for_prompt)
        )

    def post_thesis(self, ctx: RegimeContext) -> AgentThesis:
        features_str = "\n".join(f"  {k}: {v:.4f}" for k, v in ctx.price_features.items())
        tech_context = (
            f"\n=== TECHNICAL DATA ===\n"
            f"Current indicators:\n{features_str}\n"
            f"RSI: {ctx.price_features.get('rsi', 0):.2f} | "
            f"ADX: {ctx.price_features.get('adx', 0):.2f} | "
            f"MACD hist: {ctx.price_features.get('macd_histogram', 0):.4f} | "
            f"Latest close_return: {ctx.price_features.get('close_return', 0):+.4f} "
            f"({'down' if ctx.price_features.get('close_return', 0) < 0 else 'up'} move)\n"
        )

        system_prompt = self._build_system_prompt(ctx) + tech_context

        user_prompt = (
            f"Assess directional bias for {ctx.asset} from the indicators above.\n\n"
            "Reply ONLY with this JSON schema — replace every placeholder with your own analysis, "
            "do not copy the placeholder values:\n"
            '{"agent_name":"TechnicalAgent","round":1,"directional_bias":"<bullish|bearish|neutral, your judgment>",'
            '"confidence":<your confidence as a float between 0.0 and 1.0>,'
            '"reasoning":"<your 1-2 sentence reasoning>","key_factors":["<factor1>","<factor2>"],'
            '"regime_alignment":"<aligned|contrary|neutral|mixed>"}'
        )

        raw = self._call_llm(system_prompt, user_prompt)
        result = self._parse_json_response(raw, AgentThesis)
        if isinstance(result, AgentThesis):
            result.agent_name = "TechnicalAgent"
            result.round = 1
            return result
        return result
