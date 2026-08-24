"""ContrarianAgent — devil's advocate that challenges high-confidence consensus using historical evidence."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from agents.base import BaseAgent
from agents.context import RegimeContext, serialize_context_for_prompt
from agents.schemas import AgentThesis
from utils.logger import logger

_MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent)))
# Kept at the old production regime_stream.parquet — dashboard/backend/debate_stream.py
# constructs ContrarianAgent() with no arguments and hardcodes that same production
# stream elsewhere, so this default must keep matching it or live dashboard debates
# would silently search the wrong (crypto5) stream for historical analogues. Was
# previously ALWAYS this hardcoded path with no way to override — that part was the
# real bug (it ignored whatever regime_stream_df the rest of a given debate was using).
# Pass regime_stream_df to __init__ to inject the exact stream in use (preferred — also
# avoids a redundant second parquet read); crypto-only callers (run_debate.py) do this.
_REGIME_STREAM_PATH = _MARTA_ROOT / "data/processed/regime_stream.parquet"
_ACTIVATION_THRESHOLD = 0.60  # Mistral 7B confident theses typically land at 0.60-0.70
_HIGH_REGIME_PROB = 0.80
_MAX_ANALOGUE_SAMPLES = 100


class ContrarianAgent(BaseAgent):
    name = "ContrarianAgent"

    def __init__(
        self,
        model: str = "mistral:7b-instruct",
        temperature: float = 0.4,
        regime_stream_df: pd.DataFrame | None = None,
        regime_stream_path: Path | None = None,
    ) -> None:
        # Wider timeout/retry/token budget than the other agents (base.py defaults: 300s, 2
        # attempts, 600 num_predict): 600s timeout, up to 3 total attempts (1 initial + 2
        # retries), and 900 num_predict before genuinely falling back to the neutral
        # "system_error" thesis. The fallback itself is correct — it just shouldn't fire
        # until retries are truly exhausted. Retries alone don't fix this failure mode:
        # a 2026-08 audit found this agent's historical-analogue reasoning deterministically
        # overruns the shared 600-token cap for the same prompt, so it kept retrying into
        # the same truncated-JSON failure (~40% of activations across an 81-debate cache)
        # rather than a one-off timeout blip that a retry would actually help with.
        super().__init__(
            model=model, temperature=temperature, timeout_seconds=600, max_attempts=3, num_predict=900
        )
        if regime_stream_df is not None:
            self._regime_stream = regime_stream_df
        else:
            self._regime_stream = pd.read_parquet(
                regime_stream_path or _REGIME_STREAM_PATH, engine="pyarrow"
            )

    def _compute_consensus(self, theses: list[AgentThesis]) -> tuple[str, float]:
        """Return (consensus_direction, avg_confidence) from a list of theses."""
        if not theses:
            return "neutral", 0.0
        avg_conf = float(np.mean([t.confidence for t in theses]))
        bull_score = sum(t.confidence for t in theses if t.directional_bias == "bullish")
        bear_score = sum(t.confidence for t in theses if t.directional_bias == "bearish")
        if bull_score > bear_score + 0.10:
            return "bullish", avg_conf
        if bear_score > bull_score + 0.10:
            return "bearish", avg_conf
        return "neutral", avg_conf

    def _find_historical_analogues(
        self,
        ctx: RegimeContext,
        consensus_direction: str,
        aligned_df: pd.DataFrame,
    ) -> dict:
        """Find historical timestamps with same regime + high probability; compute 24h forward return stats."""
        stream = self._regime_stream.copy()
        regime_col = f"regime_prob_{ctx.regime_label}"

        same_regime = stream[
            (stream["asset"] == ctx.asset)
            & (stream["regime_label"] == ctx.regime_label)
        ].copy()

        if regime_col in same_regime.columns:
            same_regime = same_regime[same_regime[regime_col] > _HIGH_REGIME_PROB]

        if same_regime.empty:
            return {"total_cases": 0, "loss_rate": 0.0, "avg_forward_return": 0.0}

        # Normalize timestamps to UTC
        ts_col = pd.to_datetime(same_regime["timestamp"])
        same_regime = same_regime.copy()
        same_regime["timestamp"] = (
            ts_col.dt.tz_localize("UTC") if ts_col.dt.tz is None else ts_col.dt.tz_convert("UTC")
        )

        # Exclude the current debate window (±48h) to avoid data leakage
        ctx_ts = pd.Timestamp(ctx.timestamp).tz_convert("UTC") if pd.Timestamp(ctx.timestamp).tzinfo else pd.Timestamp(ctx.timestamp).tz_localize("UTC")
        same_regime = same_regime[
            abs((same_regime["timestamp"] - ctx_ts).dt.total_seconds()) > 48 * 3600
        ]

        if same_regime.empty:
            return {"total_cases": 0, "loss_rate": 0.0, "avg_forward_return": 0.0}

        sample = same_regime.head(_MAX_ANALOGUE_SAMPLES)
        aligned_sorted = aligned_df.sort_index()
        idx_tz = aligned_sorted.index.tz
        close_returns = aligned_sorted["close_return"].to_numpy(dtype=np.float64)
        n_aligned = len(aligned_sorted)
        forward_returns: list[float] = []

        # Batch all timestamp lookups in one searchsorted call instead of per-row get_indexer.
        sample_ts = pd.to_datetime(sample["timestamp"])
        if idx_tz is not None:
            sample_ts = sample_ts.dt.tz_convert(idx_tz)
        else:
            sample_ts = sample_ts.dt.tz_localize(None) if sample_ts.dt.tz is not None else sample_ts

        locs = aligned_sorted.index.searchsorted(pd.DatetimeIndex(sample_ts), side="right") - 1

        for loc in locs:
            if loc < 0 or loc + 24 >= n_aligned:
                continue
            next_rets = close_returns[loc + 1 : loc + 25]
            next_rets = next_rets[~np.isnan(next_rets)]
            if len(next_rets) == 0:
                continue
            forward_returns.append(float(np.sum(next_rets)))

        if not forward_returns:
            return {"total_cases": 0, "loss_rate": 0.0, "avg_forward_return": 0.0}

        total = len(forward_returns)
        losses = (
            sum(1 for r in forward_returns if r < 0)
            if consensus_direction == "bullish"
            else sum(1 for r in forward_returns if r > 0)
        )

        return {
            "total_cases": total,
            "loss_rate": round(losses / total, 3),
            "avg_forward_return": round(float(np.mean(forward_returns)), 6),
        }

    def _build_system_prompt(self, ctx: RegimeContext) -> str:
        return (
            "You are the Contrarian Devil's Advocate Agent for a quantitative trading system.\n"
            "Your sole purpose is to challenge the current consensus using historical evidence.\n"
            "You MUST argue the opposite position regardless of your own view.\n"
            "Present the strongest possible bear case if consensus is bullish, and vice versa.\n"
            "You MUST respond with a single JSON object — no preamble, no markdown.\n\n"
            + serialize_context_for_prompt(ctx)
        )

    def post_contrarian_thesis(
        self,
        ctx: RegimeContext,
        other_theses: list[AgentThesis],
        aligned_df: pd.DataFrame | None = None,
    ) -> AgentThesis:
        """Post a contrarian thesis opposing the consensus. Activates only when avg_confidence >= 0.60."""
        consensus_direction, avg_conf = self._compute_consensus(other_theses)

        if avg_conf < _ACTIVATION_THRESHOLD:
            logger.info(
                f"ContrarianAgent: avg_confidence={avg_conf:.2f} < threshold={_ACTIVATION_THRESHOLD} — not activating"
            )
            return AgentThesis(
                agent_name="ContrarianAgent",
                round=1,
                directional_bias="neutral",
                confidence=0.1,
                reasoning=(
                    f"Consensus confidence {avg_conf:.2f} is below the activation threshold "
                    f"{_ACTIVATION_THRESHOLD}. No strong consensus to challenge."
                ),
                key_factors=["low_consensus_confidence"],
                regime_alignment="neutral",
            )

        opposite = "bearish" if consensus_direction == "bullish" else "bullish"

        analogues = {"total_cases": 0, "loss_rate": 0.0, "avg_forward_return": 0.0}
        if aligned_df is not None and not aligned_df.empty:
            try:
                analogues = self._find_historical_analogues(ctx, consensus_direction, aligned_df)
            except Exception as e:
                logger.warning(f"ContrarianAgent: analogue retrieval failed — {e}")

        logger.info(
            f"ContrarianAgent: ACTIVATED — consensus={consensus_direction} avg_conf={avg_conf:.2f} "
            f"| arguing {opposite} | analogues={analogues['total_cases']} cases, "
            f"loss_rate={analogues['loss_rate']:.1%}"
        )

        analogue_context = (
            f"\n=== CONTRARIAN HISTORICAL EVIDENCE ===\n"
            f"Current consensus: {consensus_direction} (avg_confidence={avg_conf:.2f})\n"
            f"Your mandate: argue {opposite}\n"
            f"Historical analogues — same regime R{ctx.regime_label}, regime_prob > {_HIGH_REGIME_PROB}: "
            f"{analogues['total_cases']} cases\n"
            f"Loss rate under {consensus_direction} consensus: {analogues['loss_rate']:.1%}\n"
            f"Avg 24h forward return: {analogues['avg_forward_return']:+.4f}\n"
        )

        system_prompt = self._build_system_prompt(ctx) + analogue_context

        
        user_prompt = (
            f"Consensus is {consensus_direction} with avg_conf={avg_conf:.2f}. "
            f"Challenge it — argue {opposite} for {ctx.asset}.\n\n"
            "Reply ONLY with this JSON schema — replace <...> with your own analysis, "
            "do not copy the example confidence value:\n"
            f'{{"agent_name":"ContrarianAgent","round":1,"directional_bias":"{opposite}",'
            '"confidence":<your confidence as a float between 0.0 and 1.0, based on how strong your counter-evidence is>,'
            '"reasoning":"<strongest counter-argument citing regime risk and historical failure rate — '
            'max 2 sentences, under 400 characters, this is a hard limit>",'
            '"key_factors":["historical_loss_rate","regime_complacency","overconfident_consensus"],'
            '"regime_alignment":"contrary"}}'
        )

        raw = self._call_llm(system_prompt, user_prompt)
        result = self._parse_json_response(raw, AgentThesis)
        if isinstance(result, AgentThesis):
            result.agent_name = "ContrarianAgent"
            result.round = 1
            result.directional_bias = opposite  # enforce opposite regardless of LLM output
            result.regime_alignment = "contrary"
            return result
        return result

    def post_thesis(self, ctx: RegimeContext) -> AgentThesis:
        """Fallback for base class interface — returns neutral. Use post_contrarian_thesis for full behavior."""
        return AgentThesis(
            agent_name="ContrarianAgent",
            round=1,
            directional_bias="neutral",
            confidence=0.1,
            reasoning="Contrarian requires other_theses context — call post_contrarian_thesis instead.",
            key_factors=["context_missing"],
            regime_alignment="neutral",
        )
