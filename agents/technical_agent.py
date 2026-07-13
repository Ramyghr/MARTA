"""TechnicalAgent — regime-aware technical analysis with TFT encoder attention map."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from encoder.model import TFTRegimeEncoder
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TFTRegimeEncoder = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from agents.base import BaseAgent
from agents.context import RegimeContext, serialize_context_for_prompt
from agents.schemas import AgentThesis
from utils.logger import logger

_MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent)))
_CHECKPOINT = _MARTA_ROOT / "encoder/checkpoints/best_encoder.pt"
_FEATURES = [
    "atr", "adx", "rsi", "bb_width", "macd_histogram",
    "obv_raw", "vol_zscore", "realized_vol", "vol_ratio",
    "return_skew", "return_kurtosis", "autocorr_lag1", "realized_vol_raw", "close_return",
]
_WINDOW_SIZE = 60


class TechnicalAgent(BaseAgent):
    name = "TechnicalAgent"

    def __init__(self, model: str = "mistral:7b-instruct", temperature: float = 0.4) -> None:
        super().__init__(model=model, temperature=temperature)
        self._encoder = None
        self._device = torch.device("cpu") if _TORCH_AVAILABLE else None
        self._load_encoder()

    def _load_encoder(self) -> None:
        if not _TORCH_AVAILABLE:
            logger.warning("TechnicalAgent: torch not installed — encoder disabled, momentum-only mode")
            return
        try:
            ckpt = torch.load(_CHECKPOINT, map_location=self._device, weights_only=True)
            self._encoder = TFTRegimeEncoder(input_dim=14)
            self._encoder.load_state_dict(ckpt["model_state"])
            self._encoder.eval()
            logger.info("TechnicalAgent: encoder loaded from checkpoint")
        except Exception as e:
            logger.warning(f"TechnicalAgent: could not load encoder — {e}. Momentum-only mode.")
            self._encoder = None

    def _get_vsn_attention(self, aligned_df: pd.DataFrame, timestamp: pd.Timestamp) -> str:
        """Run 60-candle window through encoder, return top-3 attended features as string."""
        if self._encoder is None:
            return "Encoder attention: unavailable"

        idx_tz = aligned_df.index.tz
        ts_utc = pd.Timestamp(timestamp).tz_convert("UTC") if pd.Timestamp(timestamp).tzinfo else pd.Timestamp(timestamp).tz_localize("UTC")
        ts_local = ts_utc.tz_convert(idx_tz) if idx_tz else ts_utc.tz_localize(None)

        aligned_sorted = aligned_df.sort_index()
        loc = aligned_sorted.index.get_indexer([ts_local], method="pad")[0]
        if loc < 0:
            return "Encoder attention: index error"

        start = max(0, loc - _WINDOW_SIZE + 1)
        window = aligned_sorted.iloc[start : loc + 1][_FEATURES].fillna(0.0)

        if len(window) < 2:
            return "Encoder attention: insufficient data"

        # Pad to WINDOW_SIZE if needed
        if len(window) < _WINDOW_SIZE:
            pad = pd.DataFrame(0.0, index=range(_WINDOW_SIZE - len(window)), columns=_FEATURES)
            window = pd.concat([pad, window], ignore_index=True)

        if not _TORCH_AVAILABLE or self._encoder is None:
            return "Encoder attention: torch unavailable — momentum-only mode"

        x = torch.tensor(window.values, dtype=torch.float32).unsqueeze(0)  # (1, T, 14)

        try:
            with torch.no_grad():
                vsn_weights = self._encoder.get_vsn_weights(x)  # (1, T, 14)
                importance = vsn_weights.squeeze(0).mean(dim=0).numpy()  # (14,)

            top3_idx = np.argsort(importance)[::-1][:3]
            parts = [f"{_FEATURES[i]} ({importance[i]:.2f})" for i in top3_idx]
            return "Encoder attention: " + ", ".join(parts)
        except Exception as e:
            logger.warning(f"TechnicalAgent VSN error: {e}")
            return "Encoder attention: computation error"

    def _build_system_prompt(self, ctx: RegimeContext) -> str:
        return (
            "You are the Technical Pattern Agent for a quantitative trading system.\n"
            "You have access to the TFT encoder's attention map showing which features "
            "the regime encoder found most significant. Ground your technical analysis "
            "in what the encoder is attending to, combined with the indicator values.\n"
            "You MUST respond with a single JSON object — no preamble, no markdown, no explanation outside the JSON.\n\n"
            + serialize_context_for_prompt(ctx)
        )

    def post_thesis(
        self,
        ctx: RegimeContext,
        aligned_df: pd.DataFrame | None = None,
    ) -> AgentThesis:
        vsn_str = "Encoder attention: unavailable"
        if aligned_df is not None:
            vsn_str = self._get_vsn_attention(aligned_df, ctx.timestamp)

        features_str = "\n".join(f"  {k}: {v:.4f}" for k, v in ctx.price_features.items())
        tech_context = (
            f"\n=== TECHNICAL DATA ===\n"
            f"{vsn_str}\n"
            f"Current indicators:\n{features_str}\n"
            f"RSI: {ctx.price_features.get('rsi', 0):.2f} | "
            f"ADX: {ctx.price_features.get('adx', 0):.2f} | "
            f"MACD hist: {ctx.price_features.get('macd_histogram', 0):.4f}\n"
        )

        system_prompt = self._build_system_prompt(ctx) + tech_context

        user_prompt = (
            f"Assess directional bias for {ctx.asset} from the encoder attention and indicators.\n\n"
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
