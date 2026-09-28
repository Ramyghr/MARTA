"""MacroAgent — cross-asset correlation and contagion detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agents.base import BaseAgent
from agents.context import RegimeContext, serialize_context_for_prompt
from agents.schemas import AgentThesis
from utils.assets import aligned_path
from utils.logger import logger

# Kept at the original 4-asset production panel — dashboard/backend/debate_stream.py
# constructs MacroAgent() with no arguments and relies on this default matching the old
# production data/processed/regime_stream.parquet it hardcodes elsewhere; changing this
# default would silently break the live dashboard. Crypto-only callers (run_debate.py)
# pass assets=/data_dir= explicitly instead — see utils.freshness.DEBATE_ASSETS.
_ASSETS = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]
_DATA_DIR = Path("data/processed")
_CORR_WINDOW = 168  # 1 week of hourly data


class MacroAgent(BaseAgent):
    name = "MacroAgent"

    def __init__(
        self,
        model: str = "mistral:7b-instruct",
        temperature: float = 0.4,
        assets: list[str] | None = None,
        data_dir: Path | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature)
        self._assets = assets if assets is not None else _ASSETS
        self._data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
        self._aligned_cache: dict[str, pd.DataFrame] = {}

    def _load_aligned(self, asset: str) -> pd.DataFrame:
        if asset not in self._aligned_cache:
            path = aligned_path(asset, self._data_dir)
            if path.exists():
                self._aligned_cache[asset] = pd.read_parquet(path, engine="pyarrow")
            else:
                self._aligned_cache[asset] = pd.DataFrame()
        return self._aligned_cache[asset]

    def _compute_correlation_data(self, timestamp: pd.Timestamp) -> dict:
        """Pure-Python cross-asset correlation metrics. No LLM involved."""
        ts_utc = pd.Timestamp(timestamp).tz_convert("UTC") if pd.Timestamp(timestamp).tzinfo else pd.Timestamp(timestamp).tz_localize("UTC")

        returns: dict[str, pd.Series] = {}
        vol_ratios: dict[str, float] = {}

        for asset in self._assets:
            df = self._load_aligned(asset)
            if df.empty or "close_return" not in df.columns:
                continue
            idx_tz = df.index.tz
            ts_local = ts_utc.tz_convert(idx_tz) if idx_tz else ts_utc.tz_localize(None)
            loc = df.index.get_indexer([ts_local], method="pad")[0]
            if loc < 0:
                continue
            start = max(0, loc - _CORR_WINDOW + 1)
            window = df.iloc[start : loc + 1]
            returns[asset] = window["close_return"].reset_index(drop=True)
            if "vol_ratio" in window.columns:
                vol_ratios[asset] = float(window["vol_ratio"].iloc[-1])

        if len(returns) < 2:
            return {"correlation_matrix": {}, "top_pair": ("N/A", "N/A", 0.0), "contagion": False, "vol_ratios": vol_ratios}

        ret_df = pd.DataFrame(returns).dropna()
        corr_matrix = ret_df.corr() if len(ret_df) > 5 else pd.DataFrame()

        # Highest absolute correlation pair
        top_pair = ("N/A", "N/A", 0.0)
        if not corr_matrix.empty:
            # `.values`/`.to_numpy()` without copy=True can return a read-only array
            # under pandas' Copy-on-Write mode (always-on since pandas 3.0, which the
            # dashboard backend's docker image resolves via requirements.txt's
            # unpinned `pandas>=2.0.0` — the conda `marta` env still runs 2.3.3, where
            # this doesn't reproduce). np.fill_diagonal writes in place, so it needs a
            # genuinely owned, writable buffer, not a CoW view.
            corr_values = corr_matrix.to_numpy(copy=True)
            np.fill_diagonal(corr_values, 0.0)
            corr_matrix = pd.DataFrame(corr_values, index=corr_matrix.index, columns=corr_matrix.columns)
            flat = corr_matrix.abs().stack()
            if not flat.empty:
                idx_top = flat.idxmax()
                top_pair = (str(idx_top[0]), str(idx_top[1]), float(corr_matrix.loc[idx_top[0], idx_top[1]]))

        # Contagion: 3+ assets with pairwise correlation > 0.7
        contagion = False
        if not corr_matrix.empty:
            high_corr_assets = set()
            for a1 in corr_matrix.columns:
                for a2 in corr_matrix.columns:
                    if a1 != a2 and abs(corr_matrix.loc[a1, a2]) > 0.7:
                        high_corr_assets.add(a1)
                        high_corr_assets.add(a2)
            contagion = len(high_corr_assets) >= 3

        corr_summary = {}
        if not corr_matrix.empty:
            for a1 in corr_matrix.columns:
                for a2 in corr_matrix.columns:
                    if a1 < a2:
                        corr_summary[f"{a1}_{a2}"] = round(float(corr_matrix.loc[a1, a2]), 3)

        return {
            "correlation_matrix": corr_summary,
            "top_pair": top_pair,
            "contagion": contagion,
            "vol_ratios": vol_ratios,
        }

    def _build_system_prompt(self, ctx: RegimeContext) -> str:
        return (
            "You are the Macro Correlation Agent for a quantitative trading system.\n"
            "Your role is to detect cross-asset contagion and macro regime shifts.\n"
            "You receive quantitative correlation data and must assess systemic risk.\n"
            "You MUST respond with a single JSON object — no preamble, no markdown, no explanation outside the JSON.\n\n"
            + serialize_context_for_prompt(ctx)
        )

    def post_thesis(self, ctx: RegimeContext) -> AgentThesis:
        corr_data = self._compute_correlation_data(ctx.timestamp)

        top_a, top_b, top_corr = corr_data["top_pair"]
        contagion_flag = corr_data["contagion"]
        vol_ratios_str = ", ".join(f"{a}={v:.2f}" for a, v in corr_data["vol_ratios"].items())

        # Send only the 5 highest absolute correlations to keep prompt compact
        top5_pairs = sorted(corr_data["correlation_matrix"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top5_str = ", ".join(f"{k}={v}" for k, v in top5_pairs)

        macro_context = (
            f"\n=== MACRO DATA ===\n"
            f"Top corr pair: {top_a}/{top_b}={top_corr:.3f} | Contagion: {contagion_flag}\n"
            f"Vol ratios: {vol_ratios_str}\n"
            f"Top correlations: {top5_str}\n"
        )

        system_prompt = self._build_system_prompt(ctx) + macro_context

        user_prompt = (
            f"Assess directional bias for {ctx.asset} given the correlation data and regime.\n\n"
            "Reply ONLY with this JSON schema — replace every placeholder with your own analysis, "
            "do not copy the placeholder values:\n"
            '{"agent_name":"MacroAgent","round":1,"directional_bias":"<bullish|bearish|neutral, your judgment>",'
            '"confidence":<your confidence as a float between 0.0 and 1.0>,'
            '"reasoning":"<your 1-2 sentence reasoning>","key_factors":["<factor1>","<factor2>"],'
            '"regime_alignment":"<aligned|contrary|neutral|mixed>"}'
        )

        raw = self._call_llm(system_prompt, user_prompt)
        result = self._parse_json_response(raw, AgentThesis)
        if isinstance(result, AgentThesis):
            result.agent_name = "MacroAgent"
            result.round = 1
            return result
        return result  # safe default
