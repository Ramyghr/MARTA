"""RiskAgent — GARCH(1,1) volatility forecasting and hard risk limits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from agents.base import BaseAgent
from agents.context import RegimeContext, serialize_context_for_prompt
from agents.schemas import RiskLimits
from utils.assets import aligned_path
from utils.logger import logger

_DATA_DIR = Path("data/processed")
_GARCH_WINDOW = 90 * 24  # 90 days of hourly data


def _classify_risk_regime(vol: float) -> str:
    if vol < 0.01:
        return "low"
    if vol < 0.02:
        return "medium"
    if vol < 0.04:
        return "high"
    return "extreme"


def _event_risk_multiplier(upcoming_events: list | None) -> float:
    """Quantitative throttle on max_position_size ahead of high-impact economic
    releases (FOMC, CPI, NFP, ...) — a hard, auditable cap, not just something the LLM
    agents might notice in their prompt context (they also see the same event list via
    serialize_context_for_prompt, for the narrative side of this).

    `upcoming_events` is None for historical/replay debates (a past debate can't know
    the future's calendar) — correctly falls through to no throttle (1.0).
    """
    if not upcoming_events:
        return 1.0
    high_within_6h = any(
        e["impact"] == "high" and e["hours_until"] <= 6 for e in upcoming_events
    )
    if high_within_6h:
        return 0.5
    high_within_24h = any(
        e["impact"] == "high" and e["hours_until"] <= 24 for e in upcoming_events
    )
    return 0.75 if high_within_24h else 1.0


class RiskAgent(BaseAgent):
    name = "RiskAgent"

    def __init__(self, model: str = "mistral:7b-instruct", temperature: float = 0.4) -> None:
        super().__init__(model=model, temperature=temperature)
        self._aligned_cache: dict[str, pd.DataFrame] = {}

    def _load_aligned(self, asset: str) -> pd.DataFrame:
        if asset not in self._aligned_cache:
            path = aligned_path(asset, _DATA_DIR)
            if path.exists():
                self._aligned_cache[asset] = pd.read_parquet(path, engine="pyarrow")
            else:
                self._aligned_cache[asset] = pd.DataFrame()
        return self._aligned_cache[asset]

    def _fit_garch(self, returns: np.ndarray) -> float:
        """Fit GARCH(1,1) and return next-period volatility forecast. Falls back to rolling std."""
        try:
            from arch import arch_model
            # arch expects percentage returns for numerical stability
            scaled = returns * 100.0
            am = arch_model(scaled, vol="Garch", p=1, q=1, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            forecast = res.forecast(horizon=1, reindex=False)
            vol_pct = float(np.sqrt(forecast.variance.values[-1, 0]))
            return vol_pct / 100.0  # back to decimal
        except Exception as e:
            logger.warning(f"RiskAgent GARCH failed: {e} — using rolling std fallback")
            return float(np.std(returns)) if len(returns) > 1 else 0.02

    def compute_risk_limits(self, timestamp: pd.Timestamp, asset: str) -> RiskLimits:
        """Compute risk limits via GARCH. Pure Python — no LLM involved."""
        ts_utc = pd.Timestamp(timestamp).tz_convert("UTC") if pd.Timestamp(timestamp).tzinfo else pd.Timestamp(timestamp).tz_localize("UTC")

        df = self._load_aligned(asset)
        if df.empty or "close_return" not in df.columns:
            return _default_risk_limits()

        idx_tz = df.index.tz
        ts_local = ts_utc.tz_convert(idx_tz) if idx_tz else ts_utc.tz_localize(None)
        loc = df.index.get_indexer([ts_local], method="pad")[0]
        if loc < 10:
            return _default_risk_limits()

        start = max(0, loc - _GARCH_WINDOW + 1)
        returns = df.iloc[start : loc + 1]["close_return"].dropna().values

        if len(returns) < 20:
            return _default_risk_limits()

        forecast_vol = self._fit_garch(returns)
        forecast_vol = max(forecast_vol, 1e-6)  # guard against zero

        var_95 = float(norm.ppf(0.05) * forecast_vol)
        cvar_95 = float(-norm.pdf(norm.ppf(0.05)) / 0.05 * forecast_vol)
        max_pos = float(min(1.0, 0.02 / max(abs(cvar_95), 1e-6)))
        risk_regime = _classify_risk_regime(forecast_vol)

        return RiskLimits(
            max_position_size=round(max_pos, 4),
            leverage_cap=min(5.0, max(1.0, round(1.0 / max(forecast_vol * 10, 0.2), 2))),
            var_95=round(var_95, 6),
            cvar_95=round(cvar_95, 6),
            volatility_forecast=round(forecast_vol, 6),
            risk_regime=risk_regime,
        )

    def _build_system_prompt(self, ctx: RegimeContext) -> str:
        return (
            "You are the Risk Management Agent for a quantitative trading system.\n"
            "Your role is to assess risk conditions and flag unusual volatility events.\n"
            "You MUST respond with a single JSON object — no preamble, no markdown.\n\n"
            + serialize_context_for_prompt(ctx)
        )

    def get_risk_limits(self, ctx: RegimeContext) -> RiskLimits:
        """Compute GARCH-based risk limits, then throttle down for an imminent
        high-impact economic event. The LLM is not used for either step."""
        limits = self.compute_risk_limits(ctx.timestamp, ctx.asset)
        multiplier = _event_risk_multiplier(ctx.upcoming_events)
        if multiplier < 1.0:
            limits = limits.model_copy(update={
                "max_position_size": round(limits.max_position_size * multiplier, 4),
                "event_risk_multiplier": multiplier,
            })
        return limits


def _default_risk_limits() -> RiskLimits:
    return RiskLimits(
        max_position_size=0.1,
        leverage_cap=1.0,
        var_95=-0.03,
        cvar_95=0.04,
        volatility_forecast=0.02,
        risk_regime="medium",
    )
