"""Pydantic v2 output schemas for all agents in the MARTA deliberation layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentThesis(BaseModel):
    agent_name: str
    round: Literal[1, 2]
    directional_bias: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=500)
    key_factors: list[str] = Field(max_length=4)
    regime_alignment: Literal["aligned", "contrary", "neutral", "mixed"]


class AgentRebuttal(BaseModel):
    agent_name: str
    round: Literal[2]
    target_agent: str
    rebuttal_point: str = Field(max_length=400)
    maintains_bias: bool
    updated_confidence: float = Field(ge=0.0, le=1.0)


class RiskLimits(BaseModel):
    max_position_size: float = Field(ge=0.0, le=1.0)
    leverage_cap: float = Field(ge=1.0, le=5.0)
    var_95: float
    cvar_95: float
    volatility_forecast: float
    risk_regime: Literal["low", "medium", "high", "extreme"]
    event_risk_multiplier: float = 1.0  # <1.0 when a high-impact economic event is
    # imminent (see RiskAgent._event_risk_multiplier) — already folded into
    # max_position_size below; kept as its own field for auditability, same pattern
    # FundManagerDecision.regime_penalty_applied uses.


class FundManagerDecision(BaseModel):
    final_action: Literal["long", "short", "hold"]
    position_size: float = Field(ge=0.0, le=1.0)
    conviction: float = Field(ge=0.0, le=1.0)
    consensus_direction: Literal["bullish", "bearish", "neutral", "split"]
    dissenting_agents: list[str]
    rationale: str = Field(max_length=600)
    regime_override: bool
    regime_penalty_applied: float = 1.0
    regime_contradiction: bool = False


class DebateResult(BaseModel):
    timestamp: str
    asset: str
    regime_label: int
    round1_theses: list[AgentThesis]
    round2_rebuttals: list[AgentRebuttal]
    risk_limits: RiskLimits
    fund_manager_decision: FundManagerDecision
    total_tokens_used: int
    debate_duration_seconds: float
