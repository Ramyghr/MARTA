"""Tests for all 5 JSON parse stages + 5b in BaseAgent._try_parse_json.

Each test crafts a response string that passes ONLY at the targeted stage by ensuring
every earlier stage raises a parse error.
"""

from __future__ import annotations

import json

import pytest

from agents.base import BaseAgent, _safe_default
from agents.schemas import AgentRebuttal, AgentThesis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_THESIS = {
    "agent_name": "TestAgent",
    "round": 1,
    "directional_bias": "bullish",
    "confidence": 0.7,
    "reasoning": "test",
    "key_factors": ["factor1"],
    "regime_alignment": "aligned",
}

_VALID_REBUTTAL = {
    "agent_name": "TestAgent",
    "round": 2,
    "target_agent": "OtherAgent",
    "rebuttal_point": "counter-point",
    "maintains_bias": True,
    "updated_confidence": 0.6,
}


class _ConcreteAgent(BaseAgent):
    """Minimal concrete subclass so we can call _try_parse_json without LLM."""
    name = "TestAgent"

    def post_thesis(self, ctx):  # type: ignore[override]
        raise NotImplementedError


@pytest.fixture()
def agent(monkeypatch):
    """Return a BaseAgent instance with a mocked LLM to avoid Ollama dependency."""
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")  # harmless; no call made
    # Patch ChatOllama so __init__ doesn't try to connect
    import agents.base as base_mod
    from unittest.mock import MagicMock
    monkeypatch.setattr(base_mod, "ChatOllama", MagicMock)
    return _ConcreteAgent()


# ---------------------------------------------------------------------------
# Stage 1: direct JSON parse
# ---------------------------------------------------------------------------

def test_stage1_direct_json(agent):
    raw = json.dumps(_VALID_THESIS)
    result = agent._try_parse_json(raw, AgentThesis)
    assert isinstance(result, AgentThesis)
    assert result.directional_bias == "bullish"


def test_stage1_direct_json_rebuttal(agent):
    raw = json.dumps(_VALID_REBUTTAL)
    result = agent._try_parse_json(raw, AgentRebuttal)
    assert isinstance(result, AgentRebuttal)
    assert result.target_agent == "OtherAgent"


# ---------------------------------------------------------------------------
# Stage 2: markdown code fence  ```json ... ```
# ---------------------------------------------------------------------------

def test_stage2_markdown_fence(agent):
    # Wrapping in backticks makes stage-1 direct parse fail.
    inner = json.dumps(_VALID_THESIS)
    raw = f"Here is the output:\n```json\n{inner}\n```\n"
    result = agent._try_parse_json(raw, AgentThesis)
    assert isinstance(result, AgentThesis)
    assert result.confidence == pytest.approx(0.7)


def test_stage2_plain_fence(agent):
    inner = json.dumps(_VALID_THESIS)
    raw = f"```\n{inner}\n```"
    result = agent._try_parse_json(raw, AgentThesis)
    assert isinstance(result, AgentThesis)


# ---------------------------------------------------------------------------
# Stage 3: first JSON object anywhere in text
# ---------------------------------------------------------------------------

def test_stage3_json_in_text(agent):
    # Preamble + valid JSON + epilogue.  No code fences → stage 2 skips.
    inner = json.dumps(_VALID_THESIS)
    raw = f"My analysis is as follows. {inner} Let me explain."
    result = agent._try_parse_json(raw, AgentThesis)
    assert isinstance(result, AgentThesis)


# ---------------------------------------------------------------------------
# Stage 1b: well-formed JSON that fails only on a max_length constraint
# ---------------------------------------------------------------------------

def test_stage1b_clips_oversized_reasoning(agent):
    # Complete, valid JSON, but `reasoning` exceeds AgentThesis's max_length=500 —
    # found 2026-08-23: ContrarianAgent's prompt never told the model to keep this
    # short, so real Mistral responses reliably tripped this on well-formed JSON.
    long_reasoning = "x" * 600
    raw = (
        '{"agent_name":"TestAgent","round":1,"directional_bias":"bearish",'
        f'"confidence":0.6,"reasoning":"{long_reasoning}","key_factors":["a"],'
        '"regime_alignment":"neutral"}'
    )
    result = agent._try_parse_json(raw, AgentThesis)
    assert isinstance(result, AgentThesis)
    assert result.directional_bias == "bearish"  # real content preserved, not the fallback
    assert len(result.reasoning) <= 500
    assert result.reasoning.endswith("…")


def test_stage1b_clips_oversized_key_factors_list(agent):
    # key_factors has max_length=4 (a list-length constraint, not a string one).
    raw = (
        '{"agent_name":"TestAgent","round":1,"directional_bias":"neutral",'
        '"confidence":0.5,"reasoning":"r",'
        '"key_factors":["a","b","c","d","e","f"],"regime_alignment":"neutral"}'
    )
    result = agent._try_parse_json(raw, AgentThesis)
    assert isinstance(result, AgentThesis)
    assert len(result.key_factors) == 4


def test_stage1b_leaves_valid_fields_untouched(agent):
    # No field violates its constraint — clip should be a no-op, stage 1 handles it.
    raw = json.dumps(_VALID_THESIS)
    result = agent._try_parse_json(raw, AgentThesis)
    assert result.reasoning == "test"


# ---------------------------------------------------------------------------
# Stage 4: greedy extraction + trailing-comma repair
# ---------------------------------------------------------------------------

def test_stage4_trailing_comma(agent):
    # Trailing comma makes the JSON invalid for stages 1-3, but stage 4 repairs it.
    raw = (
        '{"agent_name":"TestAgent","round":1,"directional_bias":"bearish",'
        '"confidence":0.6,"reasoning":"test","key_factors":["a"],  '
        '"regime_alignment":"neutral",}'  # trailing comma before }
    )
    result = agent._try_parse_json(raw, AgentThesis)
    assert isinstance(result, AgentThesis)
    assert result.directional_bias == "bearish"


def test_stage4_trailing_comma_in_array(agent):
    raw = (
        '{"agent_name":"TestAgent","round":1,"directional_bias":"neutral",'
        '"confidence":0.5,"reasoning":"r","key_factors":["x","y",],'
        '"regime_alignment":"neutral"}'
    )
    result = agent._try_parse_json(raw, AgentThesis)
    assert isinstance(result, AgentThesis)


# ---------------------------------------------------------------------------
# Stage 5: collapse literal newlines inside JSON block
# ---------------------------------------------------------------------------

def test_stage5_literal_newlines_in_values(agent):
    # Embed real newline characters inside a string value — this breaks standard
    # JSON parsers (stages 1-4) but stage 5 collapses them and repairs.
    raw = (
        '{"agent_name":"TestAgent","round":1,"directional_bias":"bullish",'
        '"confidence":0.65,"reasoning":"line one\nline two","key_factors":["a"],'
        '"regime_alignment":"aligned"}'
    )
    result = agent._try_parse_json(raw, AgentThesis)
    assert isinstance(result, AgentThesis)
    assert result.directional_bias == "bullish"


# ---------------------------------------------------------------------------
# Stage 5b: AgentThesis field repair — missing required fields
# ---------------------------------------------------------------------------

def test_stage5b_missing_regime_alignment(agent):
    # regime_alignment is required by AgentThesis; stage 5b fills it in.
    raw = (
        '{"agent_name":"TestAgent","round":1,"directional_bias":"bullish",'
        '"confidence":0.7,"reasoning":"r","key_factors":["a"]}'
        # no regime_alignment
    )
    result = agent._try_parse_json(raw, AgentThesis)
    assert isinstance(result, AgentThesis)
    assert result.regime_alignment == "neutral"  # default filled by 5b


def test_stage5b_missing_key_factors(agent):
    raw = (
        '{"agent_name":"TestAgent","round":1,"directional_bias":"neutral",'
        '"confidence":0.5,"reasoning":"r"}'
        # no key_factors, no regime_alignment
    )
    result = agent._try_parse_json(raw, AgentThesis)
    assert isinstance(result, AgentThesis)
    assert result.key_factors == []


# ---------------------------------------------------------------------------
# Total failure → safe neutral default
# ---------------------------------------------------------------------------

def test_total_failure_returns_safe_default(agent):
    result = agent._try_parse_json("this is not json at all!!!", AgentThesis)
    assert isinstance(result, AgentThesis)
    assert result.directional_bias == "neutral"
    assert result.confidence == pytest.approx(0.1)


def test_empty_response_returns_safe_default(agent):
    result = agent._try_parse_json("", AgentThesis)
    assert isinstance(result, AgentThesis)
    assert result.directional_bias == "neutral"


def test_safe_default_rebuttal(agent):
    result = _safe_default(AgentRebuttal, "TestAgent")
    assert isinstance(result, AgentRebuttal)
    assert result.maintains_bias is True


# ---------------------------------------------------------------------------
# Agent name normalization
# ---------------------------------------------------------------------------

def test_agent_name_aliases_resolve(agent):
    result = agent._normalize_agent_name("macro agent")
    assert result == "MacroAgent"


def test_unknown_alias_passthrough(agent):
    result = agent._normalize_agent_name("SomeNewAgent")
    assert result == "SomeNewAgent"


def test_debug_aliases_removed(agent):
    """F-17 regression: 'x' and 'y' must not be in the alias table."""
    from agents.base import _AGENT_NAME_ALIASES
    assert "x" not in _AGENT_NAME_ALIASES
    assert "y" not in _AGENT_NAME_ALIASES
