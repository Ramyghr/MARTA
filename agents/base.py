"""BaseAgent — parent class for all MARTA deliberation agents."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agents.context import RegimeContext, serialize_context_for_prompt
from agents.schemas import AgentRebuttal, AgentThesis
from utils.logger import logger

_DEFAULT_MODEL = "mistral:7b-instruct"
_TIMEOUT_SECONDS = 300

# Allow overriding the Ollama base URL via env var so Docker containers can
# reach an Ollama instance running on the host (host.docker.internal).
_OLLAMA_BASE_URL: str | None = os.environ.get("OLLAMA_HOST")  # e.g. http://host.docker.internal:11434

# Mistral frequently invents agent name variants — map them back to canonical names.
# None means "unknown, leave as-is".
_AGENT_NAME_ALIASES: dict[str, str | None] = {
    "macro correlation agent": "MacroAgent",
    "macro agent": "MacroAgent",
    "macroagent": "MacroAgent",
    "technical pattern agent": "TechnicalAgent",
    "technical agent": "TechnicalAgent",
    "tfa": "TechnicalAgent",
    "technicalagent": "TechnicalAgent",
    "sentiment agent": "SentimentAgent",
    "sentimentagent": "SentimentAgent",
    "finbert": "SentimentAgent",
    "contrarian agent": "ContrarianAgent",
    "devil's advocate": "ContrarianAgent",
    "contrarian": "ContrarianAgent",
    "fund manager": "FundManagerAgent",
}


# Sized for worst-case zombie accumulation, not steady-state concurrency: a
# ThreadPoolExecutor future can only be cancelled before it starts running (see
# _call_llm below) — once an Ollama call is in flight, which it almost always is by
# the time a 300s timeout fires, future.cancel() silently no-ops and the thread keeps
# running in the background against this shared pool. A single debate can have up to
# 6 LLM-calling agents (everything except RiskAgent, which is GARCH-only) x 2 attempts
# each = 12 potential zombies in the worst case where every call times out once; the
# old max_workers=4 could be fully starved by less than one debate's worst case,
# stalling every subsequent LLM call project-wide until Ollama eventually responds (or
# hangs indefinitely). No code here can force-kill a Python thread — this just gives
# real work enough headroom that zombies from timed-out calls don't block it.
_THREAD_POOL = ThreadPoolExecutor(max_workers=32)


class BaseAgent:
    name: str = "base"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.4,
        timeout_seconds: int = _TIMEOUT_SECONDS,
        max_attempts: int = 2,
        num_predict: int = 600,
    ) -> None:
        kwargs: dict = dict(model=model, temperature=temperature, num_predict=num_predict, format="json")
        if _OLLAMA_BASE_URL:
            kwargs["base_url"] = _OLLAMA_BASE_URL
        self.llm = ChatOllama(**kwargs)
        self._token_count: int = 0
        # Per-agent overridable timeout/retry/token budget — see ContrarianAgent for an
        # example that widens all three (its prompt + historical-analogue context makes
        # slow, longer responses more likely, and losing it to a timeout or truncated-JSON
        # parse failure is more costly: the fallback neutral thesis silences the one agent
        # whose whole job is to catch groupthink). A 2026-08 audit found ContrarianAgent's
        # response was hitting the (until-now shared) 600-token num_predict cap and getting
        # cut off mid-JSON in ~40% of its activations across a validated 81-debate cache.
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
    @staticmethod
    def _repair_invalid_escapes(text: str) -> str:
        """Mistral sometimes emits Python-style \\' inside JSON strings, which is invalid JSON
        (only \\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX are legal escapes). Un-escape stray \\'."""
        return re.sub(r"\\'", "'", text)
    @property
    def tokens_used(self) -> int:
        return self._token_count

    def _build_system_prompt(self, ctx: RegimeContext) -> str:
        """Prepend regime context + agent-specific instructions. Override in subclasses."""
        return serialize_context_for_prompt(ctx)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM with timeout guard and up to self._max_attempts tries. Returns raw
        string or empty string once attempts are genuinely exhausted (the empty string
        is what makes _try_parse_json fall back to _safe_default — never fall back early).
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        def _invoke() -> str:
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self._token_count += response.usage_metadata.get("total_tokens", 0)
            else:
                self._token_count += len(content) // 4
            return content

        for attempt in range(self._max_attempts):
            try:
                future = _THREAD_POOL.submit(_invoke)
                return future.result(timeout=self._timeout_seconds)
            except FuturesTimeoutError:
                # Best-effort only — a future can't be cancelled once it's running (see
                # the pool-sizing comment above), so this is a no-op in the common case
                # where the Ollama call is already in flight. The abandoned thread keeps
                # running against _THREAD_POOL until Ollama actually responds or hangs.
                future.cancel()
                if attempt < self._max_attempts - 1:
                    logger.warning(
                        f"[{self.name}] LLM timed out (attempt {attempt + 1}/{self._max_attempts}) — retrying"
                    )
                else:
                    logger.warning(
                        f"[{self.name}] LLM timed out after {self._max_attempts} attempts — returning empty"
                    )
                    return ""
            except Exception as e:
                logger.error(f"[{self.name}] LLM call failed: {e}")
                return ""
        return ""

    def _normalize_agent_name(self, name: str) -> str:
        """Map Mistral's invented agent name variations to canonical names."""
        resolved = _AGENT_NAME_ALIASES.get(name.lower().strip())
        return resolved if resolved is not None else name

    def _normalize_agent_fields(self, result: BaseModel) -> BaseModel:
        """Normalize agent_name / target_agent fields to canonical names after any parse."""
        if isinstance(result, AgentThesis):
            result.agent_name = self._normalize_agent_name(result.agent_name)
        elif isinstance(result, AgentRebuttal):
            # agent_name for rebuttals is forced in post_rebuttal; normalize target only
            result.target_agent = self._normalize_agent_name(result.target_agent)
        return result

    def _parse_json_response(self, response: str, schema: type[BaseModel]) -> BaseModel:
        """Parse LLM response into Pydantic schema, then normalize agent names. Never raises."""
        result = self._try_parse_json(response, schema)
        return self._normalize_agent_fields(result)

    def _try_parse_json(self, response: str, schema: type[BaseModel]) -> BaseModel:
        if not response:
            return _safe_default(schema, self.name)
        response = self._repair_invalid_escapes(response)

        # 1. Try direct JSON parse
        try:
            return schema.model_validate_json(response.strip())
        except Exception:
            pass

        # 1b. Well-formed JSON that fails only on a max_length constraint (e.g. a long
        # "reasoning" string) — clip oversized fields and retry, instead of falling all
        # the way to the lossy brace-balancing repair stages below. Found 2026-08-23:
        # ContrarianAgent's prompt never told the model to keep `reasoning` short (every
        # other agent's does), so it reliably produced complete, valid JSON that Pydantic
        # rejected purely on length — a 100% miss rate on a 3-call sample, not a parsing
        # failure at all. The prompt was fixed too, but this stage makes recovery robust
        # to the same failure class on ANY max_length-constrained field, for any agent.
        try:
            data = json.loads(response.strip())
            if isinstance(data, dict):
                clipped = _clip_to_schema_constraints(data, schema)
                if clipped != data:
                    return schema.model_validate(clipped)
        except Exception:
            pass

        # 2. Extract JSON from markdown code fence ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if fence_match:
            try:
                return schema.model_validate_json(fence_match.group(1))
            except Exception:
                pass

        # 3. Extract first JSON object found anywhere in the response
        brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", response, re.DOTALL)
        if brace_match:
            try:
                return schema.model_validate_json(brace_match.group(0))
            except Exception:
                pass

        # 4. Greedy extraction: find outermost {} block + trailing-comma repair
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = response[start : end + 1]
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return schema.model_validate_json(candidate)
            except Exception:
                pass

        # 5. Last-resort: collapse literal newlines inside the extracted block
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = response[start : end + 1]
            candidate = re.sub(r"(?<!\\)\n", " ", candidate)
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return schema.model_validate_json(candidate)
            except Exception:
                pass

        # 5b. AgentThesis field repair — fill missing required fields before Pydantic validation
        if schema is AgentThesis:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    candidate = response[start : end + 1]
                    candidate = re.sub(r"(?<!\\)\n", " ", candidate)
                    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                    d = json.loads(candidate)
                    d.setdefault("regime_alignment", "neutral")
                    d.setdefault("round", 1)
                    d.setdefault("key_factors", [])
                    return schema.model_validate(d)
                except Exception:
                    pass
        # 5c. AgentRebuttal field repair — fill missing required fields before Pydantic validation
        if schema is AgentRebuttal:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    candidate = response[start : end + 1]
                    candidate = re.sub(r"(?<!\\)\n", " ", candidate)
                    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                    d = json.loads(candidate)
                    d.setdefault("round", 2)
                    d.setdefault("target_agent", "unknown")
                    d.setdefault("maintains_bias", True)
                    d.setdefault("updated_confidence", 0.5)
                    return schema.model_validate(d)
                except Exception:
                    pass

        # 6. Truncation repair: response cut off mid-string — close the string and brace.
        start = response.find("{")
        if start != -1:
            candidate = response[start:]
            # Count unescaped quotes; if odd, we're mid-string — close it.
            quote_count = len(re.findall(r'(?<!\\)"', candidate))
            if quote_count % 2 == 1:
                candidate += '"'
            # Balance braces.
            open_braces = candidate.count("{")
            close_braces = candidate.count("}")
            candidate += "}" * max(0, open_braces - close_braces)
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                d = json.loads(candidate)
                if schema is AgentThesis:
                    d.setdefault("regime_alignment", "neutral")
                    d.setdefault("round", 1)
                    d.setdefault("key_factors", [])
                elif schema is AgentRebuttal:
                    d.setdefault("round", 2)
                    d.setdefault("target_agent", "unknown")
                    d.setdefault("maintains_bias", True)
                    d.setdefault("updated_confidence", 0.5)
                return schema.model_validate(d)
            except Exception:
                pass

        logger.warning(
            f"[{self.name}] JSON parse failed — using safe default for {schema.__name__}. "
            f"Raw response (first 500 chars): {response[:500]!r}"
        )
        return _safe_default(schema, self.name)

    def post_thesis(self, ctx: RegimeContext) -> AgentThesis:
        """Round 1 — post independent thesis. Override in subclasses."""
        raise NotImplementedError

    def post_rebuttal(
        self,
        ctx: RegimeContext,
        other_theses: list[AgentThesis],
        my_r1_thesis: AgentThesis | None = None,
    ) -> AgentRebuttal | None:
        """Round 2 — rebuttal against a single opposing thesis. Return None to abstain.

        Pass my_r1_thesis to avoid a redundant post_thesis() LLM call.
        """
        if not other_theses:
            return None

        my_thesis = my_r1_thesis or self.post_thesis(ctx)
        opposing = [t for t in other_theses if t.directional_bias != my_thesis.directional_bias]
        if not opposing:
            return None
        target = max(opposing, key=lambda t: t.confidence)

        system_prompt = self._build_system_prompt(ctx) + (
            f"\n\nRound 2 rebuttal. Your view: {my_thesis.directional_bias} (conf={my_thesis.confidence:.2f}).\n"
            f"{target.agent_name} argues {target.directional_bias}.\n"
            "Reply ONLY with this JSON (fill rebuttal_point and updated_confidence):\n"
            
            + '{"agent_name":"'
            + self.name
            + '","round":2,"target_agent":"'
            + target.agent_name
            + '","rebuttal_point":"<30 words>",'
            + '"maintains_bias":<true if you still hold your original view, false if the rebuttal changed your mind>,'
            + '"updated_confidence":<your updated confidence as a float between 0.0 and 1.0>}'
        )
        user_prompt = f"Rebut {target.agent_name}."

        raw = self._call_llm(system_prompt, user_prompt)
        result = self._parse_json_response(raw, AgentRebuttal)
        if isinstance(result, AgentRebuttal):
            result.agent_name = self.name  # always canonical — LLM may output wrong name
            return result
        return None


def _clip_to_schema_constraints(data: dict, schema: type[BaseModel]) -> dict:
    """Best-effort clip of string/list fields that violate a Field(max_length=...)
    constraint, so a well-formed-but-verbose LLM response isn't discarded just because
    one field ran long. Introspects the schema's own field metadata (pydantic v2's
    annotated_types.MaxLen) rather than hardcoding field names/limits per schema, so it
    covers every current and future max_length field (AgentThesis.reasoning,
    AgentRebuttal.rebuttal_point, FundManagerDecision.rationale, etc.) with one
    implementation. Only touches fields that are actually oversized; matches JSON
    schema validation, list truncates.
    """
    clipped = dict(data)
    for name, field in schema.model_fields.items():
        if name not in clipped:
            continue
        max_len = next(
            (m.max_length for m in getattr(field, "metadata", []) if hasattr(m, "max_length")),
            None,
        )
        if max_len is None:
            continue
        value = clipped[name]
        if isinstance(value, str) and len(value) > max_len:
            clipped[name] = value[: max_len - 1].rstrip() + "…"
        elif isinstance(value, list) and len(value) > max_len:
            clipped[name] = value[:max_len]
    return clipped


def _safe_default(schema: type[BaseModel], agent_name: str) -> BaseModel:
    """Return a minimal valid instance of a schema as a fallback."""
    if schema is AgentThesis:
        return AgentThesis(
            agent_name=agent_name,
            round=1,
            directional_bias="neutral",
            confidence=0.1,
            reasoning="LLM response unavailable — defaulting to neutral.",
            key_factors=["system_error"],
            regime_alignment="neutral",
        )
    if schema is AgentRebuttal:
        return AgentRebuttal(
            agent_name=agent_name,
            round=2,
            target_agent="unknown",
            rebuttal_point="LLM response unavailable.",
            maintains_bias=True,
            updated_confidence=0.1,
        )
    # Generic fallback: try to construct with minimal fields
    try:
        return schema.model_construct()
    except Exception:
        raise ValueError(f"Cannot construct safe default for {schema.__name__}")
