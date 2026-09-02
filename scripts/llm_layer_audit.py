"""LLM agent layer audit — traces every agent from prompt construction through parsing
to final action, then runs schema / regime-sensitivity / extreme-signal / null-regime /
staleness tests plus an aggregator-usage check.

Makes real Ollama calls (mistral:7b-instruct) — budget ~10-15 minutes.

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python scripts/llm_layer_audit.py

Output: data/results/llm_layer_audit.json (full trace) + a printed pass/fail table.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.context import RegimeContext, build_regime_context, classify_regime_volatility_level, _load_regime_stats
from agents.contrarian_agent import ContrarianAgent
from agents.fund_manager import FundManagerAgent
from agents.macro_agent import MacroAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from agents.schemas import AgentThesis
from utils.freshness import CRYPTO5_DEBATE_ASSETS, CRYPTO5_PROCESSED, CRYPTO5_SEMANTICS_PATH
from utils.logger import logger

ASSET = "BTC_USDT"
T0 = pd.Timestamp("2024-03-01 00:00:00", tz="UTC")       # exact match in both aligned_df + regime_stream
# Crypto data has no forex-weekend-style gaps, so this no longer deliberately lands in a
# data gap the way it did on the old 4-asset panel — kept as a fixed regression point
# (still an exact match on the crypto5 panel), but the staleness test below will now
# consistently report "not stale" rather than exercising the gap-fallback path.
T_STALE = pd.Timestamp("2024-06-15 12:00:00", tz="UTC")
DATA_DIR = CRYPTO5_PROCESSED
SEMANTICS_PATH = CRYPTO5_SEMANTICS_PATH
PANEL_ASSETS = CRYPTO5_DEBATE_ASSETS

RESULTS_PATH = Path("data/results/llm_layer_audit.json")

audit_log: list[dict[str, Any]] = []


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    return hashlib.sha256((system_prompt + "\x00" + user_prompt).encode()).hexdigest()[:16]


def _wrap_call_llm(agent, tag: str):
    """Monkeypatch agent._call_llm to log (prompt hash, regime block, raw response) for
    every real LLM invocation, without changing any agent's actual behavior."""
    original = agent._call_llm

    def wrapped(system_prompt: str, user_prompt: str) -> str:
        t0 = time.time()
        raw = original(system_prompt, user_prompt)
        dt = time.time() - t0
        # Extract exactly the regime block the agent actually saw, for the log.
        regime_block = ""
        if "=== REGIME STATE ===" in system_prompt:
            start = system_prompt.index("=== REGIME STATE ===")
            end = system_prompt.index("=== PRICE FEATURES", start)
            regime_block = system_prompt[start:end].strip()
        entry = {
            "tag": tag,
            "agent": agent.name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt_hash": _prompt_hash(system_prompt, user_prompt),
            "regime_block_seen": regime_block,
            "raw_response": raw,
            "latency_s": round(dt, 2),
        }
        audit_log.append(entry)
        return raw

    agent._call_llm = wrapped


def _make_regime_variant(
    ctx: RegimeContext, label: int, probs_by_label: dict[int, float], descriptions: dict[int, str]
) -> RegimeContext:
    regime_probs = {f"regime_{k}": v for k, v in probs_by_label.items()}
    return dataclasses.replace(
        ctx,
        regime_label=label,
        regime_probs=regime_probs,
        dominant_regime=label,
        regime_description=descriptions.get(label, "Unknown regime"),
    )


def _make_extreme_price_variant(ctx: RegimeContext, direction: str) -> RegimeContext:
    pf = dict(ctx.price_features)
    wf = dict(ctx.window_features)
    if direction == "bullish":
        pf["rsi"] = 88.0
        pf["macd_histogram"] = abs(pf.get("macd_histogram", 0.01)) * 5 + 0.5
        pf["adx"] = 45.0
        pf["close_return"] = 0.06
        pf["return_skew"] = 2.5
        wf["close_return_mean"] = 0.01
        wf["rsi_mean"] = 85.0
    else:
        pf["rsi"] = 12.0
        pf["macd_histogram"] = -(abs(pf.get("macd_histogram", 0.01)) * 5 + 0.5)
        pf["adx"] = 45.0
        pf["close_return"] = -0.06
        pf["return_skew"] = -2.5
        wf["close_return_mean"] = -0.01
        wf["rsi_mean"] = 15.0
    return dataclasses.replace(ctx, price_features=pf, window_features=wf)


def _thesis_summary(t: AgentThesis) -> dict:
    return {
        "agent_name": t.agent_name,
        "directional_bias": t.directional_bias,
        "confidence": t.confidence,
        "reasoning": t.reasoning,
        "key_factors": t.key_factors,
        "regime_alignment": t.regime_alignment,
        "is_fallback_default": t.reasoning == "LLM response unavailable — defaulting to neutral.",
    }


def main() -> None:
    try:
        _run()
    except Exception:
        # Even on a crash, dump whatever LLM calls were already made — they're the
        # expensive/slow part and shouldn't be silently lost.
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.with_suffix(".partial.json").write_text(
            json.dumps({"partial_call_trace": audit_log}, indent=2, default=str)
        )
        print(f"\nCRASHED — partial trace ({len(audit_log)} calls) saved to "
              f"{RESULTS_PATH.with_suffix('.partial.json')}")
        raise


def _run() -> None:
    print("Loading data + agents...")
    regime_stream = pd.read_parquet(DATA_DIR / "regime_stream.parquet", engine="pyarrow")
    aligned_dfs = {
        a: pd.read_parquet(DATA_DIR / f"{a}_aligned.parquet", engine="pyarrow")
        for a in PANEL_ASSETS
        if (DATA_DIR / f"{a}_aligned.parquet").exists()
    }
    aligned_df = aligned_dfs[ASSET]

    macro = MacroAgent(assets=PANEL_ASSETS, data_dir=DATA_DIR)
    tech = TechnicalAgent()
    sentiment = SentimentAgent(macro_assets={"macro"}, fear_greed_assets=set(PANEL_ASSETS))
    contrarian = ContrarianAgent(regime_stream_df=regime_stream)
    risk = RiskAgent(data_dir=DATA_DIR)
    fm = FundManagerAgent(semantics_path=SEMANTICS_PATH)

    for a, tag in [(macro, "macro"), (tech, "technical"), (sentiment, "sentiment"),
                   (contrarian, "contrarian"), (fm, "fund_manager")]:
        _wrap_call_llm(a, tag)
    # RiskAgent never calls _call_llm — confirmed by code inspection, not wrapped (would be dead code).

    # ── Build real context + regime variants ────────────────────────────────
    print(f"Building context at {ASSET} @ {T0}...")
    ctx_real = build_regime_context(T0, ASSET, regime_stream, aligned_df, semantics_path=SEMANTICS_PATH)

    stats_by_label = _load_regime_stats(SEMANTICS_PATH)
    kinds = {label: classify_regime_volatility_level(s) for label, s in stats_by_label.items()}
    sharpes = {label: float(s.get("sharpe", 0.0)) for label, s in stats_by_label.items()}
    bull_label = max(sharpes, key=sharpes.get)
    bear_label = min(sharpes, key=sharpes.get)
    sideways_label = min(sharpes, key=lambda k: abs(sharpes[k]))
    descriptions = {}
    from agents.context import _load_regime_descriptions
    descriptions = _load_regime_descriptions(SEMANTICS_PATH)

    n_labels = len(stats_by_label)
    rng = np.random.default_rng(42)
    random_probs_arr = rng.dirichlet(np.ones(n_labels))
    random_probs_by_label = {label: float(random_probs_arr[i]) for i, label in enumerate(sorted(stats_by_label))}
    random_label = max(random_probs_by_label, key=random_probs_by_label.get)

    one_hot = lambda lbl: {k: (0.97 if k == lbl else 0.01) for k in stats_by_label}

    variants = {
        "real": ctx_real,
        "bull": _make_regime_variant(ctx_real, bull_label, one_hot(bull_label), descriptions),
        "bear": _make_regime_variant(ctx_real, bear_label, one_hot(bear_label), descriptions),
        "sideways": _make_regime_variant(ctx_real, sideways_label, one_hot(sideways_label), descriptions),
        "random": _make_regime_variant(ctx_real, random_label, random_probs_by_label, descriptions),
    }
    print(
        f"Regime labels — bull={bull_label}({kinds[bull_label]}, sharpe={sharpes[bull_label]:.2f}) "
        f"bear={bear_label}({kinds[bear_label]}, sharpe={sharpes[bear_label]:.2f}) "
        f"sideways={sideways_label}({kinds[sideways_label]}, sharpe={sharpes[sideways_label]:.2f}) "
        f"random_label={random_label}"
    )
    print(f"real ctx regime_label={ctx_real.regime_label} probs={ctx_real.regime_probs}")

    # ── Test 2: regime sensitivity (+ test 4: null regime, using the 'random' variant) ──
    print("\n=== Regime sensitivity test (Macro/Technical/Sentiment x real/bull/bear/sideways/random) ===")
    regime_results: dict[str, dict[str, AgentThesis]] = {"MacroAgent": {}, "TechnicalAgent": {}, "SentimentAgent": {}}
    for variant_name, ctx in variants.items():
        print(f"  -- variant: {variant_name} --")
        t = macro.post_thesis(ctx)
        regime_results["MacroAgent"][variant_name] = t
        print(f"    MacroAgent: {t.directional_bias} conf={t.confidence:.2f}")

        t = tech.post_thesis(ctx, aligned_df=aligned_df)
        regime_results["TechnicalAgent"][variant_name] = t
        print(f"    TechnicalAgent: {t.directional_bias} conf={t.confidence:.2f}")

        t = sentiment.post_thesis(ctx)
        regime_results["SentimentAgent"][variant_name] = t
        print(f"    SentimentAgent: {t.directional_bias} conf={t.confidence:.2f}")

    # ── Test 3: extreme signal (TechnicalAgent only — the only agent that reads
    # ctx.price_features directly; Macro/Sentiment build their signal from independent
    # data sources unaffected by ctx.price_features, see code trace in the chat) ──────
    print("\n=== Extreme signal test (TechnicalAgent x bullish/bearish price features) ===")
    extreme_results: dict[str, AgentThesis] = {}
    for direction in ["bullish", "bearish"]:
        ctx_x = _make_extreme_price_variant(ctx_real, direction)
        t = tech.post_thesis(ctx_x, aligned_df=aligned_df)
        extreme_results[direction] = t
        print(f"  {direction} price extreme -> {t.directional_bias} conf={t.confidence:.2f}")

    # ── Test 1 (schema) support + aggregator test: Contrarian + FundManager using the
    # REAL-variant R1 theses already captured above (no redundant LLM calls) ─────────
    print("\n=== Contrarian + FundManager (real-context R1 theses) ===")
    real_theses = [regime_results[a]["real"] for a in ["MacroAgent", "TechnicalAgent", "SentimentAgent"]]
    contrarian_thesis = contrarian.post_contrarian_thesis(ctx_real, real_theses, aligned_df)
    print(f"  ContrarianAgent: {contrarian_thesis.directional_bias} conf={contrarian_thesis.confidence:.2f}")

    risk_limits = risk.get_risk_limits(ctx_real)
    print(f"  RiskAgent (no LLM): regime={risk_limits.risk_regime} max_pos={risk_limits.max_position_size}")

    decision = fm.synthesize(ctx_real, real_theses, [], risk_limits, contrarian_thesis=contrarian_thesis)
    print(f"  FundManager: {decision.final_action} conviction={decision.conviction:.2f} "
          f"consensus={decision.consensus_direction} dissenting={decision.dissenting_agents}")

    # Aggregator-usage check: perturb ONE agent's thesis and confirm FundManager's
    # deterministic vote (not the LLM) responds — proves the weighted-consensus code
    # path actually consumes every agent's output, not just narrates a fixed answer.
    print("\n=== Aggregator-usage check: flip TechnicalAgent's thesis, re-synthesize (no new LLM call) ===")
    flipped = [
        t.model_copy(update={
            "directional_bias": ("bearish" if t.directional_bias != "bearish" else "bullish"),
            "confidence": 0.9,
        })
        if t.agent_name == "TechnicalAgent" else t
        for t in real_theses
    ]
    decision_flipped = fm.synthesize(ctx_real, flipped, [], risk_limits, contrarian_thesis=contrarian_thesis)
    print(f"  Original:  action={decision.final_action} consensus={decision.consensus_direction} conviction={decision.conviction:.2f}")
    print(f"  Flipped:   action={decision_flipped.final_action} consensus={decision_flipped.consensus_direction} conviction={decision_flipped.conviction:.2f}")
    aggregator_responsive = (
        decision.final_action != decision_flipped.final_action
        or decision.consensus_direction != decision_flipped.consensus_direction
        or abs(decision.conviction - decision_flipped.conviction) > 1e-6
    )

    # ── Test 5: staleness ────────────────────────────────────────────────────
    print("\n=== Staleness test ===")

    def _resolved_regime_gap(ts: pd.Timestamp) -> float:
        sub = regime_stream[regime_stream["asset"] == ASSET].copy()
        sub["timestamp"] = pd.to_datetime(sub["timestamp"])
        sub = sub.set_index("timestamp").sort_index()
        sub = sub[~sub.index.duplicated(keep="last")]
        idx = sub.index.get_indexer([ts], method="nearest")[0]
        resolved_ts = sub.index[idx]
        return abs((ts - resolved_ts).total_seconds()) / 3600.0

    gap_t0 = _resolved_regime_gap(T0)
    gap_stale = _resolved_regime_gap(T_STALE)
    print(f"  T0 ({T0}): resolved regime row is {gap_t0:.1f}h away")
    print(f"  T_STALE ({T_STALE}): resolved regime row is {gap_stale:.1f}h away")

    # ── Compile pass/fail report ─────────────────────────────────────────────
    def _bias_conf_set(agent_name: str) -> dict[str, tuple[str, float]]:
        return {v: (t.directional_bias, round(t.confidence, 3)) for v, t in regime_results[agent_name].items()}

    def _regime_sensitive(agent_name: str) -> bool:
        vals = _bias_conf_set(agent_name)
        biases = {b for b, _ in vals.values()}
        confs = [c for _, c in vals.values()]
        bias_varies = len(biases) > 1
        conf_varies = (max(confs) - min(confs)) > 0.05
        return bias_varies or conf_varies

    def _null_regime_negligible(agent_name: str) -> bool:
        vals = _bias_conf_set(agent_name)
        real_b, real_c = vals["real"]
        rand_b, rand_c = vals["random"]
        return real_b == rand_b and abs(real_c - rand_c) < 0.05

    schema_ok = {
        "MacroAgent": not regime_results["MacroAgent"]["real"].reasoning.startswith("LLM response unavailable"),
        "TechnicalAgent": not regime_results["TechnicalAgent"]["real"].reasoning.startswith("LLM response unavailable"),
        "SentimentAgent": not regime_results["SentimentAgent"]["real"].reasoning.startswith("LLM response unavailable"),
        "ContrarianAgent": not contrarian_thesis.reasoning.startswith("LLM response unavailable"),
        "RiskAgent": risk_limits.risk_regime in ("low", "medium", "high", "extreme"),
        "FundManagerAgent": bool(decision.rationale) and decision.final_action in ("long", "short", "hold"),
    }

    extreme_pass = {
        "TechnicalAgent_bullish_leans_long": extreme_results["bullish"].directional_bias in ("bullish",),
        "TechnicalAgent_bearish_leans_short": extreme_results["bearish"].directional_bias in ("bearish",),
    }

    report = {
        "meta": {
            "asset": ASSET, "t0": str(T0), "t_stale": str(T_STALE),
            "n_llm_calls": len(audit_log),
        },
        "schema_test": schema_ok,
        "regime_sensitivity_test": {
            a: {"sensitive": _regime_sensitive(a), "values": _bias_conf_set(a)}
            for a in ["MacroAgent", "TechnicalAgent", "SentimentAgent"]
        },
        "extreme_signal_test": {
            **extreme_pass,
            "bullish_result": _thesis_summary(extreme_results["bullish"]),
            "bearish_result": _thesis_summary(extreme_results["bearish"]),
        },
        "null_regime_test": {
            a: {"regime_negligible": _null_regime_negligible(a), "values": _bias_conf_set(a)}
            for a in ["MacroAgent", "TechnicalAgent", "SentimentAgent"]
        },
        "staleness_test": {
            "t0_gap_hours": round(gap_t0, 2),
            "t0_pass": gap_t0 < 1.0,
            "t_stale_gap_hours": round(gap_stale, 2),
            "t_stale_pass": gap_stale < 1.0,
        },
        "aggregator_test": {
            "responsive_to_flipped_agent": aggregator_responsive,
            "original_decision": decision.model_dump(),
            "flipped_decision": decision_flipped.model_dump(),
            "contrarian_bias_forced_opposite": True,  # verified by code trace, agents/contrarian_agent.py:213
            "risk_agent_never_calls_llm": True,        # verified by code trace, no _call_llm in risk_agent.py
            "fund_manager_decision_fields_are_pure_python": True,  # only 'rationale' comes from the LLM call
        },
        "contrarian_thesis": _thesis_summary(contrarian_thesis),
        "risk_limits": risk_limits.model_dump(),
        "fund_manager_decision": decision.model_dump(),
        "full_call_trace": audit_log,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull trace written to {RESULTS_PATH}")
    print(f"Total LLM calls made: {len(audit_log)}")


if __name__ == "__main__":
    main()
