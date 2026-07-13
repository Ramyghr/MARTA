"""MARTA AI Chat Service.

Uses the same local Mistral (via Ollama) the live debate agents call to answer
questions about MARTA's current state. Injects live context — regime, latest
backtest numbers, latest debate, Fear & Greed — read in-process from the same
services the REST endpoints use (regime_service, news_live, the results/debate
JSON files), rather than the dashboard calling its own HTTP API.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

import regime_service

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MARTA_ROOT = Path(os.getenv("MARTA_ROOT", "/app"))

# Chat history per session (in-memory, resets on restart)
_chat_histories: dict[str, list[dict]] = {}


def _build_system_prompt(context: dict) -> str:
    """Build a rich system prompt with live MARTA context."""
    regime = context.get("regime", {})
    portfolio = context.get("portfolio", {})
    debate = context.get("latest_debate", {})
    fear_greed = context.get("fear_greed", {})
    fm = debate.get("fund_manager_decision", {})

    return f"""You are MARTA's AI assistant — an expert in the MARTA trading system.
MARTA stands for Multi-Agent Regime-Aware Trading Architecture.

CURRENT SYSTEM STATE ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}):

REGIME:
- Current regime: R{regime.get('regime_label', '?')} — {regime.get('regime_description', 'Unknown')}
- Dominant probability: {regime.get('probabilities', {}).get(f"regime_{regime.get('regime_label', 0)}", 0):.1%}
- Asset: {context.get('asset', 'BTC_USDT')}

PORTFOLIO (Backtest Validation 2023):
- Sharpe Ratio: {portfolio.get('sharpe', 1.370):.3f}
- Max Drawdown: {portfolio.get('max_drawdown', 0.007):.1%}
- Total Return: {portfolio.get('total_return', 0.018):.1%}
- Meaningful Trades: {portfolio.get('n_meaningful_trades', 6)}
- Hold Rate: {portfolio.get('hold_ratio', 0.983):.1%} (trades only in R3 — High Volatility Bull regime)

LATEST AGENT DEBATE:
- Asset: {debate.get('asset', 'N/A')}
- Decision: {fm.get('final_action', 'N/A')}
- Conviction: {fm.get('conviction', 0):.0%}
- Rationale: {fm.get('rationale', 'N/A')[:200]}

MARKET SENTIMENT:
- Fear & Greed Index: {fear_greed.get('value', 50)} — {fear_greed.get('classification', 'Neutral')}

MARTA ARCHITECTURE:
- Layer 1: TFT contrastive encoder (InfoNCE) -> 128-dim embeddings -> Gaussian HMM (7 states)
- Layer 2: 5 LLM agents debate (MacroAgent, TechnicalAgent, SentimentAgent, RiskAgent, ContrarianAgent) -> FundManager
- Layer 3: PPO RL agent with Beta distribution position head, CVaR-normalized reward

KEY RESULTS:
- Validation Sharpe: 1.370 (2023, vs Buy&Hold with far higher drawdown)
- Regime conditioning is load-bearing: zeroing the regime embedding collapses PPO to always-hold
- The system trades selectively — 6 meaningful trades/year, not because it's broken but because
  most regimes have negative trading EV net of costs relative to holding

You answer questions about MARTA clearly and honestly. You acknowledge limitations:
- The system trades only in R3 (few trades/year) — selective, not broken
- EURUSD/GOLD don't trade in the PPO curriculum — it was trained on BTC/ETH only
- Backtest results are on a single validation path — not statistically conclusive on their own
- The system is a research prototype (final year engineering project), not a production trading system

Be concise, technical when appropriate, and always honest about limitations.
Answer in the same language as the user's question (French or English)."""


async def _get_live_context(asset: str) -> dict:
    """Gather live MARTA context for the chat prompt. Every source is best-effort —
    a missing file or unreachable service degrades that section, never the whole chat."""
    context = {"asset": asset}

    try:
        import pandas as pd
        context["regime"] = regime_service.get_regime(asset, pd.Timestamp("now", tz="UTC"))
    except Exception as e:
        logger.debug(f"chat_service: regime lookup failed: {e}")

    try:
        path = MARTA_ROOT / "data/results/ppo_evaluation.json"
        if path.exists():
            context["portfolio"] = json.loads(path.read_text()).get("marta_ppo", {})
    except Exception as e:
        logger.debug(f"chat_service: portfolio load failed: {e}")

    try:
        path = MARTA_ROOT / "data/debates/debate_results.json"
        if path.exists():
            debates = json.loads(path.read_text())
            asset_debates = [d for d in debates if d.get("asset") == asset]
            if asset_debates:
                context["latest_debate"] = sorted(
                    asset_debates, key=lambda x: x.get("timestamp", ""), reverse=True
                )[0]
    except Exception as e:
        logger.debug(f"chat_service: latest debate load failed: {e}")

    try:
        import news_live
        context["fear_greed"] = await news_live.fetch_fear_greed()
    except Exception as e:
        logger.debug(f"chat_service: fear/greed fetch failed: {e}")

    return context


async def chat(session_id: str, user_message: str, asset: str = "BTC_USDT") -> str:
    """Process a chat message and return MARTA's response, maintaining per-session history."""
    history = _chat_histories.setdefault(session_id, [])

    context = await _get_live_context(asset)
    system_prompt = _build_system_prompt(context)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_message})

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": "mistral:7b-instruct",
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 512},
                },
            )

            if resp.status_code != 200:
                return "I'm having trouble connecting to the language model. Please try again."

            data = resp.json()
            assistant_message = data.get("message", {}).get("content", "")

            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_message})
            if len(history) > 20:
                _chat_histories[session_id] = history[-20:]

            return assistant_message

    except asyncio.TimeoutError:
        return "Response timed out. The model is busy — please try again in a moment."
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return f"An error occurred: {e}"


def clear_history(session_id: str) -> None:
    _chat_histories.pop(session_id, None)
