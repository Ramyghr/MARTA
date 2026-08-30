"""MARTA Telegram notification service.

Sends regime shifts, trade signals, and system alerts to a Telegram chat via
the Bot API. Fires as a best-effort side channel alongside the existing
in-memory notification system (main.py._add_notification) — never raises,
never blocks a caller, silently no-ops if unconfigured.

This module is send-only (sendMessage). The interactive half — slash
commands, inline-button approve/reject — lives in telegram_bot.py, which
reuses send_telegram()'s reply_markup support to post the buttons.
"""
import asyncio
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_API = "https://api.telegram.org"


def fire_and_forget(coro) -> None:
    """Schedule an async telegram_service call from ANY context.

    live_signal.get_live_signal() — the only call site for notify_pending_trade()
    — always runs inside signal_scheduler._tick_executor (a plain ThreadPoolExecutor
    worker), which has no running asyncio event loop. asyncio.create_task() there
    raises RuntimeError and silently drops the notification (the same pattern
    main.py._add_notification uses IS acceptable to drop there, but here it would
    make the Telegram approval prompt never fire at all — the point of the feature).
    Falls back to asyncio.run() (safe: a plain worker thread never already has a
    loop of its own) when there's nothing to attach a task to.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)


async def send_telegram(
    message: str,
    parse_mode: str = "HTML",
    reply_markup: Optional[dict] = None,
) -> Optional[dict]:
    """Send a message to the configured Telegram chat.

    Returns the Bot API `result` object (contains message_id, chat) on success,
    or None if unconfigured / the send failed.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured — skipping notification")
        return None

    url = f"{TELEGRAM_API}/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                logger.info(f"Telegram notification sent: {message[:50]}...")
                return resp.json().get("result")
            logger.warning(f"Telegram error {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")
        return None


async def notify_regime_shift(asset: str, prev_regime: int, prev_desc: str,
                               new_regime: int, new_desc: str) -> None:
    """Notify on regime transition. Crisis (R2) gets its own escalated alert."""
    emoji = "🚨" if new_regime == 2 else "🔄"
    severity = "CRISIS DETECTED" if new_regime == 2 else "Regime Shift"

    msg = (
        f"{emoji} <b>MARTA — {severity}</b>\n\n"
        f"<b>Asset:</b> {asset}\n"
        f"<b>From:</b> R{prev_regime} — {prev_desc}\n"
        f"<b>To:</b> R{new_regime} — {new_desc}\n"
    )
    if new_regime == 2:
        msg += "\n⚠️ Positions forced flat in crisis regime."
    await send_telegram(msg)


async def notify_trade_signal(asset: str, action: str, position_size: float,
                               regime_label: int, regime_desc: str, source: str) -> None:
    """Notify on a non-hold trade signal from the hourly scheduler."""
    if action == "hold":
        return

    emoji = "📈" if action == "long" else "📉"
    msg = (
        f"{emoji} <b>MARTA — Trade Signal</b>\n\n"
        f"<b>Asset:</b> {asset}\n"
        f"<b>Action:</b> {action.upper()}\n"
        f"<b>Size:</b> {position_size:.1%} of portfolio\n"
        f"<b>Regime:</b> R{regime_label} — {regime_desc}\n"
        f"<b>Source:</b> {source}\n"
    )
    await send_telegram(msg)


async def notify_system_startup() -> None:
    """Notify when the MARTA backend finishes lifespan startup."""
    msg = (
        "🟢 <b>MARTA System Online</b>\n\n"
        "• Execution: LLM consensus + static rule (RL execution is an opt-in toggle)\n"
        "• Hourly scheduler: running\n"
        "• Live feeds: BTC/ETH/BNB/LTC/XRP (Binance)\n"
        "• Paper trading: enabled"
    )
    await send_telegram(msg)


async def notify_pending_trade(pending_id: int, asset: str, action: str, position_size: float,
                                price: float, regime_label: int, source: str) -> None:
    """Notify with inline Approve/Reject buttons tied to a paper_db pending_trade row.

    Fired from live_signal.py right after paper_db.create_pending_trade() — the single
    creation point shared by the hourly scheduler, "Trigger Manual Signal", and "Run
    Debate & Execute", so this covers every path that can park a trade for approval,
    not just one of them. Tapping a button hits telegram_bot.py's CallbackQueryHandler,
    which calls the same /api/trades/pending/{id}/approve|reject route the dashboard's
    approval panel uses — same code path, same effect, just reachable from a phone.
    """
    emoji = "📈" if action == "long" else "📉" if action == "short" else "⏸️"
    msg = (
        f"{emoji} <b>MARTA — Trade Awaiting Approval</b>\n\n"
        f"<b>Asset:</b> {asset}\n"
        f"<b>Action:</b> {action.upper()}\n"
        f"<b>Size:</b> {position_size:.1%} of portfolio\n"
        f"<b>Price:</b> ${price:,.2f}\n"
        f"<b>Regime:</b> R{regime_label}\n"
        f"<b>Source:</b> {source}\n"
        f"<b>Pending ID:</b> {pending_id}"
    )
    reply_markup = {
        "inline_keyboard": [[
            {"text": "✅ Approve", "callback_data": f"approve:{pending_id}"},
            {"text": "❌ Reject", "callback_data": f"reject:{pending_id}"},
        ]]
    }
    await send_telegram(msg, reply_markup=reply_markup)


async def notify_debate_complete(asset: str, action: str, conviction: float,
                                  regime_label: int, rationale: str) -> None:
    """Notify when a live agent debate completes."""
    emoji = "📈" if action == "long" else "📉" if action == "short" else "⏸️"
    msg = (
        f"{emoji} <b>MARTA — Debate Complete</b>\n\n"
        f"<b>Asset:</b> {asset}\n"
        f"<b>Decision:</b> {action.upper()} ({conviction:.0%} conviction)\n"
        f"<b>Regime:</b> R{regime_label}\n"
        f"<b>Rationale:</b> {rationale[:200]}"
    )
    await send_telegram(msg)
