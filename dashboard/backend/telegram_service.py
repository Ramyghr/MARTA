"""MARTA Telegram notification service.

Sends regime shifts, trade signals, and system alerts to a Telegram chat via
the Bot API. Fires as a best-effort side channel alongside the existing
in-memory notification system (main.py._add_notification) — never raises,
never blocks a caller, silently no-ops if unconfigured.
"""
import logging
import os

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_API = "https://api.telegram.org"


async def send_telegram(message: str, parse_mode: str = "HTML") -> bool:
    """Send a message to the configured Telegram chat."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured — skipping notification")
        return False

    url = f"{TELEGRAM_API}/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                logger.info(f"Telegram notification sent: {message[:50]}...")
                return True
            logger.warning(f"Telegram error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")
        return False


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
        "• PPO inference: active\n"
        "• Hourly scheduler: running\n"
        "• Live feeds: BTC/ETH (Binance) + EURUSD/GOLD (Twelve Data)\n"
        "• Paper trading: enabled"
    )
    await send_telegram(msg)


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
