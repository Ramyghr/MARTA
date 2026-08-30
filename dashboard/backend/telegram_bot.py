"""Interactive Telegram bot — slash commands + inline-button trade approval.

Long-polling (getUpdates), not a webhook: dashboard/docker-compose.yml only exposes
the backend on localhost:8001 with no reverse proxy or public domain, so a webhook
URL isn't reachable from Telegram's servers. Polling needs no inbound port at all —
it works from behind Docker/NAT exactly like the outbound telegram_service.py calls
already do.

Every command proxies to the existing FastAPI routes via an in-process ASGI
transport (httpx talking directly to the `app` object, no real TCP round-trip) —
this bot is just another client of the same API dashboard/frontend already calls,
so command behaviour and business logic never has to be duplicated or kept in sync.

Runs only when both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set (same gate
telegram_service.py uses) — start() no-ops otherwise, same as the send-only
notifier already does when unconfigured.

Authorization: every update is checked against TELEGRAM_CHAT_ID (comma-separated
for multiple allowed chats) before any command runs. /approve and /trigger can
move (paper) capital, so an unauthorized sender is dropped silently rather than
answered — no information leak about which commands exist.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_ALLOWED_CHAT_IDS = {c.strip() for c in os.getenv("TELEGRAM_CHAT_ID", "").split(",") if c.strip()}
DEFAULT_ASSET = "BTC_USDT"

_application: Optional[Application] = None
_client: Optional[httpx.AsyncClient] = None
_run_task: Optional[asyncio.Task] = None


def _normalize_asset(raw: str) -> str:
    return raw.strip().upper().replace("/", "_").replace("-", "_") if raw else DEFAULT_ASSET


def _authorized(update: Update) -> bool:
    chat_id = str(update.effective_chat.id) if update.effective_chat else ""
    if chat_id not in _ALLOWED_CHAT_IDS:
        logger.warning(f"telegram_bot: dropping update from unauthorized chat_id={chat_id}")
        return False
    return True


async def _reply(update: Update, text: str) -> None:
    await update.effective_message.reply_text(text, parse_mode="HTML")


async def _get(path: str, **params) -> tuple[int, dict]:
    r = await _client.get(path, params=params)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"detail": r.text[:300]}


async def _post(path: str, **params) -> tuple[int, dict]:
    r = await _client.post(path, params=params)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"detail": r.text[:300]}


def _err(status: int, data: dict) -> str:
    return f"Error ({status}): {data.get('detail', data)}"


def _num(d: dict, key: str, default: float = 0.0) -> float:
    """d.get(key, default) only falls back when the key is MISSING — several backend
    services (price_service's unavailable-price fallback in particular) return the key
    present but explicitly null, which crashes an f-string format spec on NoneType."""
    v = d.get(key)
    return v if isinstance(v, (int, float)) else default


def _str(d: dict, key: str, default: str = "n/a") -> str:
    v = d.get(key)
    return v if isinstance(v, str) and v else default


# ── Commands ─────────────────────────────────────────────────────────────────

_HELP_TEXT = (
    "<b>MARTA control bot</b>\n\n"
    "/price &lt;asset&gt; — current price\n"
    "/regime &lt;asset&gt; — current regime + probabilities\n"
    "/signal &lt;asset&gt; — latest live trading signal\n"
    "/trigger &lt;asset&gt; — fire a signal tick now\n"
    "/positions — shared capital + open exposure\n"
    "/trades [asset] — recent trade log\n"
    "/pending — trades awaiting approval\n"
    "/ablation — MARTA vs. buy-and-hold summary\n"
    "/status — GPU / Ollama / system health\n"
    "/chat &lt;question&gt; — ask MARTA's assistant\n\n"
    f"Default asset when omitted: {DEFAULT_ASSET}"
)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    await _reply(update, _HELP_TEXT)


async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    asset = _normalize_asset(context.args[0]) if context.args else DEFAULT_ASSET
    status, d = await _get(f"/api/price/{asset}")
    if status != 200:
        await _reply(update, _err(status, d))
        return
    if d.get("price") is None:
        await _reply(update, f"<b>{asset}</b>\nPrice unavailable right now (no live feed tick or fallback data yet).")
        return
    await _reply(update, (
        f"<b>{asset}</b>\n"
        f"Price: ${_num(d, 'price'):,.2f}\n"
        f"Source: {_str(d, 'source')}"
        + (" (stale)" if d.get("stale") else "")
    ))


async def cmd_regime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    asset = _normalize_asset(context.args[0]) if context.args else DEFAULT_ASSET
    status, d = await _get(f"/api/regime/{asset}")
    if status != 200:
        await _reply(update, _err(status, d))
        return
    probs = d.get("probabilities", {}) or {}
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    probs_str = ", ".join(f"R{k}={v:.0%}" for k, v in top) or "n/a"
    await _reply(update, (
        f"<b>{asset} regime</b>\n"
        f"R{d.get('regime_label', '?')} — {d.get('regime_description', 'n/a')}\n"
        f"Top probabilities: {probs_str}"
    ))


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    asset = _normalize_asset(context.args[0]) if context.args else DEFAULT_ASSET
    status, d = await _get("/api/rl/live-signal", asset=asset)
    if status != 200:
        await _reply(update, _err(status, d))
        return
    await _reply(update, (
        f"<b>{asset} signal</b>\n"
        f"Action: {_str(d, 'action').upper()}\n"
        f"Size: {_num(d, 'position_size'):.1%}\n"
        f"Regime: R{d.get('regime_label', '?')} — {_str(d, 'regime_description', '')}\n"
        f"Source: {_str(d, 'source')}\n"
        f"Pending approval: {'yes (id ' + str(d.get('pending_trade_id')) + ')' if d.get('pending_approval') else 'no'}"
    ))


async def cmd_trigger(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    asset = _normalize_asset(context.args[0]) if context.args else DEFAULT_ASSET
    await _reply(update, f"Firing a signal tick for {asset}…")
    status, d = await _post("/api/rl/trigger-signal", asset=asset)
    if status != 200:
        await _reply(update, _err(status, d))
        return
    await asyncio.sleep(4)  # mirrors the dashboard's "Trigger Manual Signal" button
    await cmd_signal(update, context)


async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    status, d = await _get("/api/account")
    if status != 200:
        await _reply(update, _err(status, d))
        return
    lines = [
        "<b>Account</b>",
        f"Capital: ${_num(d, 'shared_capital'):,.2f} ({_num(d, 'total_return'):+.2%})",
        f"Total exposure: {_num(d, 'total_exposure'):.1%}",
        "",
    ]
    positions = [p for p in d.get("positions", []) if p.get("action") in ("long", "short")]
    if not positions:
        lines.append("No open positions.")
    for p in positions:
        lines.append(f"{p['asset']}: {p['action'].upper()} {p['exposure']:.1%} ({p['n_trades']} trades)")
    await _reply(update, "\n".join(lines))


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    asset = _normalize_asset(context.args[0]) if context.args else DEFAULT_ASSET
    status, d = await _get("/api/paper/trades", asset=asset, limit=8)
    if status != 200:
        await _reply(update, _err(status, d))
        return
    trades = d.get("trades_local", [])[-8:]
    if not trades:
        await _reply(update, f"No trades yet for {asset}.")
        return
    lines = [f"<b>{asset} — last {len(trades)} trades</b>"]
    for t in reversed(trades):
        ts = str(t.get("timestamp", ""))[:16]
        lines.append(
            f"{ts} {t.get('action', '').upper()} {t.get('position_size', 0):.1%} "
            f"@ ${t.get('price', 0):,.2f} (PnL {t.get('pnl', 0):+.2%}, R{t.get('regime_label', '?')})"
        )
    await _reply(update, "\n".join(lines))


async def cmd_pending(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    status, d = await _get("/api/trades/pending", status="pending")
    if status != 200:
        await _reply(update, _err(status, d))
        return
    trades = d.get("trades", [])
    if not trades:
        await _reply(update, "No trades awaiting approval.")
        return
    for t in trades:
        msg = (
            f"<b>Pending #{t['id']}</b> — {t['asset']}\n"
            f"{t['action'].upper()} {t['position_size']:.1%} @ ${t['price']:,.2f}\n"
            f"Regime R{t['regime_label']} · {t['source']}"
        )
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Approve", callback_data=f"approve:{t['id']}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"reject:{t['id']}"),
        ]])
        await update.effective_message.reply_text(msg, parse_mode="HTML", reply_markup=keyboard)


async def cmd_ablation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    status, d = await _get("/api/ablation")
    if status != 200 or not d.get("variants"):
        await _reply(update, "No ablation results yet — run /api/ablation/refresh from the dashboard first.")
        return
    variants = d["variants"]
    marta = variants.get("marta_full") or {}
    bh = variants.get("buy_hold") or {}
    await _reply(update, (
        "<b>MARTA vs. Buy & Hold</b> (validation)\n\n"
        f"MARTA:      Sharpe {_num(marta, 'sharpe'):.2f}, return {_num(marta, 'total_return'):+.1%}, "
        f"max DD {_num(marta, 'max_drawdown'):.1%}\n"
        f"Buy & Hold: Sharpe {_num(bh, 'sharpe'):.2f}, return {_num(bh, 'total_return'):+.1%}, "
        f"max DD {_num(bh, 'max_drawdown'):.1%}"
    ))


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    status, d = await _get("/api/system/status")
    if status != 200:
        await _reply(update, _err(status, d))
        return
    gpu = d.get("gpu") or {}
    ollama = d.get("ollama") or {}
    gpu_line = (
        f"{_str(gpu, 'name', '?')} {_num(gpu, 'utilization_pct'):.0f}% "
        f"({_num(gpu, 'memory_used_mb'):.0f}/{_num(gpu, 'memory_total_mb'):.0f} MB)"
        if gpu.get("available") else "unavailable"
    )
    await _reply(update, (
        "<b>System status</b>\n"
        f"GPU: {gpu_line}\n"
        f"Ollama: {'connected' if ollama.get('connected') else 'unreachable'}"
    ))


async def cmd_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    question = " ".join(context.args) if context.args else ""
    if not question:
        await _reply(update, "Usage: /chat <question>")
        return
    session_id = f"telegram_{update.effective_chat.id}"
    r = await _client.post("/api/chat", json={"message": question, "session_id": session_id, "asset": DEFAULT_ASSET})
    if r.status_code != 200:
        await _reply(update, _err(r.status_code, {"detail": r.text[:300]}))
        return
    await _reply(update, r.json().get("response", "(no response)"))


async def on_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not _authorized(update):
        await query.answer()
        return

    action, _, pending_id = (query.data or "").partition(":")
    if action not in ("approve", "reject") or not pending_id:
        await query.answer("Unrecognized action")
        return

    status, d = await _post(f"/api/trades/pending/{pending_id}/{action}")
    if status != 200:
        await query.answer(d.get("detail", "Failed")[:180], show_alert=True)
        return

    await query.answer("Approved ✅" if action == "approve" else "Rejected ❌")
    verb = "APPROVED" if action == "approve" else "REJECTED"
    try:
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(f"Pending #{pending_id} {verb}.")
    except Exception as exc:
        logger.warning(f"telegram_bot: failed to update message after {action}: {exc}")


# ── Lifecycle ────────────────────────────────────────────────────────────────

def start(app) -> None:
    """Start long-polling in the background. No-ops if unconfigured (mirrors
    telegram_service.send_telegram's own configured-or-silent behaviour)."""
    global _application, _client

    if not TELEGRAM_TOKEN or not _ALLOWED_CHAT_IDS:
        logger.info("telegram_bot: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — bot disabled")
        return

    _client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://internal")

    def _build_application() -> Application:
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler(["start", "help"], cmd_help))
        application.add_handler(CommandHandler("price", cmd_price))
        application.add_handler(CommandHandler("regime", cmd_regime))
        application.add_handler(CommandHandler("signal", cmd_signal))
        application.add_handler(CommandHandler("trigger", cmd_trigger))
        application.add_handler(CommandHandler("positions", cmd_positions))
        application.add_handler(CommandHandler("trades", cmd_trades))
        application.add_handler(CommandHandler("pending", cmd_pending))
        application.add_handler(CommandHandler("ablation", cmd_ablation))
        application.add_handler(CommandHandler("status", cmd_status))
        application.add_handler(CommandHandler("chat", cmd_chat))
        application.add_handler(CallbackQueryHandler(on_callback_query))
        return application

    async def _run():
        global _application
        # initialize() (bot.get_me()) can race the rest of the FastAPI startup/first
        # requests, which synchronously load several 100K+-row parquet files — that
        # blocks the event loop long enough for PTB's connect timeout to trip even
        # though api.telegram.org is reachable fine a moment later (verified manually).
        # A *fresh* Application per attempt, not just a retried initialize() call, is
        # required: reusing one Application whose first initialize() timed out left its
        # underlying httpx connection pool wedged and every retry on it hung forever
        # (observed directly — reusing the object never recovered even after the
        # contention window passed, while a fresh Application connected immediately).
        # asyncio.wait_for is a second safety net against that same class of hang.
        delay = 5
        application = None
        for attempt in range(1, 6):
            application = _build_application()
            try:
                await asyncio.wait_for(application.initialize(), timeout=20)
                break
            except Exception as exc:
                logger.warning(f"telegram_bot: initialize() attempt {attempt}/5 failed ({exc}), retrying in {delay}s")
                try:
                    await application.shutdown()
                except Exception:
                    pass
                if attempt == 5:
                    logger.error("telegram_bot: giving up after 5 failed initialize() attempts — bot disabled")
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

        _application = application
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        logger.info("telegram_bot: long-polling started")

    global _run_task
    _run_task = asyncio.create_task(_run())


async def stop() -> None:
    global _application, _client, _run_task
    if _run_task is not None and not _run_task.done():
        _run_task.cancel()  # covers shutdown happening mid-retry, before _application is set
    _run_task = None
    if _application is not None:
        try:
            if _application.updater is not None and _application.updater.running:
                await _application.updater.stop()
            await _application.stop()
            await _application.shutdown()
        except Exception as exc:
            logger.warning(f"telegram_bot: error during shutdown: {exc}")
        _application = None
    if _client is not None:
        await _client.aclose()
        _client = None
