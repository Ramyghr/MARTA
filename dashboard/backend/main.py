import asyncio
import itertools
import json
import os
import subprocess
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Load .env from the same directory as this file before anything else imports os.environ
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import regime_service
import historical
import portfolio_service
import debate_stream
import live_feed
import system_service
from auth import APIKeyMiddleware
from demo_scenarios import DEMO_SCENARIOS, SCENARIO_BY_ID

limiter = Limiter(key_func=get_remote_address)
MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))


# ── Notifications (in-memory, last 50) ──────────────────────────────────────────

_notifications: deque = deque(maxlen=50)
_notification_id_seq = itertools.count(1)
_notification_ws_clients: set[WebSocket] = set()


def _add_notification(type: str, title: str, message: str, severity: str = "info") -> dict:
    """severity: info | warning | critical. Broadcasts to any connected /ws/notifications clients."""
    notif = {
        "id": str(next(_notification_id_seq)),
        "type": type,  # 'regime_shift' | 'trade_signal' | 'system'
        "title": title,
        "message": message,
        "severity": severity,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "read": False,
    }
    _notifications.appendleft(notif)
    try:
        asyncio.create_task(_broadcast_notification(notif))
    except RuntimeError:
        pass  # no running event loop (e.g. called from a sync context) — WS clients just miss this one
    return notif


async def _broadcast_notification(notif: dict) -> None:
    dead = []
    for ws in list(_notification_ws_clients):
        try:
            await ws.send_json({"type": "notification", "data": notif})
        except Exception:
            dead.append(ws)
    for ws in dead:
        _notification_ws_clients.discard(ws)


def _run_ablation_eval() -> subprocess.CompletedProcess:
    """Re-run run_rl.py --eval-only to regenerate data/results/ablation_results.json."""
    return subprocess.run(
        ["python", "run_rl.py", "--eval-only"],
        cwd=MARTA_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    regime_service.load_regime_stream()
    regime_service.load_regime_models()
    import ppo_inference
    ppo_inference.load_model()
    import signal_scheduler
    signal_scheduler.start()
    live_feed.start_forex_feed()

    # Refresh ablation on startup (background task) if clean no-regime debates exist
    no_regime_path = MARTA_ROOT / "data" / "debates" / "debate_results_no_regime.json"

    async def _startup_ablation_refresh():
        try:
            await asyncio.to_thread(_run_ablation_eval)
        except Exception:
            pass  # Don't block startup if eval fails

    if no_regime_path.exists():
        asyncio.create_task(_startup_ablation_refresh())

    import telegram_service
    asyncio.create_task(telegram_service.notify_system_startup())

    yield
    signal_scheduler.stop()


app = FastAPI(title="MARTA Dashboard API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── HTTP endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/assets")
def get_assets():
    return {
        "assets": [
            {"id": "BTC_USDT",  "label": "BTC/USDT",  "type": "crypto",    "live": True},
            {"id": "ETH_USDT",  "label": "ETH/USDT",  "type": "crypto",    "live": True},
            {"id": "EURUSD",    "label": "EUR/USD",    "type": "forex",     "live": bool(live_feed.TWELVE_DATA_KEY)},
            {"id": "GOLD",      "label": "XAU/USD",    "type": "commodity", "live": bool(live_feed.TWELVE_DATA_KEY)},
        ]
    }


@app.get("/api/data-range/{asset}")
def get_data_range(asset: str):
    return historical.get_data_range(asset)


@app.get("/api/regime/semantics")
def get_regime_semantics():
    path = MARTA_ROOT / "data" / "validated" / "regime_semantics.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    # Inject the same dynamically-generated description used everywhere else in the
    # backend (regime_service._load_regime_descriptions) so the frontend never has to
    # duplicate the vol/sharpe bucket thresholds — one source of truth for the label.
    descriptions = regime_service._load_regime_descriptions()
    for label, stats in data.get("regimes", {}).items():
        stats["description"] = descriptions.get(int(label), "Unknown")
    return data


@app.get("/api/regime/{asset}")
def get_regime(asset: str):
    """Current regime, right now — computed live (see regime_service.get_live_regime),
    not looked up in the possibly-stale precomputed regime_stream.parquet."""
    import pandas as pd
    ts = pd.Timestamp("now", tz="UTC")
    return regime_service.get_live_regime(asset, ts)


@app.get("/api/regime/{asset}/at")
def get_regime_at(asset: str, timestamp: str = Query(...)):
    import pandas as pd
    ts = pd.Timestamp(timestamp, tz="UTC")
    return regime_service.get_regime(asset, ts)


@app.get("/api/history/{asset}")
def get_history(asset: str, n: int = 200):
    return {"candles": historical.get_history(asset, n)}


@app.get("/api/history/{asset}/range")
def get_history_range(asset: str, start: str = Query(...), end: str = Query(...)):
    return {"candles": historical.get_history_range(asset, start, end)}


@app.get("/api/portfolio")
def get_portfolio():
    return portfolio_service.get_portfolio()


@app.get("/api/ablation")
def get_ablation():
    data = portfolio_service.get_ablation()
    path = MARTA_ROOT / "data" / "results" / "ablation_results.json"
    if data and path.exists():
        data["last_updated"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    return data


@app.post("/api/ablation/refresh")
def refresh_ablation():
    """Re-run --eval-only to regenerate ablation_results.json with clean no-regime debates."""
    try:
        result = _run_ablation_eval()
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Eval timed out after 5 minutes")
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=(result.stderr or result.stdout or "eval failed")[-2000:])
    return get_ablation()


@app.get("/api/test-set")
async def get_test_set_results():
    path = MARTA_ROOT / "data" / "results" / "test_set_results.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Test set results not found. Run: python run_test_set.py",
        )
    return json.loads(path.read_text())


@app.get("/api/rl/results")
def get_rl_results():
    import rl_service
    return rl_service.get_full_results()


@app.get("/api/rl/equity-curve")
def get_equity_curve():
    import rl_service
    return rl_service.get_equity_curve()


@app.get("/api/rl/positions")
def get_positions():
    import rl_service
    return rl_service.get_positions()


@app.get("/api/debates")
def list_debates():
    return {"debates": debate_stream.get_debates_list()}


@app.get("/api/debates/{debate_id}")
def get_debate(debate_id: int):
    detail = debate_stream.get_debate_detail(debate_id)
    if detail is None:
        return {"error": "not found"}
    return detail


@app.get("/api/scenarios")
def get_scenarios():
    return {"scenarios": DEMO_SCENARIOS}


@app.get("/api/demo-scenarios")
def get_demo_scenarios():
    return {"scenarios": DEMO_SCENARIOS}


@app.get("/api/regime-stream/{asset}")
def get_regime_stream(asset: str, limit: int = 500):
    """Live-aware (see get_regime_stream_tail_live_aware) — the Regime page's probability
    timeline. Found 2026-08-07: this previously called the pure static
    get_regime_stream_tail directly, so it looked permanently frozen even after the
    live-regime fixes elsewhere, since nothing routed through this endpoint at all."""
    return regime_service.get_regime_stream_tail_live_aware(asset, limit)


@app.get("/api/plots/tsne")
def get_tsne_plot():
    path = MARTA_ROOT / "regime" / "plots" / "regime_tsne.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="tsne plot not generated yet")
    return FileResponse(path, media_type="image/png")


@app.get("/api/data/freshness")
def get_data_freshness():
    import freshness_service
    return freshness_service.get_freshness()


@app.get("/api/news")
def get_news_endpoint(
    asset: str = Query("BTC_USDT"),
    timestamp: str = Query(...),
    window_days: int = Query(30),
):
    import news_service
    return {"headlines": news_service.get_news(asset, timestamp, window_days * 24)}


@app.get("/api/news/live")
@limiter.limit("20/minute")
async def get_live_news_endpoint(
    request: Request,
    asset: str = Query("BTC_USDT"),
    max_items: int = Query(40),
):
    import news_live
    headlines = await news_live.get_live_news(asset, max_items)
    return {"headlines": headlines, "live": True}


@app.get("/api/news/stocktwits/{asset}")
async def get_stocktwits(asset: str):
    """Get StockTwits messages for an asset. No auth required."""
    import news_live
    messages = await news_live.fetch_stocktwits(asset)
    return {"asset": asset, "messages": messages, "count": len(messages)}


@app.get("/api/market/fear-greed")
async def get_fear_greed():
    """Get current crypto Fear & Greed Index. No auth required."""
    import news_live
    return await news_live.fetch_fear_greed()


@app.get("/api/rl/live-signal")
def get_live_signal(
    asset: str = Query("BTC_USDT"),
    debate_action: Optional[str] = Query(None),
):
    import live_signal
    return live_signal.get_live_signal(asset, debate_action)


@app.get("/api/rl/scheduler-status")
def get_scheduler_status():
    """Return last signals emitted by the hourly scheduler + execution backend status."""
    import signal_scheduler
    import paper_exchange
    import alpaca_service
    return {
        "scheduler_running": signal_scheduler._scheduler is not None and signal_scheduler._scheduler.running,
        "testnet_connected": paper_exchange.is_live(),
        "testnet_balance":   paper_exchange.get_balance(),
        "alpaca_configured": alpaca_service.is_configured(),
        "last_signals":      signal_scheduler.get_last_signals(),
    }


@app.get("/api/alpaca/account")
async def get_alpaca_account():
    from alpaca_service import get_account
    return get_account()


@app.get("/api/alpaca/positions")
async def get_alpaca_positions():
    from alpaca_service import get_positions
    return {"positions": get_positions()}


@app.get("/api/alpaca/orders")
async def get_alpaca_orders(limit: int = Query(20)):
    from alpaca_service import get_orders_history
    return {"orders": get_orders_history(limit)}


class ChatRequest(BaseModel):
    message: str
    session_id: str
    asset: str = "BTC_USDT"


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Send a message to MARTA's AI assistant (Mistral via Ollama, live system context)."""
    from chat_service import chat as marta_chat
    response = await marta_chat(session_id=request.session_id, user_message=request.message, asset=request.asset)
    return {"response": response, "session_id": request.session_id}


@app.delete("/api/chat/{session_id}")
async def clear_chat(session_id: str):
    from chat_service import clear_history
    clear_history(session_id)
    return {"cleared": True}


@app.get("/api/chat/suggestions/{asset}")
async def get_chat_suggestions(asset: str):
    return {"suggestions": [
        f"Why is MARTA holding on {asset} right now?",
        "What is the current market regime and what does it mean?",
        "Explain the last agent debate result",
        "What would trigger MARTA to enter a trade?",
        "How does the regime encoder work?",
        "What are MARTA's limitations?",
        "Compare MARTA to a simple buy-and-hold strategy",
        "What does the Contrarian agent do?",
    ]}


@app.post("/api/rl/trigger-signal")
async def trigger_signal_now(asset: str = Query("BTC_USDT")):
    """Manually fire one signal tick for `asset` — useful for testing without waiting an hour."""
    import asyncio
    import signal_scheduler
    # Must be `async def`: a sync `def` endpoint runs in FastAPI's worker thread pool,
    # which has no running event loop, so asyncio.create_task() raised
    # "RuntimeError: no running event loop" and silently dropped the tick.
    asyncio.create_task(signal_scheduler._tick_asset(asset))
    return {"status": "tick_queued", "asset": asset}


@app.get("/api/paper/trades")
def get_paper_trades(asset: str = Query("BTC_USDT"), limit: int = Query(50)):
    """Return the paper trading ledger from SQLite (both simulated and testnet-sourced)."""
    import paper_db
    import paper_exchange
    portfolio = paper_db.get_portfolio(asset)
    trades_local = paper_db.get_trades(asset, limit=limit)
    trades_testnet = paper_exchange.get_trade_history(asset, limit=limit) if paper_exchange.is_live() else []
    return {
        "asset": asset,
        "portfolio": portfolio,
        "trades_local": trades_local,
        "trades_testnet": trades_testnet,
        "source": "testnet" if paper_exchange.is_live() else "simulated",
    }


@app.delete("/api/paper/reset")
def reset_paper_portfolio(asset: str = Query("BTC_USDT")):
    """Wipe the local paper portfolio for `asset` (dev / testing only)."""
    import paper_db
    paper_db.reset(asset)
    return {"status": "reset", "asset": asset}


@app.get("/api/trades/reasoning/{trade_id}")
def get_trade_reasoning(trade_id: int):
    """Full agent/regime reasoning chain that produced a specific trade."""
    import paper_db
    reasoning = paper_db.get_trade_reasoning(trade_id)
    if reasoning is None:
        raise HTTPException(status_code=404, detail=f"No reasoning found for trade_id={trade_id}")
    return reasoning


@app.get("/api/trades/history/{asset}")
def get_trade_history(asset: str, limit: int = Query(50)):
    """Last `limit` trades for `asset`, each with reasoning + outcome attached."""
    import paper_db
    return {"asset": asset, "trades": paper_db.get_trades_with_reasoning(asset, limit=limit)}


@app.get("/api/trades/outcomes/{asset}")
def get_trade_outcomes(asset: str):
    """Win rate, avg holding time, avg PnL from closed trade outcomes."""
    import paper_db
    return paper_db.get_trade_outcome_stats(asset)


@app.get("/api/account")
def get_account():
    """The one shared capital pool live trading draws from, plus each asset's current
    exposure as a fraction of it (see paper_db module docstring for why this replaced
    independent per-asset balances)."""
    import paper_db
    return paper_db.get_account_state()


@app.get("/api/settings/approval-mode")
def get_approval_mode_setting():
    import app_settings
    return {"approval_mode": app_settings.get_approval_mode()}


@app.post("/api/settings/approval-mode")
def set_approval_mode_setting(enabled: bool = Query(...)):
    """Toggle approval-gated execution. When on, both the hourly scheduler and manual
    'Run Debate & Execute'/'Trigger Manual Signal' park non-hold signals in the pending
    queue (see /api/trades/pending) instead of executing them immediately."""
    import app_settings
    return {"approval_mode": app_settings.set_approval_mode(enabled)}


@app.get("/api/trades/pending")
def get_pending_trades(status: str = Query("pending")):
    import paper_db
    return {"trades": paper_db.get_pending_trades(status)}


@app.post("/api/trades/pending/{pending_id}/approve")
def approve_pending_trade(pending_id: int):
    import paper_db
    try:
        result = paper_db.approve_pending_trade(pending_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "approved", "pending_id": pending_id, **result}


@app.post("/api/trades/pending/{pending_id}/reject")
def reject_pending_trade(pending_id: int):
    import paper_db
    try:
        paper_db.reject_pending_trade(pending_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "rejected", "pending_id": pending_id}


@app.get("/api/notifications")
def get_notifications(
    unread_only: bool = Query(False),
    notif_type: Optional[str] = Query(None, alias="type"),
):
    items = list(_notifications)
    if unread_only:
        items = [n for n in items if not n["read"]]
    if notif_type:
        items = [n for n in items if n["type"] == notif_type]
    return {
        "notifications": items,
        "unread_count": sum(1 for n in _notifications if not n["read"]),
    }


@app.post("/api/notifications/{notif_id}/read")
def mark_notification_read(notif_id: str):
    for n in _notifications:
        if n["id"] == notif_id:
            n["read"] = True
            return {"status": "ok", "id": notif_id}
    raise HTTPException(status_code=404, detail=f"Notification {notif_id} not found")


@app.post("/api/notifications/read-all")
def mark_all_notifications_read():
    for n in _notifications:
        n["read"] = True
    return {"status": "ok", "count": len(_notifications)}


@app.get("/api/system/status")
async def get_system_status():
    return {
        "gpu": system_service.get_gpu_status(),
        "ollama": await system_service.get_ollama_status(),
        "redis": system_service.get_redis_status(),
    }


@app.get("/api/system/models")
def get_system_models():
    return {
        "encoder": system_service.get_encoder_info(),
        "hmm": system_service.get_hmm_info(),
        "rl": system_service.get_rl_info(),
    }


@app.get("/api/system/ping")
def system_ping():
    return {"ok": True}


@app.get("/api/price/{asset}")
async def get_current_price(asset: str):
    """Current price with automatic fallback: live cache -> direct fetch -> stale cache -> parquet."""
    import price_service
    return await price_service.get_price_with_fallback(asset)


@app.get("/api/system/test-source")
async def test_source(source: str = Query(...)):
    if source == "binance":
        candles = await live_feed.fetch_live_candles("BTC_USDT", limit=2)
        return {"ok": bool(candles), "message": f"Fetched {len(candles)} candles from Binance" if candles else "No data returned from Binance"}
    if source == "rss":
        import news_live
        headlines = await news_live.get_live_news("BTC_USDT", max_items=5)
        return {"ok": bool(headlines), "message": f"Fetched {len(headlines)} headlines from RSS feeds" if headlines else "No headlines returned"}
    if source == "twelve_data":
        if not live_feed.TWELVE_DATA_KEY:
            return {"ok": False, "message": "TWELVE_DATA_KEY not set"}
        # Proxied server-side (not called from the browser) so the key is never exposed
        # to the frontend — same pattern as the binance/rss checks above.
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.twelvedata.com/api_usage",
                    params={"apikey": live_feed.TWELVE_DATA_KEY},
                )
            data = resp.json()
            # daily_usage/plan_daily_limit are the daily credit budget (e.g. 3/800) —
            # current_usage/plan_limit is a *different* pair for the per-minute rate
            # limit (e.g. 1/8) and would silently report the wrong number here.
            used = data.get("daily_usage")
            limit = data.get("plan_daily_limit")
            if used is None or limit is None:
                return {"ok": False, "message": f"Twelve Data error: {data}"}
            remaining = int(limit) - int(used)
            return {"ok": True, "message": f"Twelve Data — {remaining} credits remaining today"}
        except Exception as e:
            return {"ok": False, "message": f"Twelve Data request failed: {e}"}
    if source == "telegram":
        import telegram_service
        if not telegram_service.TELEGRAM_TOKEN or not telegram_service.TELEGRAM_CHAT_ID:
            return {"ok": False, "message": "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"}
        ok = await telegram_service.send_telegram("✅ <b>MARTA</b> — Telegram connection test successful!")
        return {"ok": ok, "message": "Test message sent — check your Telegram chat" if ok else "Send failed — see backend logs"}
    return {"ok": False, "message": f"'{source}' is not wired to a live integration in this backend yet."}


# ── WebSocket: price feed ─────────────────────────────────────────────────────

class PriceConnectionManager:
    def __init__(self):
        self._clients: dict[str, list[WebSocket]] = {}

    def connect(self, asset: str, ws: WebSocket):
        self._clients.setdefault(asset, []).append(ws)

    def disconnect(self, asset: str, ws: WebSocket):
        if asset in self._clients:
            try:
                self._clients[asset].remove(ws)
            except ValueError:
                pass

    async def broadcast(self, asset: str, data: dict):
        dead = []
        for ws in list(self._clients.get(asset, [])):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(asset, ws)


price_manager = PriceConnectionManager()


def _add_regime(candle: dict, asset: str, live: bool = False) -> dict:
    """Attach regime_label and regime_description to a candle dict.

    ``live=True`` uses get_live_regime (fresh PCA->HMM recompute) for a genuinely
    real-time tick; the default historical nearest-lookup (get_regime) is for backfilling
    a batch of already-past candles, where recomputing a live sequence per candle would be
    both wasteful (up to 300 HMM calls per connection) and semantically wrong (those bars
    aren't "now"). Found during the 2026-07-31 audit: this previously always used the
    historical lookup, so the live-tick candles streamed over /ws/price carried a regime
    label frozen at whatever regime_stream.parquet's last precomputed row was.
    """
    try:
        import pandas as pd
        ts = pd.Timestamp(candle["timestamp"], tz="UTC")
        r = regime_service.get_live_regime(asset, ts) if live else regime_service.get_regime(asset, ts)
        return {**candle, "regime_label": r["regime_label"], "regime_description": r.get("regime_description", "")}
    except Exception:
        return {**candle, "regime_label": 0, "regime_description": ""}


@app.websocket("/ws/price/{asset}")
async def ws_price(websocket: WebSocket, asset: str, mode: str = "demo"):
    await websocket.accept()

    if mode == "live":
        # Try to pull real CCXT data first; fall back to historical parquet
        live_candles = await live_feed.fetch_live_candles(asset, limit=300)
        if live_candles:
            enriched = [_add_regime(c, asset) for c in live_candles]
            await websocket.send_json({"type": "history", "candles": enriched})
        else:
            try:
                candles = historical.get_history(asset, 300)
                await websocket.send_json({"type": "history", "candles": candles})
            except WebSocketDisconnect:
                return
            except Exception as e:
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except Exception:
                    pass

        async def forward(candle: dict):
            try:
                await websocket.send_json({"type": "candle", **_add_regime(candle, asset, live=True)})
            except Exception:
                pass

        live_feed.subscribe(asset, forward)
        live_feed.ensure_feed(asset)
        try:
            while True:
                await asyncio.sleep(30)
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            live_feed.unsubscribe(asset, forward)

    else:
        # Historical / demo mode — serve from parquet
        try:
            candles = historical.get_history(asset, 200)
            await websocket.send_json({"type": "history", "candles": candles})
        except WebSocketDisconnect:
            return
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
            return
        try:
            await websocket.receive_text()
        except (WebSocketDisconnect, Exception):
            pass


def _lookup_regime(asset: str, ts: "pd.Timestamp", explicit_timestamp: bool) -> dict:
    """explicit_timestamp=False means the caller asked for "now" (no timestamp
    supplied) — treat that as a live query. An explicit timestamp is always a
    historical "what was true on this date" question, even if it happens to
    resolve close to the present moment."""
    if explicit_timestamp:
        return regime_service.get_regime(asset, ts)
    return regime_service.get_live_regime(asset, ts)


@app.websocket("/ws/regime/{asset}")
async def ws_regime(websocket: WebSocket, asset: str, timestamp: Optional[str] = None):
    await websocket.accept()
    try:
        import pandas as pd
        ts = pd.Timestamp(timestamp, tz="UTC") if timestamp else pd.Timestamp("now", tz="UTC")
        data = _lookup_regime(asset, ts, explicit_timestamp=bool(timestamp))
        await websocket.send_json({"type": "regime", **data})
        # Keep alive; client sends new timestamps to get updated regime
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                payload = json.loads(msg)
                new_ts_raw = payload.get("timestamp")
                new_ts = pd.Timestamp(new_ts_raw, tz="UTC") if new_ts_raw else pd.Timestamp("now", tz="UTC")
                data = _lookup_regime(asset, new_ts, explicit_timestamp=bool(new_ts_raw))
                await websocket.send_json({"type": "regime", **data})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except (WebSocketDisconnect, Exception):
        pass


# ── WebSocket: notifications ───────────────────────────────────────────────────

@app.websocket("/ws/notifications")
async def ws_notifications(websocket: WebSocket):
    await websocket.accept()
    _notification_ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _notification_ws_clients.discard(websocket)


# ── WebSocket: debate stream ──────────────────────────────────────────────────

@app.websocket("/ws/debate")
async def ws_debate(
    websocket: WebSocket,
    mode: str = "replay",
    debate_id: int = 0,
    scenario_id: Optional[str] = None,
    asset: Optional[str] = None,
    timestamp: Optional[str] = None,
):
    await websocket.accept()
    try:
        if scenario_id and scenario_id in SCENARIO_BY_ID:
            scenario = SCENARIO_BY_ID[scenario_id]
            idx = scenario.get("debate_index", 0)
            await debate_stream.stream_debate_replay(websocket, idx)
        elif mode == "live" and asset and timestamp:
            await debate_stream.stream_debate_live(websocket, asset, timestamp)
        else:
            await debate_stream.stream_debate_replay(websocket, debate_id)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
