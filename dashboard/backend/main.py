import asyncio
import itertools
import json
import os
import socket
import subprocess
import sys
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# This container has no IPv6 route (confirmed 2026-09-01: AF_INET6 connect() raises
# "Network is unreachable"), but several external hosts we poll in the background
# (api.telegram.org, api.twelvedata.com) still publish AAAA records. httpx/httpcore
# pick whichever address getaddrinfo returns first with no automatic IPv4 fallback,
# so every outbound call had a ~50% chance of picking the dead IPv6 address —
# surfacing as intermittent httpx.ConnectError in the Telegram poll loop and blank/
# DNS-error lines in live_feed's Twelve Data loop. Forcing AF_INET here (before any
# network client is constructed) makes resolution deterministic.
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_only_getaddrinfo

# Load .env from the same directory as this file before anything else imports os.environ
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
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
import strategy_backtest
import auth_db
from auth import APIKeyMiddleware
from session_auth import SessionAuthMiddleware
from oauth_google import router as google_oauth_router, google_oauth_enabled
from oauth_github import router as github_oauth_router, github_oauth_enabled
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
    """Re-run run_rl.py --eval-only to regenerate data/results/ablation_results.json.

    The deployed rl/checkpoints/ppo_best.zip has observation_space Box(-10,10,(868,))
    (confirmed via PPO.load(...).observation_space, 2026-09-03) — that's
    window_size(60)*n_raw_features(14) + CONSENSUS_DIM(16) + portfolio_dim(8) +
    asset_id_dim(4) = 868, i.e. the OLD 4-asset production panel's obs_mode="raw"
    (rl/env.py's default CANONICAL_ASSETS = [BTC,ETH,EURUSD,GOLD], 14 raw features),
    NOT the crypto5_multihorizon panel (22 features, 5 assets -> 1350-dim, confirmed by
    reproducing that exact shape mismatch when --crypto5 was tried here first).
    run_rl.py --eval-only defaults to obs_mode="encoder" (152/156-dim), which crashes
    with a SB3 "Unexpected observation shape" ValueError against this checkpoint — only
    --obs-mode raw (no --crypto5) matches how it was actually trained.
    """
    return subprocess.run(
        ["python", "run_rl.py", "--eval-only", "--obs-mode", "raw"],
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

    # Warm historical.py's per-asset candle cache immediately on boot rather than
    # waiting for the first hourly tick (signal_scheduler._warm_historical_cache) — a
    # fresh restart is exactly when a real user's first dashboard page load would
    # otherwise pay the multi-second-to-10s ccxt backfill cold-cache cost (see
    # historical.warm_cache's docstring).
    asyncio.create_task(asyncio.to_thread(historical.warm_cache, signal_scheduler._ACTIVE_ASSETS))
    # live_feed.start_forex_feed() — NOT started (2026-09-01). EURUSD/GOLD were dropped
    # from /api/assets by the crypto-only migration (RESULTS_AND_METHODOLOGY.md Section 7)
    # and have no remaining frontend consumer, but this Twelve Data poll loop kept running
    # anyway — pure background noise (intermittent DNS/connect errors in the backend logs
    # with nothing reading the result). live_feed.get_latest_price() / _fetch_forex_price()
    # are left in place (historical.py's _append_live_candle degrades harmlessly to a no-op
    # without a live tick) in case EURUSD/GOLD are ever reinstated as tradeable assets.

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

    import telegram_bot
    telegram_bot.start(app)

    yield
    signal_scheduler.stop()
    await telegram_bot.stop()


app = FastAPI(title="MARTA Dashboard API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(APIKeyMiddleware)
app.add_middleware(SessionAuthMiddleware)
# Signs its own "session" cookie (distinct from the marta_session app-login cookie
# above) — only used to hold the short-lived OAuth state/nonce between
# /api/auth/google/login and .../callback (authlib's starlette integration
# requires request.session to exist). SESSION_SECRET should be a random secret in
# .env; the fallback below is fine for localhost-only dev, not for a real deployment.
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "dev-insecure-secret-change-me"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001",
        # .claude/launch.json's marta-frontend-preview — separate ports so the preview
        # tooling doesn't collide with a running docker compose stack on 3001/8001.
        "http://localhost:3010", "http://127.0.0.1:3010",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(google_oauth_router)
app.include_router(github_oauth_router)


# ── Auth ───────────────────────────────────────────────────────────────────────
# Real per-user accounts — SessionAuthMiddleware (session_auth.py) gates every other
# /api/ route on the `marta_session` cookie this sets. auth_db.py holds the actual
# users/sessions sqlite tables (same raw-sqlite3 convention as paper_db.py).

class SignupRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class ProfileUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None


def _set_session_cookie(response, token: str) -> None:
    response.set_cookie(
        key=auth_db.SESSION_COOKIE_NAME,
        value=token,
        max_age=auth_db.SESSION_TTL_DAYS * 24 * 3600,
        httponly=True,
        samesite="lax",
        # Not marked secure= — this dashboard is served over plain http on localhost/
        # a LAN docker deployment (same assumption auth.py's MARTA_API_KEY docstring
        # already makes). Revisit if this is ever put behind real TLS.
        path="/",
    )


def _require_user(request: Request) -> dict:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@app.post("/api/auth/signup")
@limiter.limit("10/minute")
def signup(request: Request, body: SignupRequest):
    import sqlite3
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    try:
        user = auth_db.create_user(body.email, body.password, body.display_name)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with that email already exists")
    token = auth_db.create_session(user["id"])
    response = Response(content=json.dumps({"user": user}), media_type="application/json")
    _set_session_cookie(response, token)
    return response


@app.post("/api/auth/login")
@limiter.limit("20/minute")
def login(request: Request, body: LoginRequest):
    user = auth_db.authenticate(body.email, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = auth_db.create_session(user["id"])
    response = Response(content=json.dumps({"user": user}), media_type="application/json")
    _set_session_cookie(response, token)
    return response


@app.post("/api/auth/logout")
def logout(request: Request):
    token = request.cookies.get(auth_db.SESSION_COOKIE_NAME)
    if token:
        auth_db.delete_session(token)
    response = Response(content=json.dumps({"ok": True}), media_type="application/json")
    response.delete_cookie(auth_db.SESSION_COOKIE_NAME, path="/")
    return response


@app.get("/api/auth/me")
def get_me(request: Request):
    return {"user": _require_user(request)}


@app.get("/api/auth/providers")
def get_auth_providers():
    """Lets the login/signup pages hide the Google/GitHub buttons until the
    corresponding CLIENT_ID/SECRET are actually configured, instead of showing
    a button that 500s."""
    return {"google": google_oauth_enabled(), "github": github_oauth_enabled()}


@app.patch("/api/auth/profile")
def update_profile(request: Request, body: ProfileUpdateRequest):
    user = _require_user(request)
    if body.new_password is not None:
        if not body.current_password:
            raise HTTPException(status_code=400, detail="current_password is required to set a new password")
        if auth_db.authenticate(user["email"], body.current_password) is None:
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        if len(body.new_password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    updated = auth_db.update_profile(user["id"], display_name=body.display_name, new_password=body.new_password)
    return {"user": updated}


# ── HTTP endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/assets")
def get_assets():
    # Crypto-only panel (2026-08 migration) — EURUSD/GOLD dropped from the live asset
    # list (see docs/RESULTS_AND_METHODOLOGY.md Section 7-9); all 5 are genuinely
    # live-tradeable via the same Binance/ccxt feed, no Twelve Data dependency.
    return {
        "assets": [
            {"id": "BTC_USDT",  "label": "BTC/USDT",  "type": "crypto", "live": True},
            {"id": "ETH_USDT",  "label": "ETH/USDT",  "type": "crypto", "live": True},
            {"id": "BNB_USDT",  "label": "BNB/USDT",  "type": "crypto", "live": True},
            {"id": "LTC_USDT",  "label": "LTC/USDT",  "type": "crypto", "live": True},
            {"id": "XRP_USDT",  "label": "XRP/USDT",  "type": "crypto", "live": True},
        ]
    }


@app.get("/api/data-range/{asset}")
def get_data_range(asset: str):
    return historical.get_data_range(asset)


@app.get("/api/regime/semantics")
def get_regime_semantics(asset: str = "BTC_USDT"):
    # Crypto-only panel (2026-08 migration): crypto5 assets read the crypto5
    # multihorizon HMM's own semantics (raw features, no encoder/PCA — see
    # regime_service.py's dispatch); anything else falls back to the legacy live
    # semantics file. The Regime page displays this alongside get_live_regime(), so its
    # stats/descriptions must come from the same model that produced the regime label,
    # not a mismatched one.
    is_crypto5 = asset in regime_service._CRYPTO5_ASSETS
    path = (
        regime_service._CRYPTO5_SEMANTICS_PATH if is_crypto5
        else MARTA_ROOT / "data" / "validated" / "regime_semantics_live.json"
    )
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    # Inject the same dynamically-generated description used everywhere else in the
    # backend so the frontend never has to duplicate the vol/sharpe bucket thresholds —
    # one source of truth for the label.
    descriptions = (
        regime_service._load_crypto5_regime_descriptions() if is_crypto5
        else regime_service._load_regime_descriptions()
    )
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
def get_history(asset: str, n: int = 720):
    # 720h (30 days) — was 200h (~8 days), which read as "the chart only goes back a
    # week" by default. tail(n) is a cheap slice regardless of n, no perf reason to
    # keep it small; the user can still scrub/zoom further back via getHistoryRange.
    return {"candles": historical.get_history(asset, n)}


@app.get("/api/history/{asset}/range")
def get_history_range(asset: str, start: str = Query(...), end: str = Query(...)):
    return {"candles": historical.get_history_range(asset, start, end)}


@app.get("/api/indicators/{asset}/latest")
def get_latest_indicators(asset: str):
    """Latest real (non-extension) indicator values — see historical.get_latest_indicators
    for why this is not simply "the latest candle": recent candles are frequently still
    inside the OHLCV-only live-extension window and would report every indicator as null."""
    try:
        return historical.get_latest_indicators(asset)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


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
    """Re-run --eval-only (--obs-mode raw, matching the deployed checkpoint) and copy its
    suffixed output over the canonical ablation_results.json/ppo_evaluation.json the
    dashboard actually reads. run_rl.py's own suffix naming (checkpoint_suffix -> "_raw")
    exists to stop different experiment configs from clobbering each other's results on
    disk, so eval-only never writes the bare canonical filename itself for a non-default
    obs_mode run — this endpoint is the one place that deliberately promotes that
    suffixed run's output to canonical, since it's regenerating the exact thing the
    dashboard displays.
    """
    try:
        result = _run_ablation_eval()
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Eval timed out after 5 minutes")
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=(result.stderr or result.stdout or "eval failed")[-2000:])
    results_dir = MARTA_ROOT / "data" / "results"
    for stem in ("ablation_results", "ppo_evaluation"):
        suffixed = results_dir / f"{stem}_raw.json"
        if suffixed.exists():
            (results_dir / f"{stem}.json").write_bytes(suffixed.read_bytes())
    return get_ablation()


@app.get("/api/backtest/strategies")
def get_backtest_strategies(asset: str | None = Query(None)):
    return strategy_backtest.list_strategies(asset)


@app.get("/api/backtest/run")
def run_strategy_backtest(
    asset: str = Query(...),
    strategy: str = Query(...),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    try:
        return strategy_backtest.run_backtest(asset, strategy, start, end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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


# Must be registered before /api/debates/{debate_id} — that route's `debate_id: int`
# would otherwise try (and fail) to parse the literal path segment "agreement" as an int,
# since FastAPI/Starlette match routes in registration order and a bare {debate_id}
# path converter matches any string at the routing layer before type validation runs.
@app.get("/api/debates/agreement")
def get_debates_agreement():
    return {"agreement": debate_stream.get_agent_agreement()}


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
    # Live-only t-SNE (run_regime_live.py) — same model as get_live_regime()/the Regime
    # Statistics cards, spanning 2018 -> today. Falls back to the sacred (train<2024)
    # plot only if the live one hasn't been generated yet.
    path = MARTA_ROOT / "regime" / "plots" / "regime_tsne_live.png"
    if not path.exists():
        path = MARTA_ROOT / "regime" / "plots" / "regime_tsne.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="tsne plot not generated yet")
    # No-cache: FileResponse's default headers (Last-Modified/ETag only) let browsers
    # heuristically cache this indefinitely, so a run_regime.py refit's new plot never
    # shows up without a hard refresh. Forces revalidation on every load — cheap, since
    # the browser still gets a 304 when the file hasn't actually changed.
    return FileResponse(
        path, media_type="image/png",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


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
    request: Request,
    asset: str = Query("BTC_USDT"),
    debate_action: Optional[str] = Query(None),
):
    import live_signal
    user = _require_user(request)
    return live_signal.get_live_signal(user["id"], asset, debate_action)


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


@app.get("/api/chat/conversations")
def list_chat_conversations(limit: int = Query(50)):
    """Past conversations for the history panel — title, asset, message count,
    most-recently-updated first. Empty conversations (created but never messaged)
    are excluded by chat_db.list_conversations itself."""
    import chat_db
    return {"conversations": chat_db.list_conversations(limit)}


@app.get("/api/chat/conversations/{session_id}")
def get_chat_conversation(session_id: str):
    """Full message history for one conversation — used to reload it into the chat
    widget when the user picks it from the history panel."""
    import chat_db
    return {"session_id": session_id, "messages": chat_db.get_conversation(session_id)}


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
    import signal_scheduler
    # Was fire-and-forget (asyncio.create_task, returning immediately with no result data)
    # — the frontend then guessed completion with a hardcoded 4s setTimeout before
    # refetching, so a slow tick (PPO/GARCH inference in a thread executor, can run
    # longer than 4s under load) silently refetched stale state and the button looked
    # like it did nothing. _tick_asset is already `async def` and does its blocking work
    # via run_in_executor, so it's safe to await here directly — returns the real fresh
    # signal instead of an opaque "queued" ack.
    await signal_scheduler._tick_asset(asset)
    return {"status": "tick_complete", "asset": asset, "signal": signal_scheduler.get_last_signals().get(asset)}


@app.get("/api/paper/trades")
def get_paper_trades(request: Request, asset: str = Query("BTC_USDT"), limit: int = Query(50)):
    """Return the paper trading ledger from SQLite (both simulated and testnet-sourced)."""
    import paper_db
    import paper_exchange
    user = _require_user(request)
    portfolio = paper_db.get_portfolio(user["id"], asset)
    trades_local = paper_db.get_trades(user["id"], asset, limit=limit)
    trades_testnet = paper_exchange.get_trade_history(asset, limit=limit) if paper_exchange.is_live() else []
    return {
        "asset": asset,
        "portfolio": portfolio,
        "trades_local": trades_local,
        "trades_testnet": trades_testnet,
        "source": "testnet" if paper_exchange.is_live() else "simulated",
    }


@app.delete("/api/paper/reset")
def reset_paper_portfolio(request: Request, asset: str = Query("BTC_USDT")):
    """Wipe the local paper portfolio for `asset` (dev / testing only)."""
    import paper_db
    user = _require_user(request)
    paper_db.reset(user["id"], asset)
    return {"status": "reset", "asset": asset}


@app.get("/api/trades/reasoning/{trade_id}")
def get_trade_reasoning(request: Request, trade_id: int):
    """Full agent/regime reasoning chain that produced a specific trade."""
    import paper_db
    user = _require_user(request)
    reasoning = paper_db.get_trade_reasoning(trade_id, user["id"])
    if reasoning is None:
        raise HTTPException(status_code=404, detail=f"No reasoning found for trade_id={trade_id}")
    return reasoning


@app.get("/api/trades/history/{asset}")
def get_trade_history(request: Request, asset: str, limit: int = Query(50)):
    """Last `limit` trades for `asset`, each with reasoning + outcome attached."""
    import paper_db
    user = _require_user(request)
    return {"asset": asset, "trades": paper_db.get_trades_with_reasoning(user["id"], asset, limit=limit)}


@app.get("/api/trades/outcomes/{asset}")
def get_trade_outcomes(request: Request, asset: str):
    """Win rate, avg holding time, avg PnL from closed trade outcomes."""
    import paper_db
    user = _require_user(request)
    return paper_db.get_trade_outcome_stats(user["id"], asset)


@app.get("/api/account")
def get_account(request: Request):
    """This user's own shared capital pool, plus each asset's current exposure as a
    fraction of it (see paper_db module docstring — per-user since the multi-user
    migration, previously one pool for the whole app)."""
    import paper_db
    user = _require_user(request)
    return paper_db.get_account_state(user["id"])


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


@app.get("/api/settings/rl-execution")
def get_rl_execution_setting():
    """RL execution toggle — default off. See docs/RESULTS_AND_METHODOLOGY.md Section 9:
    a complete, controlled ablation on the crypto5 panel found no RL configuration with
    a statistically defensible edge over buy-and-hold. LLM consensus + static
    (rule-based) execution is the default signal source; RL is opt-in only."""
    import app_settings
    return {
        "rl_execution_enabled": app_settings.get_rl_execution_enabled(),
        "warning": (
            "RL execution is experimental. A controlled ablation found no crypto5 "
            "configuration with a confirmed edge over buy-and-hold — enabling this "
            "trades on a model without a validated advantage."
        ),
    }


@app.post("/api/settings/rl-execution")
def set_rl_execution_setting(enabled: bool = Query(...)):
    import app_settings
    return {"rl_execution_enabled": app_settings.set_rl_execution_enabled(enabled)}


@app.get("/api/settings/weekly-ppo")
def get_weekly_ppo_setting():
    """Weekly-cadence (decision_interval=168) PPO toggle — a different checkpoint from
    rl-execution above, validated with a positive Sharpe on BTC_USDT
    (rl/train_weekly_smoketest.py). BTC_USDT-only; takes over that asset's signal
    entirely when on (see live_signal.py)."""
    import app_settings
    return {
        "weekly_ppo_enabled": app_settings.get_weekly_ppo_enabled(),
        "weekly_ppo_obs_mode": app_settings.get_weekly_ppo_obs_mode(),
    }


@app.post("/api/settings/weekly-ppo")
def set_weekly_ppo_setting(enabled: bool = Query(...), obs_mode: Optional[str] = Query(None)):
    import app_settings
    import weekly_ppo_live
    was_enabled = app_settings.get_weekly_ppo_enabled()
    if obs_mode is not None:
        app_settings.set_weekly_ppo_obs_mode(obs_mode)
    result = app_settings.set_weekly_ppo_enabled(enabled)
    if enabled and not was_enabled:
        # Fresh on-switch: discard any stale decision state from a previous enabled
        # period (see weekly_ppo_live.reset_state's docstring — a stale multi-week-old
        # timestamp would catch up a long, possibly-irrelevant return history under the
        # wrong regime context on the very next tick).
        weekly_ppo_live.reset_state()
    return {"weekly_ppo_enabled": result, "weekly_ppo_obs_mode": app_settings.get_weekly_ppo_obs_mode()}


@app.get("/api/trades/pending")
def get_pending_trades(request: Request, status: str = Query("pending")):
    import paper_db
    user = _require_user(request)
    return {"trades": paper_db.get_pending_trades(user["id"], status)}


@app.post("/api/trades/pending/{pending_id}/approve")
def approve_pending_trade(request: Request, pending_id: int):
    import paper_db
    user = _require_user(request)
    try:
        result = paper_db.approve_pending_trade(pending_id, user["id"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "approved", "pending_id": pending_id, **result}


@app.post("/api/trades/pending/{pending_id}/reject")
def reject_pending_trade(request: Request, pending_id: int):
    import paper_db
    user = _require_user(request)
    try:
        paper_db.reject_pending_trade(pending_id, user["id"])
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
    if source == "newsapi":
        api_key = os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWS_API_KEY")
        if not api_key:
            return {"ok": False, "message": "NEWSAPI_KEY not set"}
        # pipeline/ is mounted at MARTA_ROOT in the container, not on sys.path by
        # default — same fix live_feature_bridge.py needed for the same reason.
        if str(MARTA_ROOT) not in sys.path:
            sys.path.insert(0, str(MARTA_ROOT))
        from pipeline.live.news import _fetch_newsapi_sync
        articles = await asyncio.to_thread(_fetch_newsapi_sync, "BTC_USDT")
        return {"ok": bool(articles), "message": f"Fetched {len(articles)} articles from NewsAPI" if articles else "No articles returned (bad key, rate limit, or network)"}
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


async def _ws_require_user(websocket: WebSocket) -> Optional[dict]:
    """WebSocket routes bypass SessionAuthMiddleware entirely (BaseHTTPMiddleware
    doesn't wrap the WS handshake — same caveat auth.py's APIKeyMiddleware already
    documents), so every `@app.websocket` route checks the session cookie directly
    before accepting the connection. The WS handshake is a regular HTTP GET under
    the hood, so the browser still attaches the httpOnly `marta_session` cookie
    (SameSite=Lax, same-site since frontend/backend are both on localhost just on
    different ports) — no need to smuggle the token through a query param.
    """
    token = websocket.cookies.get(auth_db.SESSION_COOKIE_NAME)
    user = auth_db.get_session_user(token) if token else None
    if user is None:
        await websocket.close(code=4401)
    return user


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


def _add_regime_batch(candles: list[dict], asset: str) -> list[dict]:
    """Batch counterpart to _add_regime(..., live=False) for a whole history backfill.

    Found 2026-09-16: the /ws/price live-mode history batch (below) was tagging each
    candle with the plain _add_regime default, i.e. regime_service.get_regime() — a pure
    nearest-neighbor lookup into the frozen regime_stream.parquet. Since that static
    stream stops growing the moment run_regime.py last ran (2025-12-31 for the crypto5
    causal stream), every candle timestamp after that boundary — which by now is the
    entire visible 720h/30-day chart window — resolves via "nearest" to that exact same
    last frozen row, so the whole chart showed one identical regime label. historical.py's
    get_history()/_build_records() already avoided this via
    regime_service.get_regime_series_live_aware (nearest hit in the incrementally-growing
    regime_live_log.parquet past the frozen boundary, falling back to one shared fresh
    get_live_regime() compute only if the log has no nearby entry) — this reuses the same
    function so the live-mode WS history batch gets the same fix instead of a
    reimplementation.
    """
    try:
        import pandas as pd
        timestamps = [pd.Timestamp(c["timestamp"], tz="UTC") for c in candles]
        regimes = regime_service.get_regime_series_live_aware(asset, timestamps)
        return [
            {**c, "regime_label": r["regime_label"], "regime_description": r.get("regime_description", "")}
            for c, r in zip(candles, regimes)
        ]
    except Exception:
        return [{**c, "regime_label": 0, "regime_description": ""} for c in candles]


@app.websocket("/ws/price/{asset}")
async def ws_price(websocket: WebSocket, asset: str, mode: str = "demo"):
    if await _ws_require_user(websocket) is None:
        return
    await websocket.accept()

    if mode == "live":
        # Try to pull real CCXT data first; fall back to historical parquet.
        # 720h (30 days) — was 300h (~12.5 days); Binance's fetch_ohlcv caps limit at
        # 1000, so this is well within range. See /api/history's get_history for the
        # same "chart only shows a week" fix.
        live_candles = await live_feed.fetch_live_candles(asset, limit=720)
        if live_candles:
            enriched = _add_regime_batch(live_candles, asset)
            await websocket.send_json({"type": "history", "candles": enriched})
        else:
            try:
                candles = historical.get_history(asset, 720)
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
        # Historical / demo mode — serve from parquet. 720h, same reasoning as above.
        try:
            candles = historical.get_history(asset, 720)
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
    if await _ws_require_user(websocket) is None:
        return
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
    if await _ws_require_user(websocket) is None:
        return
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
    if await _ws_require_user(websocket) is None:
        return
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
