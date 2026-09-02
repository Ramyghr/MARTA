"""Autonomous hourly signal loop for MARTA live paper trading.

APScheduler runs `_run_signal_tick()` every 60 minutes (aligned to the top of
the hour so signals land on the same hourly candle boundary as training data).

Each tick:
  0. Refreshes all 4 assets' live-bridge normalized features (live_feature_bridge.py)
     so this hour's regime lookups — trading signal below, and every other
     get_live_regime() caller — read current data, not last hour's.
  1. Calls live_signal.get_live_signal() for each active crypto asset.
  2. If a real testnet exchange is connected (paper_exchange.is_live()):
       — cancels stale open orders
       — places a market order matching the PPO/regime action
  3. Always updates the local SQLite paper_db (even without testnet).
  4. Logs each tick to MLflow (experiment: marta_live_paper).
  5. Logs EURUSD/GOLD's live regime (they skip steps 1-3 — no trading loop for them).

Disabled assets: EURUSD / GOLD have no real CCXT live feed from Binance, so they stay
in the dashboard's historical-replay mode only for *trading* — their regime data is
still kept live via step 0/5 above.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_ACTIVE_ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]

_scheduler: Optional["AsyncIOScheduler"] = None
_last_signals: dict[str, dict] = {}

# live_signal.get_live_signal()/_fetch_latest_price() are fully synchronous — regime
# lookup, PPO inference, a live Binance network call, and a SQLite write, with no
# internal awaits. Calling them directly from this async function blocked the entire
# event loop for the whole chain (measured ~9s), stalling every other request/websocket
# on the server for that long — same class of bug as the live-debate blocking issue
# fixed in debate_stream.py. max_workers=2 so a scheduled hourly tick and a manual
# "Run Debate & Execute" trigger-signal call don't queue behind each other.
_tick_executor = ThreadPoolExecutor(max_workers=2)


def start(app=None) -> None:  # noqa: ARG001
    """Start the APScheduler background job. Call once from main.py lifespan."""
    global _scheduler
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    import paper_exchange
    paper_exchange.init()

    _scheduler = AsyncIOScheduler(timezone="UTC")
    # Top of every hour  (minute=0) — matches hourly training candle boundary
    _scheduler.add_job(
        _run_signal_tick,
        CronTrigger(minute=0),
        id="hourly_signal",
        replace_existing=True,
        misfire_grace_time=300,   # fire up to 5 min late if server was busy
    )
    _scheduler.start()
    logger.info("signal_scheduler: started — firing every hour on the :00")


def stop() -> None:
    """Shutdown the scheduler gracefully. Call from main.py lifespan cleanup."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("signal_scheduler: stopped")


def get_last_signals() -> dict[str, dict]:
    """Return the most recent signal dict per asset (for the dashboard status bar)."""
    return dict(_last_signals)


# ── Internal tick ─────────────────────────────────────────────────────────────

async def _run_signal_tick() -> None:
    """One hour = one tick. Runs for all active assets sequentially."""
    now = datetime.now(timezone.utc).isoformat()
    logger.info(f"signal_scheduler: tick at {now}")

    # Refresh all 5 crypto5 assets' live-bridge normalized features FIRST, before
    # generating any signal below — so this hour's trading signal (and every
    # get_live_regime() caller this hour: current-regime badge, chart tagging, chat)
    # reads this hour's data, not last hour's. A failure here must never block the
    # trading loop that follows.
    await _refresh_live_bridges()

    for asset in _ACTIVE_ASSETS:
        try:
            await _tick_asset(asset)
        except Exception as exc:
            logger.warning(f"signal_scheduler: tick failed for {asset} — {exc}")

    # _log_nontrading_regimes() (EURUSD/GOLD hourly regime-only logging) — NOT called
    # (2026-09-01). Confirmed no frontend path can reach these assets anymore
    # (dropped from /api/assets by the crypto-only migration; PriceChart.tsx/
    # dashboard/page.tsx's remaining EURUSD/GOLD references are inert formatting
    # branches an unreachable asset id never triggers), so this was hourly compute
    # spent updating regime_live_log.parquet rows nothing reads. Left defined below
    # in case EURUSD/GOLD trading is ever reinstated.


async def _refresh_live_bridges() -> None:
    """Keep all 5 crypto5 assets' live-bridge normalized features current (see
    live_feature_bridge.py — this is what get_live_regime() reads). Runs in the executor:
    network calls (Binance) + pandas recomputation are blocking.
    """
    loop = asyncio.get_running_loop()
    for asset in _ACTIVE_ASSETS:
        try:
            await loop.run_in_executor(_tick_executor, _update_live_bridge_sync, asset)
        except Exception as exc:
            logger.warning(f"signal_scheduler: live bridge refresh failed for {asset} — {exc}")


def _update_live_bridge_sync(asset: str) -> None:
    import yaml
    import live_feature_bridge

    with open(live_feature_bridge.MARTA_ROOT / "configs" / "assets.yaml") as f:
        config = yaml.safe_load(f)
    result = live_feature_bridge.update_live_bridge(asset, config)
    if result:
        logger.info(f"signal_scheduler: live bridge refreshed for {asset} -> {result}")


async def _log_nontrading_regimes() -> None:
    """EURUSD/GOLD don't go through _tick_asset (no trading loop), which is what logs
    BTC/ETH's live regime to regime_live_log.parquet every hour (via live_signal.py's
    log=True call). Log theirs here instead, same cadence — see _run_signal_tick.
    """
    loop = asyncio.get_running_loop()
    for asset in ("EURUSD", "GOLD"):
        try:
            await loop.run_in_executor(_tick_executor, _log_live_regime_sync, asset)
        except Exception as exc:
            logger.warning(f"signal_scheduler: live regime logging failed for {asset} — {exc}")


def _log_live_regime_sync(asset: str) -> None:
    import pandas as pd
    import regime_service

    regime_service.get_live_regime(asset, pd.Timestamp.now(tz="UTC"), log=True)


async def _tick_asset(asset: str) -> None:
    import live_signal
    import paper_exchange
    import paper_db
    import alpaca_service

    previous = _last_signals.get(asset, {})
    previous_regime = previous.get("regime_label")
    previous_desc = previous.get("regime_description", "")

    loop = asyncio.get_running_loop()

    # Close out any open positions using this tick's price before opening a new signal,
    # so an outcome never straddles the trade that's about to replace it.
    price_now = await loop.run_in_executor(_tick_executor, live_signal._fetch_latest_price, asset)
    if price_now > 0:
        _update_trade_outcomes(asset, price_now)

    # Generate signal (runs PPO or regime_policy internally)
    signal = await loop.run_in_executor(_tick_executor, live_signal.get_live_signal, asset)
    _last_signals[asset] = {**signal, "scheduler_fired_at": datetime.now(timezone.utc).isoformat()}

    action  = signal["action"]
    size    = signal["position_size"]
    price   = signal["price"]
    regime  = signal["regime_label"]
    regime_desc = signal.get("regime_description", "")
    source  = signal["source"]

    logger.info(
        f"signal_scheduler [{asset}]: {action.upper()} size={size:.4f} "
        f"price={price:.2f} regime={regime} source={source}"
    )

    pending_approval = bool(signal.get("pending_approval"))

    _emit_notifications(
        asset=asset, action=action, size=size, regime=regime, regime_desc=regime_desc,
        previous_regime=previous_regime, previous_desc=previous_desc, source=source,
        pending_approval=pending_approval,
    )

    # Real/practice order placement (only when credentials are configured). Skipped while
    # this signal is sitting in the approval queue — real order placement must wait for
    # the same human approval gate as the local paper_db execution (live_signal.py); it
    # fires from the /approve endpoint instead, once a person confirms the trade.
    # Alpaca takes priority over the Binance testnet when both are configured — it's the
    # dedicated paper-trading API, testnet is the fallback. paper_db is always updated by
    # live_signal.get_live_signal() above regardless of which (if either) of these fire;
    # it remains the dashboard's source of truth for portfolio value / trade log either way.
    if not pending_approval and price > 0:
        if alpaca_service.is_configured():
            order = alpaca_service.place_order(asset, action, size, price)
            if order.get("source") == "alpaca" and order.get("order_id"):
                logger.info(f"signal_scheduler [{asset}]: Alpaca order placed — id={order.get('order_id')}")
        elif paper_exchange.is_live():
            paper_exchange.cancel_all_orders(asset)
            portfolio = paper_db.get_portfolio(asset)
            order = paper_exchange.place_order(
                asset=asset,
                action=action,
                position_fraction=size,
                portfolio_value=float(portfolio.get("portfolio_value", 10000.0)),
                price=price,
            )
            if order:
                logger.info(f"signal_scheduler [{asset}]: testnet order placed — id={order.get('id')}")

    # MLflow logging (best-effort — never blocks the tick)
    _log_mlflow(asset, signal)


def _emit_notifications(
    asset: str, action: str, size: float, regime: int, regime_desc: str,
    previous_regime: Optional[int], previous_desc: str, source: str,
    pending_approval: bool = False,
) -> None:
    """Push regime-shift / crisis / trade-signal notifications for this tick.

    Crisis alerts fire only on entry into regime 2 (inside the shift check), not on
    every tick the asset happens to still be sitting in it — the latter would spam
    a new "CRISIS DETECTED" notification every hour for as long as the regime holds.
    """
    import main as main_module
    import telegram_service

    if previous_regime is not None and regime != previous_regime:
        severity = "critical" if regime == 2 else "warning"
        main_module._add_notification(
            type="regime_shift",
            title=f"Regime Shift: {previous_desc} → {regime_desc}",
            message=f"{asset} regime changed from R{previous_regime} to R{regime}",
            severity=severity,
        )
        asyncio.create_task(telegram_service.notify_regime_shift(
            asset=asset, prev_regime=previous_regime, prev_desc=previous_desc,
            new_regime=regime, new_desc=regime_desc,
        ))
        if regime == 2:
            main_module._add_notification(
                type="regime_shift",
                title="⚠ CRISIS REGIME DETECTED",
                message=f"{asset} entered Stress/Crisis regime — positions forced flat",
                severity="critical",
            )

    if action != "hold" and pending_approval:
        main_module._add_notification(
            type="approval_required",
            title=f"Approval needed: {action.upper()} {asset}",
            message=f"Position: {size:.1%} | Regime: {regime_desc} | Source: {source} — review on the Trading page",
            severity="warning",
        )
    elif action != "hold":
        main_module._add_notification(
            type="trade_signal",
            title=f"Trade Signal: {action.upper()} {asset}",
            message=f"Position: {size:.1%} | Regime: {regime_desc} | Source: {source}",
            severity="info",
        )
        asyncio.create_task(telegram_service.notify_trade_signal(
            asset=asset, action=action, position_size=size,
            regime_label=regime, regime_desc=regime_desc, source=source,
        ))


def _update_trade_outcomes(asset: str, current_price: float) -> None:
    """Close out open paper trades whose price has moved >0.5% since entry.

    Direction-aware: a "profit" for a short position means price fell, not rose.
    """
    import paper_db
    import regime_service
    import pandas as pd

    open_trades = paper_db.get_open_trades(asset)
    if not open_trades:
        return

    regime_label = regime_service.get_live_regime(asset, pd.Timestamp("now", tz="UTC")).get("regime_label")

    for trade in open_trades:
        direction = 1 if trade["action"] == "long" else -1 if trade["action"] == "short" else 0
        if direction == 0 or trade["price"] <= 0:
            continue
        pnl = direction * (current_price - trade["price"]) / trade["price"]
        if abs(pnl) > 0.005:  # 0.5% threshold to close
            paper_db.close_trade_outcome(
                trade_id=trade["id"],
                exit_price=current_price,
                realized_pnl=pnl,
                outcome="profit" if pnl > 0 else "loss",
                regime_at_exit=regime_label,
            )


def _log_mlflow(asset: str, signal: dict) -> None:
    try:
        import mlflow
        with mlflow.start_run(
            experiment_id=_get_or_create_experiment("marta_live_paper"),
            run_name=f"signal_{asset}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M')}",
            nested=False,
        ):
            mlflow.log_params({
                "asset":  asset,
                "source": signal.get("source", ""),
                "regime": signal.get("regime_label", -1),
            })
            mlflow.log_metrics({
                "action_long":  1.0 if signal.get("action") == "long"  else 0.0,
                "action_short": 1.0 if signal.get("action") == "short" else 0.0,
                "position_size": float(signal.get("position_size", 0)),
                "price":         float(signal.get("price", 0)),
                "paper_return":  float(signal.get("paper_total_return", 0)),
            })
    except Exception:
        pass   # MLflow is optional — never crash the scheduler tick


def _get_or_create_experiment(name: str) -> str:
    try:
        import mlflow
        exp = mlflow.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        return mlflow.create_experiment(name)
    except Exception:
        return "0"
