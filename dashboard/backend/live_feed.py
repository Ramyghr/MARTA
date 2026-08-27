"""Live price feed — real CCXT data for the dashboard's live mode.

On first connect the WebSocket handler calls `fetch_live_candles()` to pull the
latest N hourly candles straight from Binance.  After that, `ensure_feed()` starts
a background task that polls every 30 s and pushes the newest candle to all
subscribed callbacks.
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Callable

SYMBOL_MAP = {
    "BTC_USDT": "BTC/USDT",
    "ETH_USDT": "ETH/USDT",
    # Crypto-only panel (2026-08 migration) — same Binance/ccxt path as BTC/ETH above,
    # no bridging needed (unlike EURUSD/GOLD's Twelve Data workaround below): all three
    # are genuinely live-tradeable pairs on the same exchange.
    "BNB_USDT": "BNB/USDT",
    "LTC_USDT": "LTC/USDT",
    "XRP_USDT": "XRP/USDT",
}

# Twelve Data /price symbols, keyed by the dashboard's internal asset id (matches
# historical.ASSET_FILE_MAP and every other asset-id usage in this codebase: GOLD ==
# XAUUSD — NOT "XAUUSD" as the dict key, or the frontend's subscribe("GOLD", ...) would
# never see these candles).
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "")
FOREX_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "GOLD":   "XAU/USD",
}

_subscribers: dict[str, list[Callable]] = {}
_tasks: dict[str, asyncio.Task] = {}
_forex_task: asyncio.Task | None = None

# Most recent Twelve Data tick per asset, kept so a REST caller (historical.get_history)
# can bridge the gap between the parquet's last bar and today without waiting on a WS
# subscriber to be connected.
_latest_price: dict[str, dict] = {}


# ── Public helpers ────────────────────────────────────────────────────────────

def get_latest_price(asset: str) -> dict | None:
    """Most recent Twelve Data tick for `asset`, or None if the forex feed hasn't ticked yet."""
    return _latest_price.get(asset)


def subscribe(asset: str, callback: Callable) -> None:
    _subscribers.setdefault(asset, []).append(callback)


def unsubscribe(asset: str, callback: Callable) -> None:
    if asset in _subscribers:
        try:
            _subscribers[asset].remove(callback)
        except ValueError:
            pass


async def fetch_live_candles(asset: str, limit: int = 300) -> list[dict]:
    """Pull the latest `limit` hourly candles from Binance via CCXT REST.

    Returns an empty list if the asset is not crypto or CCXT is unavailable.
    """
    symbol = SYMBOL_MAP.get(asset)
    if not symbol:
        return []
    try:
        import ccxt.async_support as ccxt  # type: ignore
        exchange = ccxt.binance({"enableRateLimit": True})
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, "1h", limit=limit)
        finally:
            await exchange.close()
        return [_fmt(ts, o, h, l, c, v, asset) for ts, o, h, l, c, v in ohlcv]
    except Exception as e:
        print(f"[live_feed] fetch_live_candles({asset}): {e}")
        return []


def ensure_feed(asset: str) -> None:
    """Start the background polling loop for `asset` if not already running."""
    if asset not in _tasks or _tasks[asset].done():
        loop = asyncio.get_event_loop()
        _tasks[asset] = loop.create_task(_poll_loop(asset))


def start_forex_feed() -> None:
    """Start the Twelve Data forex/commodity polling loop (once, at backend startup)."""
    global _forex_task
    if _forex_task is None or _forex_task.done():
        loop = asyncio.get_event_loop()
        _forex_task = loop.create_task(_poll_forex_loop())


# ── Internal ──────────────────────────────────────────────────────────────────

def _fmt(ts: int, o: float, h: float, l: float, c: float, v: float, asset: str) -> dict:
    return {
        "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "asset": asset,
    }


async def _broadcast(asset: str, candle: dict) -> None:
    dead = []
    for cb in list(_subscribers.get(asset, [])):
        try:
            await cb(candle)
        except Exception:
            dead.append(cb)
    for cb in dead:
        unsubscribe(asset, cb)


async def _poll_loop(asset: str) -> None:
    """Poll Binance REST every 30 s for the latest 1-h candle."""
    symbol = SYMBOL_MAP.get(asset)
    if not symbol:
        return
    try:
        import ccxt.async_support as ccxt  # type: ignore
        exchange = ccxt.binance({"enableRateLimit": True})
        last_ts: int | None = None
        while True:
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, "1h", limit=2)
                if ohlcv:
                    ts, o, h, l, c, v = ohlcv[-1]
                    import price_service
                    price_service._update_cache(asset, c, "binance")
                    if ts != last_ts:
                        last_ts = ts
                        await _broadcast(asset, _fmt(ts, o, h, l, c, v, asset))
            except Exception as e:
                print(f"[live_feed] poll error {asset}: {e}")
            await asyncio.sleep(30)
    except ImportError:
        print("[live_feed] ccxt not available")
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await exchange.close()
        except Exception:
            pass


async def _fetch_forex_price(client, asset: str, symbol: str) -> dict | None:
    """One Twelve Data /price call → a same-OHLC candle at the current price, or None.

    /price returns just the latest trade price (no OHLC bar), so open=high=low=close —
    same shape the old Alpha Vantage FX_INTRADAY candle produced, just a point sample
    instead of a completed bar.
    """
    resp = await client.get(
        "https://api.twelvedata.com/price",
        params={"symbol": symbol, "apikey": TWELVE_DATA_KEY},
    )
    data = resp.json()

    if "price" not in data:
        print(f"[live_feed] Twelve Data {asset} error: {data}")
        return None

    price = float(data["price"])
    now = datetime.now(timezone.utc)
    return {
        "timestamp": now.isoformat(),
        "open": price, "high": price, "low": price, "close": price,
        "volume": 0.0,
        "asset": asset,
    }


async def _poll_forex_loop() -> None:
    """Poll Twelve Data for EURUSD/GOLD.

    Cadence: 8s stagger between the two symbols (well under the 8 req/min limit), then
    the remainder of a ~5-minute cycle. That's 2 requests/~300s = ~576 requests/day —
    comfortably under the 800 credits/day free-tier budget with headroom for the
    Settings page's manual "Test Connection" checks, unlike a tighter cycle (e.g. a
    flat 60s full-cycle) which would run ~2,300 requests/day and exhaust the daily
    budget in well under 24h — the exact failure mode this replaces Alpha Vantage for.
    """
    if not TWELVE_DATA_KEY:
        print("[live_feed] TWELVE_DATA_KEY not set — forex live feed disabled")
        return

    import httpx

    last_ts: dict[str, str] = {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            while True:
                for asset, symbol in FOREX_SYMBOLS.items():
                    try:
                        candle = await _fetch_forex_price(client, asset, symbol)
                        if candle:
                            _latest_price[asset] = candle
                            import price_service
                            price_service._update_cache(asset, candle["close"], "twelvedata")
                            if candle["timestamp"] != last_ts.get(asset):
                                last_ts[asset] = candle["timestamp"]
                                await _broadcast(asset, candle)
                    except Exception as e:
                        print(f"[live_feed] Twelve Data {asset} error: {e}")

                    await asyncio.sleep(8)  # stagger between symbols — stay under 8 req/min

                await asyncio.sleep(284)  # ~292s total per 2-symbol cycle ≈ 5 minutes
    except asyncio.CancelledError:
        pass
