"""pipeline/live/prices.py — real-time OHLCV feed for all 5 MARTA assets.

Sources:
  BTC/USDT, ETH/USDT  — Binance WebSocket via ccxt.pro (watch_ohlcv)
  EURUSD, XAUUSD — Alpha Vantage REST (FX_INTRADAY / commodity endpoints)
                           with yfinance fallback on rate-limit or API key absence

Logical-to-symbol mapping mirrors utils/assets.py:
  GOLD → XAUUSD

Cache backend: Redis (65-second TTL) with in-memory dict fallback when Redis is
unavailable, so the feed works out of the box without a running Redis instance.

Requirements (all optional — feed degrades gracefully if missing):
  ccxt[pro]   — pip install ccxt[pro]      (Binance WebSocket)
  yfinance    — pip install yfinance        (fallback for FX/commodities)
  redis       — pip install redis           (caching)
  ALPHA_VANTAGE_KEY env var                  (free at alphavantage.co)
  REDIS_URL   env var (default: redis://localhost:6379)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Asset configuration
# ---------------------------------------------------------------------------

# Maps MARTA logical names to exchange/data-source symbols.
_BINANCE_SYMBOLS: dict[str, str] = {
    "BTC_USDT": "BTC/USDT",
    "ETH_USDT": "ETH/USDT",
}

# Alpha Vantage function + symbol for each traditional asset.
_ALPHA_VANTAGE_CONFIG: dict[str, dict[str, str]] = {
    "EURUSD": {"function": "FX_INTRADAY", "from_symbol": "EUR", "to_symbol": "USD"},
    "GOLD":   {"function": "FX_INTRADAY", "from_symbol": "XAU", "to_symbol": "USD"},
}

# yfinance tickers used as fallback.
_YFINANCE_TICKERS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GOLD":   "GC=F",
}

_TRADITIONAL_ASSETS = list(_ALPHA_VANTAGE_CONFIG.keys())
_POLL_INTERVAL_S = 60          # Alpha Vantage: 25 req/min free tier
_CACHE_TTL_S     = 65          # slightly longer than poll interval
_HISTORY_MAXLEN  = 500         # per-asset in-memory ring buffer (candles)


# ---------------------------------------------------------------------------
# Redis / memory cache
# ---------------------------------------------------------------------------

def _make_redis() -> Any | None:
    """Return a Redis client, or None if unavailable."""
    try:
        import redis as _redis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        client = _redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        logger.info(f"[prices] Redis connected at {url}")
        return client
    except Exception as exc:
        logger.warning(f"[prices] Redis not available ({exc}), using in-memory cache")
        return None


class _MemCache:
    """Minimal in-memory TTL cache used when Redis is unavailable."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}

    def set(self, key: str, value: str, ex: int = _CACHE_TTL_S) -> None:
        self._store[key] = (value, time.monotonic() + ex)

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        val, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return val


# ---------------------------------------------------------------------------
# Candle helpers
# ---------------------------------------------------------------------------

def _candle_from_ohlcv(row: list | tuple, symbol: str = "") -> dict[str, Any]:
    """Normalise a CCXT OHLCV list [ts_ms, o, h, l, c, v] to a candle dict."""
    ts_ms, o, h, l, c, v = row[:6]
    return {
        "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
        "open":   float(o),
        "high":   float(h),
        "low":    float(l),
        "close":  float(c),
        "volume": float(v),
        "symbol": symbol,
    }


def _candle_from_av(row: pd.Series, symbol: str = "") -> dict[str, Any]:
    """Build a candle dict from an Alpha Vantage OHLCV row."""
    return {
        "timestamp": row.name.isoformat() if hasattr(row.name, "isoformat") else str(row.name),
        "open":   float(row.get("1. open",  row.get("open",  0))),
        "high":   float(row.get("2. high",  row.get("high",  0))),
        "low":    float(row.get("3. low",   row.get("low",   0))),
        "close":  float(row.get("4. close", row.get("close", 0))),
        "volume": float(row.get("5. volume",row.get("volume",0))),
        "symbol": symbol,
    }


# ---------------------------------------------------------------------------
# Alpha Vantage fetcher (synchronous — called in executor from async loop)
# ---------------------------------------------------------------------------

def _fetch_alpha_vantage_sync(asset: str, api_key: str) -> pd.DataFrame | None:
    """Fetch the latest intraday candles from Alpha Vantage.

    Returns a DataFrame sorted ascending by timestamp, or None on failure.
    """
    try:
        import requests
    except ImportError:
        logger.warning("[prices] requests not installed — Alpha Vantage fetch skipped")
        return None

    cfg = _ALPHA_VANTAGE_CONFIG[asset]
    base = "https://www.alphavantage.co/query"

    if cfg["function"] == "FX_INTRADAY":
        params = {
            "function":    "FX_INTRADAY",
            "from_symbol": cfg["from_symbol"],
            "to_symbol":   cfg["to_symbol"],
            "interval":    "60min",
            "outputsize":  "compact",
            "apikey":      api_key,
        }
        ts_key = "Time Series FX (60min)"
    else:
        params = {
            "function":   "TIME_SERIES_INTRADAY",
            "symbol":     cfg.get("symbol", asset),
            "interval":   "60min",
            "outputsize": "compact",
            "apikey":     api_key,
        }
        ts_key = "Time Series (60min)"

    try:
        resp = requests.get(base, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(f"[prices] Alpha Vantage request failed for {asset}: {exc}")
        return None

    if "Note" in data:
        logger.warning(f"[prices] Alpha Vantage rate limit hit for {asset}")
        return None
    if "Error Message" in data:
        logger.warning(f"[prices] Alpha Vantage error for {asset}: {data['Error Message']}")
        return None

    raw = data.get(ts_key, {})
    if not raw:
        logger.warning(f"[prices] Alpha Vantage: no data for {asset} (key={ts_key!r})")
        return None

    df = pd.DataFrame.from_dict(raw, orient="index")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return df


def _fetch_yfinance_sync(asset: str, n_candles: int = 60) -> pd.DataFrame | None:
    """Fetch recent hourly candles from yfinance as fallback."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("[prices] yfinance not installed — fallback unavailable")
        return None

    ticker = _YFINANCE_TICKERS.get(asset)
    if ticker is None:
        return None

    try:
        df = yf.download(ticker, period="7d", interval="1h", progress=False)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        return df.tail(n_candles)
    except Exception as exc:
        logger.warning(f"[prices] yfinance fallback failed for {asset}: {exc}")
        return None


# ---------------------------------------------------------------------------
# LivePriceFeed
# ---------------------------------------------------------------------------

class LivePriceFeed:
    """Real-time OHLCV feed for all 5 MARTA assets.

    Usage::

        feed = LivePriceFeed()
        await feed.start_streaming()

        candle = await feed.get_latest_candle("BTC_USDT")
        df     = await feed.get_recent_candles("GOLD", n=60)

    Streaming starts background tasks for Binance WebSocket connections and
    Alpha Vantage polling loops.  Call ``stop()`` to cancel them cleanly.
    """

    def __init__(self) -> None:
        self._av_key: str = os.environ.get("ALPHA_VANTAGE_KEY", "")
        self._cache: Any = _make_redis()
        if self._cache is None:
            self._cache = _MemCache()

        # Per-asset in-memory candle ring buffers (most recent last).
        self._history: dict[str, deque[dict[str, Any]]] = {
            a: deque(maxlen=_HISTORY_MAXLEN)
            for a in list(_BINANCE_SYMBOLS) + _TRADITIONAL_ASSETS
        }

        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------ cache

    def _cache_key(self, asset: str) -> str:
        return f"marta:live:candle:{asset}"

    def _cache_set(self, asset: str, candle: dict[str, Any]) -> None:
        import json
        self._cache.set(self._cache_key(asset), json.dumps(candle), ex=_CACHE_TTL_S)

    def _cache_get(self, asset: str) -> dict[str, Any] | None:
        import json
        raw = self._cache.get(self._cache_key(asset))
        return json.loads(raw) if raw else None

    # ------------------------------------------------- internal update helpers

    def _ingest_candle(self, asset: str, candle: dict[str, Any]) -> None:
        """Store a freshly-received candle in both cache and ring buffer."""
        self._cache_set(asset, candle)
        buf = self._history[asset]
        if buf and buf[-1]["timestamp"] == candle["timestamp"]:
            buf[-1] = candle          # update in-flight candle
        else:
            buf.append(candle)

    # ----------------------------------------------- Binance WebSocket tasks

    async def _stream_binance(self, asset: str) -> None:
        """Maintain a persistent Binance 1h kline WebSocket for one asset."""
        try:
            import ccxt.pro as ccxtpro
        except ImportError:
            logger.warning(
                "[prices] ccxt[pro] not installed — Binance WebSocket unavailable. "
                "Install with: pip install ccxt[pro]"
            )
            return

        symbol = _BINANCE_SYMBOLS[asset]
        exchange = ccxtpro.binance({"enableRateLimit": True})
        logger.info(f"[prices] Binance WebSocket starting for {symbol}")

        try:
            while True:
                try:
                    ohlcv_list = await exchange.watch_ohlcv(symbol, "1h")
                    if ohlcv_list:
                        candle = _candle_from_ohlcv(ohlcv_list[-1], symbol=asset)
                        self._ingest_candle(asset, candle)
                        logger.debug(
                            f"[prices] {asset} WebSocket: {candle['timestamp']} "
                            f"close={candle['close']}"
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(f"[prices] Binance WebSocket error for {asset}: {exc}")
                    await asyncio.sleep(5)
        finally:
            await exchange.close()

    # -------------------------------------------- Alpha Vantage polling tasks

    async def _poll_traditional(self, asset: str) -> None:
        """Poll Alpha Vantage every 60 s for one traditional asset; fall back to yfinance."""
        loop = asyncio.get_event_loop()
        logger.info(f"[prices] Alpha Vantage polling starting for {asset}")

        while True:
            try:
                # Try Alpha Vantage first (requires API key).
                if self._av_key:
                    df = await loop.run_in_executor(
                        None, _fetch_alpha_vantage_sync, asset, self._av_key
                    )
                else:
                    df = None

                # Fallback to yfinance.
                if df is None or df.empty:
                    df = await loop.run_in_executor(
                        None, _fetch_yfinance_sync, asset
                    )

                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    candle = _candle_from_av(latest, symbol=asset)
                    self._ingest_candle(asset, candle)
                    logger.debug(
                        f"[prices] {asset}: {candle['timestamp']} close={candle['close']}"
                    )
                else:
                    logger.warning(f"[prices] No data received for {asset}")

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"[prices] Poll error for {asset}: {exc}")

            await asyncio.sleep(_POLL_INTERVAL_S)

    # ----------------------------------------------------------------- public

    async def start_streaming(self) -> None:
        """Start Binance WebSocket tasks and Alpha Vantage polling loops."""
        for asset in _BINANCE_SYMBOLS:
            task = asyncio.create_task(self._stream_binance(asset), name=f"binance_{asset}")
            self._tasks.append(task)

        for asset in _TRADITIONAL_ASSETS:
            task = asyncio.create_task(self._poll_traditional(asset), name=f"av_{asset}")
            self._tasks.append(task)

        logger.info(
            f"[prices] Streaming started: {len(self._tasks)} tasks "
            f"({len(_BINANCE_SYMBOLS)} Binance WS + {len(_TRADITIONAL_ASSETS)} AV polls)"
        )

    async def stop(self) -> None:
        """Cancel all background tasks."""
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("[prices] All streaming tasks stopped")

    async def get_latest_candle(self, asset: str) -> dict[str, Any]:
        """Return the most recent candle for *asset*.

        Returns a dict with keys: timestamp, open, high, low, close, volume, symbol.
        Raises ValueError for unknown assets.
        """
        known = set(_BINANCE_SYMBOLS) | set(_TRADITIONAL_ASSETS)
        if asset not in known:
            raise ValueError(f"Unknown asset: {asset!r}. Known: {sorted(known)}")

        # 1. Try Redis / memory cache first (fast path).
        cached = self._cache_get(asset)
        if cached:
            return cached

        # 2. Fall back to ring buffer.
        buf = self._history.get(asset)
        if buf:
            return buf[-1]

        # 3. Fetch synchronously (cold start).
        loop = asyncio.get_event_loop()
        if asset in _BINANCE_SYMBOLS:
            candle = await self._fetch_binance_once(asset)
        else:
            df = None
            if self._av_key:
                df = await loop.run_in_executor(
                    None, _fetch_alpha_vantage_sync, asset, self._av_key
                )
            if df is None or df.empty:
                df = await loop.run_in_executor(None, _fetch_yfinance_sync, asset)
            candle = _candle_from_av(df.iloc[-1], symbol=asset) if (df is not None and not df.empty) else {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0, "symbol": asset,
            }
        self._ingest_candle(asset, candle)
        return candle

    async def _fetch_binance_once(self, asset: str) -> dict[str, Any]:
        """One-shot Binance REST fetch for a single asset (cold start / fallback)."""
        try:
            import ccxt.async_support as ccxt_async
            symbol = _BINANCE_SYMBOLS[asset]
            exchange = ccxt_async.binance()
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, "1h", limit=1)
                return _candle_from_ohlcv(ohlcv[-1], symbol=asset) if ohlcv else {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0, "symbol": asset,
                }
            finally:
                await exchange.close()
        except Exception as exc:
            logger.warning(f"[prices] Binance one-shot fetch failed for {asset}: {exc}")
            # Try yfinance as last resort for BTC/ETH.
            yf_ticker = {"BTC_USDT": "BTC-USD", "ETH_USDT": "ETH-USD"}.get(asset)
            if yf_ticker:
                try:
                    import yfinance as yf
                    df = yf.download(yf_ticker, period="1d", interval="1h", progress=False)
                    if not df.empty:
                        row = df.iloc[-1]
                        return {
                            "timestamp": df.index[-1].isoformat(),
                            "open": float(row.get("Open", 0)),
                            "high": float(row.get("High", 0)),
                            "low":  float(row.get("Low",  0)),
                            "close":float(row.get("Close",0)),
                            "volume": float(row.get("Volume", 0)),
                            "symbol": asset,
                        }
                except Exception:
                    pass
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0, "symbol": asset,
            }

    async def get_recent_candles(self, asset: str, n: int = 60) -> pd.DataFrame:
        """Return the last *n* candles for *asset* as a DataFrame.

        Falls back to a one-shot historical fetch if the ring buffer has fewer
        than *n* rows.  Used to build the 60-candle encoder input window.
        """
        buf = list(self._history.get(asset, []))
        if len(buf) < n:
            # Attempt a bulk historical pull to warm the buffer.
            loop = asyncio.get_event_loop()
            if asset in _BINANCE_SYMBOLS:
                try:
                    import ccxt.async_support as ccxt_async
                    symbol = _BINANCE_SYMBOLS[asset]
                    exchange = ccxt_async.binance()
                    try:
                        rows = await exchange.fetch_ohlcv(symbol, "1h", limit=n)
                        for row in rows:
                            self._ingest_candle(asset, _candle_from_ohlcv(row, symbol=asset))
                    finally:
                        await exchange.close()
                except Exception as exc:
                    logger.warning(f"[prices] Binance bulk fetch failed for {asset}: {exc}")
            else:
                df_hist = await loop.run_in_executor(None, _fetch_yfinance_sync, asset, n)
                if df_hist is not None and not df_hist.empty:
                    for ts, row in df_hist.iterrows():
                        c = _candle_from_av(row, symbol=asset)
                        c["timestamp"] = ts.isoformat()
                        self._ingest_candle(asset, c)
            buf = list(self._history.get(asset, []))

        recent = buf[-n:] if len(buf) >= n else buf
        if not recent:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(recent).set_index("timestamp")
