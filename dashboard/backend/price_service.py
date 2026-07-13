"""MARTA price service with automatic fallback chain.

Primary:   live_feed's already-running cache (Binance poll loop for crypto,
           Twelve Data poll loop for forex/gold — both update `_price_cache`
           below via `_update_cache`, so a fresh tick is usually already here).
Secondary: a direct Twelve Data call (forex/gold) or yfinance (any asset).
Tertiary:  last known cached price, marked stale.
Last resort: last `close` from the processed features parquet.
"""
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

import live_feed
from historical import ASSET_FILE_MAP, MARTA_ROOT

TWELVE_DATA_KEY = live_feed.TWELVE_DATA_KEY
TWELVEDATA_SYMBOLS = live_feed.FOREX_SYMBOLS  # {"EURUSD": "EUR/USD", "GOLD": "XAU/USD"}

YFINANCE_SYMBOLS = {
    "BTC_USDT": "BTC-USD",
    "ETH_USDT": "ETH-USD",
    "EURUSD":   "EURUSD=X",
    "GOLD":     "GC=F",
}

_CACHE_FRESH_S = 60  # a cache entry this new is treated as a live primary read, not a fallback

# Last known good price per asset — populated by live_feed's poll loops via _update_cache,
# and by this module's own successful fetches.
_price_cache: dict[str, dict] = {}


async def get_price_with_fallback(asset: str) -> dict:
    """Get current price, trying successively staler sources until one works."""
    cached = _price_cache.get(asset)
    if cached and _age_s(cached) < _CACHE_FRESH_S:
        return cached

    if asset in TWELVEDATA_SYMBOLS and TWELVE_DATA_KEY:
        try:
            price = await _fetch_twelvedata(asset)
            if price:
                return _update_cache(asset, price, "twelvedata")
        except Exception as e:
            print(f"[price_service] Twelve Data failed for {asset}: {e}")

    yf_symbol = YFINANCE_SYMBOLS.get(asset)
    if yf_symbol:
        try:
            price = await asyncio.to_thread(_fetch_yfinance, yf_symbol)
            if price:
                return _update_cache(asset, price, "yfinance")
        except Exception as e:
            print(f"[price_service] yfinance failed for {asset}: {e}")

    if cached:
        print(f"[price_service] using stale cached price for {asset}")
        return {**cached, "source": "cache", "stale": True}

    try:
        price = _fetch_from_parquet(asset)
        if price:
            return _update_cache(asset, price, "parquet")
    except Exception as e:
        print(f"[price_service] parquet fallback failed for {asset}: {e}")

    return {"asset": asset, "price": None, "source": "unavailable", "stale": True}


async def _fetch_twelvedata(asset: str) -> float | None:
    symbol = TWELVEDATA_SYMBOLS[asset]
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.twelvedata.com/price",
            params={"symbol": symbol, "apikey": TWELVE_DATA_KEY},
        )
        data = resp.json()
    if "price" in data:
        return float(data["price"])
    return None


def _fetch_yfinance(symbol: str) -> float | None:
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    price = ticker.fast_info.last_price
    if price and price > 0:
        return float(price)
    return None


def _fetch_from_parquet(asset: str) -> float | None:
    """Last `close` from the processed features parquet — absolute last resort."""
    filename = ASSET_FILE_MAP.get(asset)
    if not filename:
        return None
    path = MARTA_ROOT / "data" / "processed" / filename
    if not path.exists():
        return None
    df = pd.read_parquet(path, engine="pyarrow", columns=["close"])
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


def _age_s(entry: dict) -> float:
    ts = datetime.fromisoformat(entry["timestamp"])
    return (datetime.now(timezone.utc) - ts).total_seconds()


def _update_cache(asset: str, price: float, source: str) -> dict:
    entry = {
        "asset": asset,
        "price": price,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stale": False,
    }
    _price_cache[asset] = entry
    return entry


def get_cached_price(asset: str) -> dict | None:
    return _price_cache.get(asset)
