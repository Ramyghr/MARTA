import os
import time
import pandas as pd
from pathlib import Path
from regime_service import get_regime_series, REGIME_DESCRIPTIONS
import live_feed

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))

# Parquet stems for the features files (OHLCV + indicators).
# GOLD uses the v2 histdata file name (14-feature layout).
ASSET_FILE_MAP = {
    "BTC_USDT": "BTC_USDT_features.parquet",
    "ETH_USDT": "ETH_USDT_features.parquet",
    "EURUSD":   "EURUSD_features.parquet",
    "GOLD":     "XAUUSD_features.parquet",
}

# yfinance tickers used to backfill traditional assets.
_YF_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GOLD":   "GC=F",
}

# Binance symbols for crypto assets.
_BINANCE_SYMBOLS = {
    "BTC_USDT": "BTC/USDT",
    "ETH_USDT": "ETH/USDT",
}

_CACHE_TTL_S = 2 * 3600   # re-fetch live extension every 2 h
_STALE_THRESHOLD_H = 2    # consider parquet stale if last bar > 2 h old

# In-memory cache: asset → (DataFrame, loaded_at_monotonic)
_cache: dict[str, tuple[pd.DataFrame, float]] = {}


# ---------------------------------------------------------------------------
# Extension fetch (synchronous)
# ---------------------------------------------------------------------------

_EXTENSION_FETCH_TIMEOUT_MS = 5_000  # bound the live network call — this runs synchronously
# in the request path of /api/data-range and friends. Without an explicit timeout, ccxt's
# default (10s) can still leave a request that's the *first* hit after cache expiry (see
# _CACHE_TTL_S) hanging long enough to trip a frontend AbortController (e.g. the Regime
# page's 15s full-history timeout) if Binance is unreachable from wherever this runs.


def _fetch_extension_binance(symbol: str, since_ms: int, limit: int = 30_000) -> pd.DataFrame:
    """Fetch hourly OHLCV from Binance since `since_ms` (epoch ms)."""
    try:
        import ccxt
        ex = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
            "timeout": _EXTENSION_FETCH_TIMEOUT_MS,
        })
        rows = []
        end_ms = int(time.time() * 1000)
        while since_ms < end_ms and len(rows) < limit:
            batch = ex.fetch_ohlcv(symbol, "1h", since=since_ms, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            since_ms = batch[-1][0] + 1
            if len(batch) < 1000:
                break
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp").sort_index()
    except Exception as exc:
        print(f"[historical] Binance extension fetch failed ({symbol}): {exc}")
        return pd.DataFrame()


def _fetch_extension_yfinance(ticker: str, since: pd.Timestamp) -> pd.DataFrame:
    """Fetch hourly OHLCV from yfinance since `since` (max 730 days back)."""
    try:
        import yfinance as yf
        now = pd.Timestamp.now(tz="UTC")
        days = min(int((now - since).days) + 2, 729)
        period = f"{max(days, 7)}d"
        df = yf.download(ticker, period=period, interval="1h", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        rename = {"adj_close": "close", "adj close": "close"}
        df = df.rename(columns=rename)
        df = df[df.index > since]
        return df[["open", "high", "low", "close", "volume"]].sort_index()
    except Exception as exc:
        print(f"[historical] yfinance extension fetch failed ({ticker}): {exc}")
        return pd.DataFrame()


def _extend(asset: str, base: pd.DataFrame) -> pd.DataFrame:
    """Fetch candles from the exchange/yfinance and append them to `base`.

    Only OHLCV columns are filled for the extension rows; indicator columns
    remain NaN (the dashboard only reads OHLCV for price charts).
    """
    last_ts = base.index.max()
    age_hours = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 3600
    if age_hours < _STALE_THRESHOLD_H:
        return base

    print(f"[historical] {asset}: parquet ends {last_ts.date()} ({age_hours:.0f}h ago) — fetching extension")

    if asset in _BINANCE_SYMBOLS:
        since_ms = int(last_ts.timestamp() * 1000) + 3_600_000
        ext = _fetch_extension_binance(_BINANCE_SYMBOLS[asset], since_ms)
    else:
        ticker = _YF_TICKERS.get(asset)
        if ticker is None:
            return base
        ext = _fetch_extension_yfinance(ticker, last_ts)

    if ext.empty:
        print(f"[historical] {asset}: no extension data returned")
        return base

    # Align extension to base column schema (NaN for missing indicator cols).
    ext = ext.reindex(columns=base.columns)
    combined = pd.concat([base, ext])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    print(f"[historical] {asset}: +{len(ext)} candles → total {len(combined):,} rows (now → {combined.index.max().date()})")
    return combined


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _load(asset: str) -> pd.DataFrame:
    now = time.monotonic()
    cached = _cache.get(asset)
    if cached is not None:
        df, loaded_at = cached
        if now - loaded_at < _CACHE_TTL_S:
            return df

    filename = ASSET_FILE_MAP.get(asset)
    if filename is None:
        raise ValueError(f"Unknown asset: {asset}")
    path = MARTA_ROOT / "data" / "processed" / filename
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)

    df = _extend(asset, df)
    _cache[asset] = (df, now)
    return df


def _invalidate(asset: str | None = None) -> None:
    """Force next access to re-fetch the live extension."""
    if asset:
        _cache.pop(asset, None)
    else:
        _cache.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LIVE_BRIDGE_MAX_AGE_MIN = 10


def _append_live_candle(asset: str, records: list[dict]) -> list[dict]:
    """Bridge the gap between the last historical bar and today's live price.

    live_feed's Twelve Data poll loop keeps the latest EURUSD/GOLD tick in memory;
    if it's fresh (<10 min old) and newer than the last historical bar, append it
    as a synthetic same-OHLC candle so the chart doesn't stop dead at the parquet's
    cutoff (2024-12-31) while waiting on the next full data refresh.
    """
    if asset not in live_feed.FOREX_SYMBOLS:
        return records

    tick = live_feed.get_latest_price(asset)
    if tick is None:
        return records

    tick_ts = pd.Timestamp(tick["timestamp"])
    age_min = (pd.Timestamp.now(tz="UTC") - tick_ts).total_seconds() / 60
    if age_min > _LIVE_BRIDGE_MAX_AGE_MIN:
        return records

    last = records[-1] if records else None
    if last is not None and tick_ts <= pd.Timestamp(last["timestamp"]):
        return records

    synthetic = {
        "timestamp": tick_ts.isoformat(),
        "open": tick["open"], "high": tick["high"], "low": tick["low"], "close": tick["close"],
        "volume": tick.get("volume", 0.0),
        "regime_label": last["regime_label"] if last else 0,
        "regime_description": last["regime_description"] if last else "",
        "is_live": True,
    }
    return records + [synthetic]


def get_history(asset: str, n: int = 200) -> list[dict]:
    df = _load(asset).tail(n)
    records = _build_records(asset, df)
    return _append_live_candle(asset, records)


def get_history_range(asset: str, start: str, end: str) -> list[dict]:
    df = _load(asset)
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    df = df[(df.index >= s) & (df.index <= e)]
    if len(df) > 2000:
        step = len(df) // 2000
        df = df.iloc[::step]
    return _build_records(asset, df)


def get_data_range(asset: str) -> dict:
    df = _load(asset)
    return {
        "asset": asset,
        "start": df.index[0].isoformat(),
        "end":   df.index[-1].isoformat(),
        "n_rows": len(df),
    }


def get_replay_candles(asset: str, around_timestamp: str, n: int = 200) -> list[dict]:
    """Return n candles centred around a timestamp for demo replay."""
    df = _load(asset)
    ts = pd.Timestamp(around_timestamp, tz="UTC")
    idx = df.index.get_indexer([ts], method="nearest")[0]
    start = max(0, idx - n // 2)
    end = min(len(df), start + n)
    return _build_records(asset, df.iloc[start:end])


def _build_records(asset: str, df: pd.DataFrame) -> list[dict]:
    timestamps = df.index.tolist()
    regimes = get_regime_series(asset, timestamps)
    records = []
    for i, (ts, row) in enumerate(df.iterrows()):
        regime_label = regimes[i]["regime_label"] if i < len(regimes) else 0
        records.append({
            "timestamp":          ts.isoformat(),
            "open":               float(row["open"])            if pd.notna(row.get("open"))   else None,
            "high":               float(row["high"])            if pd.notna(row.get("high"))   else None,
            "low":                float(row["low"])             if pd.notna(row.get("low"))    else None,
            "close":              float(row["close"])           if pd.notna(row.get("close"))  else None,
            "volume":             float(row.get("volume", 0))  if pd.notna(row.get("volume")) else 0.0,
            "regime_label":       regime_label,
            "regime_description": REGIME_DESCRIPTIONS.get(regime_label, ""),
        })
    return records
