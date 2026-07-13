"""Nightly CCXT data refresh for BTC/ETH — designed to run as a cron job.

Fetches any new hourly candles since the last cached row, appends them, and
emits a staleness report to data/freshness.json for the dashboard.

Schedule (add to crontab -e):
    0 2 * * *  /path/to/marta_env/bin/python /path/to/marta/scripts/nightly_refresh.py

Or run manually:
    python scripts/nightly_refresh.py
    python scripts/nightly_refresh.py --dry-run   # report only, no writes
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from utils.logger import logger

_RAW_DIR     = _ROOT / "data" / "raw"
_FRESHNESS   = _ROOT / "data" / "freshness.json"
_CRYPTO_ASSETS = [
    {"symbol": "BTC/USDT", "safe_name": "BTC_USDT", "exchange": "binance"},
    {"symbol": "ETH/USDT", "safe_name": "ETH_USDT", "exchange": "binance"},
]
_STALE_HOURS = 26   # alert if last bar is more than this many hours old


def _last_timestamp(safe_name: str) -> datetime | None:
    """Return the UTC datetime of the last cached candle, or None."""
    import pandas as pd
    path = _RAW_DIR / f"{safe_name}_1h_raw.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        if df.empty:
            return None
        ts = df.index.max()
        return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _hours_stale(last: datetime | None) -> float:
    if last is None:
        return float("inf")
    now = datetime.now(timezone.utc)
    return (now - last).total_seconds() / 3600


def _refresh_asset(symbol: str, safe_name: str, exchange: str, dry_run: bool) -> dict:
    """Fetch new candles since the last cached bar.  Returns a status dict."""
    import pandas as pd

    last = _last_timestamp(safe_name)
    age_h = _hours_stale(last)
    status = {
        "symbol": symbol,
        "last_timestamp": last.isoformat() if last else None,
        "age_hours": round(age_h, 1),
        "stale": age_h > _STALE_HOURS,
        "refreshed": False,
        "new_rows": 0,
        "error": None,
    }

    if age_h <= 1.5:
        logger.info(f"[{symbol}] Already fresh (age={age_h:.1f}h) — skipping fetch")
        return status

    if dry_run:
        logger.info(f"[{symbol}] DRY RUN — would fetch since {last}")
        return status

    try:
        import ccxt
        ex_cls = getattr(ccxt, exchange)
        ex = ex_cls({"enableRateLimit": True, "options": {"defaultType": "spot"}})

        # Start from 1h after the last cached bar, or 30 days ago if no cache
        if last:
            since_ms = int((last.timestamp() + 3600) * 1000)
        else:
            since_ms = int((datetime.now(timezone.utc).timestamp() - 30 * 86400) * 1000)

        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        all_candles = []

        while since_ms < end_ms:
            batch = ex.fetch_ohlcv(symbol, timeframe="1h", since=since_ms, limit=1000)
            if not batch:
                break
            all_candles.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= since_ms:
                break
            since_ms = last_ts + 1
            if last_ts >= end_ms:
                break

        if not all_candles:
            logger.info(f"[{symbol}] No new candles")
            return status

        new_df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
        new_df.set_index("timestamp", inplace=True)
        new_df.sort_index(inplace=True)
        new_df = new_df[~new_df.index.duplicated(keep="last")]

        # Append to existing parquet
        path = _RAW_DIR / f"{safe_name}_1h_raw.parquet"
        if path.exists():
            existing = pd.read_parquet(path, engine="pyarrow")
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new_df

        combined.to_parquet(path, engine="pyarrow")
        new_rows = len(new_df)
        logger.success(f"[{symbol}] +{new_rows} new rows → {path}")
        status["refreshed"] = True
        status["new_rows"] = new_rows
        status["last_timestamp"] = combined.index.max().isoformat()
        status["age_hours"] = 0.0
        status["stale"] = False

    except Exception as exc:
        logger.error(f"[{symbol}] Refresh failed: {exc}")
        status["error"] = str(exc)

    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()

    logger.info(f"Nightly refresh — dry_run={args.dry_run}")
    results = []
    for asset in _CRYPTO_ASSETS:
        r = _refresh_asset(asset["symbol"], asset["safe_name"], asset["exchange"], args.dry_run)
        results.append(r)
        if r["stale"] and not r["refreshed"]:
            logger.warning(f"[{asset['symbol']}] STALE: last bar {r['age_hours']:.1f}h ago")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assets": results,
        "any_stale": any(r["stale"] and not r["refreshed"] for r in results),
    }
    if not args.dry_run:
        _FRESHNESS.parent.mkdir(parents=True, exist_ok=True)
        _FRESHNESS.write_text(json.dumps(report, indent=2))
        logger.success(f"Freshness report saved → {_FRESHNESS}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
