"""2025+ data extension — fetch through today and extend the existing raw parquets.

Run:  /home/ramy/miniconda3/envs/marta/bin/python pipeline/fetch_2025.py   (from repo root)

Sources per asset:
  - BTC/USDT, ETH/USDT  — CCXT/Binance hourly, 2025-01-01 → today, appended to the
    existing raw parquet (2018-2024 is never re-fetched).
  - EURUSD, XAUUSD      — histdata.com only ships whole-year ZIPs, so this extends
    through 2025-12-31 only (full-year 2025 ZIPs at the repo root:
    DAT_ASCII_{PAIR}_M1_2025.zip). Closing the remaining 2026 partial-year gap for
    forex needs a month-by-month histdata fetch, which fetch_histdata() does not
    support yet — out of scope here.

Idempotent: an asset whose raw parquet already reaches the target end is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow "python pipeline/fetch_2025.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fetch import (  # noqa: E402
    RAW_DIR,
    _cached_raw_covers_end,
    _fetch_crypto_ccxt,
    _sanitize_ohlcv,
    fetch_histdata,
)
from utils.logger import logger  # noqa: E402

FETCH_START_CRYPTO = "2025-01-01"
FETCH_END_CRYPTO = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
FETCH_END_HISTDATA = "2026-01-01"  # covers full-year 2025 only
_MIN_ACCEPTABLE_END_CRYPTO = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
_MIN_ACCEPTABLE_END_HISTDATA = pd.Timestamp("2025-12-20", tz="UTC")

CRYPTO = [("BTC/USDT", "binance"), ("ETH/USDT", "binance")]
HISTDATA_PAIRS = ["eurusd", "xauusd"]  # 2025 ZIPs already at repo root


def _raw_path(asset: str) -> Path:
    return RAW_DIR / f"{asset.replace('/', '_')}_1h_raw.parquet"


def _merge_and_save(asset: str, new_data: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    """Append ``new_data`` to the existing raw parquet, dedupe on timestamp, save."""
    existing = pd.read_parquet(raw_path, engine="pyarrow")
    merged = pd.concat([existing, new_data])
    merged = merged[~merged.index.duplicated(keep="first")]
    merged = merged.sort_index()
    merged = _sanitize_ohlcv(merged)
    merged.to_parquet(raw_path, engine="pyarrow")
    print(f"{asset}: {len(existing)} → {len(merged)} rows, now ends {merged.index[-1]}")
    return merged


def fetch_2025_extension() -> dict[str, pd.DataFrame]:
    """Extend all 4 raw parquets through today (crypto) / end-2025 (forex)."""
    results: dict[str, pd.DataFrame] = {}

    # --- BTC / ETH (CCXT, append 2025-01-01 → today) ---
    for symbol, exchange in CRYPTO:
        asset = symbol.replace("/", "_")
        raw_path = _raw_path(symbol)
        cached = _cached_raw_covers_end(raw_path, FETCH_END_CRYPTO)
        if cached is not None:
            print(f"{asset}: already ends {cached.index[-1]} — skipping fetch")
            results[asset] = cached
            continue
        logger.info(f"[{symbol}] Fetching via CCXT ({FETCH_START_CRYPTO} → {FETCH_END_CRYPTO})")
        new_data = _fetch_crypto_ccxt(symbol, exchange, FETCH_START_CRYPTO, FETCH_END_CRYPTO)
        results[asset] = _merge_and_save(asset, new_data, raw_path)

    # --- EURUSD / XAUUSD (histdata; root 2025 ZIPs + per-year caches reused) ---
    for pair in HISTDATA_PAIRS:
        asset = pair.upper()
        raw_path = _raw_path(asset)
        cached = _cached_raw_covers_end(raw_path, FETCH_END_HISTDATA)
        if cached is not None:
            print(f"{asset}: already ends {cached.index[-1]} — skipping fetch")
            results[asset] = cached
            continue
        zip_path = Path.cwd() / f"DAT_ASCII_{asset}_M1_2025.zip"
        if not zip_path.exists():
            raise FileNotFoundError(
                f"{zip_path} not found — the 2025 histdata ZIP must sit at the repo root"
            )
        # fetch_histdata reuses per-year 1-min caches for 2018-2024 and the root ZIP
        # for 2025, resamples the whole history once, and writes the raw parquet.
        df = fetch_histdata(pair, 2018, 2025, RAW_DIR)
        print(f"{asset}: {len(df)} rows, now ends {df.index[-1]}")
        results[asset] = df

    # --- Verify crypto reaches ~today, forex reaches end-2025 ---
    failures = []
    for asset, df in results.items():
        floor = (
            _MIN_ACCEPTABLE_END_CRYPTO
            if asset in {"BTC_USDT", "ETH_USDT"}
            else _MIN_ACCEPTABLE_END_HISTDATA
        )
        if df.index[-1] < floor:
            failures.append(asset)

    print("\n=== 2025+ extension summary ===")
    for asset, df in results.items():
        print(f"  {asset:<10} {len(df):>7,} rows   {df.index[0]} → {df.index[-1]}")
    if failures:
        raise RuntimeError(f"Assets not reaching target end: {failures}")
    print("BTC/ETH extended through today; EURUSD/XAUUSD extended through 2025-12-31 ✓")
    return results


if __name__ == "__main__":
    fetch_2025_extension()
