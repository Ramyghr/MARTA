"""2024 data extension — fetch ONLY the 2024 year and extend the existing raw parquets.

Run:  /home/ramy/miniconda3/envs/marta/bin/python pipeline/fetch_2024.py   (from repo root)

Sources per asset:
  - BTC/USDT, ETH/USDT  — CCXT/Binance hourly, 2024-01-01 → 2025-01-01 only,
    appended to the existing raw parquet (2018-2023 is never re-fetched).
  - EURUSD, XAUUSD      — histdata.com 2024 M1 ZIPs already sitting at the repo root
    (DAT_ASCII_{PAIR}_M1_2024.zip). Handled by the existing cache-aware
    ``fetch_histdata`` which reuses root ZIPs + per-year 1-minute parquet caches and
    resamples the full 1-minute history to hourly in a single pass (avoids a
    year-boundary resampling seam that a separate append would create).

WTIUSD was removed from the project (2026-07): histdata has no WTI coverage past
2023-12-01 and the only fallback (yfinance) cannot provide genuine hourly bars for
the full test year — synthetic daily-upsampled intraday data undermines the
encoder's integrity. MARTA runs on 4 genuine assets.

Idempotent: an asset whose raw parquet already reaches the end of 2024 is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow "python pipeline/fetch_2024.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fetch import (  # noqa: E402
    RAW_DIR,
    _cached_raw_covers_end,
    _fetch_crypto_ccxt,
    _sanitize_ohlcv,
    fetch_histdata,
)
from utils.logger import logger  # noqa: E402

FETCH_START = "2024-01-01"
FETCH_END = "2025-01-01"
# An asset is considered complete when its raw reaches within ~2 days of FETCH_END
# (markets close before midnight New Year's Eve; forex stops Friday etc.).
_MIN_ACCEPTABLE_END = pd.Timestamp("2024-12-20", tz="UTC")

CRYPTO = [("BTC/USDT", "binance"), ("ETH/USDT", "binance")]
HISTDATA_PAIRS = ["eurusd", "xauusd"]  # 2024 ZIPs already at repo root


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
    print(
        f"{asset}: {len(existing)} → {len(merged)} rows, now ends {merged.index[-1]}"
    )
    return merged


def fetch_2024_extension() -> dict[str, pd.DataFrame]:
    """Extend all 4 raw parquets through 2024. Never re-fetches 2018-2023."""
    results: dict[str, pd.DataFrame] = {}

    # --- BTC / ETH (CCXT, append 2024 only) ---
    for symbol, exchange in CRYPTO:
        asset = symbol.replace("/", "_")
        raw_path = _raw_path(symbol)
        cached = _cached_raw_covers_end(raw_path, FETCH_END)
        if cached is not None:
            print(f"{asset}: already ends {cached.index[-1]} — skipping fetch")
            results[asset] = cached
            continue
        logger.info(f"[{symbol}] Fetching 2024 via CCXT ({FETCH_START} → {FETCH_END})")
        new_data = _fetch_crypto_ccxt(symbol, exchange, FETCH_START, FETCH_END)
        results[asset] = _merge_and_save(asset, new_data, raw_path)

    # --- EURUSD / XAUUSD (histdata; root 2024 ZIPs + per-year caches reused) ---
    for pair in HISTDATA_PAIRS:
        asset = pair.upper()
        raw_path = _raw_path(asset)
        cached = _cached_raw_covers_end(raw_path, FETCH_END)
        if cached is not None:
            print(f"{asset}: already ends {cached.index[-1]} — skipping fetch")
            results[asset] = cached
            continue
        zip_path = Path.cwd() / f"DAT_ASCII_{asset}_M1_2024.zip"
        if not zip_path.exists():
            raise FileNotFoundError(
                f"{zip_path} not found — the 2024 histdata ZIP must sit at the repo root"
            )
        # fetch_histdata reuses per-year 1-min caches for 2018-2023 and the root ZIP
        # for 2024, resamples the whole history once, and writes the raw parquet.
        df = fetch_histdata(pair, 2018, 2024, RAW_DIR)
        print(f"{asset}: {len(df)} rows, now ends {df.index[-1]}")
        results[asset] = df

    # --- Verify every asset reaches end-2024 ---
    failures = [
        a for a, df in results.items() if df.index[-1] < _MIN_ACCEPTABLE_END
    ]
    print("\n=== 2024 extension summary ===")
    for asset, df in results.items():
        print(f"  {asset:<10} {len(df):>7,} rows   {df.index[0]} → {df.index[-1]}")
    if failures:
        raise RuntimeError(f"Assets not reaching end-2024: {failures}")
    print("All 4 assets extended through 2024 ✓")
    return results


if __name__ == "__main__":
    fetch_2024_extension()
