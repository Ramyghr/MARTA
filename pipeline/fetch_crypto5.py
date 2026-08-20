"""Fetches OHLCV for the 3 NEW crypto-only-panel assets not yet in data/raw/:
BNB/USDT, LTC/USDT, XRP/USDT. BTC/ETH are already fetched (data/raw/BTC_USDT_1h_raw.parquet,
ETH_USDT_1h_raw.parquet) and untouched by this script.

Regime-layer audit (2026-08-19): crypto-only 5-asset panel (BTC/ETH/BNB/LTC/XRP),
replacing the BTC/ETH/EURUSD/GOLD panel for regime-representation testing — EURUSD/
GOLD's degenerate zero-volume features were found to make the raw-feature HMM key on
asset identity rather than market regime. Verified via a live CCXT first-available-
candle check: BTC/ETH/BNB/LTC all have genuine data from 2017, XRP from 2018-05-04
(SOL was excluded — no data before 2020-08-11, would truncate the whole panel's usable
history by 2.5 years and lose the 2018 bear market / COVID crash).

Reuses pipeline/fetch.py's existing fetch_all_assets/_cached_raw_covers_end machinery —
same CCXT/Binance path, same OHLCV sanitization, same cache-aware behavior. Writes to
the SAME data/raw/{SYMBOL}_1h_raw.parquet convention BTC/ETH already use — additive
only, BTC/ETH/EURUSD/XAUUSD's existing raw parquets are untouched. Does NOT modify
configs/assets.yaml (the production 4-asset config) — the crypto asset list is
overridden in-memory only.

Run: conda run -n marta python pipeline/fetch_crypto5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.fetch import fetch_all_assets

NEW_CRYPTO_ASSETS = [
    {"symbol": "BNB/USDT", "exchange": "binance", "source": "ccxt"},
    {"symbol": "LTC/USDT", "exchange": "binance", "source": "ccxt"},
    {"symbol": "XRP/USDT", "exchange": "binance", "source": "ccxt"},
]


def main() -> None:
    with open(ROOT / "configs" / "assets.yaml") as f:
        config = yaml.safe_load(f)

    config["assets"]["crypto"] = NEW_CRYPTO_ASSETS
    config["assets"]["traditional"] = []  # skip histdata entirely — not needed here

    fetch_all_assets(config)


if __name__ == "__main__":
    main()
