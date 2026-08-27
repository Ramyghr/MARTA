"""Keeps all 4 assets' normalized features current, bridging the gap between each
asset's training-time historical cutoff and today — via Twelve Data for EURUSD/GOLD,
via Binance/ccxt for BTC/ETH.

Produces {file_stem}_normalized_live.parquet: the exact same 14-column schema as the
training-time {file_stem}_normalized.parquet that ppo_inference.py's live embedding path
reads (_ALIGNED_PATTERN), just extended forward with real market bars run through the
same features.py -> normalize.py pipeline training used. Does NOT touch the canonical
{file_stem}_normalized.parquet or any other training-time artifact.

Why this exists: ppo_inference.py's "live" regime computation was confirmed (2026-07-31
audit) to be silently frozen — it reads the static training-time normalized parquet, which
stops advancing the moment nothing re-runs fetch/features/normalize for that asset, so
every query after that date resolves to the same last window regardless of when it's made.

Two different root causes, one fix shape:
  - EURUSD/GOLD: genuinely capped by histdata.com's whole-year-only ZIPs (no 2026 ZIP
    exists yet as of this pass) — a real external data-source ceiling. Bridged via
    Twelve Data's /time_series.
  - BTC/ETH: NOT a data-source ceiling — Binance has every hour up to now. The static
    {file_stem}_normalized.parquet was just never refreshed on a schedule (confirmed
    2026-08-01: the pre-join per-asset file was already 3 weeks fresher than EURUSD/GOLD's
    ceiling, not stuck at the same 2025-12-31 boundary — align.py's join truncation only
    affects the POST-join {file_stem}_aligned.parquet, which ppo_inference.py never reads).
    Bridged via `historical._fetch_extension_binance` (the same ccxt batch-fetch already
    proven correct for the price-chart's live extension — reused here rather than a second
    implementation of the same batching logic).

Design note — three files, not one, and why:
  {file_stem}_raw_live.parquet         — pure OHLCV, append-only, NEVER trimmed.
  {file_stem}_livebridge_features.parquet — recomputed fresh each run from a trailing
                                          slice of the above (debug/intermediate artifact).
  {file_stem}_normalized_live.parquet  — the accumulated output ppo_inference.py reads.
An earlier version of this script fed its own *already-warmup-trimmed* features output
back in as next cycle's raw context. compute_features() drops ~20-50 rows of indicator
warmup from the START of whatever block it's given, every time it's called — feeding a
shrinking window back in as input each cycle loses ~20-50 rows/cycle against maybe 1
new bar/cycle at hourly cadence, eroding the trailing context to nothing within days.
The raw accumulator is append-only and never passed through compute_features itself, so
`.tail(N)` on it always yields a full, untrimmed N-row window regardless of cycle count.

Run standalone: conda run -n marta python dashboard/backend/live_feature_bridge.py
Run programmatically (e.g. from the hourly scheduler tick): update_live_bridge(asset, config)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
if str(MARTA_ROOT) not in sys.path:
    sys.path.insert(0, str(MARTA_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.features import compute_features
from pipeline.normalize import normalize_features, normalize_features_multihorizon
from utils.assets import ASSET_FILE_STEM
from live_feed import FOREX_SYMBOLS, TWELVE_DATA_KEY, SYMBOL_MAP as CRYPTO_SYMBOLS

_PROCESSED = MARTA_ROOT / "data" / "processed"
# Crypto-only panel (2026-08 migration) — the "latest regime model" (crypto5_multihorizon's
# raw-feature HMM, no encoder/PCA) needs the 22-column multihorizon schema, not the old
# 14-column one. Every asset in CRYPTO_SYMBOLS (now 5, after live_feed.py's migration) is
# bridged into this schema instead — EURUSD/GOLD (FOREX_SYMBOLS) are untouched, still 14-col.
_CRYPTO5_PROCESSED = _PROCESSED / "crypto5_multihorizon"

# Rows of trailing raw-OHLCV context fed into compute_features each cycle, so the 720h
# rolling z-score has full history right at whatever the current boundary is — without
# this, freshly-bridged rows would be computed against a truncated (<720h) window and be
# needlessly degraded. Comfortably larger than the 720h/min_periods=168h normalization
# window even accounting for forex's ~120h/week (not 168h/week) trading cadence.
_TRAILING_CONTEXT_ROWS = 3000

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _fetch_twelvedata_bars(symbol: str, outputsize: int) -> pd.DataFrame:
    """Real hourly OHLC bars from Twelve Data's /time_series, chronological order.

    Twelve Data forex/commodity bars carry no volume field at all (not even zero) —
    set to 0.0, matching the existing historical convention (histdata forex/gold also
    has zero centralised volume; obv_raw/vol_zscore are already degenerate for these
    assets in the training data, so this introduces no new inconsistency).
    """
    import httpx

    resp = httpx.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": symbol, "interval": "1h", "outputsize": outputsize, "apikey": TWELVE_DATA_KEY},
        timeout=30,
    )
    data = resp.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data error for {symbol}: {data}")

    rows = [
        {
            "timestamp": pd.Timestamp(v["datetime"], tz="UTC"),
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low": float(v["low"]),
            "close": float(v["close"]),
            "volume": 0.0,
        }
        for v in data["values"]
    ]
    # Twelve Data returns newest-first; sort restores chronological order.
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def _fetch_binance_bars(symbol: str, since_ms: int) -> pd.DataFrame:
    """Real hourly OHLCV bars from Binance since `since_ms`, chronological order.

    Reuses historical._fetch_extension_binance (the same ccxt batch-fetch already
    exercised by the price chart's live extension) instead of a second implementation
    of the same batching/pagination logic. Unlike Twelve Data's outputsize-based
    /time_series, this is a since-cursor fetch — it naturally returns "however much is
    new" whether that's a routine 1-bar catch-up or a multi-week first-run backfill, so
    (unlike the forex path below) no is_first_run size branching is needed here.
    Real (nonzero) volume, unlike the forex/gold bridge below.
    """
    from historical import _fetch_extension_binance
    df = _fetch_extension_binance(symbol, since_ms, limit=30_000)
    if df.empty:
        # _fetch_extension_binance returns a bare pd.DataFrame() (RangeIndex, no
        # columns) on the common "nothing new since last tick" path — reshape to the
        # same typed-empty schema as a real result so `live_bars.index > cutoff`
        # downstream doesn't blow up comparing an int RangeIndex to a Timestamp.
        return pd.DataFrame(columns=_OHLCV_COLS, index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))
    return df


def _raw_live_path(file_stem: str) -> Path:
    return _PROCESSED / f"{file_stem}_raw_live.parquet"


def _load_raw_accumulator(file_stem: str) -> pd.DataFrame:
    """The append-only raw-OHLCV store. Bootstrapped from data/raw/{file_stem}_1h_raw.parquet
    (the common fetch-stage output every asset has, crypto5 or legacy) rather than a
    feature-stage file — crypto5's new assets (BNB/LTC/XRP) were never run through the
    standard pipeline.features.py caching path build_crypto5_features.py uses instead, so
    no {file_stem}_features.parquet exists for them, but the raw fetch file always does."""
    path = _raw_live_path(file_stem)
    if path.exists():
        return pd.read_parquet(path)
    raw = pd.read_parquet(MARTA_ROOT / "data" / "raw" / f"{file_stem}_1h_raw.parquet")
    return raw[_OHLCV_COLS].tail(_TRAILING_CONTEXT_ROWS)


def update_live_bridge(asset: str, config: dict, outputsize: int | None = None) -> Path | None:
    """Extend `asset`'s live-bridge normalized features up to the latest available bar —
    Twelve Data for EURUSD/GOLD (14-col legacy schema), Binance for every crypto5 asset
    (22-col multihorizon schema — see module-level _CRYPTO5_PROCESSED comment). Self-
    selecting: the raw accumulator bootstraps from historical data on the first-ever call;
    every call after that is a cheap incremental extension.

    Returns the written normalized-live path, or None if there was nothing new to add
    (e.g. market closed since the last run, or the source hasn't ticked yet) — not an
    error condition.
    """
    is_crypto = asset in CRYPTO_SYMBOLS
    is_forex = asset in FOREX_SYMBOLS
    if not is_crypto and not is_forex:
        raise ValueError(f"{asset} is not a live-bridge asset (not in CRYPTO_SYMBOLS or FOREX_SYMBOLS)")
    if is_forex and not TWELVE_DATA_KEY:
        raise RuntimeError("TWELVE_DATA_KEY not set")

    # Crypto assets need no ASSET_FILE_STEM aliasing (BTC_USDT etc. are already their own
    # file stem); only GOLD->XAUUSD (forex/legacy path) needs the translation.
    file_stem = asset if is_crypto else ASSET_FILE_STEM.get(asset, asset)
    processed_dir = _CRYPTO5_PROCESSED if is_crypto else _PROCESSED
    normalized_live_path = processed_dir / f"{file_stem}_normalized_live.parquet"
    is_first_run = not normalized_live_path.exists()
    if is_first_run:
        # Crypto5 bootstraps from the multihorizon aligned parquet (22-col); legacy
        # forex bootstraps from the old training-time normalized parquet (14-col).
        bootstrap_path = (
            _CRYPTO5_PROCESSED / f"{file_stem}_aligned.parquet"
            if is_crypto else _PROCESSED / f"{file_stem}_normalized.parquet"
        )
        base_normalized = pd.read_parquet(bootstrap_path)
    else:
        base_normalized = pd.read_parquet(normalized_live_path)
    cutoff = base_normalized.index.max()

    if is_crypto:
        # Since-cursor fetch (see _fetch_binance_bars) — no outputsize concept and no
        # external ceiling, so first-run vs. routine needs no size branching: it just
        # returns however much is new, whether that's one bar or a multi-week backfill.
        since_ms = int(cutoff.timestamp() * 1000) + 1
        live_bars = _fetch_binance_bars(CRYPTO_SYMBOLS[asset], since_ms)
    else:
        # First run needs to close a potentially large gap (5000 bars ~= 7 months on a
        # fresh deploy); a routine refresh only needs however long it's been since the
        # last successful run, with generous headroom in case the scheduler missed a
        # few ticks.
        default_size = 5000 if is_first_run else 500
        live_bars = _fetch_twelvedata_bars(FOREX_SYMBOLS[asset], outputsize=outputsize or default_size)

    live_bars = live_bars[live_bars.index > cutoff]
    if live_bars.empty:
        print(f"[{asset}] no new bars past {cutoff} — nothing to do")
        return None

    # Extend the append-only raw accumulator (never itself passed through
    # compute_features, so it never loses rows to warmup trimming — see module docstring).
    raw_accum = pd.concat([_load_raw_accumulator(file_stem), live_bars[_OHLCV_COLS]]).sort_index()
    raw_accum = raw_accum[~raw_accum.index.duplicated(keep="last")]
    raw_accum.to_parquet(_raw_live_path(file_stem), engine="pyarrow")

    # Recompute features/normalization fresh over a trailing slice of the (untrimmed)
    # accumulator — same functions training used, not a reimplementation. Distinct
    # `symbol` so compute_features' internal save doesn't overwrite the canonical
    # {file_stem}_features.parquet. Crypto5 uses the 22-col multihorizon normalizer to
    # match the latest regime model's input; legacy forex keeps the 14-col one.
    context = raw_accum.tail(_TRAILING_CONTEXT_ROWS + len(live_bars))
    features = compute_features(context, symbol=f"{file_stem}_livebridge")
    normalized = (
        normalize_features_multihorizon(features, config) if is_crypto
        else normalize_features(features, config)
    )

    new_rows = normalized[normalized.index > cutoff]
    if new_rows.empty:
        print(
            f"[{asset}] source had newer bars but the bridged normalization produced "
            f"no rows past {cutoff} (still inside rolling warmup) — nothing to do"
        )
        return None

    combined = pd.concat([base_normalized, new_rows]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.to_parquet(normalized_live_path, engine="pyarrow")

    print(
        f"[{asset}] +{len(new_rows):,} new rows -> {len(combined):,} total, "
        f"now current through {combined.index.max()}"
    )
    return normalized_live_path


def main() -> None:
    with open(MARTA_ROOT / "configs" / "assets.yaml") as f:
        config = yaml.safe_load(f)
    for asset in ["EURUSD", "GOLD", "BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]:
        try:
            update_live_bridge(asset, config)
        except Exception as exc:
            print(f"[{asset}] FAILED: {exc}")


if __name__ == "__main__":
    main()
