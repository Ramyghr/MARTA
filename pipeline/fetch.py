"""
pipeline/fetch.py
Fetches OHLCV data for all configured assets.

  - Crypto (BTC/USDT, ETH/USDT): CCXT/Binance hourly.
  - Traditional (EURUSD, XAUUSD): histdata.com genuine 1-minute data,
    resampled to hourly. No more daily-upsampling — all data is genuine OHLCV.

Both fetchers are cache-aware: an existing raw parquet that already covers the
configured end date is reused rather than re-downloaded. histdata per-year ZIPs
and 1-minute parquets are cached under data/raw/histdata/ and never re-downloaded
if present (do not delete them — see CLAUDE.md).
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.logger import logger

RAW_DIR = Path("data/raw")
HISTDATA_DIR = RAW_DIR / "histdata"


def _safe_symbol_name(symbol: str) -> str:
    """Convert 'BTC/USDT' -> 'BTC_USDT', 'EURUSD' -> 'EURUSD'."""
    return symbol.replace("/", "_")


def _detect_gaps(df: pd.DataFrame, max_gap_hours: float, symbol: str) -> int:
    """Log gaps exceeding threshold; return gap count."""
    if df.index.empty:
        return 0
    diffs = df.index.to_series().diff().dropna()
    threshold = pd.Timedelta(hours=max_gap_hours)
    gaps = diffs[diffs > threshold]
    if not gaps.empty:
        logger.warning(
            f"[{symbol}] {len(gaps)} gaps > {max_gap_hours}h detected. "
            f"Largest: {gaps.max()}"
        )
        for ts, gap in gaps.nlargest(5).items():
            logger.warning(f"  Gap at {ts}: {gap}")
    return len(gaps)


# ---------------------------------------------------------------------------
# OHLCV sanitization — shared by crypto and histdata
# ---------------------------------------------------------------------------

def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clamp OHLCV to be internally consistent and remove bad rows:
    - high = max(open, high, close)
    - low  = min(open, low,  close)
    - Replace zero/negative close prices with NaN and forward-fill
    - Drop any remaining rows with NaN close after forward-fill
    """
    df = df.copy()

    # Fix zero/negative close prices — forward-fill then drop leading NaN
    zero_price_mask = df["close"] <= 0
    if zero_price_mask.any():
        count = int(zero_price_mask.sum())
        df.loc[zero_price_mask, ["open", "high", "low", "close"]] = float("nan")
        df[["open", "high", "low", "close"]] = (
            df[["open", "high", "low", "close"]].ffill()
        )
        logger.warning(f"Replaced {count} zero/negative close price rows via forward-fill")

    # Clamp high/low to be consistent with open/close
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    # Drop rows that still have NaN (e.g., leading rows before any valid price)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    return df


def _cached_raw_covers_end(path: Path, end: str) -> pd.DataFrame | None:
    """
    Return the cached raw DataFrame if it exists and already extends to (or past)
    the requested end date minus a small slack; otherwise None. Cache reuse avoids
    re-hitting the exchange / histdata servers on every pipeline run.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return None
    if df.empty:
        return None
    end_ts = pd.Timestamp(end, tz="UTC")
    # Hourly data: accept the cache if it reaches within ~2 days of the requested end
    if df.index.max() >= end_ts - pd.Timedelta(days=2):
        return df
    return None


# ---------------------------------------------------------------------------
# Crypto — CCXT / Binance
# ---------------------------------------------------------------------------

def _fetch_crypto_ccxt(
    symbol: str,
    exchange_id: str,
    start: str,
    end: str,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Batch-fetch OHLCV from a CCXT exchange between start and end.

    Loop termination (the ETH-bug fix): the loop breaks ONLY when
      (a) the exchange returns an empty batch, or
      (b) the last returned candle timestamp reaches the configured end.
    It does NOT break on `len(batch) < batch_size` — that condition was the
    silent killer that capped ETH at 2021-10-12. A genuine no-progress guard
    (last_ts not advancing) remains to prevent an infinite loop.
    """
    exchange_cls = getattr(ccxt, exchange_id)
    exchange: ccxt.Exchange = exchange_cls(
        {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    )

    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    batch_size = 1000
    all_candles: list[list] = []

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]Fetching {symbol}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("fetch", total=None)
        while since_ms < end_ms:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=batch_size,
                )
            except Exception as exc:
                logger.error(f"[{symbol}] CCXT fetch error: {exc}")
                break

            # (a) Break on a truly empty batch — there is no more data.
            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]

            # No-progress guard: the exchange returned only candles at or before
            # where we already are. Advancing past it prevents an infinite loop.
            # (This is NOT the `len < batch_size` break — short batches continue.)
            if last_ts < since_ms:
                break
            if last_ts == since_ms:
                since_ms += batch_size * 3_600_000  # nudge forward by one batch-width
                continue

            since_ms = last_ts + 1  # avoid off-by-one overlap

            advance = len(candles)
            progress.update(task, advance=advance)
            time.sleep(0.3)

            # (b) Stop once we have reached the end boundary.
            if last_ts >= end_ms:
                break

    if not all_candles:
        raise RuntimeError(f"No data returned for {symbol}")

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Deduplicate — last candle of batch N sometimes repeats as first of batch N+1
    df = df[~df.index.duplicated(keep="last")]

    # Trim to requested range
    df = df.loc[
        (df.index >= pd.Timestamp(start, tz="UTC"))
        & (df.index < pd.Timestamp(end, tz="UTC"))
    ]

    return _sanitize_ohlcv(df)


# ---------------------------------------------------------------------------
# Traditional — histdata.com (genuine 1-minute → hourly)
# ---------------------------------------------------------------------------

def _read_histdata_zip(zip_path: Path) -> pd.DataFrame:
    """
    Read a histdata GENERIC_ASCII M1 ZIP into a UTC-indexed 1-minute OHLCV frame.

    The CSV inside has no header, is semicolon-separated with 6 columns:
        datetime_str(YYYYMMDD HHMMSS), open, high, low, close, volume
    Note: histdata FX/commodity volume is always 0 (no centralised volume).
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as fh:
            df = pd.read_csv(
                fh,
                sep=";",
                header=None,
                names=["datetime_str", "open", "high", "low", "close", "volume"],
            )
    df["timestamp"] = pd.to_datetime(df["datetime_str"], format="%Y%m%d %H%M%S", utc=True)
    df = df.drop(columns=["datetime_str"]).set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


@retry(
    retry=retry_if_exception_type((OSError, IOError, Exception)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _download_histdata_zip(
    pair_lower: str, pair_upper: str, year: int, download_dir: Path
) -> Path:
    """Download one histdata year ZIP with exponential-backoff retry (up to 4 attempts)."""
    returned = dl(
        year=str(year),
        month=None,
        pair=pair_lower,
        platform=P.GENERIC_ASCII,
        time_frame=TF.ONE_MINUTE,
        output_directory=str(download_dir),
    )
    zip_path = download_dir / returned
    time.sleep(2)  # be polite to histdata servers
    return zip_path


def fetch_histdata(
    pair: str,
    start_year: int,
    end_year: int,
    output_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """
    Fetch genuine 1-minute OHLCV for a histdata.com instrument across
    [start_year, end_year] inclusive, resample to hourly, and return it.

    Reproducibility / politeness:
      - Per-year 1-minute parquets are cached at
        data/raw/histdata/{pair}_{year}_1min.parquet and reused if present.
      - The downloaded year ZIPs (DAT_ASCII_{PAIR}_M1_{year}.zip) are reused if
        already on disk; only missing years are downloaded (2s sleep between
        downloads to be polite to histdata servers).

    Saves the final hourly frame to data/raw/{PAIR}_1h_raw.parquet and returns it.
    """
    HISTDATA_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_lower = pair.lower()
    pair_upper = pair.upper()
    download_dir = Path.cwd()  # histdata `dl` writes/reads the ZIP here

    yearly_minute_frames: list[pd.DataFrame] = []
    total_min_rows = 0
    years = list(range(start_year, end_year + 1))

    for year in years:
        minute_parquet = HISTDATA_DIR / f"{pair_lower}_{year}_1min.parquet"

        if minute_parquet.exists():
            df_year = pd.read_parquet(minute_parquet, engine="pyarrow")
        else:
            zip_path = download_dir / f"DAT_ASCII_{pair_upper}_M1_{year}.zip"
            if not zip_path.exists():
                logger.info(f"[{pair_upper}] Downloading histdata {year} (1-minute)…")
                zip_path = _download_histdata_zip(
                    pair_lower, pair_upper, year, download_dir
                )
            else:
                logger.info(f"[{pair_upper}] Reusing cached ZIP {zip_path.name}")

            df_year = _read_histdata_zip(zip_path)
            df_year.to_parquet(minute_parquet, engine="pyarrow")
            logger.info(f"[{pair_upper}] {year}: {len(df_year):,} 1-min rows → {minute_parquet}")

        yearly_minute_frames.append(df_year)
        total_min_rows += len(df_year)

    df_1min = pd.concat(yearly_minute_frames).sort_index()
    df_1min = df_1min[~df_1min.index.duplicated(keep="last")]

    df_hourly = df_1min.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()

    df_hourly = _sanitize_ohlcv(df_hourly)
    df_hourly.index.name = "timestamp"

    out_path = output_dir / f"{pair_upper}_1h_raw.parquet"
    df_hourly.to_parquet(out_path, engine="pyarrow")

    logger.success(
        f"[{pair_upper}] histdata {years[0]}–{years[-1]}: "
        f"{total_min_rows:,} 1-min rows → {len(df_hourly):,} hourly rows | "
        f"{df_hourly.index.min()} → {df_hourly.index.max()} | saved {out_path}"
    )
    return df_hourly


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all_assets(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for all assets defined in config.
    Returns dict mapping symbol name -> DataFrame with columns [open, high, low, close, volume].
    Saves raw parquet files to data/raw/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    start: str = config["date_range"]["start"]
    end: str = config["date_range"]["end"]
    timeframe: str = config["timeframe"]["candle"]
    max_gap_crypto: float = config["validation"]["max_gap_hours_crypto"]
    max_gap_trad: float = config["validation"]["max_gap_hours_traditional"]

    start_year = int(pd.Timestamp(start).year)
    # date_range.end is exclusive (e.g. 2024-01-01); the last full year to fetch is end-1.
    end_year = int((pd.Timestamp(end) - pd.Timedelta(days=1)).year)

    result: dict[str, pd.DataFrame] = {}

    # --- Crypto (CCXT) ---
    for asset in config["assets"]["crypto"]:
        symbol: str = asset["symbol"]
        exchange_id: str = asset["exchange"]
        safe_name = _safe_symbol_name(symbol)
        out_path = RAW_DIR / f"{safe_name}_1h_raw.parquet"
        logger.info(f"Fetching crypto: {symbol} from {exchange_id}")

        try:
            cached = _cached_raw_covers_end(out_path, end)
            if cached is not None:
                logger.info(
                    f"[{symbol}] Reusing cached raw ({len(cached):,} rows, "
                    f"to {cached.index.max()}) — skipping CCXT fetch."
                )
                df = cached
            else:
                df = _fetch_crypto_ccxt(symbol, exchange_id, start, end, timeframe)
                df.to_parquet(out_path, engine="pyarrow")

            gap_count = _detect_gaps(df, max_gap_crypto, symbol)
            logger.success(
                f"[{symbol}] {len(df):,} rows | "
                f"{df.index.min()} → {df.index.max()} | gaps: {gap_count}"
            )
            logger.info(f"[{symbol}] Raw → {out_path}")
            result[safe_name] = df
        except Exception as exc:
            logger.error(f"[{symbol}] Fetch failed: {exc}. Skipping.")

    # --- Traditional (histdata.com) ---
    for asset in config["assets"]["traditional"]:
        symbol = asset["symbol"]
        histdata_pair: str = asset["histdata_pair"]
        safe_name = _safe_symbol_name(symbol)
        out_path = RAW_DIR / f"{safe_name}_1h_raw.parquet"
        logger.info(f"Fetching traditional: {symbol} (histdata pair '{histdata_pair}')")

        try:
            cached = _cached_raw_covers_end(out_path, end)
            if cached is not None:
                logger.info(
                    f"[{symbol}] Reusing cached raw ({len(cached):,} rows, "
                    f"to {cached.index.max()}) — skipping histdata fetch."
                )
                df = cached
            else:
                df = fetch_histdata(histdata_pair, start_year, end_year, RAW_DIR)

            gap_count = _detect_gaps(df, max_gap_trad, symbol)
            logger.success(
                f"[{symbol}] {len(df):,} rows | "
                f"{df.index.min()} → {df.index.max()} | gaps: {gap_count}"
            )
            logger.info(f"[{symbol}] Raw → {out_path}")
            result[safe_name] = df
        except Exception as exc:
            logger.error(f"[{symbol}] Fetch failed: {exc}. Skipping.")

    total_assets = len(config["assets"]["crypto"]) + len(config["assets"]["traditional"])
    logger.success(f"Fetch complete. {len(result)}/{total_assets} assets loaded.")
    return result
