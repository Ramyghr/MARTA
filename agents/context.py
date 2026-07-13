"""RegimeContext — shared context object injected into every agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

_REGIME_DESCRIPTIONS: dict[int, str] = {
    0: "Mixed/transitional regime — unclear directional bias",
    1: "Calm bull regime — low volatility, positive drift",
    2: "Stress/crisis regime — high volatility, GOLD-dominant, negative returns",
    3: "Bear trend regime — sustained negative returns, bearish momentum",
    4: "Calm bear regime — low volatility, negative drift",
    5: "Trending forex regime A — moderate volatility, positive Sharpe",
    6: "Trending forex regime B — moderate volatility, EURUSD-dominant",
}

_WINDOW_INDICATORS = ["realized_vol", "rsi", "adx", "close_return"]

USE_PER_ASSET_REGIME: bool = True

# TODO: Fill in BTC-specific regime descriptions after running run_per_asset_regime.py.
# Check the per-asset semantic validation table output for regime character assignments.
# Example structure: {0: 'BTC calm/transitional', 1: 'BTC calm bull', 2: 'BTC crash/crisis', ...}
_BTC_REGIME_DESCRIPTIONS: dict[int, str] = {
    0: "BTC bear/crisis — strong downtrend (ann. return -189.5%), elevated volatility, "
       "worst risk-adjusted performance (Sharpe -2.77), trending (ADX+)",
    1: "BTC calm/range-bound — mild positive drift (ann. return +15.7%), moderate volatility, "
       "near-neutral risk-adjusted return (Sharpe 0.23)",
    2: "BTC high-volatility/transitional — near-zero net return (ann. +0.4%) despite the "
       "highest raw volatility band, non-trending (ADX-), choppy price action; this is the "
       "regime BTC's March 2020 crash occurs within (pre-crash and crash windows both fall "
       "in R2, distinguished by a sharp internal probability shift rather than a label change)",
    3: "BTC volatile bull — strong uptrend (ann. return +94.9%) with high realized volatility, "
       "trending (ADX+), solid risk-adjusted return (Sharpe 0.98)",
    4: "BTC calm bull — strongest uptrend (ann. return +203.5%) combined with the lowest "
       "volatility of any BTC regime, best risk-adjusted performance (Sharpe 3.36)",
}
_PER_ASSET_STREAM_PATH = _ROOT / 'data' / 'processed' / 'regime_stream_per_asset.parquet'
_per_asset_stream_cache: dict[str, pd.DataFrame] = {}


def _resolve_regime_stream(regime_stream_df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Return per-asset stream only for BTC when USE_PER_ASSET_REGIME is True."""
    if not USE_PER_ASSET_REGIME or asset != "BTC_USDT":
        return regime_stream_df
    if "data" not in _per_asset_stream_cache:
        if _PER_ASSET_STREAM_PATH.exists():
            _per_asset_stream_cache["data"] = pd.read_parquet(
                _PER_ASSET_STREAM_PATH, engine="pyarrow"
            )
        else:
            return regime_stream_df
    return _per_asset_stream_cache["data"]



@dataclass
class RegimeContext:
    timestamp: pd.Timestamp
    asset: str
    regime_label: int
    regime_probs: dict[str, float]
    dominant_regime: int
    regime_description: str
    price_features: dict[str, float]
    window_features: dict[str, float]
    live_headlines: list | None = None  # injected by dashboard live mode


def build_regime_context(
    timestamp: pd.Timestamp,
    asset: str,
    regime_stream_df: pd.DataFrame,
    aligned_df: pd.DataFrame,
) -> RegimeContext:
    """Build RegimeContext for a given timestamp and asset."""
    ts_utc = pd.Timestamp(timestamp).tz_convert("UTC") if pd.Timestamp(timestamp).tzinfo else pd.Timestamp(timestamp).tz_localize("UTC")

    # Resolve aligned index timezone
    aligned_tz = aligned_df.index.tz
    if aligned_tz is not None:
        ts_aligned = ts_utc.tz_convert(aligned_tz)
    else:
        ts_aligned = ts_utc.tz_localize(None)

    # Look up regime row
    active_stream = _resolve_regime_stream(regime_stream_df, asset)
    mask = (active_stream["asset"] == asset) & (active_stream["timestamp"] == ts_utc)
    regime_rows = active_stream[mask]
    if regime_rows.empty:
        # Fall back to nearest available timestamp. get_indexer(method="nearest") requires
        # a uniquely-valued index — regime_stream_per_asset.parquet has one duplicated
        # BTC_USDT timestamp (a boundary artifact from however the file was built), which
        # makes pandas treat the *whole* index as non-unique and raise InvalidIndexError
        # for every nearest-lookup, not just ones near that specific timestamp. This is
        # the path a live "now" request always takes once it's past the data's end date.
        asset_df = active_stream[active_stream["asset"] == asset].copy()
        asset_df = asset_df.set_index("timestamp").sort_index()
        asset_df = asset_df[~asset_df.index.duplicated(keep="last")]
        idx = asset_df.index.get_indexer([ts_utc], method="nearest")[0]
        row = asset_df.iloc[idx]
    else:
        row = regime_rows.iloc[0]

    regime_label = int(row["regime_label"])
    prob_cols = [c for c in active_stream.columns if c.startswith("regime_prob_")]
    regime_probs = {col.replace("regime_prob_", "regime_"): float(row[col]) for col in prob_cols}
    dominant_regime = max(regime_probs, key=regime_probs.get)
    dominant_regime_int = int(dominant_regime.replace("regime_", ""))

    # Extract 13 features at the timestamp
    if ts_aligned in aligned_df.index:
        feature_row = aligned_df.loc[ts_aligned]
    else:
        idx = aligned_df.index.get_indexer([ts_aligned], method="nearest")[0]
        feature_row = aligned_df.iloc[idx]
    price_features = {col: float(feature_row[col]) for col in aligned_df.columns}

    # 60-candle window stats
    aligned_sorted = aligned_df.sort_index()
    loc = aligned_sorted.index.get_indexer([ts_aligned], method="pad")[0]
    start = max(0, loc - 59)
    window = aligned_sorted.iloc[start : loc + 1]
    window_features: dict[str, float] = {}
    for ind in _WINDOW_INDICATORS:
        if ind in window.columns:
            window_features[f"{ind}_mean"] = float(window[ind].mean())
            window_features[f"{ind}_std"] = float(window[ind].std(ddof=1) if len(window) > 1 else 0.0)

    return RegimeContext(
        timestamp=ts_utc,
        asset=asset,
        regime_label=regime_label,
        regime_probs=regime_probs,
        dominant_regime=dominant_regime_int,
        regime_description=(
            _BTC_REGIME_DESCRIPTIONS.get(regime_label)
            if USE_PER_ASSET_REGIME
            and asset == "BTC_USDT"
            and regime_label in _BTC_REGIME_DESCRIPTIONS
            else _REGIME_DESCRIPTIONS.get(regime_label, "Unknown regime")
        ),
        price_features=price_features,
        window_features=window_features,
    )


def serialize_context_for_prompt(ctx: RegimeContext, include_regime: bool = True) -> str:
    """Serialize RegimeContext into a compact prompt-ready string (<400 tokens).

    ``include_regime=False`` drops the regime label, regime description, and probability
    vector entirely — used to run "clean" no-regime-context ablation debates (see
    ``run_clean_ablation_debates.py``). Price features and window stats are unaffected.
    """
    price_lines = "\n".join(
        f"  {k}: {v:.4f}" for k, v in list(ctx.price_features.items())[:8]
    )

    window_lines = "\n".join(
        f"  {k}: {v:.4f}" for k, v in ctx.window_features.items()
    )

    if not include_regime:
        return (
            f"Asset: {ctx.asset} | Timestamp: {ctx.timestamp}\n\n"
            f"=== PRICE FEATURES (latest) ===\n{price_lines}\n\n"
            f"=== MARKET CHARACTER (60-candle window) ===\n{window_lines}\n"
        )

    probs_str = ", ".join(
        f"R{k.split('_')[-1]}={v:.2f}" for k, v in sorted(ctx.regime_probs.items()) if v > 0.01
    )

    return (
        f"=== REGIME STATE ===\n"
        f"Asset: {ctx.asset} | Timestamp: {ctx.timestamp}\n"
        f"Regime: R{ctx.regime_label} — {ctx.regime_description}\n"
        f"Regime probabilities: {probs_str}\n\n"
        f"=== PRICE FEATURES (latest) ===\n{price_lines}\n\n"
        f"=== MARKET CHARACTER (60-candle window) ===\n{window_lines}\n"
    )
