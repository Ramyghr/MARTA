import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))

# Static fallback only — used if regime_semantics.json is missing (pipeline never run).
# Whenever that file exists, _load_regime_descriptions() below is the source of truth:
# regime semantics shift on every HMM refit and this dict has no way to track that itself.
_FALLBACK_DESCRIPTIONS = {
    0: "Mixed/Transitional",
    1: "Calm Bull",
    2: "Stress/Crisis",
    3: "Bear Trend",
    4: "Calm Bear",
    5: "Trending Forex A",
    6: "Trending Forex B",
}

_descriptions_cache: dict[int, str] | None = None


def _load_regime_descriptions() -> dict[int, str]:
    """Generate regime descriptions from the current HMM's actual validated statistics
    (data/validated/regime_semantics.json) instead of a hand-maintained label table that
    silently goes stale every time the HMM is refit. Cached for the process lifetime.
    """
    global _descriptions_cache
    if _descriptions_cache is not None:
        return _descriptions_cache

    sem_path = MARTA_ROOT / "data" / "validated" / "regime_semantics.json"
    if not sem_path.exists():
        _descriptions_cache = dict(_FALLBACK_DESCRIPTIONS)
        return _descriptions_cache

    try:
        sem = json.loads(sem_path.read_text())
    except Exception:
        _descriptions_cache = dict(_FALLBACK_DESCRIPTIONS)
        return _descriptions_cache

    descriptions: dict[int, str] = {}
    for label, stats in sem.get("regimes", {}).items():
        rvol = float(stats.get("realized_vol_mean", 0.0))
        sharpe = float(stats.get("sharpe", 0.0))
        dom = str(stats.get("dominant_asset", "")).replace("_USDT", "")

        if rvol > 0.8:
            vol_desc = "High Volatility"
        elif rvol > 0.2:
            vol_desc = "Moderate Volatility"
        elif rvol < -0.2:
            vol_desc = "Low Volatility"
        else:
            vol_desc = "Neutral Volatility"

        if sharpe > 1.5:
            perf_desc = "Strong Bull"
        elif sharpe > 0.5:
            perf_desc = "Positive"
        elif sharpe > 0:
            perf_desc = "Mildly Positive"
        elif sharpe > -0.5:
            perf_desc = "Mildly Negative"
        else:
            perf_desc = "Bear"

        descriptions[int(label)] = f"{vol_desc} / {perf_desc}" + (f" ({dom}-dominant)" if dom else "")

    _descriptions_cache = descriptions or dict(_FALLBACK_DESCRIPTIONS)
    return _descriptions_cache


def invalidate_regime_descriptions_cache() -> None:
    """Call after regime_semantics.json is regenerated so the next lookup re-reads it."""
    global _descriptions_cache
    _descriptions_cache = None


# Module attribute kept for other modules that import it directly (e.g. historical.py).
# Computed once at import time; call invalidate_regime_descriptions_cache() + re-read
# _load_regime_descriptions() if semantics are regenerated within a running process.
REGIME_DESCRIPTIONS = _load_regime_descriptions()

_df: pd.DataFrame | None = None


def load_regime_stream() -> None:
    global _df
    path = MARTA_ROOT / "data" / "processed" / "regime_stream.parquet"
    _df = pd.read_parquet(path)
    if "timestamp" in _df.columns and not isinstance(_df.index, pd.DatetimeIndex):
        _df = _df.set_index("timestamp")
    _df.index = pd.to_datetime(_df.index, utc=True)
    print(f"[regime_service] loaded {len(_df):,} rows from {path}")


def get_regime(asset: str, timestamp: pd.Timestamp) -> dict:
    if _df is None:
        load_regime_stream()
    ts = pd.Timestamp(timestamp, tz="UTC") if not isinstance(timestamp, pd.Timestamp) else timestamp
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    asset_df = _df[_df["asset"] == asset]
    if asset_df.empty:
        return _empty_regime()
    idx = asset_df.index.get_indexer([ts], method="nearest")
    row = asset_df.iloc[idx[0]]
    label = int(row["regime_label"])
    probs = {f"regime_{i}": float(row.get(f"regime_prob_{i}", 0.0)) for i in range(7)}
    return {
        "regime_label": label,
        "regime_description": _load_regime_descriptions().get(label, "Unknown"),
        "probabilities": probs,
        "dominant_regime": label,
    }


def get_regime_series(asset: str, timestamps: list[pd.Timestamp]) -> list[dict]:
    """Batch regime lookup for a list of timestamps."""
    if _df is None:
        load_regime_stream()
    asset_df = _df[_df["asset"] == asset]
    results = []
    for ts in timestamps:
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts, tz="UTC")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        idx = asset_df.index.get_indexer([ts], method="nearest")
        if len(idx) == 0 or asset_df.empty:
            results.append(_empty_regime())
            continue
        row = asset_df.iloc[idx[0]]
        label = int(row["regime_label"])
        results.append({"regime_label": label, "regime_description": _load_regime_descriptions().get(label, "Unknown")})
    return results


def get_regime_stream_tail(asset: str, limit: int = 500) -> list[dict]:
    """Last `limit` regime-stream rows for `asset`, oldest first, decimated to ~2000 points."""
    if _df is None:
        load_regime_stream()
    asset_df = _df[_df["asset"] == asset].tail(limit)
    if len(asset_df) > 2000:
        # Keep every Nth row — 33k+ points overwhelm both the JSON encoder and the chart
        step = len(asset_df) // 2000
        asset_df = asset_df.iloc[::step]
    out = asset_df.reset_index().rename(columns={"index": "timestamp"})
    # Sanitize non-finite values that crash the JSON encoder
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], 0).fillna(0)
    out["timestamp"] = out["timestamp"].astype(str)
    return out.to_dict(orient="records")


def get_available_assets() -> list[str]:
    if _df is None:
        load_regime_stream()
    return sorted(_df["asset"].unique().tolist())


def _empty_regime() -> dict:
    return {
        "regime_label": 0,
        "regime_description": _load_regime_descriptions().get(0, "Unknown"),
        "probabilities": {f"regime_{i}": 0.0 for i in range(7)},
        "dominant_regime": 0,
    }
