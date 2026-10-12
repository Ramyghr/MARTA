import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))

# Static fallback only — used if regime_semantics.json is missing (pipeline never run).
# Whenever that file exists, _load_regime_descriptions() below is the source of truth:
# regime semantics shift on every HMM refit and this dict has no way to track that itself.
# Worded as volatility/trend, not bull/bear — see _load_crypto5_regime_descriptions()'s
# docstring for why (2026-09-01 validation found sharpe-derived direction labels wrong
# more than half the time at the per-instance level).
_FALLBACK_DESCRIPTIONS = {
    0: "Mixed/Transitional",
    1: "Low Volatility / Trending",
    2: "Extreme Volatility / Shock",
    3: "Elevated Volatility / Trending",
    4: "Low Volatility / Range-bound",
    5: "Trending Forex A",
}

_descriptions_cache: dict[int, str] | None = None

# ── Live regime inference (PCA + HMM applied to a fresh embedding) ─────────────
# regime_stream.parquet is a static snapshot from the last run_regime.py pass — it
# stops growing the moment that script stops running, and nothing currently refreshes
# it (scripts/nightly_refresh.py only refreshes raw crypto candles). get_regime()'s
# nearest-timestamp lookup then silently snaps to that frozen last row forever, which
# looks like a stable "current regime" but is actually whatever was true on the day the
# stream was last generated. get_live_regime() below fixes this for "what's the regime
# right now" callers by recomputing PCA -> HMM directly on a fresh embedding instead.


# Live-only PCA+HMM (run_regime_live.py) — fit on embeddings spanning 2018 -> today,
# NOT the sacred pca_model.pkl/hmm_model.pkl (train<2024 only, used by agents/context.py
# and RL/backtest — never touched here). Found 2026-08-08: the sacred model is 2+ years
# out-of-distribution by mid-2026 and every live query since January 2026, across all 4
# assets independently, was landing nearest a single state (~100% R3) — verified as real
# distributional drift (embeddings vary day to day, genuinely closest to one 3-year-old
# cluster), not a frozen/stale-data bug. This whole dashboard-backend module is entirely
# separate from the research/RL pipeline, so repointing it here doesn't touch anything
# the sacred split protects.
_PCA_PATH = MARTA_ROOT / "regime" / "checkpoints" / "pca_model_live.pkl"
_HMM_PATH = MARTA_ROOT / "regime" / "checkpoints" / "hmm_model_live.pkl"
_LIVE_LOG_PATH = MARTA_ROOT / "data" / "processed" / "regime_live_log.parquet"
_STALE_THRESHOLD_HOURS = 48.0

_pca_model = None
_hmm_model = None
_regime_models_loaded = False

# Crypto-only panel (2026-08 migration) — the "latest regime model": a GaussianHMM fit
# directly on 22 raw multihorizon features, NO encoder, NO PCA (see
# regime/run_hmm_crypto5_test.py / regime/hmm.py's pca_n_components=None path). This is
# architecturally different from the encoder->PCA->HMM pipeline above, not a path swap —
# get_live_regime() below dispatches to whichever one matches the asset.
_CRYPTO5_ASSETS = {"BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"}
_CRYPTO5_HMM_PATH = MARTA_ROOT / "regime" / "checkpoints" / "crypto5_multihorizon" / "hmm_model.pkl"
_CRYPTO5_PROCESSED = MARTA_ROOT / "data" / "processed" / "crypto5_multihorizon"
_CRYPTO5_SEMANTICS_PATH = MARTA_ROOT / "data" / "validated" / "regime_semantics_crypto5_multihorizon.json"
# Same 22-column order normalize_features_multihorizon() always produces — verified
# identical to what the HMM was fit on (both paths go through the same function).
_CRYPTO5_FEATURE_COLS = [
    "atr", "adx", "rsi", "bb_width", "macd_histogram", "obv_raw", "vol_zscore",
    "realized_vol", "vol_ratio", "return_skew", "return_kurtosis", "autocorr_lag1",
    "realized_vol_raw", "close_return", "realized_vol_z90d", "realized_vol_z180d",
    "atr_z90d", "atr_z180d", "bb_width_z90d", "bb_width_z180d", "vol_ratio_z90d",
    "vol_ratio_z180d",
]

_crypto5_hmm_model = None
_crypto5_descriptions_cache: dict[int, str] | None = None


def load_regime_models() -> None:
    """Load the fitted PCA(16) + GaussianHMM once, for live regime inference. Call
    once from main.py's startup lifespan, alongside load_regime_stream().
    """
    global _pca_model, _hmm_model, _crypto5_hmm_model, _regime_models_loaded
    _regime_models_loaded = True
    try:
        import joblib
        _pca_model = joblib.load(_PCA_PATH)
        _hmm_model = joblib.load(_HMM_PATH)
        print(
            f"[regime_service] loaded live regime models: "
            f"PCA({_pca_model.n_components_}) + HMM({_hmm_model.n_components} states)"
        )
    except Exception as exc:
        print(f"[regime_service] could not load live regime models ({exc}) — "
              f"get_live_regime() will fall back to the historical lookup")
        _pca_model = None
        _hmm_model = None

    try:
        import joblib
        _crypto5_hmm_model = joblib.load(_CRYPTO5_HMM_PATH)
        print(f"[regime_service] loaded crypto5 live HMM ({_crypto5_hmm_model.n_components} states, no PCA)")
    except Exception as exc:
        print(f"[regime_service] could not load crypto5 HMM ({exc}) — "
              f"get_live_regime() for crypto5 assets will fall back to the historical lookup")
        _crypto5_hmm_model = None


def _load_crypto5_regime_descriptions() -> dict[int, str]:
    """Was `f"{vol_desc} / {perf_desc}"` with perf_desc a sharpe-derived "Strong Bull"/
    "Bear" label. Removed 2026-09-01: an objective test (regime/validate_regime_trend_
    alignment.py — zigzag-segmented price legs vs. dominant regime, 111 legs across all
    5 crypto5 assets, 2021) found only 49% of legs had a dominant regime whose
    *historical* Sharpe sign matched the leg's actual price direction. R4 (the state
    with the highest historical Sharpe, so it read as "Strong Bull") was dominant on
    MORE down-legs than up-legs (40.6% up) — i.e. the label was actively backwards more
    than half the time it mattered. realized_vol_mean and adx_mean describe HOW MUCH a
    regime moves and how persistently (ADX is trend strength regardless of sign in
    classic TA) — that held up in validation; sharpe-derived direction did not.
    """
    global _crypto5_descriptions_cache
    if _crypto5_descriptions_cache is not None:
        return _crypto5_descriptions_cache
    if not _CRYPTO5_SEMANTICS_PATH.exists():
        _crypto5_descriptions_cache = dict(_FALLBACK_DESCRIPTIONS)
        return _crypto5_descriptions_cache
    try:
        sem = json.loads(_CRYPTO5_SEMANTICS_PATH.read_text())
    except Exception:
        _crypto5_descriptions_cache = dict(_FALLBACK_DESCRIPTIONS)
        return _crypto5_descriptions_cache

    descriptions: dict[int, str] = {}
    for label, stats in sem.get("regimes", {}).items():
        rvol = float(stats.get("realized_vol_mean", 0.0))
        adx = float(stats.get("adx_mean", 0.0))
        dom = str(stats.get("dominant_asset", "")).replace("_USDT", "")
        vol_desc = (
            "Extreme Volatility" if rvol > 1.5 else
            "Elevated Volatility" if rvol > 0.5 else
            "Moderate Volatility" if rvol > -0.3 else
            "Low Volatility" if rvol > -0.9 else
            "Very Low Volatility"
        )
        trend_desc = "Trending" if adx >= 0 else "Range-bound"
        descriptions[int(label)] = f"{vol_desc} / {trend_desc}" + (f" ({dom}-dominant)" if dom else "")
    _crypto5_descriptions_cache = descriptions or dict(_FALLBACK_DESCRIPTIONS)
    return _crypto5_descriptions_cache


def get_live_regime_crypto5(asset: str, timestamp: pd.Timestamp, log: bool = False) -> dict:
    """Live regime for a crypto5 asset — raw multihorizon features straight into the
    HMM, no encoder/PCA step (architecturally distinct from get_live_regime() above).
    Same 168h-lookback-sequence-then-take-last-position approach and the same reasoning
    for why (a single-frame predict_proba() is dominated by the model's prior — see
    get_live_regime()'s docstring, the same finding applies here since it's a property
    of HMM forward-filtering in general, not specific to which features are used).

    Reads dashboard/backend/live_feature_bridge.py's {asset}_normalized_live.parquet —
    falls back to the historical lookup if that file doesn't exist yet, is too short for
    a full lookback window, or the crypto5 HMM failed to load.
    """
    if not _regime_models_loaded:
        load_regime_models()
    if _crypto5_hmm_model is None:
        return {**get_regime(asset, timestamp), "live": False}

    path = _CRYPTO5_PROCESSED / f"{asset}_normalized_live.parquet"
    if not path.exists():
        return {**get_regime(asset, timestamp), "live": False}

    try:
        df = pd.read_parquet(path, engine="pyarrow")
        df = df[df.index <= timestamp]
        if len(df) < _LIVE_LOOKBACK_HOURS:
            return {**get_regime(asset, timestamp), "live": False}
        window = df[_CRYPTO5_FEATURE_COLS].tail(_LIVE_LOOKBACK_HOURS).to_numpy(dtype=np.float64)
        probs_seq = _crypto5_hmm_model.predict_proba(window)
        probs_arr = probs_seq[-1]
    except Exception as exc:
        print(f"[regime_service] crypto5 live regime computation failed for {asset} ({exc})")
        return {**get_regime(asset, timestamp), "live": False}

    label = int(probs_arr.argmax())
    probs = {f"regime_{i}": float(p) for i, p in enumerate(probs_arr)}
    result = {
        "regime_label": label,
        "regime_description": _load_crypto5_regime_descriptions().get(label, "Unknown"),
        "probabilities": probs,
        "dominant_regime": label,
        "stale": False,
        "age_hours": 0.0,
        "live": True,
        "computed_at": pd.Timestamp(timestamp, tz="UTC").isoformat() if timestamp.tzinfo is None else timestamp.isoformat(),
        "lookback_hours": _LIVE_LOOKBACK_HOURS,
    }
    if log:
        _append_live_regime_log(asset, timestamp, result)
    return result


def _n_regime_states() -> int:
    """Number of HMM states, read from the loaded model rather than hardcoded — the
    HMM has been refit with different n_states over this project's life (was 7, is
    currently 6), and hardcoding it here would go stale the same way the old
    range(7) below already had.
    """
    if _hmm_model is not None:
        return _hmm_model.n_components
    return 6  # current production HMM (see CLAUDE.md) — used only if models aren't loaded


def _load_regime_descriptions() -> dict[int, str]:
    """Generate regime descriptions from the current HMM's actual validated statistics
    (data/validated/regime_semantics.json) instead of a hand-maintained label table that
    silently goes stale every time the HMM is refit. Cached for the process lifetime.

    Worded as volatility/trend, not bull/bear (2026-09-01) — mirrors
    _load_crypto5_regime_descriptions()'s fix, same anti-pattern (sharpe sign used as a
    direction proxy). NOTE: the zigzag trend-leg validation that motivated this fix
    (regime/validate_regime_trend_alignment.py) was run against the crypto5 panel only;
    this legacy EURUSD/GOLD encoder+PCA+HMM path hasn't been independently re-checked on
    forex data. Fixed on the strength of "same mechanism, same anti-pattern" rather than
    a fresh direction-alignment test on this specific HMM — re-run the validation here
    if this path becomes load-bearing again.
    """
    global _descriptions_cache
    if _descriptions_cache is not None:
        return _descriptions_cache

    sem_path = MARTA_ROOT / "data" / "validated" / "regime_semantics_live.json"
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
        adx = float(stats.get("adx_mean", 0.0))
        dom = str(stats.get("dominant_asset", "")).replace("_USDT", "")

        if rvol > 1.5:
            vol_desc = "Extreme Volatility"
        elif rvol > 0.5:
            vol_desc = "Elevated Volatility"
        elif rvol > -0.3:
            vol_desc = "Moderate Volatility"
        elif rvol > -0.9:
            vol_desc = "Low Volatility"
        else:
            vol_desc = "Very Low Volatility"

        trend_desc = "Trending" if adx >= 0 else "Range-bound"

        descriptions[int(label)] = f"{vol_desc} / {trend_desc}" + (f" ({dom}-dominant)" if dom else "")

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
_crypto5_df: pd.DataFrame | None = None


def load_regime_stream() -> None:
    global _df, _crypto5_df
    # Live-only stream (regime_stream_live.parquet) — same model as get_live_regime(),
    # self-consistent 2018 -> today. The sacred regime_stream.parquet (train<2024 model)
    # is read independently by agents/context.py and RL/backtest, never through here.
    path = MARTA_ROOT / "data" / "processed" / "regime_stream_live.parquet"
    if not path.exists():
        path = MARTA_ROOT / "data" / "processed" / "regime_stream.parquet"
    _df = pd.read_parquet(path)
    if "timestamp" in _df.columns and not isinstance(_df.index, pd.DatetimeIndex):
        _df = _df.set_index("timestamp")
    _df.index = pd.to_datetime(_df.index, utc=True)
    print(f"[regime_service] loaded {len(_df):,} rows from {path}")

    # Crypto-only panel — kept as a SEPARATE table (not merged into _df): a different
    # HMM's regime labels mean different things at the same integer index, and this
    # stream has 5 assets _df has never heard of (BNB/LTC/XRP). get_regime()/get_regime_
    # series() dispatch to whichever table actually matches the requested asset.
    #
    # Causal, not the leaky forward-backward-smoothed stream — even a "genuinely
    # historical" lookup here is used as evaluation-relevant context (the debate layer's
    # ContrarianAgent historical-analogue retrieval, backtests reading get_regime_series)
    # and must not be informed by data after its own timestamp. See
    # utils.freshness.resolve_regime_stream_path / regime/causal_regime_stream.py.
    from utils.freshness import resolve_regime_stream_path
    crypto5_path = resolve_regime_stream_path(_CRYPTO5_PROCESSED)
    if crypto5_path.exists():
        _crypto5_df = pd.read_parquet(crypto5_path, engine="pyarrow")
        if "timestamp" in _crypto5_df.columns and not isinstance(_crypto5_df.index, pd.DatetimeIndex):
            _crypto5_df = _crypto5_df.set_index("timestamp")
        _crypto5_df.index = pd.to_datetime(_crypto5_df.index, utc=True)
        print(f"[regime_service] loaded {len(_crypto5_df):,} rows from {crypto5_path}")
    else:
        print(f"[regime_service] crypto5 regime stream not found at {crypto5_path}")


def _resolve_stream_and_descriptions(asset: str):
    """Return (stream_df, descriptions_fn, n_states) for whichever regime model
    actually covers `asset` — crypto5's raw-feature HMM for the 5 crypto5 assets,
    the legacy encoder-based one otherwise. Centralizes the dispatch every lookup
    function below needs so a new caller can't accidentally read the wrong table."""
    if asset in _CRYPTO5_ASSETS:
        if _crypto5_df is None:
            load_regime_stream()
        n_states = _crypto5_hmm_model.n_components if _crypto5_hmm_model is not None else 6
        return _crypto5_df, _load_crypto5_regime_descriptions, n_states
    if _df is None:
        load_regime_stream()
    return _df, _load_regime_descriptions, _n_regime_states()


def get_regime(asset: str, timestamp: pd.Timestamp) -> dict:
    """Historical regime lookup — nearest precomputed row in regime_stream.parquet.

    Correct for genuinely historical queries ("what was the regime on this past
    date"). For "what's the regime right now," prefer get_live_regime() below —
    this function has no way to know its own answer might be looking at a row from
    over a year ago, so it reports that gap explicitly via `stale`/`age_hours`
    instead of presenting an old row as if it were current.
    """
    stream, descriptions_fn, n_states = _resolve_stream_and_descriptions(asset)
    ts = pd.Timestamp(timestamp, tz="UTC") if not isinstance(timestamp, pd.Timestamp) else timestamp
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    asset_df = stream[stream["asset"] == asset] if stream is not None else pd.DataFrame()
    if asset_df.empty:
        return _empty_regime(descriptions_fn, n_states)
    idx = asset_df.index.get_indexer([ts], method="nearest")
    row = asset_df.iloc[idx[0]]
    matched_ts = asset_df.index[idx[0]]
    label = int(row["regime_label"])
    probs = {f"regime_{i}": float(row.get(f"regime_prob_{i}", 0.0)) for i in range(n_states)}
    age_hours = abs((ts - matched_ts).total_seconds()) / 3600
    return {
        "regime_label": label,
        "regime_description": descriptions_fn().get(label, "Unknown"),
        "probabilities": probs,
        "dominant_regime": label,
        "stale": age_hours > _STALE_THRESHOLD_HOURS,
        "age_hours": round(age_hours, 1),
    }


def get_regime_series(asset: str, timestamps: list[pd.Timestamp]) -> list[dict]:
    """Batch regime lookup for a list of timestamps."""
    stream, descriptions_fn, n_states = _resolve_stream_and_descriptions(asset)
    asset_df = stream[stream["asset"] == asset] if stream is not None else pd.DataFrame()
    results = []
    for ts in timestamps:
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts, tz="UTC")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        idx = asset_df.index.get_indexer([ts], method="nearest")
        if len(idx) == 0 or asset_df.empty:
            results.append(_empty_regime(descriptions_fn, n_states))
            continue
        row = asset_df.iloc[idx[0]]
        label = int(row["regime_label"])
        results.append({"regime_label": label, "regime_description": descriptions_fn().get(label, "Unknown")})
    return results


def _format_stream_rows(df: pd.DataFrame, limit: int) -> list[dict]:
    """Shared tail/decimate/sanitize step for get_regime_stream_tail and its live-aware
    variant below — same output shape either way."""
    df = df.tail(limit)
    if len(df) > 2000:
        # Keep every Nth row — 33k+ points overwhelm both the JSON encoder and the chart
        step = len(df) // 2000
        df = df.iloc[::step]
    out = df.reset_index().rename(columns={"index": "timestamp"})
    # Sanitize non-finite values that crash the JSON encoder
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], 0).fillna(0)
    out["timestamp"] = out["timestamp"].astype(str)
    return out.to_dict(orient="records")


def get_regime_stream_tail(asset: str, limit: int = 500) -> list[dict]:
    """Last `limit` regime-stream rows for `asset`, oldest first, decimated to ~2000 points.

    Pure regime_stream.parquet — frozen at whatever run_regime.py last produced. Use
    get_regime_stream_tail_live_aware for anything user-facing (the Regime page); this is
    kept for callers that specifically want the static historical stream.
    """
    stream, _, _ = _resolve_stream_and_descriptions(asset)
    asset_df = stream[stream["asset"] == asset] if stream is not None else pd.DataFrame()
    return _format_stream_rows(asset_df, limit)


def get_regime_stream_tail_live_aware(asset: str, limit: int = 500) -> list[dict]:
    """Like get_regime_stream_tail, but appends regime_live_log.parquet's rows for
    `asset` past regime_stream.parquet's frozen boundary, instead of silently stopping
    there — this is what made the Regime page's probability timeline look permanently
    stuck (found 2026-08-07: /api/regime-stream/{asset} called get_regime_stream_tail
    directly, so the earlier live-aware fix to historical.py's *chart-candle tagging*
    never touched this endpoint at all).

    The live log only has rows from whenever hourly logging for `asset` began (see
    signal_scheduler.py) — there's no way to reconstruct real per-hour regime for the
    period before that started, so a visible gap between the frozen boundary and the
    first live-log entry is expected, not a bug. It fills in and gets denser as more
    hourly ticks land; it never repeats one frozen value across the gap the way the old
    behavior did.
    """
    stream, _, _ = _resolve_stream_and_descriptions(asset)
    asset_df = stream[stream["asset"] == asset] if stream is not None else pd.DataFrame()
    if asset_df.empty:
        return _format_stream_rows(asset_df, limit)
    frozen_max = asset_df.index.max()

    live_tail = pd.DataFrame()
    if _LIVE_LOG_PATH.exists():
        try:
            log = pd.read_parquet(_LIVE_LOG_PATH, engine="pyarrow")
            live_tail = log[(log["asset"] == asset) & (log.index > frozen_max)].sort_index()
        except Exception:
            live_tail = pd.DataFrame()

    if live_tail.empty:
        return _format_stream_rows(asset_df, limit)

    # Decimate the (much larger) static history and the live tail SEPARATELY, then
    # concatenate the formatted results — decimating them together strides over the
    # combined series, and since the live tail is a tiny fraction of a 180k+ row static
    # history, that stride reliably lands nowhere near the last few rows and silently
    # drops the entire live tail (confirmed: BTC's full-history request, limit=200000,
    # produced a step of ~90 and dropped every single live-log row — found 2026-08-07,
    # the exact reason the full-history chart still looked frozen after the first pass
    # of this fix). The live tail is small enough it rarely needs decimation at all in
    # practice, but still routed through _format_stream_rows defensively in case the log
    # grows large over a long-running deployment.
    static_rows = _format_stream_rows(asset_df, limit)
    live_rows = _format_stream_rows(live_tail, len(live_tail))
    return static_rows + live_rows


def get_available_assets() -> list[str]:
    if _df is None:
        load_regime_stream()
    assets = set(_df["asset"].unique().tolist())
    if _crypto5_df is not None:
        assets |= set(_crypto5_df["asset"].unique().tolist())
    return sorted(assets)


def _empty_regime(descriptions_fn=None, n_states: int | None = None) -> dict:
    descriptions_fn = descriptions_fn or _load_regime_descriptions
    n_states = n_states if n_states is not None else _n_regime_states()
    return {
        "regime_label": 0,
        "regime_description": descriptions_fn().get(0, "Unknown"),
        "probabilities": {f"regime_{i}": 0.0 for i in range(n_states)},
        "dominant_regime": 0,
        "stale": True,
        "age_hours": None,
    }


_LIVE_LOOKBACK_HOURS = 168  # 1 week — see get_live_regime() for why a sequence, not a point


def get_live_regime(asset: str, timestamp: pd.Timestamp | None = None, log: bool = False) -> dict:
    """Compute the regime right now, directly from fresh embeddings, instead of
    looking backward at regime_stream.parquet's last precomputed row.

    Queries the HMM with a `_LIVE_LOOKBACK_HOURS`-long ordered sequence ending at
    `timestamp`, not a single isolated point. A single-frame predict_proba() call was
    tried first and verified (empirically, against real data) to be dominated entirely
    by the model's startprob_ prior rather than the actual observation — every
    single-frame query returned the same state regardless of input. regime_stream.parquet
    itself was always built the same way, as one predict_proba() call over the full
    ordered historical sequence; a live query needs the same sequence context to get a
    meaningful answer out of this model, which is why this fetches `_LIVE_LOOKBACK_HOURS`
    hours of history and reads the LAST position of the resulting posterior sequence
    (mathematically equivalent to pure forward-filtering for that last position, so
    there's no leakage from data past `timestamp`).

    Falls back to the historical nearest-lookup (get_regime, with its `stale` flag
    intact) if the live models aren't loaded or there isn't enough history before
    `timestamp` for a full lookback window — e.g. GOLD/EURUSD when their normalized
    parquet hasn't been refreshed close to `timestamp`.

    Crypto-only panel (2026-08 migration): dispatches to get_live_regime_crypto5() for
    the 5 crypto5 assets instead — a different model (raw features, no encoder/PCA), not
    just a different data path. EURUSD/GOLD still use the encoder-based path below.
    """
    ts = timestamp if isinstance(timestamp, pd.Timestamp) else pd.Timestamp("now", tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    if asset in _CRYPTO5_ASSETS:
        return get_live_regime_crypto5(asset, ts, log=log)

    if not _regime_models_loaded:
        load_regime_models()

    if _pca_model is None or _hmm_model is None:
        return {**get_regime(asset, ts), "live": False}

    import ppo_inference
    embeddings = ppo_inference.get_live_embedding_sequence(asset, ts, lookback=_LIVE_LOOKBACK_HOURS)
    if embeddings is None:
        return {**get_regime(asset, ts), "live": False}

    reduced = _pca_model.transform(embeddings.astype(np.float64))
    probs_seq = _hmm_model.predict_proba(reduced)
    probs_arr = probs_seq[-1]  # last position = "now", filtered only by the past
    label = int(probs_arr.argmax())
    probs = {f"regime_{i}": float(p) for i, p in enumerate(probs_arr)}

    result = {
        "regime_label": label,
        "regime_description": _load_regime_descriptions().get(label, "Unknown"),
        "probabilities": probs,
        "dominant_regime": label,
        "stale": False,
        "age_hours": 0.0,
        "live": True,
        "computed_at": ts.isoformat(),
        "lookback_hours": _LIVE_LOOKBACK_HOURS,
    }

    if log:
        _append_live_regime_log(asset, ts, result)

    return result


def _append_live_regime_log(asset: str, ts: pd.Timestamp, result: dict) -> None:
    """Best-effort append of a live regime computation to a small, separate rolling
    log — so the historical record keeps growing instead of staying frozen at
    regime_stream.parquet's last precomputed row. Never raises: this is an audit
    trail, not a dependency of the live signal path. Deliberately opt-in (`log=`
    defaults False in get_live_regime) rather than logging on every call — the
    dashboard's "current regime" GET endpoint and the /ws/regime websocket can be
    polled arbitrarily often and shouldn't each be rewriting this file; only
    live_signal's once-per-hour-per-asset tick opts in.

    n_states is taken from len(result["probabilities"]) — NOT _n_regime_states()
    (that reads the legacy encoder-based _hmm_model's component count regardless of
    which model actually produced `result`). Found 2026-09-03: crypto5 assets route
    through get_live_regime_crypto5()'s 6-state model, but whenever the legacy 4-state
    _hmm_model happened to also be loaded in-process, _n_regime_states() silently
    truncated the logged row to 4 probabilities, leaving regime_prob_4/5 NaN even
    though regime_label (computed from the full 6-state argmax) could legitimately be
    4 or 5. result["probabilities"] always has exactly the right number of keys for
    whichever model computed it (get_live_regime_crypto5, get_live_regime, and
    get_regime's crypto5 dispatch all size it correctly), so deriving n_states from it
    directly is correct for every caller and immune to which model(s) happen to be
    loaded.
    """
    try:
        n_states = len(result.get("probabilities", {})) or _n_regime_states()
        row = {
            "timestamp": ts,
            "asset": asset,
            "regime_label": result["regime_label"],
            **{
                f"regime_prob_{i}": result["probabilities"].get(f"regime_{i}", 0.0)
                for i in range(n_states)
            },
        }
        new_row = pd.DataFrame([row]).set_index("timestamp")
        if _LIVE_LOG_PATH.exists():
            existing = pd.read_parquet(_LIVE_LOG_PATH, engine="pyarrow")
            combined = pd.concat([existing, new_row])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new_row
        _LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(_LIVE_LOG_PATH, engine="pyarrow")
    except Exception as exc:
        print(f"[regime_service] live regime log append failed (non-fatal): {exc}")


# ── Live-aware chart tagging ────────────────────────────────────────────────────
# get_regime_series (above) is a pure nearest-lookup into regime_stream.parquet, whose
# newest row is whatever run_regime.py last produced (2025-12-31 as of this pass, but
# the exact date isn't the point — it stops growing the moment that script stops
# running, same staleness class documented at the top of this file). historical.py's
# chart-candle tagging used to call get_regime_series directly, which meant every
# extension candle fetched live past that boundary (BTC/ETH via ccxt, EURUSD/GOLD via
# yfinance/Twelve Data — see historical.py's _extend()) still got tagged with that one
# frozen last row, for however many months/candles have passed since. Found + fixed
# 2026-08-07 audit, alongside the BTC/ETH live bridge above.

_LIVE_LOG_MAX_AGE_DAYS = 14.0  # nearest-log-entry tolerance for chart tagging — beyond
# this, a "nearest" live_log hit would itself be too stale to trust; fall back to one
# fresh get_live_regime() computation instead (shared across the whole request, not
# recomputed per candle — see get_regime_series_live_aware).


def _nearest_live_log_regime(asset: str, ts: pd.Timestamp, max_age_days: float = _LIVE_LOG_MAX_AGE_DAYS) -> Optional[dict]:
    """Nearest-neighbor lookup in regime_live_log.parquet — the incrementally-growing
    per-hour audit trail (_append_live_regime_log), NOT the frozen regime_stream.parquet
    snapshot. Populated once/hour/asset by the scheduler (live_signal.py for BTC/ETH,
    signal_scheduler._log_nontrading_regimes for EURUSD/GOLD). Returns None if the log
    is missing, has no rows for `asset`, or its nearest entry is more than `max_age_days`
    from `ts` — all three cases are handled by the caller falling back to a fresh compute.
    """
    if not _LIVE_LOG_PATH.exists():
        return None
    try:
        log = pd.read_parquet(_LIVE_LOG_PATH, engine="pyarrow")
    except Exception:
        return None
    asset_log = log[log["asset"] == asset].sort_index()
    if asset_log.empty:
        return None

    pos = int(asset_log.index.searchsorted(ts))
    candidates = [i for i in (pos - 1, pos) if 0 <= i < len(asset_log)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(asset_log.index[i] - ts))
    if abs(asset_log.index[best] - ts) > pd.Timedelta(days=max_age_days):
        return None
    return {"regime_label": int(asset_log.iloc[best]["regime_label"])}


def get_regime_series_live_aware(asset: str, timestamps: list[pd.Timestamp]) -> list[dict]:
    """Like get_regime_series, but candles past regime_stream.parquet's frozen boundary
    use live data instead of silently repeating its last row — for historical.py's chart
    tagging, so a stale regime_stream doesn't make every recent candle look like it's
    stuck in one regime.

    For each timestamp:
      - at/before the frozen boundary: identical to get_regime_series (cheap static
        nearest-lookup — these are genuinely past candles, the fast path is correct).
      - after it: nearest hit in the live audit log (regime_live_log.parquet) if one
        exists within `_LIVE_LOG_MAX_AGE_DAYS`; otherwise ONE shared fresh
        get_live_regime() computation, reused for every such candle in this call.

    Cost bound: at most one full PCA->HMM live recompute per call, never one per candle
    — recomputing per-candle would mean up to hundreds of 168-window HMM passes per
    chart load (see _add_regime's docstring in main.py for the same reasoning applied
    to the live-tick websocket path).
    """
    stream, descriptions_fn, n_states = _resolve_stream_and_descriptions(asset)
    asset_df = stream[stream["asset"] == asset] if stream is not None else pd.DataFrame()
    if asset_df.empty:
        return [_empty_regime(descriptions_fn, n_states) for _ in timestamps]
    frozen_max = asset_df.index.max()

    fresh_live: Optional[dict] = None  # lazily computed at most once, shared below
    results = []
    for ts in timestamps:
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts, tz="UTC")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        if ts <= frozen_max:
            idx = asset_df.index.get_indexer([ts], method="nearest")
            label = int(asset_df.iloc[idx[0]]["regime_label"])
        else:
            hit = _nearest_live_log_regime(asset, ts)
            if hit is None:
                if fresh_live is None:
                    fresh_live = get_live_regime(asset, pd.Timestamp.now(tz="UTC"))
                hit = fresh_live
            label = hit["regime_label"]

        results.append({"regime_label": label, "regime_description": descriptions_fn().get(label, "Unknown")})
    return results
