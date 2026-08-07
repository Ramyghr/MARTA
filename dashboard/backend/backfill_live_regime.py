"""One-time backfill: real per-hour regime history for the gap between
regime_stream.parquet's frozen boundary and today, for all 4 assets.

Why this exists: get_regime_series_live_aware / get_regime_stream_tail_live_aware
(regime_service.py) fall back to a single shared "current regime" value for any
timestamp in that gap that has no nearby regime_live_log.parquet entry — an honest
but flat placeholder, since the hourly scheduler tick only started accumulating real
log entries once the live-bridge fixes shipped (2026-08-07). That flat fallback reads,
visually, almost exactly like the original bug (one color from the gap's start to
today) even though the *value* itself is now correct — not what a regime *timeline*
should show.

This script closes that gap for real instead of waiting weeks for the hourly log to
fill it in on its own: it builds the embedding for every hour in the gap (fast — ONNX,
~5ms/call) and runs ONE hmm.predict_proba() over the whole ordered sequence, batched —
the same method regime_stream.parquet itself was originally built with (one call over
the full ordered historical sequence lets the transition matrix inform the posterior,
unlike a single-frame call, which is dominated by startprob_ — see
regime_service.get_live_regime's docstring for the same point applied to a live query).
This is a genuine historical reconstruction using the real, already-elapsed feature
sequence — not a live "current regime" guess repeated backward in time.

Writes directly into regime_live_log.parquet (the same file get_regime_series_live_aware
/ get_regime_stream_tail_live_aware already read from), merged with whatever hourly
entries the scheduler has already logged since deployment (dedup keep="last" on exact
timestamp collisions — harmless, both sources agree on those in practice).

Run once after deploying the live bridges: conda run -n marta python dashboard/backend/backfill_live_regime.py
Safe to re-run — it always recomputes the full gap fresh and re-merges (idempotent).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
for p in (str(MARTA_ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ppo_inference
import regime_service

_CANONICAL_ASSETS = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]
_LIVE_LOG_PATH = MARTA_ROOT / "data" / "processed" / "regime_live_log.parquet"


def _backfill_asset(asset: str) -> pd.DataFrame | None:
    if regime_service._df is None:
        regime_service.load_regime_stream()
    asset_df = regime_service._df[regime_service._df["asset"] == asset]
    if asset_df.empty:
        print(f"[{asset}] no regime_stream rows — skipping")
        return None
    frozen_max = asset_df.index.max()

    aligned = ppo_inference._get_aligned_df(asset)
    if aligned is None or aligned.empty:
        print(f"[{asset}] no aligned data available — skipping")
        return None
    data_max = aligned.index.max()

    gap_hours = pd.date_range(frozen_max, data_max, freq="1h", inclusive="right")
    if len(gap_hours) == 0:
        print(f"[{asset}] no gap to backfill (data ends at/before {frozen_max})")
        return None

    print(f"[{asset}] backfilling {len(gap_hours):,} hours: {gap_hours[0]} -> {gap_hours[-1]}")

    embeddings, kept_ts = [], []
    for ts in gap_hours:
        emb = ppo_inference._build_regime_embedding(asset, ts)
        if not np.any(emb):
            continue  # insufficient history for this window (shouldn't happen mid-gap)
        embeddings.append(emb)
        kept_ts.append(ts)

    if not embeddings:
        print(f"[{asset}] no valid embeddings in the gap — skipping")
        return None

    stacked = np.stack(embeddings).astype(np.float64)
    reduced = regime_service._pca_model.transform(stacked)
    probs_seq = regime_service._hmm_model.predict_proba(reduced)  # one batched call —
    # forward-backward over the whole ordered gap, not per-hour single-frame calls.

    n_states = probs_seq.shape[1]
    rows = {
        "timestamp": kept_ts,
        "asset": [asset] * len(kept_ts),
        "regime_label": probs_seq.argmax(axis=1).astype(int),
    }
    for i in range(n_states):
        rows[f"regime_prob_{i}"] = probs_seq[:, i]

    out = pd.DataFrame(rows).set_index("timestamp")
    print(f"[{asset}] backfilled {len(out):,} rows")
    return out


def main() -> None:
    ppo_inference.load_model()
    regime_service.load_regime_stream()
    regime_service.load_regime_models()
    if regime_service._pca_model is None or regime_service._hmm_model is None:
        raise RuntimeError("PCA/HMM models failed to load — check regime/checkpoints/")

    backfilled = [df for asset in _CANONICAL_ASSETS if (df := _backfill_asset(asset)) is not None]
    if not backfilled:
        print("Nothing to backfill.")
        return

    new_rows = pd.concat(backfilled).sort_index()
    if _LIVE_LOG_PATH.exists():
        existing = pd.read_parquet(_LIVE_LOG_PATH, engine="pyarrow")
        combined = pd.concat([existing, new_rows])
    else:
        combined = new_rows
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    _LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_LIVE_LOG_PATH, engine="pyarrow")
    print(f"Wrote {len(combined):,} total rows -> {_LIVE_LOG_PATH}")


if __name__ == "__main__":
    main()
