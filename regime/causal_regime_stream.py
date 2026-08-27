"""
regime/causal_regime_stream.py

Walk-forward, CAUSAL replacement for regime/hmm.py's generate_regime_stream() — fixes a
real lookahead bias confirmed in the 2026-08-26 pre-RL audit.

The bug: generate_regime_stream() (regime/hmm.py:282-283) calls
`hmm_model.predict_proba(emb_reduced, lengths=lengths)` on an asset's ENTIRE history in
one shot. hmmlearn's predict_proba is forward-BACKWARD (smoothed) — the posterior it
returns for a 2019 row is influenced by 2025 data, because the backward pass runs from
the end of the sequence back to the start. This is real, not theoretical: it's the
documented root cause behind why obs_mode="raw_regime" underperformed obs_mode="raw" in
the crypto5 RL ablation (docs/RESULTS_AND_METHODOLOGY.md Section 9, rows 1-2) — the
regime-probability feature the policy saw during training was quietly conditioned on
future market data it would never have access to live.

The fix is NOT a new algorithm — dashboard/backend/regime_service.py's
get_live_regime_crypto5() already computes a provably causal quantity: it slices history
strictly `<= timestamp`, runs predict_proba() on that bounded trailing window, and keeps
only the LAST position of the returned posterior sequence. That last position is
mathematically identical to pure forward-filtering (P(state_t | obs_1..t)), because there
is no "future" inside a window that ends at t — the backward pass's contribution at the
window's own last step is uniform. This module applies that exact same trick to every
row of history instead of just "right now", by walking forward one step at a time and
re-running the model at each step — same function call, same window length, just at
every historical timestamp instead of once per live serving request. This keeps the
offline stream and live serving computing the identical quantity, rather than
introducing a second, hand-rolled forward-filter implementation that could silently
diverge from hmmlearn's own (log-space, numerically careful) internals.

This is deliberately NOT split-boundary-restricted the way regime/hmm.py's model FITTING
is (configs/splits.py's TestSetContaminationError guards there exist to keep 2025 test
data out of parameter estimation). This module does no fitting — it loads the existing,
already train-only-fit checkpoint (regime/checkpoints/{tag}/hmm_model.pkl) and only
*decodes* with it. Causal decoding at row t reads nothing after t by construction, so
running it across the full train+val+test history is safe: whichever split later reads a
given row's probability will only ever see information that was actually available by
that row's own timestamp, no matter which split the row happens to fall in.

Run:
  conda run -n marta python regime/causal_regime_stream.py
  conda run -n marta python regime/causal_regime_stream.py --tag crypto5_multihorizon --n-jobs 5
  conda run -n marta python regime/causal_regime_stream.py --assets BTC_USDT --lookback-hours 168
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import logger  # noqa: E402

# Matches dashboard/backend/regime_service.py's _LIVE_LOOKBACK_HOURS — same window length
# so this offline stream and the live serving path compute the identical quantity, just
# at different points in time.
LOOKBACK_HOURS = 168

_DEFAULT_TAG = "crypto5_multihorizon"
_DEFAULT_ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]
_PROGRESS_EVERY = 10_000


def _causal_probs_for_asset(
    args: tuple[str, np.ndarray, Path, int],
) -> tuple[str, np.ndarray, np.ndarray]:
    """Worker: walk forward through one asset's feature matrix, one row at a time.
    At row t, run predict_proba() on feature_matrix[max(0, t-lookback+1) : t+1] (a
    growing window until a full lookback is available, then a fixed-length sliding
    window) and keep only the last row of the returned posterior — see module docstring
    for why that one position is provably causal. Must be module-level (picklable for
    ProcessPoolExecutor), matching regime/hmm.py:_fit_one_seed's pattern.

    Growing window (rather than skipping/NaN-ing the first lookback-1 rows) still never
    reads a future row — it just has less context very early in an asset's history,
    which is the correct, honest behavior rather than an artificial gap.
    """
    asset, feature_matrix, hmm_path, lookback = args
    hmm_model = joblib.load(hmm_path)
    n_states = hmm_model.n_components
    n_rows = feature_matrix.shape[0]

    assert feature_matrix.shape[1] == hmm_model.means_.shape[1], (
        f"[{asset}] feature matrix has {feature_matrix.shape[1]} cols, "
        f"HMM was fit on {hmm_model.means_.shape[1]}"
    )

    labels = np.empty(n_rows, dtype=np.int64)
    probs = np.empty((n_rows, n_states), dtype=np.float64)

    t0 = time.time()
    for t in range(n_rows):
        start = max(0, t - lookback + 1)
        window = feature_matrix[start : t + 1]
        posterior_seq = hmm_model.predict_proba(window)
        probs[t] = posterior_seq[-1]
        labels[t] = int(np.argmax(probs[t]))

        if (t + 1) % _PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate = (t + 1) / elapsed
            eta = (n_rows - t - 1) / rate if rate > 0 else float("nan")
            print(
                f"[causal_regime_stream][{asset}] {t + 1:,}/{n_rows:,} rows "
                f"({rate:.0f} rows/s, ETA {eta:.0f}s)",
                flush=True,
            )

    print(f"[causal_regime_stream][{asset}] done: {n_rows:,} rows in {time.time() - t0:.1f}s", flush=True)
    return asset, labels, probs


def generate_causal_regime_stream(
    processed_dir: Path,
    hmm_path: Path,
    assets: list[str],
    lookback_hours: int = LOOKBACK_HOURS,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Build a causal regime-probability stream for every asset in `assets`.

    Reads {asset}_aligned.parquet per asset (written by regime/build_crypto5_features.py)
    and derives the feature-column order from the file itself rather than hardcoding a
    third copy of the 22-col list (already duplicated once in
    dashboard/backend/regime_service.py's _CRYPTO5_FEATURE_COLS and once implicitly via
    build_crypto5_features.py's own column ordering — deriving it here avoids adding a
    fourth independent copy that could drift out of sync).

    Assets are independent series, so they're computed in parallel worker processes
    (n_jobs, default: min(len(assets), cpu_count)) using a spawn context — same
    fork+BLAS-deadlock reasoning as regime/hmm.py:_fit_multiseed (predict_proba on a
    freshly-unpickled model still touches BLAS via numpy).
    """
    n_jobs = n_jobs or min(len(assets), os.cpu_count() or 1)

    ref_cols: Optional[list[str]] = None
    feature_matrices: dict[str, np.ndarray] = {}
    timestamps_by_asset: dict[str, pd.DatetimeIndex] = {}

    for asset in assets:
        path = processed_dir / f"{asset}_aligned.parquet"
        df = pd.read_parquet(path, engine="pyarrow")
        if ref_cols is None:
            ref_cols = list(df.columns)
        else:
            assert list(df.columns) == ref_cols, (
                f"[{asset}] column order {list(df.columns)} does not match "
                f"reference {ref_cols} (from {assets[0]})"
            )
        feature_matrices[asset] = df[ref_cols].to_numpy(dtype=np.float64)
        timestamps_by_asset[asset] = df.index

    logger.info(
        f"Causal regime stream: {len(assets)} asset(s), lookback={lookback_hours}h, "
        f"n_jobs={n_jobs}, cols={ref_cols}"
    )

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    spawn_ctx = multiprocessing.get_context("spawn")

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=spawn_ctx) as executor:
        futures = {
            executor.submit(
                _causal_probs_for_asset,
                (asset, feature_matrices[asset], hmm_path, lookback_hours),
            ): asset
            for asset in assets
        }
        for future in as_completed(futures):
            asset, labels, probs = future.result()
            results[asset] = (labels, probs)

    frames = []
    n_states = next(iter(results.values()))[1].shape[1]
    for asset in assets:
        labels, probs = results[asset]
        idx = timestamps_by_asset[asset]
        df = pd.DataFrame({"timestamp": idx, "asset": asset, "regime_label": labels})
        for i in range(n_states):
            df[f"regime_prob_{i}"] = probs[:, i]
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True)
    dist = out["regime_label"].value_counts(normalize=True).sort_index()
    logger.info(f"Causal regime distribution (all assets):\n{dist.to_string()}")
    return out


def _compare_to_leaky(processed_dir: Path, causal_df: pd.DataFrame, asset: str, sample_ts: str) -> None:
    """Sanity check: at a fixed historical timestamp deep in train, log the causal vs.
    leaky (regime/hmm.py's generate_regime_stream) probability vectors side by side, so
    the fix's effect is a verified, logged number rather than an assumption. Best-effort
    — the leaky stream may not exist for this tag, in which case this just logs a note.
    """
    leaky_path = processed_dir / "regime_stream.parquet"
    if not leaky_path.exists():
        logger.info(f"No existing regime_stream.parquet at {processed_dir} — skipping leaky comparison")
        return
    leaky = pd.read_parquet(leaky_path, engine="pyarrow")
    ts = pd.Timestamp(sample_ts, tz="UTC")

    causal_asset = causal_df[causal_df["asset"] == asset].copy()
    causal_asset["timestamp"] = pd.to_datetime(causal_asset["timestamp"], utc=True)
    leaky_asset = leaky[leaky["asset"] == asset].copy()
    leaky_asset["timestamp"] = pd.to_datetime(leaky_asset["timestamp"], utc=True)

    c_idx = (causal_asset["timestamp"] - ts).abs().idxmin()
    l_idx = (leaky_asset["timestamp"] - ts).abs().idxmin()
    c_row, l_row = causal_asset.loc[c_idx], leaky_asset.loc[l_idx]

    n_states = sum(1 for c in causal_asset.columns if c.startswith("regime_prob_"))
    c_probs = np.array([c_row[f"regime_prob_{i}"] for i in range(n_states)])
    l_probs = np.array([l_row.get(f"regime_prob_{i}", 0.0) for i in range(n_states)])
    l1_diff = float(np.abs(c_probs - l_probs).sum())

    logger.info(
        f"[{asset} @ {sample_ts}] causal label={int(c_row['regime_label'])} "
        f"probs={np.round(c_probs, 3).tolist()}  |  "
        f"leaky label={int(l_row['regime_label'])} probs={np.round(l_probs, 3).tolist()}  |  "
        f"L1 diff={l1_diff:.3f}"
    )
    mlflow.log_metric(f"causal_vs_leaky_l1diff_{asset}", l1_diff)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=_DEFAULT_TAG,
                         help="data/processed/{tag}/ + regime/checkpoints/{tag}/hmm_model.pkl (default: crypto5_multihorizon)")
    parser.add_argument("--assets", type=str, default=None,
                         help="Comma-separated asset ids (default: the 5 crypto5 assets)")
    parser.add_argument("--lookback-hours", type=int, default=LOOKBACK_HOURS)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--compare-ts", type=str, default="2020-03-12",
                         help="Timestamp to log a causal-vs-leaky comparison at (BTC March 2020 crash by default)")
    args = parser.parse_args()

    processed_dir = ROOT / "data" / "processed" / args.tag
    hmm_path = ROOT / "regime" / "checkpoints" / args.tag / "hmm_model.pkl"
    assets = args.assets.split(",") if args.assets else list(_DEFAULT_ASSETS)

    if not hmm_path.exists():
        raise FileNotFoundError(f"{hmm_path} not found — run regime/run_hmm_crypto5_test.py first.")

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment("marta_regime_hmm")
    with mlflow.start_run(run_name=f"causal_regime_stream_{args.tag}"):
        mlflow.log_param("tag", args.tag)
        mlflow.log_param("assets", assets)
        mlflow.log_param("lookback_hours", args.lookback_hours)

        t0 = time.time()
        causal_df = generate_causal_regime_stream(
            processed_dir, hmm_path, assets,
            lookback_hours=args.lookback_hours, n_jobs=args.n_jobs,
        )
        elapsed = time.time() - t0

        out_path = processed_dir / "regime_stream_causal.parquet"
        causal_df.to_parquet(out_path, engine="pyarrow", index=False)
        logger.success(f"Saved causal regime stream -> {out_path}  ({len(causal_df):,} rows, {elapsed:.0f}s)")

        mlflow.log_metric("n_rows", len(causal_df))
        mlflow.log_metric("wall_clock_seconds", elapsed)
        mlflow.log_artifact(str(out_path))

        if args.compare_ts and "BTC_USDT" in assets:
            _compare_to_leaky(processed_dir, causal_df, "BTC_USDT", args.compare_ts)


if __name__ == "__main__":
    main()
