"""scripts/regime_direction_horizon_test.py

Does the CURRENT regime label predict price direction over multi-hour forward horizons
(1h out to 72h) — not just the literal next bar? This is a distinct, more specific claim
than "regime predicts next-bar direction" (already established at ~chance,
docs/RESULTS_AND_METHODOLOGY.md) and the zigzag-leg test (project memory, 49% match) —
neither directly tested "does regime X's historical directional lean generalize
out-of-sample at a several-hour horizon."

Uses the CAUSAL regime stream (regime/causal_regime_stream.py) — this question is about
what a live regime reading could actually tell you about the near future, so the leaky
stream (which would answer a contaminated, different question) is never touched.

Methodology (mirrors rl/crypto5_feature_predictability.py's train<2024/val=2024 split,
same "fit on train, test out-of-sample on val" discipline used throughout this project):
  1. For each asset, compute the forward cumulative return over each horizon h at every
     timestamp t, using ONLY bars t+1..t+h (never bar t itself or anything before it).
  2. TRAIN split: for each regime label, compute the fraction of TRAIN rows where the
     h-hour forward return was positive — a "directional lean" score in [0,1] per
     regime, per horizon.
  3. VAL split: use the TRAIN-derived per-regime lean as a continuous prediction score,
     compute ROC-AUC against the actual out-of-sample forward-direction label.

PLACEBO CONTROL (added after the raw run showed AUC creeping to ~0.52-0.528 at 36-72h —
worth checking whether that's real or an artifact of long-horizon forward-return windows
overlapping heavily between consecutive rows, which makes "n_val" rows far less
independent than the count suggests). For each horizon, the regime_label column is
randomly permuted WITHIN each asset's own row order (breaks any real temporal link to
price direction; preserves each asset's regime-label frequency distribution exactly) and
the identical train-lean/val-AUC pipeline is rerun N_SHUFFLES times. If the real AUC sits
inside the resulting null distribution, the "signal" is an artifact of the AUC
computation/overlap structure, not the regime label. If it sits clearly outside, that's
real evidence the label carries information the placebo can't reproduce.

Run: conda run -n marta python scripts/regime_direction_horizon_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.splits import TRAIN_END_EXCLUSIVE, VAL_END_EXCLUSIVE, VAL_START  # noqa: E402
from utils.logger import logger  # noqa: E402

_PROCESSED = ROOT / "data" / "processed" / "crypto5_multihorizon"
_REGIME_STREAM_CAUSAL = _PROCESSED / "regime_stream_causal.parquet"
ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]
# Extended out to 72h (3 days) — well past the regime layer's own ~11h median dwell time
# (project memory: "Crypto5 Regime-vs-History Check"), to check whether a longer horizon
# than the regime's own typical duration reveals anything the shorter ones didn't.
HORIZONS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 36, 48, 72]
N_SHUFFLES = 40
_SEED = 42


def forward_cum_return(returns: np.ndarray, h: int) -> np.ndarray:
    """fwd[t] = cumulative return over bars t+1..t+h inclusive (causal — never bar t
    itself or anything before it). NaN where there isn't enough future data left."""
    n = len(returns)
    log_r = np.log1p(np.nan_to_num(returns, nan=0.0))
    cumsum = np.concatenate([[0.0], np.cumsum(log_r)])
    fwd = np.full(n, np.nan)
    valid_n = n - h
    if valid_n > 0:
        fwd[:valid_n] = np.expm1(cumsum[h + 1 : h + 1 + valid_n] - cumsum[1 : 1 + valid_n])
    return fwd


def load_asset_frame(asset: str, regime_stream: pd.DataFrame) -> pd.DataFrame:
    aligned = pd.read_parquet(_PROCESSED / f"{asset}_aligned.parquet", engine="pyarrow")
    reg = regime_stream[regime_stream["asset"] == asset].set_index("timestamp")["regime_label"]
    reg.index = pd.to_datetime(reg.index, utc=True)
    df = aligned[["close_return"]].join(reg, how="inner")
    return df


def build_horizon_df(frames: dict[str, pd.DataFrame], h: int) -> pd.DataFrame:
    rows = []
    for asset, df in frames.items():
        fwd = forward_cum_return(df["close_return"].to_numpy(), h)
        sub = pd.DataFrame({
            "timestamp": df.index, "asset": asset,
            "regime_label": df["regime_label"].to_numpy(),
            "up": (fwd > 0).astype(float),
        })
        sub = sub[~np.isnan(fwd)]
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def compute_auc(
    df: pd.DataFrame, label_col: str, train_end: pd.Timestamp, val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> tuple[float, pd.Series]:
    train_df = df[df["timestamp"] < train_end]
    val_df = df[(df["timestamp"] >= val_start) & (df["timestamp"] < val_end)]
    lean = train_df.groupby(label_col)["up"].mean()
    val_df = val_df[val_df[label_col].isin(lean.index)]
    if len(val_df) == 0 or val_df["up"].nunique() < 2:
        return float("nan"), lean
    score = val_df[label_col].map(lean)
    return float(roc_auc_score(val_df["up"], score)), lean


def shuffled_labels(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Permute regime_label independently, ROW BY ROW, within each asset. Breaks any
    temporal link to price direction and preserves each asset's regime label frequency
    distribution — but ALSO destroys the regime process's own persistence/block structure
    (median dwell ~11h, project memory) entirely, scattering it into i.i.d. noise. That
    makes this a weaker null than it looks: a real, persistent (autocorrelated) label
    naturally has SOME correlation with an autocorrelated forward-return series just from
    both having multi-hour timescales, independent of whether the label's CONTENT says
    anything about direction. This placebo can't detect that confound because it removes
    persistence entirely rather than only breaking the direction-specific content — see
    circular_shift_labels for the stricter, more appropriate null."""
    out = df["regime_label"].to_numpy().copy()
    asset_codes = df["asset"].to_numpy()
    for asset in np.unique(asset_codes):
        idx = np.flatnonzero(asset_codes == asset)
        out[idx] = rng.permutation(out[idx])
    return out


def circular_shift_labels(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Circularly shift each asset's regime_label series by a random offset. Preserves
    the label's own serial correlation / block (dwell-time) structure and marginal
    distribution EXACTLY — only destroys its temporal alignment with the return series.
    This is the appropriate null for an autocorrelated label: it isolates "does the
    CONTENT of the label matter" from "does a persistent, multi-hour-scale process just
    happen to line up with a persistent, multi-hour-scale return series regardless of
    content" (the confound row-shuffling can't rule out)."""
    out = df["regime_label"].to_numpy().copy()
    asset_codes = df["asset"].to_numpy()
    orig = df["regime_label"].to_numpy()
    for asset in np.unique(asset_codes):
        idx = np.flatnonzero(asset_codes == asset)
        n = len(idx)
        low, high = max(1, n // 10), max(2, n - n // 10)
        shift = int(rng.integers(low, high)) if high > low else 1
        out[idx] = np.roll(orig[idx], shift)
    return out


def main() -> None:
    regime_stream = pd.read_parquet(_REGIME_STREAM_CAUSAL, engine="pyarrow")
    regime_stream["timestamp"] = pd.to_datetime(regime_stream["timestamp"], utc=True)
    train_end = pd.Timestamp(TRAIN_END_EXCLUSIVE, tz="UTC")
    val_start = pd.Timestamp(VAL_START, tz="UTC")
    val_end = pd.Timestamp(VAL_END_EXCLUSIVE, tz="UTC")

    frames = {a: load_asset_frame(a, regime_stream) for a in ASSETS}
    rng = np.random.default_rng(_SEED)

    print(f"\n{'='*100}\nReal AUC vs. TWO placebo nulls (row-shuffle: weak/destroys persistence; "
          f"circular-shift: strict/preserves it), {N_SHUFFLES} draws each, per horizon\n{'='*100}")

    def _placebo_stats(df: pd.DataFrame, shuffle_fn, real_auc: float) -> dict:
        aucs = []
        for _ in range(N_SHUFFLES):
            df["_shuffled"] = shuffle_fn(df, rng)
            auc, _ = compute_auc(df, "_shuffled", train_end, val_start, val_end)
            if not np.isnan(auc):
                aucs.append(auc)
        aucs = np.array(aucs)
        mean, std = aucs.mean(), aucs.std()
        p95 = np.percentile(aucs, 95)
        z = (real_auc - mean) / std if std > 0 else float("nan")
        return {"mean": mean, "std": std, "p95": p95, "z": z, "outside_95": bool(real_auc > p95)}

    all_results = []
    for h in HORIZONS:
        df = build_horizon_df(frames, h)
        real_auc, lean = compute_auc(df, "regime_label", train_end, val_start, val_end)

        row = _placebo_stats(df, shuffled_labels, real_auc)
        circ = _placebo_stats(df, circular_shift_labels, real_auc)

        print(f"\n--- horizon = {h}h ---  real AUC = {real_auc:.4f}")
        print(f"  row-shuffle    placebo: mean={row['mean']:.4f} std={row['std']:.4f} z={row['z']:+.2f}  {'REAL?' if row['outside_95'] else 'artifact'}")
        print(f"  circular-shift placebo: mean={circ['mean']:.4f} std={circ['std']:.4f} z={circ['z']:+.2f}  {'REAL?' if circ['outside_95'] else 'artifact'}")
        all_results.append({"horizon_h": h, "real_auc": real_auc, "row": row, "circ": circ})

    print(f"\n{'='*100}\nSUMMARY\n{'='*100}")
    print(f"{'horizon':<10}{'real AUC':>10}{'row-shuf z':>12}{'row verdict':>14}{'circ-shift z':>14}{'circ verdict':>14}")
    for r in all_results:
        rv = "REAL?" if r["row"]["outside_95"] else "artifact"
        cv = "REAL?" if r["circ"]["outside_95"] else "artifact"
        print(f"{r['horizon_h']:<10}{r['real_auc']:>10.4f}{r['row']['z']:>12.2f}{rv:>14}{r['circ']['z']:>14.2f}{cv:>14}")

    logger.info("z = (real AUC - placebo mean) / placebo std. 'REAL?' means the real AUC "
                "exceeds the placebo distribution's 95th percentile — a candidate for real "
                "signal, still not a substitute for a proper multi-seed/multi-window "
                "significance test, just stronger evidence than the raw AUC alone.")


if __name__ == "__main__":
    main()
