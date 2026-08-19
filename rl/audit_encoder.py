"""Encoder diagnostic — items 3-6 of the 2026-08-11 encoder audit follow-up.

Does NOT retrain the encoder, change InfoNCE, or touch PPO/RL. Pure inspection of the
CURRENTLY DEPLOYED encoder/checkpoints/best_encoder.pt (frozen, loaded read-only).

Item 3: multi-horizon predictability (1h/4h/12h/24h) — raw 14-feature vs 128-dim
        encoder embedding, strict chronological OOS split.
Item 4: reconstruction diagnostic — 128-dim embedding -> MLP -> reconstruct the 14 raw
        features (Z -> X_raw), vs a trivial train-mean baseline.
Item 5: embedding collapse statistics — per-dim mean/std/min/max, pairwise cosine
        similarity, covariance/effective rank, PCA explained variance, near-zero-variance
        dim count, cross-window comparison.
Item 6: raw -> Z predictability (companion to item 4's Z -> raw).

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.audit_encoder
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from rl.env import TRAIN_END, VAL_END, VAL_START
from rl.minimal_btc_env import REGIME_DIM, SIMPLE_FEATURE_COLS, load_shared_data
from rl.supervised_predictability import _valid_indices, build_marta_features, build_simple_features
from utils.logger import logger

_console = Console()
_RESULTS_DIR = Path("data/results")
_SEED = 42
_HORIZONS = [1, 4, 12, 24]


# --------------------------------------------------------------------------- item 3
def build_horizon_labels(df: pd.DataFrame, idx: np.ndarray, horizon: int) -> np.ndarray:
    """Cumulative log return over the next `horizon` bars after t = idx[i]:
    sum(close_return[t+1 : t+1+horizon]) = log(close[t+horizon]/close[t]).
    Requires t+horizon < len(df) — callers filter idx accordingly."""
    ret = df["close_return"].to_numpy(dtype=np.float64)
    n = len(ret)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(ret))])  # cs[i] = sum(ret[0:i])
    out = np.full(len(idx), np.nan)
    valid = idx + horizon < n
    t = idx[valid]
    out[valid] = cs[t + horizon + 1] - cs[t + 1]
    return out


def multi_horizon_report(X_train, X_val, y_train_ret, y_val_ret, name: str, horizon: int) -> dict[str, Any]:
    y_train_bin = (y_train_ret > 0).astype(np.int64)
    y_val_bin = (y_val_ret > 0).astype(np.int64)

    scaler = StandardScaler().fit(X_train)
    Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000, random_state=_SEED).fit(Xtr, y_train_bin)
    pred = clf.predict(Xva)
    proba = clf.predict_proba(Xva)[:, 1]

    reg = LinearRegression().fit(Xtr, y_train_ret)
    pred_ret = reg.predict(Xva)

    pearson_r, _ = pearsonr(pred_ret, y_val_ret)
    spearman_r, _ = spearmanr(pred_ret, y_val_ret)

    result = {
        "observation": name,
        "horizon_hours": horizon,
        "n_train": int(len(y_train_ret)),
        "n_val": int(len(y_val_ret)),
        "accuracy": float(accuracy_score(y_val_bin, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val_bin, pred)),
        "roc_auc": float(roc_auc_score(y_val_bin, proba)) if len(np.unique(y_val_bin)) > 1 else None,
        "baseline_accuracy": float(max(np.mean(y_val_bin == 1), np.mean(y_val_bin == 0))),
        "pearson_r_predicted_vs_actual_return": float(pearson_r),
        "spearman_r_predicted_vs_actual_return": float(spearman_r),
        "linreg_r2_oos": float(r2_score(y_val_ret, pred_ret)),
    }
    logger.info(f"[{name}/{horizon}h] {result}")
    return result


def run_multi_horizon(shared: dict[str, Any]) -> list[dict[str, Any]]:
    _console.rule("[bold]Item 3 — multi-horizon predictability (1h/4h/12h/24h): raw vs encoder embedding")
    df = shared["df"]
    rc = shared.get("require_contiguous", False)
    train_idx = _valid_indices(df.index, "train", require_contiguous=rc, max_horizon=max(_HORIZONS))
    val_idx = _valid_indices(df.index, "val", require_contiguous=rc, max_horizon=max(_HORIZONS))

    X_train_simple = build_simple_features(df, train_idx)
    X_val_simple = build_simple_features(df, val_idx)
    logger.info("Encoding train/val windows through frozen encoder (shared across all horizons)...")
    X_train_marta = build_marta_features(df, train_idx, shared["encoder"], device="cpu")[:, :REGIME_DIM]
    X_val_marta = build_marta_features(df, val_idx, shared["encoder"], device="cpu")[:, :REGIME_DIM]

    results: list[dict[str, Any]] = []
    for h in _HORIZONS:
        y_train = build_horizon_labels(df, train_idx, h)
        y_val = build_horizon_labels(df, val_idx, h)
        tr_mask = ~np.isnan(y_train)
        va_mask = ~np.isnan(y_val)

        results.append(multi_horizon_report(
            X_train_simple[tr_mask], X_val_simple[va_mask], y_train[tr_mask], y_val[va_mask], "raw_14", h,
        ))
        results.append(multi_horizon_report(
            X_train_marta[tr_mask], X_val_marta[va_mask], y_train[tr_mask], y_val[va_mask], "encoder_128", h,
        ))

    table = Table(title="Multi-horizon predictability (out-of-sample, 2024 val)")
    table.add_column("Horizon")
    table.add_column("Obs")
    table.add_column("Acc", justify="right")
    table.add_column("Bal.Acc", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Pearson r", justify="right")
    table.add_column("Spearman r", justify="right")
    table.add_column("R2", justify="right")
    for r in results:
        table.add_row(
            f"{r['horizon_hours']}h", r["observation"], f"{r['accuracy']:.4f}", f"{r['balanced_accuracy']:.4f}",
            f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "n/a", f"{r['baseline_accuracy']:.4f}",
            f"{r['pearson_r_predicted_vs_actual_return']:.4f}", f"{r['spearman_r_predicted_vs_actual_return']:.4f}",
            f"{r['linreg_r2_oos']:.4f}",
        )
    _console.print(table)
    return results


# --------------------------------------------------------------------------- item 4 + 6
def run_reconstruction(shared: dict[str, Any]) -> dict[str, Any]:
    _console.rule("[bold]Item 4/6 — reconstruction: Z(128-d embedding) -> raw 14 features, and raw -> Z")
    df = shared["df"]
    rc = shared.get("require_contiguous", False)
    train_idx = _valid_indices(df.index, "train", require_contiguous=rc, max_horizon=0)
    val_idx = _valid_indices(df.index, "val", require_contiguous=rc, max_horizon=0)

    X_train_simple = build_simple_features(df, train_idx)
    X_val_simple = build_simple_features(df, val_idx)
    X_train_marta = build_marta_features(df, train_idx, shared["encoder"], device="cpu")[:, :REGIME_DIM]
    X_val_marta = build_marta_features(df, val_idx, shared["encoder"], device="cpu")[:, :REGIME_DIM]

    z_scaler = StandardScaler().fit(X_train_marta)
    Ztr, Zva = z_scaler.transform(X_train_marta), z_scaler.transform(X_val_marta)
    raw_scaler = StandardScaler().fit(X_train_simple)
    Xtr, Xva = raw_scaler.transform(X_train_simple), raw_scaler.transform(X_val_simple)

    # ---- Z -> X_raw (the important direction) ----
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=_SEED, early_stopping=True)
    mlp.fit(Ztr, Xtr)
    pred_val = mlp.predict(Zva)

    per_feature_r2 = {}
    per_feature_mse = {}
    for i, col in enumerate(SIMPLE_FEATURE_COLS):
        per_feature_r2[col] = float(r2_score(Xva[:, i], pred_val[:, i]))
        per_feature_mse[col] = float(np.mean((Xva[:, i] - pred_val[:, i]) ** 2))

    # trivial baseline: predict the TRAIN MEAN (=0 after StandardScaler fit on train) for every val row
    baseline_pred = np.zeros_like(Xva)
    baseline_r2 = {col: float(r2_score(Xva[:, i], baseline_pred[:, i])) for i, col in enumerate(SIMPLE_FEATURE_COLS)}

    overall_r2 = float(r2_score(Xva, pred_val))  # multioutput='uniform_average' default
    overall_mse = float(np.mean((Xva - pred_val) ** 2))
    baseline_overall_r2 = float(r2_score(Xva, baseline_pred))

    z_to_raw = {
        "per_feature_r2_mlp": per_feature_r2,
        "per_feature_mse_mlp": per_feature_mse,
        "per_feature_r2_trivial_baseline": baseline_r2,
        "overall_r2_mlp": overall_r2,
        "overall_mse_mlp": overall_mse,
        "overall_r2_trivial_baseline": baseline_overall_r2,
    }
    _console.print(f"Z -> X_raw reconstruction: overall R2 = {overall_r2:.4f} (MLP) vs {baseline_overall_r2:.4f} (predict train mean)")
    table = Table(title="Z -> X_raw: per-feature reconstruction R2 (val, OOS)")
    table.add_column("Feature")
    table.add_column("MLP R2", justify="right")
    table.add_column("Baseline R2", justify="right")
    for col in SIMPLE_FEATURE_COLS:
        table.add_row(col, f"{per_feature_r2[col]:.4f}", f"{baseline_r2[col]:.4f}")
    _console.print(table)

    # ---- X_raw -> Z (companion direction, item 6 — secondary/lower priority) ----
    # Hard-timeboxed: an MLPRegressor(64,128).fit(34026 samples) hung indefinitely (0 CPU
    # progress over repeated samples) during the 2026-08-11 E2 batch eval, likely BLAS/OpenMP
    # thread contention from running immediately after other sklearn-heavy processes in the
    # same session — not reproduced on the very first (isolated) run of this script hours
    # earlier. Since this is explicitly the less important direction, skip gracefully on
    # timeout rather than block the whole batch.
    raw_to_z: dict[str, Any] = {"overall_r2_mlp": None, "skipped_reason": None}
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("X_raw -> Z MLP fit exceeded 90s timeout")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(90)
    try:
        mlp2 = MLPRegressor(hidden_layer_sizes=(64, 128), max_iter=500, random_state=_SEED, early_stopping=True)
        mlp2.fit(Xtr, Ztr)
        pred_z_val = mlp2.predict(Xva)
        raw_to_z_overall_r2 = float(r2_score(Zva, pred_z_val))
        raw_to_z = {"overall_r2_mlp": raw_to_z_overall_r2, "skipped_reason": None}
        _console.print(f"X_raw -> Z: overall R2 = {raw_to_z_overall_r2:.4f} (predicting the 128-dim embedding from the SAME single current-bar raw features the embedding's 60-bar window ends on)")
    except TimeoutError as exc:
        raw_to_z = {"overall_r2_mlp": None, "skipped_reason": str(exc)}
        _console.print(f"[yellow]X_raw -> Z: SKIPPED — {exc}[/yellow]")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return {"z_to_raw": z_to_raw, "raw_to_z": raw_to_z}


# --------------------------------------------------------------------------- item 5
def run_collapse_diagnostics(shared: dict[str, Any]) -> dict[str, Any]:
    _console.rule("[bold]Item 5 — embedding collapse statistics")
    df = shared["df"]
    val_idx = _valid_indices(df.index, "val", require_contiguous=shared.get("require_contiguous", False), max_horizon=0)
    Z = build_marta_features(df, val_idx, shared["encoder"], device="cpu")[:, :REGIME_DIM]

    per_dim_mean = Z.mean(axis=0)
    per_dim_std = Z.std(axis=0)
    per_dim_min = Z.min(axis=0)
    per_dim_max = Z.max(axis=0)
    near_zero_var_dims = int(np.sum(per_dim_std < 1e-4))

    # Pairwise cosine similarity on a random sample (full 5928x5928 matrix is fine but sample for speed)
    rng = np.random.default_rng(_SEED)
    sample_idx = rng.choice(len(Z), size=min(1500, len(Z)), replace=False)
    Zs = Z[sample_idx]
    norms = np.linalg.norm(Zs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    Zn = Zs / norms
    cos_sim = Zn @ Zn.T
    iu = np.triu_indices_from(cos_sim, k=1)
    cos_vals = cos_sim[iu]

    # Covariance / effective rank via SVD on centered embeddings
    Zc = Z - per_dim_mean
    cov = np.cov(Zc, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    total_var = eigvals.sum()
    sorted_eig = np.sort(eigvals)[::-1]
    explained_ratio = sorted_eig / max(total_var, 1e-12)
    cum_explained = np.cumsum(explained_ratio)
    n_dims_for_90pct = int(np.searchsorted(cum_explained, 0.90) + 1)
    n_dims_for_95pct = int(np.searchsorted(cum_explained, 0.95) + 1)
    # effective rank (exp of entropy of normalized eigenvalue spectrum, Roy & Vetterli)
    p = eigvals / max(eigvals.sum(), 1e-12)
    p_nonzero = p[p > 1e-12]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    effective_rank = float(np.exp(entropy))

    # Cross-window comparison: embeddings from within the same week vs a year apart
    ts = df.index[val_idx]
    same_week_mask = (ts >= pd.Timestamp("2024-03-01", tz="UTC")) & (ts < pd.Timestamp("2024-03-08", tz="UTC"))
    other_period_mask = (ts >= pd.Timestamp("2024-09-01", tz="UTC")) & (ts < pd.Timestamp("2024-09-08", tz="UTC"))
    same_week_Z = Z[same_week_mask]
    other_Z = Z[other_period_mask]
    within_week_cos = None
    cross_period_cos = None
    if len(same_week_Z) > 5 and len(other_Z) > 5:
        sw_n = same_week_Z / np.maximum(np.linalg.norm(same_week_Z, axis=1, keepdims=True), 1e-8)
        ot_n = other_Z / np.maximum(np.linalg.norm(other_Z, axis=1, keepdims=True), 1e-8)
        within_week_cos = float(np.mean((sw_n @ sw_n.T)[np.triu_indices(len(sw_n), k=1)]))
        cross_period_cos = float(np.mean((sw_n @ ot_n.T)))

    result = {
        "n_val_embeddings": int(len(Z)),
        "embedding_dim": int(Z.shape[1]),
        "per_dim_mean_summary": {
            "mean_of_means": float(per_dim_mean.mean()), "std_of_means": float(per_dim_mean.std()),
        },
        "per_dim_std_summary": {
            "mean_of_stds": float(per_dim_std.mean()), "min_std": float(per_dim_std.min()),
            "max_std": float(per_dim_std.max()), "median_std": float(np.median(per_dim_std)),
        },
        "per_dim_min": float(per_dim_min.min()),
        "per_dim_max": float(per_dim_max.max()),
        "near_zero_variance_dims_count": near_zero_var_dims,
        "near_zero_variance_threshold": 1e-4,
        "pairwise_cosine_similarity": {
            "mean": float(cos_vals.mean()), "std": float(cos_vals.std()),
            "min": float(cos_vals.min()), "max": float(cos_vals.max()),
            "p05": float(np.percentile(cos_vals, 5)), "p95": float(np.percentile(cos_vals, 95)),
        },
        "pca_effective_rank": {
            "n_dims_for_90pct_variance": n_dims_for_90pct,
            "n_dims_for_95pct_variance": n_dims_for_95pct,
            "effective_rank_entropy_based": effective_rank,
            "total_dims": REGIME_DIM,
        },
        "cross_window_comparison": {
            "within_same_week_mean_cosine_sim": within_week_cos,
            "across_6_months_apart_mean_cosine_sim": cross_period_cos,
        },
    }
    _console.print(result)
    if within_week_cos is not None and cross_period_cos is not None:
        _console.print(
            f"[bold]Within-week cosine sim = {within_week_cos:.4f} vs cross-6-months cosine sim = "
            f"{cross_period_cos:.4f} — {'embeddings DO vary across genuinely different market periods (not collapsed)' if abs(within_week_cos - cross_period_cos) > 0.02 else 'little difference — possible collapse or the two periods are similarly regimed'}"
        )
    return result


# --------------------------------------------------------------------------- main
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Encoder diagnostic — items 3-6")
    parser.add_argument(
        "--encoder-checkpoint", default=None,
        help="Path to an encoder checkpoint (default: production best_encoder.pt). "
             "Pass an E2 periodic/best/final checkpoint to evaluate it with the identical protocol.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Suffix for the output json filename (e.g. 'e2_best') so results from "
             "different checkpoints don't overwrite each other.",
    )
    parser.add_argument(
        "--file-suffix", default="_aligned",
        help="Data source suffix (default '_aligned', exact prior behavior). Pass "
             "'_normalized' to read the pre-alignment, gap-free file — see "
             "encoder/train.py's _load_asset_split docstring. Should be paired with "
             "--require-contiguous.",
    )
    parser.add_argument(
        "--require-contiguous", action="store_true",
        help="Reject any window/label whose span isn't genuinely hourly-contiguous "
             "(2026-08-14 gap-corruption fix). Default off preserves exact prior results.",
    )
    args = parser.parse_args()

    torch.manual_seed(_SEED)
    np.random.seed(_SEED)

    encoder_path = Path(args.encoder_checkpoint) if args.encoder_checkpoint else None
    shared = load_shared_data(device="cpu", encoder_path=encoder_path, file_suffix=args.file_suffix)
    shared["require_contiguous"] = args.require_contiguous
    logger.info(f"Evaluating encoder checkpoint: {shared['encoder_path']}  "
                f"(file_suffix={args.file_suffix!r}, require_contiguous={args.require_contiguous})")

    horizon_results = run_multi_horizon(shared)
    reconstruction_results = run_reconstruction(shared)
    collapse_results = run_collapse_diagnostics(shared)

    out = {
        "encoder_checkpoint": shared["encoder_path"],
        "item3_multi_horizon": horizon_results,
        "item4_6_reconstruction": reconstruction_results,
        "item5_collapse": collapse_results,
    }
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = _RESULTS_DIR / f"encoder_diagnostic{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    _console.rule("[bold]Encoder diagnostic complete")
    logger.success(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
