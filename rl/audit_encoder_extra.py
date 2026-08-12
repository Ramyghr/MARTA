"""Incremental follow-up to rl/audit_encoder.py — fills in the additional metrics
requested in the 2026-08-11 encoder audit follow-up round: per-feature MAE and
true-vs-reconstructed correlation (item 4), and per-dim median/percentiles, condition
number, individual PC1-PC5 variance, and the 99% PCA threshold (item 5). Does not
retrain the encoder or touch PPO.

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.audit_encoder_extra
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
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from rl.minimal_btc_env import REGIME_DIM, SIMPLE_FEATURE_COLS, load_shared_data
from rl.supervised_predictability import _valid_indices, build_marta_features, build_simple_features
from utils.logger import logger

_console = Console()
_RESULTS_DIR = Path("data/results")
_SEED = 42


def run_reconstruction_extra(shared: dict[str, Any]) -> dict[str, Any]:
    _console.rule("[bold]Item 4 extra — MAE + true-vs-reconstructed correlation, Z -> X_raw")
    df = shared["df"]
    train_idx = _valid_indices(df.index, "train")
    val_idx = _valid_indices(df.index, "val")

    X_train_simple = build_simple_features(df, train_idx)
    X_val_simple = build_simple_features(df, val_idx)
    X_train_marta = build_marta_features(df, train_idx, shared["encoder"], device="cpu")[:, :REGIME_DIM]
    X_val_marta = build_marta_features(df, val_idx, shared["encoder"], device="cpu")[:, :REGIME_DIM]

    z_scaler = StandardScaler().fit(X_train_marta)
    Ztr, Zva = z_scaler.transform(X_train_marta), z_scaler.transform(X_val_marta)
    raw_scaler = StandardScaler().fit(X_train_simple)
    Xtr, Xva = raw_scaler.transform(X_train_simple), raw_scaler.transform(X_val_simple)

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=_SEED, early_stopping=True)
    mlp.fit(Ztr, Xtr)
    pred_val = mlp.predict(Zva)

    # trivial baseline: predict the train mean (0 in standardized space) for every val row
    baseline_pred = np.zeros_like(Xva)

    per_feature: dict[str, dict[str, float]] = {}
    for i, col in enumerate(SIMPLE_FEATURE_COLS):
        true_col = Xva[:, i]
        pred_col = pred_val[:, i]
        r, _ = pearsonr(true_col, pred_col)
        per_feature[col] = {
            "mlp_mse": float(mean_squared_error(true_col, pred_col)),
            "mlp_mae": float(mean_absolute_error(true_col, pred_col)),
            "mlp_r2": float(r2_score(true_col, pred_col)),
            "mlp_pearson_r_true_vs_reconstructed": float(r),
            "baseline_mse": float(mean_squared_error(true_col, baseline_pred[:, i])),
            "baseline_mae": float(mean_absolute_error(true_col, baseline_pred[:, i])),
            "baseline_r2": float(r2_score(true_col, baseline_pred[:, i])),
        }

    overall = {
        "mlp_mse": float(mean_squared_error(Xva, pred_val)),
        "mlp_mae": float(mean_absolute_error(Xva, pred_val)),
        "mlp_r2": float(r2_score(Xva, pred_val)),
        "baseline_mse": float(mean_squared_error(Xva, baseline_pred)),
        "baseline_mae": float(mean_absolute_error(Xva, baseline_pred)),
        "baseline_r2": float(r2_score(Xva, baseline_pred)),
    }

    table = Table(title="Z -> X_raw: full per-feature reconstruction diagnostic (val, OOS)")
    table.add_column("Feature")
    table.add_column("MLP R2", justify="right")
    table.add_column("MLP MAE", justify="right")
    table.add_column("MLP MSE", justify="right")
    table.add_column("True-vs-Recon r", justify="right")
    table.add_column("Baseline R2", justify="right")
    table.add_column("Baseline MAE", justify="right")
    for col in SIMPLE_FEATURE_COLS:
        r = per_feature[col]
        table.add_row(
            col, f"{r['mlp_r2']:.4f}", f"{r['mlp_mae']:.4f}", f"{r['mlp_mse']:.4f}",
            f"{r['mlp_pearson_r_true_vs_reconstructed']:.4f}", f"{r['baseline_r2']:.4f}", f"{r['baseline_mae']:.4f}",
        )
    _console.print(table)
    _console.print(f"Overall: {overall}")

    return {"per_feature": per_feature, "overall": overall}


def run_collapse_extra(shared: dict[str, Any]) -> dict[str, Any]:
    _console.rule("[bold]Item 5 extra — per-dim percentiles, condition number, PC1-5, 99% threshold")
    df = shared["df"]
    val_idx = _valid_indices(df.index, "val")
    Z = build_marta_features(df, val_idx, shared["encoder"], device="cpu")[:, :REGIME_DIM]

    per_dim_median = np.median(Z, axis=0)
    percentiles = {p: np.percentile(Z, p, axis=0) for p in [5, 25, 50, 75, 95]}

    cov = np.cov(Z, rowvar=False)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
    sorted_eig = np.sort(eigvals)[::-1]
    total_var = sorted_eig.sum()
    explained_ratio = sorted_eig / max(total_var, 1e-12)
    cum_explained = np.cumsum(explained_ratio)

    nonzero_eig = sorted_eig[sorted_eig > 1e-12]
    condition_number = float(nonzero_eig[0] / nonzero_eig[-1]) if len(nonzero_eig) > 1 else None

    rng = np.random.default_rng(_SEED)
    sample_idx = rng.choice(len(Z), size=min(1500, len(Z)), replace=False)
    Zs = Z[sample_idx]
    norms = np.maximum(np.linalg.norm(Zs, axis=1, keepdims=True), 1e-8)
    Zn = Zs / norms
    cos_sim = Zn @ Zn.T
    iu = np.triu_indices_from(cos_sim, k=1)
    cos_vals = cos_sim[iu]

    result = {
        "per_dim_median_summary": {
            "mean_of_medians": float(per_dim_median.mean()), "std_of_medians": float(per_dim_median.std()),
        },
        "per_dim_percentiles_summary": {
            f"p{p}_mean_across_dims": float(v.mean()) for p, v in percentiles.items()
        },
        "cosine_similarity_median": float(np.median(cos_vals)),
        "pca": {
            "pc1_explained_variance_ratio": float(explained_ratio[0]),
            "pc2_explained_variance_ratio": float(explained_ratio[1]),
            "pc3_explained_variance_ratio": float(explained_ratio[2]),
            "pc4_explained_variance_ratio": float(explained_ratio[3]),
            "pc5_explained_variance_ratio": float(explained_ratio[4]),
            "cumulative_pc1_5": float(cum_explained[4]),
            "n_dims_for_90pct": int(np.searchsorted(cum_explained, 0.90) + 1),
            "n_dims_for_95pct": int(np.searchsorted(cum_explained, 0.95) + 1),
            "n_dims_for_99pct": int(np.searchsorted(cum_explained, 0.99) + 1),
            "condition_number_nonzero_eigenvalues": condition_number,
            "n_nonzero_eigenvalues_gt_1e-12": int(len(nonzero_eig)),
        },
    }
    _console.print(result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Encoder diagnostic extra — items 4/5 detail")
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    torch.manual_seed(_SEED)
    np.random.seed(_SEED)
    encoder_path = Path(args.encoder_checkpoint) if args.encoder_checkpoint else None
    shared = load_shared_data(device="cpu", encoder_path=encoder_path)
    logger.info(f"Evaluating encoder checkpoint: {shared['encoder_path']}")

    reconstruction_extra = run_reconstruction_extra(shared)
    collapse_extra = run_collapse_extra(shared)

    out = {
        "encoder_checkpoint": shared["encoder_path"],
        "item4_extra_reconstruction": reconstruction_extra,
        "item5_extra_collapse": collapse_extra,
    }
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = _RESULTS_DIR / f"encoder_diagnostic_extra{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    logger.success(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
