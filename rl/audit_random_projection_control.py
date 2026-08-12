"""Random-projection sanity control — item 8 of the 2026-08-11 E2 pre-flight audit.

Not a useful model on its own. Purpose: is the trained encoder actually better than a
fixed, untrained, information-preserving-ish linear transform of the same raw window, or
is it destroying information relative to even this trivial baseline? Answers "is training
helping at all" independent of "is training helping in the way we want."

Fixed (seeded) random Gaussian projection: flatten each 60x14=840-dim raw window, project
to 128 dims via a fixed W ~ N(0, 1/840) (variance-scaled so the projection is
approximately norm-preserving, standard Johnson-Lindenstrauss-style construction). No
training involved — W is drawn once and reused for every window.

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.audit_random_projection_control
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from rl.audit_encoder import _HORIZONS, build_horizon_labels, multi_horizon_report
from rl.minimal_btc_env import REGIME_DIM, load_shared_data
from rl.supervised_predictability import _valid_indices, build_marta_features, build_simple_features
from utils.logger import logger

_console = Console()
_RESULTS_DIR = Path("data/results")
_SEED = 42
_WINDOW = 60


def build_random_projection_features(df, idx: np.ndarray, window_dim: int = 14, window_size: int = _WINDOW) -> np.ndarray:
    """Flatten each window and project to REGIME_DIM via a FIXED seeded random matrix."""
    arr = df.to_numpy(dtype=np.float64)
    flat_dim = window_size * window_dim
    rng = np.random.default_rng(_SEED)
    W = rng.normal(0.0, 1.0 / np.sqrt(flat_dim), size=(flat_dim, REGIME_DIM))

    X = np.zeros((len(idx), flat_dim), dtype=np.float64)
    for i, t in enumerate(idx):
        window = arr[t - window_size + 1 : t + 1]  # (60, 14)
        X[i] = window.flatten()
    return (X @ W).astype(np.float32)


def main() -> None:
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)

    _console.rule("[bold]Item 8 — random-projection sanity control (no training)")
    shared = load_shared_data(device="cpu")
    df = shared["df"]
    train_idx = _valid_indices(df.index, "train")
    val_idx = _valid_indices(df.index, "val")

    logger.info("Building raw, random-projection, and current-encoder features...")
    X_train_simple = build_simple_features(df, train_idx)
    X_val_simple = build_simple_features(df, val_idx)
    X_train_rand = build_random_projection_features(df, train_idx)
    X_val_rand = build_random_projection_features(df, val_idx)
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
            X_train_rand[tr_mask], X_val_rand[va_mask], y_train[tr_mask], y_val[va_mask], "random_projection_128", h,
        ))
        results.append(multi_horizon_report(
            X_train_marta[tr_mask], X_val_marta[va_mask], y_train[tr_mask], y_val[va_mask], "current_encoder_128", h,
        ))

    table = Table(title="Predictability control: raw vs. random projection (untrained) vs. current encoder")
    table.add_column("Horizon")
    table.add_column("Representation")
    table.add_column("AUC", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Pearson r", justify="right")
    table.add_column("Spearman r", justify="right")
    for r in results:
        table.add_row(
            f"{r['horizon_hours']}h", r["observation"],
            f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "n/a",
            f"{r['baseline_accuracy']:.4f}", f"{r['pearson_r_predicted_vs_actual_return']:.4f}",
            f"{r['spearman_r_predicted_vs_actual_return']:.4f}",
        )
    _console.print(table)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / "encoder_random_projection_control.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    logger.success(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
