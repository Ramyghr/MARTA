"""Supervised predictability diagnostic — NOT an RL test. Checks whether the 3A (raw
14-feature) and 3B (156-dim MARTA) observations have ANY static relationship with
sign(next_return), independent of PPO's ability to exploit it.

    observation at t  ->  Logistic Regression / tiny MLP  ->  predict sign(next_return)

Chronological split matching rl/env.py exactly (TRAIN_END/VAL_START/VAL_END) — train on
< 2024, evaluate out-of-sample on 2024 only. For the MARTA representation, the
portfolio block (self-referential: own position/P&L/drawdown) is ZEROED — there is no
"current position" in a static, non-sequential prediction task, so those dims would be
constant/meaningless noise here; only the market-derived parts (128-dim encoder embedding
+ 16-dim consensus, itself always zero in TEST 3 + 4-dim constant asset-id) are used.

If both come back ~50% accuracy / ~0.50 ROC-AUC, that's strong evidence the representation
(not just PPO) lacks one-bar directional information. If MARTA is materially above 50% but
PPO still failed to learn/collapsed to hold, the bottleneck is more likely in the RL
formulation (reward scale, GAE/value-function issues — see rl/audit_test3.py item 5) than
in the representation itself.

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.supervised_predictability
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from rl.env import ASSET_ID_DIM, CANONICAL_ASSETS, CONSENSUS_DIM, TRAIN_END, VAL_END, VAL_START
from rl.minimal_btc_env import REGIME_DIM, SIMPLE_FEATURE_COLS, load_shared_data
from utils.logger import logger

_console = Console()
_RESULTS_DIR = Path("data/results")
_SEED = 42
_WINDOW = 60
_ENCODE_BATCH = 1024


def _valid_indices(index: pd.DatetimeIndex, split: str) -> np.ndarray:
    if split == "train":
        mask = np.asarray(index <= TRAIN_END)
    else:
        mask = np.asarray((index >= VAL_START) & (index <= VAL_END))
    n = len(index)
    valid = np.zeros(n, dtype=bool)
    for t in range(_WINDOW, n - 1):
        if mask[t] and mask[t + 1]:
            valid[t] = True
    return np.flatnonzero(valid)


def build_labels(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """y[i] = sign(close_return[t+1]) for t = idx[i] — the exact next_return the
    MinimalBTCEnv reward function realizes for a decision made using data through t."""
    ret = df["close_return"].to_numpy(dtype=np.float64)
    y = np.sign(ret[idx + 1])
    return (y > 0).astype(np.int64)  # binary: 1 = up, 0 = down/flat


def build_simple_features(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    arr = df[SIMPLE_FEATURE_COLS].to_numpy(dtype=np.float32)
    return arr[idx]


def build_marta_features(df: pd.DataFrame, idx: np.ndarray, encoder, device: str) -> np.ndarray:
    """128-dim encoder embedding + 16 zeros (consensus) + 4 constant asset-id one-hot.
    Portfolio block (8 dims) is DELIBERATELY OMITTED — see module docstring."""
    arr = df.to_numpy(dtype=np.float32)
    n = len(idx)
    embeddings = np.zeros((n, REGIME_DIM), dtype=np.float32)
    encoder = encoder.to(device).eval()
    with torch.no_grad():
        for start in range(0, n, _ENCODE_BATCH):
            batch_idx = idx[start : start + _ENCODE_BATCH]
            windows = np.stack([arr[t - _WINDOW + 1 : t + 1] for t in batch_idx], axis=0)
            tensor = torch.from_numpy(windows).to(device)
            emb = encoder.encode(tensor).cpu().numpy().astype(np.float32)
            embeddings[start : start + len(batch_idx)] = emb

    consensus = np.zeros((n, CONSENSUS_DIM), dtype=np.float32)
    asset_id = np.zeros((n, ASSET_ID_DIM), dtype=np.float32)
    asset_id[:, CANONICAL_ASSETS.index("BTC_USDT")] = 1.0
    return np.concatenate([embeddings, consensus, asset_id], axis=1)


def evaluate_model(model, X_train, y_train, X_val, y_val, name: str) -> dict[str, Any]:
    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    proba_val = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    acc = float(accuracy_score(y_val, pred_val))
    bal_acc = float(balanced_accuracy_score(y_val, pred_val))
    auc = float(roc_auc_score(y_val, proba_val)) if proba_val is not None and len(np.unique(y_val)) > 1 else None
    cm = confusion_matrix(y_val, pred_val).tolist()
    train_acc = float(accuracy_score(y_train, model.predict(X_train)))

    baseline_acc = float(max(np.mean(y_val == 1), np.mean(y_val == 0)))

    result = {
        "model": name,
        "train_accuracy": train_acc,
        "val_accuracy": acc,
        "val_balanced_accuracy": bal_acc,
        "val_roc_auc": auc,
        "baseline_majority_class_accuracy": baseline_acc,
        "confusion_matrix_val": cm,  # [[TN, FP], [FN, TP]]
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }
    logger.info(f"[{name}] {result}")
    return result


def main() -> None:
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)

    _console.rule("[bold]Supervised predictability diagnostic (NOT an RL test)")
    shared = load_shared_data(device="cpu")
    df = shared["df"]
    encoder = shared["encoder"]

    train_idx = _valid_indices(df.index, "train")
    val_idx = _valid_indices(df.index, "val")
    logger.info(f"train n={len(train_idx)} (<{TRAIN_END}), val n={len(val_idx)} ({VAL_START}..{VAL_END})")

    y_train = build_labels(df, train_idx)
    y_val = build_labels(df, val_idx)
    _console.print(
        f"Label balance — train: {y_train.mean():.4f} up, val: {y_val.mean():.4f} up "
        "(sanity check: should be close to 50/50, matching item 4's next-return positive_pct)"
    )

    results: dict[str, list[dict[str, Any]]] = {}

    # ---------------------------------------------------------------- 3A: simple features
    _console.rule("[bold]3A features (14-dim raw indicators)")
    X_train_simple = build_simple_features(df, train_idx)
    X_val_simple = build_simple_features(df, val_idx)
    scaler = StandardScaler().fit(X_train_simple)
    X_train_s = scaler.transform(X_train_simple)
    X_val_s = scaler.transform(X_val_simple)

    results["3A_simple"] = [
        evaluate_model(
            LogisticRegression(max_iter=1000, random_state=_SEED), X_train_s, y_train, X_val_s, y_val,
            "LogisticRegression",
        ),
        evaluate_model(
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=_SEED, early_stopping=True),
            X_train_s, y_train, X_val_s, y_val, "MLP(32)",
        ),
    ]

    # ---------------------------------------------------------------- 3B: MARTA representation
    _console.rule("[bold]3B MARTA representation (128-dim encoder embedding + zeros; portfolio omitted)")
    logger.info("Batch-encoding train windows through frozen encoder (this takes a while)...")
    X_train_marta = build_marta_features(df, train_idx, encoder, device="cpu")
    logger.info("Batch-encoding val windows through frozen encoder...")
    X_val_marta = build_marta_features(df, val_idx, encoder, device="cpu")

    scaler_m = StandardScaler().fit(X_train_marta)
    X_train_m = scaler_m.transform(X_train_marta)
    X_val_m = scaler_m.transform(X_val_marta)

    results["3B_marta"] = [
        evaluate_model(
            LogisticRegression(max_iter=1000, random_state=_SEED), X_train_m, y_train, X_val_m, y_val,
            "LogisticRegression",
        ),
        evaluate_model(
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=_SEED, early_stopping=True),
            X_train_m, y_train, X_val_m, y_val, "MLP(32)",
        ),
    ]

    # ---------------------------------------------------------------- report
    table = Table(title="Supervised predictability of sign(next_return) — OUT-OF-SAMPLE (2024 val)")
    table.add_column("Observation")
    table.add_column("Model")
    table.add_column("Val Acc.", justify="right")
    table.add_column("Bal. Acc.", justify="right")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("Baseline Acc.", justify="right")
    table.add_column("Train Acc.", justify="right")
    for obs_name, entries in results.items():
        for r in entries:
            table.add_row(
                obs_name, r["model"], f"{r['val_accuracy']:.4f}", f"{r['val_balanced_accuracy']:.4f}",
                f"{r['val_roc_auc']:.4f}" if r["val_roc_auc"] is not None else "n/a",
                f"{r['baseline_majority_class_accuracy']:.4f}", f"{r['train_accuracy']:.4f}",
            )
    _console.print(table)

    for obs_name, entries in results.items():
        for r in entries:
            _console.print(f"[{obs_name}/{r['model']}] confusion matrix (val) [[TN,FP],[FN,TP]]: {r['confusion_matrix_val']}")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / "test3_supervised_predictability.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    logger.success(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
