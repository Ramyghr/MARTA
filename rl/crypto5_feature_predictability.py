"""Per-feature predictability ceiling check for the crypto5 panel (2026-08-24).

Answers a narrower, more diagnostic question than rl/supervised_predictability.py's
aggregate 14-dim-vs-128-dim test: does ANY single feature in the crypto5_multihorizon
22-column panel, using its own last 20 lags, carry predictive signal for next-bar
direction? If nothing clears a real margin over 50%, the crypto5 RL results this session
(buy-hold-equivalent at best, significantly worse at worst) are a structural ceiling, not
an architecture/reward-shaping problem — matching the same "raw features beat the
contrastive encoder, but the edge itself is small" finding already established for the
old 4-asset panel (see project memory / docs/RESULTS_AND_METHODOLOGY.md), now checked
directly on this panel instead of assumed to carry over.

Per feature f:
    X = [f[t], f[t-1], ..., f[t-19]]  (20 lags, that feature only)
    y_reg = close_return[t+1]
    y_dir = 1 if close_return[t+1] > 0 else 0

Ridge regression for R² (continuous), LogisticRegression for directional accuracy —
both fit on train (< 2024-01-01), evaluated out-of-sample on val (2024). The "trading
Sharpe" column simulates going long when the val-set logistic model predicts up, short
when it predicts down, using the model's own out-of-sample prediction (not the true
label) — this is what the feature could actually deliver as a trading signal, not a
theoretical ceiling.

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.crypto5_feature_predictability
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from configs.splits import TRAIN_END_EXCLUSIVE, VAL_END_EXCLUSIVE, VAL_START
from utils.logger import logger

_console = Console()
_DATA_PATH = Path("data/processed/crypto5_multihorizon/BTC_USDT_aligned.parquet")
_RESULTS_PATH = Path("data/results/crypto5_feature_predictability.json")
_N_LAGS = 20
_SEED = 42


def _build_lag_matrix(series: np.ndarray, n_lags: int) -> np.ndarray:
    """Row t (for t >= n_lags-1) = [series[t], series[t-1], ..., series[t-n_lags+1]]."""
    n = len(series)
    mat = np.full((n, n_lags), np.nan, dtype=np.float64)
    for lag in range(n_lags):
        mat[lag:, lag] = series[: n - lag]
    return mat


def _annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or returns.std(ddof=1) < 1e-12:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(24 * 365))


def main() -> None:
    np.random.seed(_SEED)
    _console.rule("[bold]Crypto5 per-feature predictability ceiling check (BTC_USDT)")

    df = pd.read_parquet(_DATA_PATH, engine="pyarrow")
    logger.info(f"Loaded {_DATA_PATH}: {df.shape}")

    ts = df.index
    train_mask = ts < pd.Timestamp(TRAIN_END_EXCLUSIVE, tz="UTC")
    val_mask = (ts >= pd.Timestamp(VAL_START, tz="UTC")) & (ts < pd.Timestamp(VAL_END_EXCLUSIVE, tz="UTC"))

    next_return = df["close_return"].to_numpy(dtype=np.float64)
    next_return = np.concatenate([next_return[1:], [np.nan]])  # y[t] = close_return[t+1]
    y_dir_all = (next_return > 0).astype(np.int64)

    features = list(df.columns)
    results: list[dict] = []

    for feat in features:
        series = df[feat].to_numpy(dtype=np.float64)
        X = _build_lag_matrix(series, _N_LAGS)

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(next_return))
        train_idx = np.flatnonzero(np.asarray(train_mask) & valid)
        val_idx = np.flatnonzero(np.asarray(val_mask) & valid)
        if len(train_idx) < 100 or len(val_idx) < 100:
            logger.warning(f"[{feat}] too few valid rows (train={len(train_idx)}, val={len(val_idx)}) — skipping")
            continue

        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_val = scaler.transform(X[train_idx]), scaler.transform(X[val_idx])
        y_reg_train, y_reg_val = next_return[train_idx], next_return[val_idx]
        y_dir_train, y_dir_val = y_dir_all[train_idx], y_dir_all[val_idx]

        ridge = Ridge(alpha=1.0, random_state=_SEED).fit(X_train, y_reg_train)
        r2 = float(r2_score(y_reg_val, ridge.predict(X_val)))

        logit = LogisticRegression(max_iter=1000, random_state=_SEED).fit(X_train, y_dir_train)
        pred_dir = logit.predict(X_val)
        dir_acc = float((pred_dir == y_dir_val).mean())

        # Trading Sharpe from the model's own out-of-sample predicted direction —
        # long (+1) when predicted up, short (-1) when predicted down.
        strategy_returns = np.where(pred_dir == 1, 1.0, -1.0) * y_reg_val
        sharpe = _annualized_sharpe(strategy_returns)

        results.append({
            "feature": feat,
            "r2_val": r2,
            "directional_accuracy_val": dir_acc,
            "trading_sharpe_val": sharpe,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
        })
        logger.info(f"[{feat}] R2={r2:.5f} dir_acc={dir_acc:.4f} sharpe={sharpe:.3f}")

    results.sort(key=lambda r: r["directional_accuracy_val"], reverse=True)

    table = Table(title="Per-feature predictability of next-bar direction — crypto5 BTC_USDT (2024 val, OOS)")
    table.add_column("Feature")
    table.add_column("R2 (val)", justify="right")
    table.add_column("Dir. accuracy", justify="right")
    table.add_column("Trading Sharpe", justify="right")
    for r in results:
        table.add_row(r["feature"], f"{r['r2_val']:.5f}", f"{r['directional_accuracy_val']:.4f}", f"{r['trading_sharpe_val']:.3f}")
    _console.print(table)

    best = results[0] if results else None
    n_above_505 = sum(1 for r in results if r["directional_accuracy_val"] > 0.505)
    n_above_52 = sum(1 for r in results if r["directional_accuracy_val"] > 0.52)
    _console.print(f"\nFeatures with directional accuracy > 50.5%: {n_above_505}/{len(results)}")
    _console.print(f"Features with directional accuracy > 52.0%: {n_above_52}/{len(results)}")
    if best:
        _console.print(f"Best single feature: {best['feature']} ({best['directional_accuracy_val']:.4f})")

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float))
    logger.success(f"Results -> {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
