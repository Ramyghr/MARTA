"""
regime/hmm.py
Gaussian HMM fitting pipeline for MARTA regime discovery.
PCA(16) on training embeddings → BIC-selected GaussianHMM → regime probability stream.

BIC search strategy (fast):
  - diag covariance, 100 iters, tol=1e-3, 20K-sample subsample of training data
Final refit (accurate):
  - diag covariance, 200 iters, tol=1e-4, all training data

Covariance type note (fixed 2026-07-20): the final refit previously used
covariance_type="full" (136 free params/state in 16-dim space) while the BIC search
above it, and _compute_bic's own parameter-count formula (n_states**2 + 2*n_states*
pca_dim), both assume diagonal covariance (16 params/state). That mismatch let one
state's full covariance run away unconstrained during EM — verified on the live model:
one state's covariance trace was 20-250x every other state's, making it an "absorbing"
catch-all that any live (non-full-history) HMM query collapsed into regardless of the
actual input, because the model was never fit with any pressure keeping per-state
covariances comparable. Diagonal covariance for both stages closes that gap and
keeps the BIC formula's complexity accounting consistent with what's actually deployed.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA

_ROOT = Path(__file__).resolve().parent.parent

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

from utils.logger import logger  # noqa: E402


def fit_pca(
    embeddings: np.ndarray,
    train_mask: np.ndarray,
    n_components: int = 16,
    random_state: int = 42,
) -> tuple[PCA, np.ndarray]:
    """Fit PCA on training embeddings only; transform the full matrix. Asserts >75% variance."""
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(embeddings[train_mask].astype(np.float64))

    explained_var = float(pca.explained_variance_ratio_.sum())
    logger.info(f"PCA({n_components}) retained variance: {explained_var:.4f}")
    mlflow.log_metric("pca_explained_variance_ratio", explained_var)
    mlflow.log_param("pca_n_components", n_components)

    assert explained_var > 0.75, (
        f"PCA retained only {explained_var:.1%} variance — below 75% threshold. "
        "Increase n_components."
    )

    emb_reduced = pca.transform(embeddings.astype(np.float64))
    return pca, emb_reduced


def _compute_bic(
    log_likelihood: float,
    n_samples: int,
    n_states: int,
    pca_dim: int,
) -> float:
    # simplified: transition matrix + means + diagonal-covariance approximation
    n_params = n_states ** 2 + 2 * n_states * pca_dim
    return -2.0 * log_likelihood + n_params * np.log(n_samples)


def _compute_aic(
    log_likelihood: float,
    n_states: int,
    pca_dim: int,
) -> float:
    n_params = n_states ** 2 + 2 * n_states * pca_dim
    return -2.0 * log_likelihood + 2.0 * n_params


_BIC_SUBSAMPLE = 20_000  # rows used for BIC search; full data used for final refit


def fit_hmm_bic(
    emb_reduced: np.ndarray,
    train_mask: np.ndarray,
    n_states_range: tuple[int, int] = (2, 9),
) -> tuple[GaussianHMM, int, dict[int, dict[str, float]]]:
    """
    Grid-search GaussianHMM over n_states; select by BIC.

    Speed strategy:
      - diag covariance, 100 iters, tol=1e-3
      - subsample training data to _BIC_SUBSAMPLE rows
    Both BIC fit and BIC scoring use the subsample so the LL scale is consistent.
    run_hmm_pipeline refits the winner on the full dataset with full covariance.
    """
    train_emb_full = emb_reduced[train_mask]
    n_train_full = len(train_emb_full)
    pca_dim = emb_reduced.shape[1]

    # subsample for fast BIC search
    rng = np.random.default_rng(42)
    sub_size = min(_BIC_SUBSAMPLE, n_train_full)
    sub_idx = rng.choice(n_train_full, size=sub_size, replace=False)
    sub_idx.sort()
    train_emb = train_emb_full[sub_idx]
    n_train = len(train_emb)

    logger.info(
        f"BIC search: n_states {n_states_range[0]}–{n_states_range[1]-1}  "
        f"subsample={n_train:,}/{n_train_full:,}  pca_dim={pca_dim}  "
        f"cov=diag  n_iter=100"
    )

    results: dict[int, dict[str, float]] = {}
    models: dict[int, GaussianHMM] = {}

    for n in range(n_states_range[0], n_states_range[1]):
        try:
            model = GaussianHMM(
                n_components=n,
                covariance_type="diag",
                n_iter=100,
                random_state=42,
                tol=1e-3,
            )
            model.fit(train_emb)

            if not model.monitor_.converged:
                logger.warning(
                    f"  n_states={n}: did not converge ({model.monitor_.iter} iters)"
                )

            ll = model.score(train_emb)
            bic = _compute_bic(ll, n_train, n, pca_dim)
            aic = _compute_aic(ll, n, pca_dim)

            results[n] = {"log_likelihood": ll, "bic": bic, "aic": aic}
            models[n] = model

            logger.info(
                f"  n={n}: LL={ll:,.1f}  BIC={bic:,.1f}  AIC={aic:,.1f}"
            )
            mlflow.log_metric(f"hmm_ll_n{n}", ll)
            mlflow.log_metric(f"hmm_bic_n{n}", bic)
            mlflow.log_metric(f"hmm_aic_n{n}", aic)

        except Exception as exc:
            logger.warning(f"  n_states={n} failed: {exc}")
            results[n] = {
                "log_likelihood": float("-inf"),
                "bic": float("inf"),
                "aic": float("inf"),
            }

    valid = {n: r for n, r in results.items() if r["bic"] < float("inf")}
    if not valid:
        raise RuntimeError("All HMM fits failed — cannot select best n_states.")

    sorted_by_bic = sorted(valid, key=lambda k: valid[k]["bic"])
    best_n = sorted_by_bic[0]
    best_bic = valid[best_n]["bic"]

    if len(sorted_by_bic) > 1:
        second_n = sorted_by_bic[1]
        second_bic = valid[second_n]["bic"]
        if abs(best_bic - second_bic) < 0.05 * abs(best_bic):
            simpler = min(best_n, second_n)
            logger.warning(
                f"BIC curve flat (Δ={abs(best_bic - second_bic):.2f} < 5% of "
                f"|{best_bic:.2f}|). Picking simpler model: n_states={simpler}"
            )
            best_n = simpler

    logger.success(
        f"Selected n_states={best_n}  BIC={results[best_n]['bic']:,.2f}"
    )
    mlflow.log_param("best_n_states", best_n)
    mlflow.log_metric("best_bic", results[best_n]["bic"])

    return models[best_n], best_n, results


def generate_regime_stream(
    embeddings: np.ndarray,
    timestamps: pd.DataFrame,
    asset_labels: Optional[np.ndarray],
    hmm_model: GaussianHMM,
    pca: PCA,
) -> pd.DataFrame:
    """Apply PCA + HMM to full embeddings; return labelled DataFrame."""
    emb_reduced = pca.transform(embeddings.astype(np.float64))

    regime_labels = hmm_model.predict(emb_reduced)
    regime_probs = hmm_model.predict_proba(emb_reduced)
    n_states = hmm_model.n_components

    df = timestamps[["timestamp", "asset"]].copy().reset_index(drop=True)
    df["regime_label"] = regime_labels

    for i in range(n_states):
        df[f"regime_prob_{i}"] = regime_probs[:, i]

    dist = pd.Series(regime_labels).value_counts(normalize=True).sort_index()
    logger.info(f"Regime distribution (full set):\n{dist.to_string()}")
    for state_idx, pct in dist.items():
        mlflow.log_metric(f"regime_{state_idx}_pct_total", float(pct))

    mlflow.log_metric("regime_stream_n_rows", len(df))
    mlflow.log_param("regime_stream_n_states", n_states)

    return df


def run_hmm_pipeline(config: dict) -> dict[str, Any]:
    """
    Orchestrate the full Week 3 HMM pipeline.

    Returns dict with keys:
        regime_df   : pd.DataFrame  — full regime probability stream
        bic_results : dict[int, dict[str, float]]
        best_n      : int
        hmm_model   : GaussianHMM   — fitted on full dataset
        pca         : PCA
    """
    processed_dir = _ROOT / config["data"]["processed_dir"]

    embeddings = np.load(processed_dir / "embeddings.npy")
    timestamps = pd.read_parquet(
        processed_dir / "embedding_timestamps.parquet"
    )

    logger.info(
        f"Embeddings: {embeddings.shape}  dtype={embeddings.dtype}"
    )
    logger.info(
        f"Timestamps: {timestamps.shape}  "
        f"[{timestamps['timestamp'].min()} → {timestamps['timestamp'].max()}]"
    )

    # The 14-feature embed.py stores asset identity in a separate asset-map
    # parquet (start_idx/end_idx/asset_name) rather than an `asset` column on
    # the timestamps frame. Reconstruct the column, preserving the downstream
    # naming convention (GOLD — used by the old regime stream, rl/env.py
    # CANONICAL_ASSETS, and the agents) by aliasing the histdata instrument names.
    if "asset" not in timestamps.columns:
        asset_map = pd.read_parquet(
            processed_dir / "embedding_asset_map.parquet"
        )
        alias = {"XAUUSD": "GOLD"}
        asset_col = np.empty(len(timestamps), dtype=object)
        for _, row in asset_map.iterrows():
            name = alias.get(row["asset_name"], row["asset_name"])
            asset_col[int(row["start_idx"]):int(row["end_idx"])] = name
        if (asset_col == None).any():  # noqa: E711 — object-array None check
            raise ValueError("asset_map does not cover all embedding rows")
        timestamps = timestamps.copy()
        timestamps["asset"] = asset_col

    # training mask: rows before config["training"]["train_end"] (2023-01-01 →
    # PCA/HMM fit on 2018-2022 only; val 2023 and test 2024 stay out of the fit)
    train_cutoff = pd.Timestamp(config["training"]["train_end"], tz="UTC")
    ts_series = pd.to_datetime(timestamps["timestamp"], utc=True)
    train_mask = (ts_series < train_cutoff).values

    n_train = int(train_mask.sum())
    n_total = len(train_mask)
    logger.info(f"Train samples: {n_train:,} / {n_total:,} ({n_train/n_total:.1%})")
    mlflow.log_metric("n_train_samples", n_train)
    mlflow.log_metric("n_total_samples", n_total)

    # Step 1 — PCA (16 dims; fit on training rows only to avoid leakage)
    pca, emb_reduced = fit_pca(embeddings, train_mask, n_components=16)

    # Step 2 — BIC selection (training data only)
    _train_model, best_n, bic_results = fit_hmm_bic(emb_reduced, train_mask)

    # Step 3 — Refit winner on TRAINING DATA ONLY (diag covariance, more iterations —
    # see module docstring for why this is diag, not full)
    logger.info(f"Refitting best HMM (n_states={best_n}) on training data only …")
    final_hmm = GaussianHMM(
        n_components=best_n,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
        tol=1e-4,
    )
    final_hmm.fit(emb_reduced[train_mask])          # <-- was emb_reduced (full dataset)
    if not final_hmm.monitor_.converged:
        logger.warning(
            f"Final HMM (n_states={best_n}) did not converge "
            f"({final_hmm.monitor_.iter} iters)"
        )

    # save checkpoints
    ckpt_dir = _ROOT / "regime" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_hmm, ckpt_dir / "hmm_model.pkl")
    joblib.dump(pca, ckpt_dir / "pca_model.pkl")
    logger.success(f"Saved checkpoints → {ckpt_dir}")

    # Step 4 — regime probability stream
    regime_df = generate_regime_stream(
        embeddings, timestamps, None, final_hmm, pca
    )

    out_path = processed_dir / "regime_stream.parquet"
    regime_df.to_parquet(out_path, engine="pyarrow", index=False)
    logger.success(f"Saved regime stream → {out_path}  ({len(regime_df):,} rows)")

    # persist BIC results as JSON artifact
    bic_json_path = ckpt_dir / "bic_results.json"
    with open(bic_json_path, "w") as fh:
        json.dump({str(k): v for k, v in bic_results.items()}, fh, indent=2)

    for path in [ckpt_dir / "hmm_model.pkl", ckpt_dir / "pca_model.pkl",
                 out_path, bic_json_path]:
        mlflow.log_artifact(str(path))

    return {
        "regime_df": regime_df,
        "bic_results": bic_results,
        "best_n": best_n,
        "hmm_model": final_hmm,
        "pca": pca,
    }
