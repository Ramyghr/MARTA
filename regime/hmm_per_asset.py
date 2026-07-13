"""
regime/hmm_per_asset.py
Per-asset Gaussian HMM pipeline for MARTA regime discovery.

Each asset gets its own PCA(16) + BIC-selected GaussianHMM fitted independently,
avoiding the shared-embedding-space bias where GOLD's crisis signature anchors the
shared HMM and causes BTC crash periods to be misclassified.

BIC search:  diag covariance, 100 iters, tol=1e-3, ≤20K-row subsample, n_states 2–7
Final refit: full covariance, 200 iters, tol=1e-4, full asset data

Saves per-asset models to regime/checkpoints/hmm_{asset}.pkl and pca_{asset}.pkl.
Shared HMM checkpoints (hmm_model.pkl, pca_model.pkl) are NOT touched.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

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

_BIC_SUBSAMPLE = 20_000
_TRAIN_CUTOFF = pd.Timestamp("2021-07-01", tz="UTC")
_PCA_N_COMPONENTS = 16


def _build_asset_map_from_timestamps(timestamp_df: pd.DataFrame) -> pd.DataFrame:
    """Derive contiguous asset index ranges from the grouped timestamp DataFrame."""
    records: list[dict[str, Any]] = []
    arr = timestamp_df["asset"].values
    for asset in timestamp_df["asset"].unique():
        indices = np.where(arr == asset)[0]
        records.append(
            {
                "asset_name": asset,
                "start_idx": int(indices[0]),
                "end_idx": int(indices[-1]),
            }
        )
    return pd.DataFrame(records)


def _compute_bic(
    log_likelihood: float,
    n_samples: int,
    n_states: int,
    pca_dim: int,
) -> float:
    n_params = n_states ** 2 + 2 * n_states * pca_dim
    return -2.0 * log_likelihood + n_params * np.log(n_samples)


def _compute_aic(
    log_likelihood: float,
    n_states: int,
    pca_dim: int,
) -> float:
    n_params = n_states ** 2 + 2 * n_states * pca_dim
    return -2.0 * log_likelihood + 2.0 * n_params


def _fit_asset_pca(
    asset_emb: np.ndarray,
    asset_ts: pd.DataFrame,
    train_cutoff: pd.Timestamp,          # NEW — passed in instead of module constant
    n_components: int = _PCA_N_COMPONENTS,
    random_state: int = 42,
) -> tuple[PCA, np.ndarray]:
    """Fit PCA on this asset's training rows only; transform full asset embeddings."""
    ts_utc = pd.to_datetime(asset_ts["timestamp"], utc=True)
    train_mask = (ts_utc < train_cutoff).values

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(asset_emb[train_mask].astype(np.float64))

    explained_var = float(pca.explained_variance_ratio_.sum())
    if explained_var <= 0.75:
        logger.warning(
            f"  PCA({n_components}) retained only {explained_var:.1%} variance "
            "(threshold 75%) — proceeding for per-asset fit"
        )
    else:
        logger.info(f"  PCA({n_components}) explained variance: {explained_var:.4f}")

    emb_reduced = pca.transform(asset_emb.astype(np.float64))
    return pca, emb_reduced


def _bic_search(
    emb_reduced: np.ndarray,
    train_mask: np.ndarray,
    asset_name: str,
    n_states_range: tuple[int, int] = (2, 8),
) -> tuple[GaussianHMM, int, dict[int, dict[str, float]]]:
    """BIC grid-search over n_states for one asset's PCA-reduced embeddings."""
    train_emb_full = emb_reduced[train_mask]
    n_train_full = len(train_emb_full)
    pca_dim = emb_reduced.shape[1]

    rng = np.random.default_rng(42)
    sub_size = min(_BIC_SUBSAMPLE, n_train_full)
    sub_idx = rng.choice(n_train_full, size=sub_size, replace=False)
    sub_idx.sort()
    train_emb = train_emb_full[sub_idx]
    n_train = len(train_emb)

    logger.info(
        f"  [{asset_name}] BIC search n_states {n_states_range[0]}–{n_states_range[1]-1}  "
        f"subsample={n_train:,}/{n_train_full:,}  pca_dim={pca_dim}  cov=diag  n_iter=100"
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
                    f"  [{asset_name}] n={n}: did not converge ({model.monitor_.iter} iters)"
                )

            ll = model.score(train_emb)
            bic = _compute_bic(ll, n_train, n, pca_dim)
            aic = _compute_aic(ll, n, pca_dim)

            results[n] = {"log_likelihood": ll, "bic": bic, "aic": aic}
            models[n] = model

            logger.info(f"  [{asset_name}] n={n}: LL={ll:,.1f}  BIC={bic:,.1f}  AIC={aic:,.1f}")
            mlflow.log_metric(f"{asset_name}_bic_n{n}", bic)
            mlflow.log_metric(f"{asset_name}_ll_n{n}", ll)
            mlflow.log_metric(f"{asset_name}_aic_n{n}", aic)

        except Exception as exc:
            logger.warning(f"  [{asset_name}] n={n} failed: {exc}")
            results[n] = {
                "log_likelihood": float("-inf"),
                "bic": float("inf"),
                "aic": float("inf"),
            }

    valid = {n: r for n, r in results.items() if r["bic"] < float("inf")}
    if not valid:
        raise RuntimeError(f"All HMM fits failed for {asset_name}.")

    sorted_by_bic = sorted(valid, key=lambda k: valid[k]["bic"])
    best_n = sorted_by_bic[0]
    best_bic = valid[best_n]["bic"]

    if len(sorted_by_bic) > 1:
        second_n = sorted_by_bic[1]
        second_bic = valid[second_n]["bic"]
        if abs(best_bic - second_bic) < 0.05 * abs(best_bic):
            simpler = min(best_n, second_n)
            logger.warning(
                f"  [{asset_name}] BIC curve flat (Δ={abs(best_bic - second_bic):.2f} < 5%). "
                f"Picking simpler model: n_states={simpler}"
            )
            best_n = simpler

    logger.success(
        f"  [{asset_name}] Selected n_states={best_n}  BIC={results[best_n]['bic']:,.2f}"
    )
    mlflow.log_param(f"{asset_name}_best_n_states", best_n)
    mlflow.log_metric(f"{asset_name}_best_bic", results[best_n]["bic"])

    return models[best_n], best_n, results


def fit_per_asset_hmms(
    embeddings: np.ndarray,
    timestamp_df: pd.DataFrame,
    asset_map_df: pd.DataFrame | None,
    config: dict,
) -> dict[str, tuple[GaussianHMM, PCA]]:
    """
    Fit independent PCA(16) + GaussianHMM for each asset.

    For each asset:
      - Extracts the asset's embedding slice using asset_map_df
      - Fits PCA(16) on that asset's training rows only (timestamps before 2021-07-01)
      - Runs BIC model selection for n_states in 2–7
      - Refits winner on full asset data with full covariance
      - Saves models to regime/checkpoints/hmm_{asset}.pkl and pca_{asset}.pkl

    Returns:
        dict mapping asset_name → (fitted_hmm, fitted_pca)
    """
    train_cutoff = pd.Timestamp(config["training"]["train_end"], tz="UTC")
    if asset_map_df is None or len(asset_map_df) == 0:
        asset_map_df = _build_asset_map_from_timestamps(timestamp_df)

    ckpt_dir = _ROOT / "regime" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    per_asset_models: dict[str, tuple[GaussianHMM, PCA]] = {}

    for _, map_row in asset_map_df.iterrows():
        asset_name: str = str(map_row["asset_name"])
        start_idx: int = int(map_row["start_idx"])
        end_idx: int = int(map_row["end_idx"])

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Per-asset HMM — {asset_name}  "
            f"(embedding rows {start_idx}–{end_idx}, {end_idx - start_idx + 1} samples)"
        )

        asset_emb = embeddings[start_idx : end_idx + 1]
        asset_ts = timestamp_df.iloc[start_idx : end_idx + 1].reset_index(drop=True)

        pca_model, emb_reduced = _fit_asset_pca(asset_emb, asset_ts, train_cutoff) 
        mlflow.log_metric(
            f"{asset_name}_pca_explained_variance",
            float(pca_model.explained_variance_ratio_.sum()),
        )

        ts_utc = pd.to_datetime(asset_ts["timestamp"], utc=True)
        train_mask = (ts_utc < train_cutoff).values
        n_train = int(train_mask.sum())
        logger.info(f"  [{asset_name}] Train samples: {n_train:,} / {len(asset_ts):,}")
        mlflow.log_metric(f"{asset_name}_n_train", n_train)

        _bic_winner, best_n, _bic_results = _bic_search(
            emb_reduced, train_mask, asset_name, n_states_range=(2, 8)
        )

        logger.info(
            f"  [{asset_name}] Refitting n_states={best_n} on full asset data "
            "(full cov, 200 iters) …"
        )
        final_hmm = GaussianHMM(
            n_components=best_n,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        final_hmm.fit(emb_reduced[train_mask]) 
        if not final_hmm.monitor_.converged:
            logger.warning(
                f"  [{asset_name}] Final HMM did not converge "
                f"({final_hmm.monitor_.iter} iters)"
            )

        hmm_path = ckpt_dir / f"hmm_{asset_name}.pkl"
        pca_path = ckpt_dir / f"pca_{asset_name}.pkl"
        joblib.dump(final_hmm, hmm_path)
        joblib.dump(pca_model, pca_path)
        mlflow.log_artifact(str(hmm_path))
        mlflow.log_artifact(str(pca_path))
        logger.success(
            f"  [{asset_name}] Saved → {hmm_path.name}, {pca_path.name}"
        )

        per_asset_models[asset_name] = (final_hmm, pca_model)

    return per_asset_models


def generate_per_asset_regime_stream(
    embeddings: np.ndarray,
    timestamp_df: pd.DataFrame,
    asset_map_df: pd.DataFrame | None,
    per_asset_models: dict[str, tuple[GaussianHMM, PCA]],
) -> pd.DataFrame:
    """
    Generate a unified regime stream from per-asset HMMs.

    Different assets may have different n_states; probability columns are padded
    with 0.0 up to max_states so all rows share the same schema.

    Output schema: timestamp, asset, regime_label, regime_prob_0 … regime_prob_{max-1}
    Saves to data/processed/regime_stream_per_asset.parquet.
    """
    if asset_map_df is None or len(asset_map_df) == 0:
        asset_map_df = _build_asset_map_from_timestamps(timestamp_df)

    max_states = max(hmm.n_components for hmm, _ in per_asset_models.values())
    logger.info(f"Generating per-asset regime stream  (max_states={max_states})")

    frames: list[pd.DataFrame] = []

    for asset_name, (hmm_model, pca_model) in per_asset_models.items():
        asset_rows = asset_map_df[asset_map_df["asset_name"] == asset_name]
        if asset_rows.empty:
            logger.warning(f"  [{asset_name}] Not found in asset_map — skipping")
            continue

        map_row = asset_rows.iloc[0]
        start_idx = int(map_row["start_idx"])
        end_idx = int(map_row["end_idx"])

        asset_emb = embeddings[start_idx : end_idx + 1]
        asset_ts = timestamp_df.iloc[start_idx : end_idx + 1].reset_index(drop=True)

        emb_reduced = pca_model.transform(asset_emb.astype(np.float64))
        labels = hmm_model.predict(emb_reduced)
        probs = hmm_model.predict_proba(emb_reduced)
        n_states = hmm_model.n_components

        df = asset_ts[["timestamp"]].copy()
        df["asset"] = asset_name 
        df["regime_label"] = labels

        for i in range(max_states):
            df[f"regime_prob_{i}"] = probs[:, i] if i < n_states else 0.0

        frames.append(df)

        dist = pd.Series(labels).value_counts(normalize=True).sort_index()
        logger.info(f"  [{asset_name}] n_states={n_states}  dist={dist.to_dict()}")
        for state_idx, pct in dist.items():
            mlflow.log_metric(f"{asset_name}_regime_{state_idx}_pct", float(pct))

    combined = pd.concat(frames, ignore_index=True)

    out_path = _ROOT / "data" / "processed" / "regime_stream_per_asset.parquet"
    combined.to_parquet(out_path, engine="pyarrow", index=False)
    logger.success(
        f"Saved per-asset regime stream → {out_path}  ({len(combined):,} rows)"
    )
    mlflow.log_artifact(str(out_path))
    mlflow.log_metric("per_asset_n_rows", len(combined))
    mlflow.log_param("per_asset_max_states", max_states)

    return combined
