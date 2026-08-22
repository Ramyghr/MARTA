"""
regime/hmm.py
Gaussian HMM fitting pipeline for MARTA regime discovery.
PCA(16) on training embeddings → BIC-selected GaussianHMM → regime probability stream.

BIC search: diag covariance, 100 iters, tol=1e-3, full training data (no subsampling).
Final refit: diag covariance, 200 iters, tol=1e-4, all training data, best-of-N seeds.

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

Sequence-boundary + robustness note (fixed 2026-08-19, regime-layer audit): two more
correctness gaps, same family as the one above.

  1. Every multi-asset embedding matrix is 4 (or fewer) assets' full histories
     concatenated back-to-back. Every `.fit()`/`.score()`/`.predict()` call now passes
     `lengths=` (via `_compute_lengths`) so hmmlearn treats each asset's block as an
     independent chain — previously the model was fit and Viterbi-decoded as if BTC's
     last training row transitioned directly into ETH's first training row.
  2. `fit_hmm_bic` used to subsample 20,000 RANDOM, non-contiguous rows and fit them as
     ONE sequence for the BIC search — consecutive "adjacent" rows in that subsample
     could be hours or days apart in real time, corrupting the estimated self-transition
     probability (biasing it toward looking like faster regime-switching than reality)
     for every n_states candidate BIC scored. Removed entirely; BIC search now fits each
     candidate on the full, correctly-segmented training data. Slower, not wrong.

Also added: the final refit is now best-of-N (`n_seeds`, default 5) independent EM runs
from different random initializations — hmmlearn has no built-in multi-restart, so this
is done manually, keeping the highest-training-log-likelihood model — plus a seed-
sensitivity report (pairwise Adjusted Rand Index / Normalized Mutual Information between
every pair of seeds' decoded labels) saved alongside the checkpoint, so "how stable is
this regime structure to EM initialization" is an artifact, not an assumption. The N
seeds are fit in parallel worker processes (`n_jobs`, default: all cores) — hmmlearn/
numpy have no GPU backend, and this workload (many small, sequential Baum-Welch
iterations on tiny n_states x n_features matrices) is the wrong shape for a GPU to help
with even if they did; independent per-seed CPU processes are the real lever.
"""
from __future__ import annotations

import itertools
import json
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

_ROOT = Path(__file__).resolve().parent.parent

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

from configs.splits import TEST_START, TRAIN_END_EXCLUSIVE, TestSetContaminationError  # noqa: E402
from utils.logger import logger  # noqa: E402


def _compute_lengths(asset_labels: np.ndarray) -> np.ndarray:
    """Contiguous per-asset run lengths, in order of appearance — the `lengths` array
    hmmlearn needs to fit/score/decode a multi-asset embedding matrix as independent
    chains instead of one continuous sequence. `asset_labels` must already be ordered
    so each asset's rows form one contiguous block (true of every embeddings.npy this
    module reads — encoder/embed.py and regime/build_crypto5_features.py both write
    one contiguous block per asset). Works unchanged for a single-asset input (returns
    one length covering everything).
    """
    asset_labels = np.asarray(asset_labels)
    if len(asset_labels) == 0:
        return np.array([], dtype=int)
    change_points = np.flatnonzero(asset_labels[1:] != asset_labels[:-1]) + 1
    boundaries = np.concatenate(([0], change_points, [len(asset_labels)]))
    return np.diff(boundaries)


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


def fit_hmm_bic(
    emb_reduced: np.ndarray,
    train_mask: np.ndarray,
    lengths: np.ndarray,
    n_states_range: tuple[int, int] = (2, 9),
    force_n_states: Optional[int] = None,
) -> tuple[GaussianHMM, int, dict[int, dict[str, float]]]:
    """
    Grid-search GaussianHMM over n_states; select by BIC.

    diag covariance, 100 iters, tol=1e-3, fit on the FULL training data (see module
    docstring — this used to subsample 20K random non-contiguous rows, which corrupted
    the estimated self-transition probability; removed, not worth the speedup).
    run_hmm_pipeline refits the winner on the full dataset via best-of-N seeds.

    `lengths` must be `_compute_lengths()` of the asset labels for
    `emb_reduced[train_mask]`, in the same row order — passed to every `.fit()`/
    `.score()` call so multi-asset sequences are modeled as independent chains, not one
    continuous (and at the asset boundaries, nonsensical) sequence.

    force_n_states: bypass BIC selection entirely and fit only this n_states. Added
    2026-08-08 after a regime-quality audit (regime/regime_quality_audit.py) found the
    BIC flat-curve guard unreliable at this data scale — the log-likelihood term
    dominates the complexity penalty so heavily that relative-BIC comparisons are
    nearly meaningless (see regime/n_states_sweep.py for the empirical alternative:
    fit each candidate and check for near-duplicate states / single-bar dwell times
    directly). Use this to commit a specific, empirically-validated n_states.
    """
    if force_n_states is not None:
        n_states_range = (force_n_states, force_n_states + 1)
    train_emb = emb_reduced[train_mask]
    n_train = len(train_emb)
    pca_dim = emb_reduced.shape[1]

    assert lengths.sum() == n_train, (
        f"lengths sum to {lengths.sum()}, expected {n_train} (len(emb_reduced[train_mask]))"
    )

    logger.info(
        f"BIC search: n_states {n_states_range[0]}–{n_states_range[1]-1}  "
        f"n_train={n_train:,} (full training data, {len(lengths)} asset segment(s))  "
        f"pca_dim={pca_dim}  cov=diag  n_iter=100"
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
            model.fit(train_emb, lengths=lengths)

            if not model.monitor_.converged:
                logger.warning(
                    f"  n_states={n}: did not converge ({model.monitor_.iter} iters)"
                )

            ll = model.score(train_emb, lengths=lengths)
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
    raw_best_n = sorted_by_bic[0]
    raw_best_bic = valid[raw_best_n]["bic"]

    # Flat-curve guard (fixed 2026-08-08): previously only compared the top-2
    # candidates, which is meaningless at this BIC scale (LL ~500K, BIC ~1M) — every
    # adjacent pair in the n=2..8 grid was trivially within 5% of each other, so the
    # guard just knocked the raw argmin down by exactly one state every time (n=8 ->
    # n=7) without ever checking whether a much simpler model fit just as well by the
    # same criterion. A regime-quality audit (KS/ANOVA distinctness + dwell time +
    # predictive signal, see regime/regime_quality_audit.py) confirmed n=7 produced
    # two pairs of statistically-indistinguishable, single-bar-dwell duplicate states
    # (1≈4, 2≈3) — exactly what an under-penalized state count looks like. Now compares
    # every candidate against the GLOBAL best BIC and picks the smallest n_states within
    # 5% of it, which is what "the curve is flat, prefer simplicity" actually means.
    within_threshold = [
        n for n in valid if abs(valid[n]["bic"] - raw_best_bic) < 0.05 * abs(raw_best_bic)
    ]
    best_n = min(within_threshold)
    best_bic = valid[best_n]["bic"]

    if best_n != raw_best_n:
        logger.warning(
            f"BIC curve flat across n_states={sorted(within_threshold)} (all within "
            f"5% of global best BIC={raw_best_bic:,.2f} at n={raw_best_n}). "
            f"Picking most parsimonious model within threshold: n_states={best_n}"
        )

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
    pca: Optional[PCA],
) -> pd.DataFrame:
    """Apply PCA (if given) + HMM to full embeddings; return labelled DataFrame.

    pca=None means the HMM was fit directly on `embeddings` with no reduction
    (raw-feature baseline mode — see run_hmm_pipeline's pca_n_components).

    Decodes with `lengths=` (per-asset contiguous run lengths from `timestamps["asset"]`)
    so Viterbi decoding and the forward-backward posterior never bleed across an asset
    boundary — the caller-supplied `asset_labels` param is unused for this (kept for
    call-site compatibility with run_regime_live.py); `timestamps["asset"]` is what's
    actually always populated by the time this is called.
    """
    emb_reduced = pca.transform(embeddings.astype(np.float64)) if pca is not None else embeddings.astype(np.float64)
    lengths = _compute_lengths(timestamps["asset"].to_numpy())

    regime_labels = hmm_model.predict(emb_reduced, lengths=lengths)
    regime_probs = hmm_model.predict_proba(emb_reduced, lengths=lengths)
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


def _fit_one_seed(
    args: tuple[np.ndarray, np.ndarray, int, int],
) -> tuple[int, GaussianHMM, float, bool, np.ndarray]:
    """Worker for _fit_multiseed's process pool — must be a module-level function so
    it's picklable for ProcessPoolExecutor. Fits one GaussianHMM at one seed.

    Runs single-threaded (BLAS thread env vars are set by the parent before the pool
    is created — see _fit_multiseed) since n_jobs workers each spinning up their own
    multi-threaded BLAS would oversubscribe the machine's cores against each other,
    on top of being what triggers the fork-deadlock this function's callsite avoids
    via a spawn context in the first place.
    """
    train_emb, lengths, n_states, seed = args
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag",
        n_iter=200, random_state=seed, tol=1e-4,
    )
    model.fit(train_emb, lengths=lengths)
    ll = float(model.score(train_emb, lengths=lengths))
    converged = bool(model.monitor_.converged)
    labels = model.predict(train_emb, lengths=lengths)
    return seed, model, ll, converged, labels


def _fit_multiseed(
    train_emb: np.ndarray,
    lengths: np.ndarray,
    n_states: int,
    n_seeds: int = 5,
    seed_base: int = 42,
    n_jobs: Optional[int] = None,
) -> tuple[GaussianHMM, dict[str, Any]]:
    """Fit n_seeds independent GaussianHMMs (diag cov, 200 iter, tol=1e-4) at the given
    n_states, each from a different EM initialization, and return the highest-training-
    log-likelihood one — hmmlearn has no built-in n_init/multi-restart, so this is the
    manual equivalent.

    The n_seeds fits are independent of each other, so they run in parallel worker
    processes (n_jobs, default: all cores) rather than sequentially — hmmlearn/numpy
    have no GPU backend, so process-level CPU parallelism across seeds is the actual
    lever here, not hardware.

    Uses a "spawn" (not the Linux default "fork") multiprocessing context. By the time
    this runs, the caller (run_hmm_pipeline's BIC search) has already done enough
    NumPy work on a large array to trigger OpenBLAS's multi-threaded worker pool in
    THIS process — fork()ing a process pool at that point duplicates only the calling
    thread, leaving the BLAS worker threads' locks "phantom-held" in every child, which
    deadlocks the first BLAS call each child makes (confirmed: workers sat at ~0% CPU
    indefinitely under fork; spawn starts each worker as a fresh interpreter with no
    inherited thread state). BLAS thread env vars are set here, before the pool is
    created, so spawned children inherit them at OS process-creation time (before their
    own NumPy import) and each runs single-threaded — n_jobs single-threaded workers is
    the intended parallelism, not n_jobs workers each also spinning up its own
    multi-threaded BLAS pool.

    Also returns a seed-sensitivity report: per-seed convergence/log-likelihood, and
    pairwise Adjusted Rand Index / Normalized Mutual Information between every pair of
    seeds' Viterbi-decoded training labels. ARI/NMI near 1.0 means every seed converged
    to essentially the same state assignment (the regime structure is stable); values
    well below 1.0 mean the "regimes" are sensitive to EM initialization, not a robust
    property of the data.
    """
    seeds = [seed_base + i for i in range(n_seeds)]
    n_jobs = n_jobs or min(n_seeds, os.cpu_count() or 1)

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    spawn_ctx = multiprocessing.get_context("spawn")

    by_seed: dict[int, tuple[GaussianHMM, float, bool, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=spawn_ctx) as executor:
        futures = {
            executor.submit(_fit_one_seed, (train_emb, lengths, n_states, seed)): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            seed, model, ll, converged, labels = future.result()
            by_seed[seed] = (model, ll, converged, labels)
            logger.info(f"  seed={seed}: LL={ll:,.1f}  converged={converged}")

    models = [by_seed[s][0] for s in seeds]
    lls = [by_seed[s][1] for s in seeds]
    converged = [by_seed[s][2] for s in seeds]
    labels = [by_seed[s][3] for s in seeds]

    best_idx = int(np.argmax(lls))
    best_model = models[best_idx]

    ari_pairs: dict[str, float] = {}
    nmi_pairs: dict[str, float] = {}
    for i, j in itertools.combinations(range(n_seeds), 2):
        key = f"{seeds[i]}_vs_{seeds[j]}"
        ari_pairs[key] = float(adjusted_rand_score(labels[i], labels[j]))
        nmi_pairs[key] = float(normalized_mutual_info_score(labels[i], labels[j]))

    report = {
        "n_states": n_states,
        "seeds": seeds,
        "log_likelihoods": lls,
        "converged": converged,
        "best_seed": seeds[best_idx],
        "pairwise_ari": ari_pairs,
        "pairwise_nmi": nmi_pairs,
        "mean_ari": float(np.mean(list(ari_pairs.values()))) if ari_pairs else 1.0,
        "min_ari": float(np.min(list(ari_pairs.values()))) if ari_pairs else 1.0,
        "mean_nmi": float(np.mean(list(nmi_pairs.values()))) if nmi_pairs else 1.0,
        "min_nmi": float(np.min(list(nmi_pairs.values()))) if nmi_pairs else 1.0,
    }
    logger.info(
        f"Seed sensitivity (n_states={n_states}, {n_seeds} seeds): "
        f"mean ARI={report['mean_ari']:.3f}  min ARI={report['min_ari']:.3f}  "
        f"mean NMI={report['mean_nmi']:.3f}  min NMI={report['min_nmi']:.3f}  "
        f"(1.0 = identical across seeds; best_seed={report['best_seed']})"
    )
    return best_model, report


def run_hmm_pipeline(
    config: dict,
    force_n_states: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    pca_n_components: Optional[int] = 16,
    n_seeds: int = 5,
) -> dict[str, Any]:
    """
    Orchestrate the full Week 3 HMM pipeline.

    force_n_states: skip BIC search, fit only this n_states (see fit_hmm_bic docstring).
    checkpoint_dir: where hmm_model.pkl/pca_model.pkl/bic_results.json are written.
        Defaults to regime/checkpoints/ (production location) — pass an isolated path
        for experimental runs so they don't overwrite the production HMM/PCA.
    pca_n_components: PCA dimensionality before the HMM. Defaults to 16 (production
        behavior, unchanged). Pass None to skip PCA entirely and fit the HMM directly
        on `embeddings` as-is — for a raw-feature baseline where embeddings.npy is
        already low-dimensional (e.g. 14 raw features) and PCA would either fail
        (n_components > n_features) or be pointless.

    n_seeds: number of independent EM initializations for the final refit (best-of-N by
        training log-likelihood). Also produces the seed-sensitivity report saved as
        seed_sensitivity.json (see _fit_multiseed).

    Returns dict with keys:
        regime_df       : pd.DataFrame  — full regime probability stream
        bic_results     : dict[int, dict[str, float]]
        best_n          : int
        hmm_model       : GaussianHMM   — fitted on full dataset (best-of-n_seeds)
        pca             : PCA
        seed_sensitivity: dict — see _fit_multiseed's report
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

    # training mask: rows before the canonical TRAIN_END_EXCLUSIVE (configs/splits.py) —
    # PCA/HMM fit on train only; val and test stay out of the fit.
    train_cutoff = pd.Timestamp(TRAIN_END_EXCLUSIVE, tz="UTC")
    ts_series = pd.to_datetime(timestamps["timestamp"], utc=True)
    train_mask = (ts_series < train_cutoff).values

    # Contamination guard (P0.2): verify the resulting mask actually respects the
    # sacred test boundary, not just that train_cutoff was computed correctly.
    if train_mask.any():
        max_train_ts = ts_series[train_mask].max()
        if max_train_ts >= pd.Timestamp(TEST_START, tz="UTC"):
            raise TestSetContaminationError(
                f"HMM/PCA training mask includes timestamps up to {max_train_ts}, which "
                f"is >= TEST_START ({TEST_START}). Sacred 2025 test data must never reach "
                f"HMM/PCA fitting."
            )

    n_train = int(train_mask.sum())
    n_total = len(train_mask)
    logger.info(f"Train samples: {n_train:,} / {n_total:,} ({n_train/n_total:.1%})")
    mlflow.log_metric("n_train_samples", n_train)
    mlflow.log_metric("n_total_samples", n_total)

    # Step 1 — PCA (fit on training rows only to avoid leakage), or skip entirely
    # when pca_n_components is None (raw-feature baseline mode).
    if pca_n_components is not None:
        pca, emb_reduced = fit_pca(embeddings, train_mask, n_components=pca_n_components)
    else:
        logger.info("pca_n_components=None — skipping PCA, fitting HMM directly on embeddings")
        pca = None
        emb_reduced = embeddings.astype(np.float64)

    # lengths for the train-masked subset — every asset's train rows form one
    # contiguous block (see _compute_lengths), needed so BIC search and the final
    # refit model multi-asset sequences as independent chains, not one continuous one.
    train_lengths = _compute_lengths(timestamps.loc[train_mask, "asset"].to_numpy())

    # Step 2 — BIC selection (training data only)
    _train_model, best_n, bic_results = fit_hmm_bic(
        emb_reduced, train_mask, train_lengths, force_n_states=force_n_states
    )

    # Step 3 — Refit winner on TRAINING DATA ONLY, best-of-n_seeds (diag covariance,
    # more iterations — see module docstring for why this is diag, not full, and why
    # this is now multiple seeds rather than one fixed random_state=42 fit)
    logger.info(
        f"Refitting best HMM (n_states={best_n}) on training data only — "
        f"{n_seeds} seeds, best-of-N by log-likelihood …"
    )
    final_hmm, seed_report = _fit_multiseed(
        emb_reduced[train_mask], train_lengths, best_n, n_seeds=n_seeds, seed_base=42,
    )
    if not all(seed_report["converged"]):
        n_failed = sum(1 for c in seed_report["converged"] if not c)
        logger.warning(f"{n_failed}/{n_seeds} seeds did not converge for n_states={best_n}")

    mlflow.log_metric("seed_sensitivity_mean_ari", seed_report["mean_ari"])
    mlflow.log_metric("seed_sensitivity_min_ari", seed_report["min_ari"])
    mlflow.log_metric("seed_sensitivity_mean_nmi", seed_report["mean_nmi"])
    mlflow.log_metric("seed_sensitivity_min_nmi", seed_report["min_nmi"])
    mlflow.log_param("best_seed", seed_report["best_seed"])

    # save checkpoints
    ckpt_dir = checkpoint_dir or (_ROOT / "regime" / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_hmm, ckpt_dir / "hmm_model.pkl")
    if pca is not None:
        joblib.dump(pca, ckpt_dir / "pca_model.pkl")
    logger.success(f"Saved checkpoints → {ckpt_dir}")

    # Step 4 — regime probability stream
    regime_df = generate_regime_stream(
        embeddings, timestamps, None, final_hmm, pca
    )

    out_path = processed_dir / "regime_stream.parquet"
    regime_df.to_parquet(out_path, engine="pyarrow", index=False)
    logger.success(f"Saved regime stream → {out_path}  ({len(regime_df):,} rows)")

    # persist BIC results + seed-sensitivity report as JSON artifacts
    bic_json_path = ckpt_dir / "bic_results.json"
    with open(bic_json_path, "w") as fh:
        json.dump({str(k): v for k, v in bic_results.items()}, fh, indent=2)

    seed_json_path = ckpt_dir / "seed_sensitivity.json"
    with open(seed_json_path, "w") as fh:
        json.dump(seed_report, fh, indent=2)

    artifact_paths = [ckpt_dir / "hmm_model.pkl", out_path, bic_json_path, seed_json_path]
    if pca is not None:
        artifact_paths.append(ckpt_dir / "pca_model.pkl")
    for path in artifact_paths:
        mlflow.log_artifact(str(path))

    return {
        "regime_df": regime_df,
        "bic_results": bic_results,
        "best_n": best_n,
        "hmm_model": final_hmm,
        "pca": pca,
        "seed_sensitivity": seed_report,
    }
