"""
run_regime_live.py
Fits a SEPARATE, live-only PCA(16) + GaussianHMM(4) on embeddings spanning
2018 -> today, used exclusively by the dashboard backend (regime_service.py) for
live regime display and paper trading. The "sacred" regime/checkpoints/{pca,hmm}_model.pkl
and data/processed/regime_stream.parquet (fit only on data before configs/assets.yaml's
train_end, 2024-01-01) are never touched by this script — those remain frozen for
RL/backtest/research reproducibility.

Why this exists: the sacred model's PCA+HMM were fit on 2018-2023 data. By mid-2026,
live queries are 2+ years out-of-distribution — verified empirically (2026-08-08 audit)
that every live embedding since January 2026, across all 4 assets independently, was
landing nearest a single HMM state (~100% R3), even though the embeddings themselves
genuinely vary day to day (this ruled out a computation bug — it's real distributional
drift from a 3-year-frozen cluster map, not frozen/stale data). Refitting on an extended
window that actually includes 2025-2026 conditions fixes the collapse for the live display
without touching anything the sacred split policy protects.

Run: conda run -n marta python run_regime_live.py
Outputs:
  regime/checkpoints/{pca,hmm}_model_live.pkl
  data/processed/regime_stream_live.parquet   (self-consistent, 2018 -> today, ONE model
                                                throughout — no blending of two HMMs'
                                                incompatible state semantics)
  data/validated/regime_semantics_live.json
  regime/plots/regime_tsne_live.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM

_ROOT = Path(__file__).resolve().parent
for p in (str(_ROOT), str(_ROOT / "dashboard" / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from regime.hmm import fit_pca, generate_regime_stream  # noqa: E402
from regime.validate import validate_regime_semantics  # noqa: E402
from regime.visualize import plot_regime_tsne  # noqa: E402
from utils.logger import logger  # noqa: E402

_CANONICAL_ASSETS = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]
_CKPT_DIR = _ROOT / "regime" / "checkpoints"
_PLOTS_DIR = _ROOT / "regime" / "plots"
_PROCESSED = _ROOT / "data" / "processed"


def _build_all_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    """Build embeddings for the ENTIRE 2018 -> today window through ONE pipeline —
    ppo_inference._build_regime_embedding, the exact function get_live_regime() and
    backfill_live_regime.py use for real live decisions.

    Originally this reused the bulk embeddings.npy (built by regime/regenerate_embeddings.py
    for 2018-2025) and only live-built the new 2026 rows. Found 2026-08-08: for the SAME
    (asset, timestamp), the bulk pipeline and the live builder produce embeddings 0.45-0.53
    L2 apart — bigger than the actual inter-regime cluster distances (~0.15-0.28). Mixing
    the two pipelines meant the HMM was clustering on "which code path built this vector,"
    not real market structure — it produced an isolated ~10%-of-data state that was
    (almost) exactly the new 2026 rows and nothing else, which looked like a 2026 regime
    shift but was actually this artifact. Rebuilding everything through the live pipeline
    uniformly removes that confound entirely.
    """
    import ppo_inference

    timestamps_ref = pd.read_parquet(_PROCESSED / "embedding_timestamps.parquet")
    if "asset" not in timestamps_ref.columns:
        asset_map = pd.read_parquet(_PROCESSED / "embedding_asset_map.parquet")
        alias = {"XAUUSD": "GOLD"}
        asset_col = np.empty(len(timestamps_ref), dtype=object)
        for _, row in asset_map.iterrows():
            name = alias.get(row["asset_name"], row["asset_name"])
            asset_col[int(row["start_idx"]):int(row["end_idx"])] = name
        timestamps_ref = timestamps_ref.copy()
        timestamps_ref["asset"] = asset_col
    timestamps_ref["timestamp"] = pd.to_datetime(timestamps_ref["timestamp"], utc=True)

    ppo_inference.load_model()
    all_embs, all_rows = [], []
    for asset in _CANONICAL_ASSETS:
        hist_ts = timestamps_ref.loc[timestamps_ref["asset"] == asset, "timestamp"].sort_values()
        asset_max = hist_ts.max()
        aligned = ppo_inference._get_aligned_df(asset)
        if aligned is None or aligned.empty:
            logger.warning(f"[{asset}] no aligned data — skipping")
            continue
        data_max = aligned.index.max()
        new_hours = pd.date_range(asset_max, data_max, freq="1h", inclusive="right")
        all_hours = pd.DatetimeIndex(hist_ts).union(new_hours).sort_values()

        logger.info(f"[{asset}] building {len(all_hours):,} embeddings (uniform live pipeline): "
                    f"{all_hours[0]} -> {all_hours[-1]}")
        n_done = 0
        for ts in all_hours:
            emb = ppo_inference._build_regime_embedding(asset, ts)
            if not np.any(emb):
                continue
            all_embs.append(emb)
            all_rows.append({"timestamp": ts, "asset": asset})
            n_done += 1
            if n_done % 20000 == 0:
                logger.info(f"[{asset}] {n_done:,}/{len(all_hours):,} built …")

    embeddings = np.stack(all_embs).astype(np.float32)
    timestamps = pd.DataFrame(all_rows)
    logger.success(f"Built embeddings: {embeddings.shape}  timestamps: {len(timestamps):,} rows")
    return embeddings, timestamps


def main() -> None:
    config = yaml.safe_load(open(_ROOT / "configs" / "assets.yaml"))

    mlflow.set_tracking_uri(str(_ROOT / "mlruns"))
    mlflow.set_experiment("marta_regime_hmm_live")
    with mlflow.start_run(run_name="regime_live_refit"):
        embeddings, timestamps = _build_all_embeddings()

        # No train/val/test split here — this model is never used for backtest/RL
        # evaluation, only live dashboard classification, so there's no leakage concern
        # in fitting on the full 2018 -> today window.
        train_mask = np.ones(len(embeddings), dtype=bool)
        pca, emb_reduced = fit_pca(embeddings, train_mask, n_components=16)

        n_states = 4  # matches the sacred model's empirically-validated structure
        # (regime/n_states_sweep.py, 2026-08-08 audit) — no flicker/duplicate states.
        logger.info(f"Fitting live HMM (n_states={n_states}) on {len(embeddings):,} rows …")
        hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=42, tol=1e-4)
        hmm.fit(emb_reduced)
        if not hmm.monitor_.converged:
            logger.warning(f"Live HMM did not converge ({hmm.monitor_.iter} iters)")

        _CKPT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(hmm, _CKPT_DIR / "hmm_model_live.pkl")
        joblib.dump(pca, _CKPT_DIR / "pca_model_live.pkl")
        logger.success(f"Saved live checkpoints -> {_CKPT_DIR}")

        regime_df = generate_regime_stream(embeddings, timestamps, None, hmm, pca)
        out_path = _PROCESSED / "regime_stream_live.parquet"
        regime_df.to_parquet(out_path, engine="pyarrow", index=False)
        logger.success(f"Saved live regime stream -> {out_path}  ({len(regime_df):,} rows)")

        validate_regime_semantics(regime_df, config, output_filename="regime_semantics_live.json")

        tsne_path = _PLOTS_DIR / "regime_tsne_live.png"
        logger.info("Running t-SNE on live embeddings (this may take a few minutes) …")
        plot_regime_tsne(embeddings, regime_df["regime_label"].values, tsne_path)
        logger.success(f"Live t-SNE -> {tsne_path}")


if __name__ == "__main__":
    main()
