"""Fits the HMM on the BTC-only-v3 encoder's embeddings (2026-08-14) — the downstream
regime-quality check for the best-converged, gap-bug-fixed encoder from this investigation
(val top-1 contrastive accuracy 61.77%, reconstruction R2=0.79, both the best of the whole
investigation; modest but real AUC edge over raw_14 at 12h/24h on a clean baseline).

Fresh BIC search (not force_n_states) — this is a fundamentally different single-asset
pipeline branch, not a controlled single-variable comparison against production's 4-asset
n=4 HMM, so there's no "match production" reason to fix n_states here.

Reads embeddings.npy/embedding_timestamps.parquet/embedding_asset_map.parquet from
data/processed/encoder_experiments/e1_predictive_btc_only_v3/ (already generated as a
side effect of that encoder's own training run — extract_embeddings ran on the
gap-free BTC_USDT_normalized.parquet, per configs/encoder_e1_predictive_btc_only_v3.yaml's
file_suffix). Writes to fully isolated paths — never touches production's
regime_stream.parquet or regime/checkpoints/{hmm,pca}_model.pkl.

Run: conda run -n marta python regime/run_hmm_btc_only_v3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from regime.hmm import run_hmm_pipeline

PROCESSED_DIR = "data/processed/encoder_experiments/e1_predictive_btc_only_v3"
CHECKPOINT_DIR = ROOT / "regime" / "checkpoints" / "btc_only_v3"


def main() -> None:
    with open(ROOT / "configs" / "assets.yaml") as f:
        config = yaml.safe_load(f)
    config["data"]["processed_dir"] = PROCESSED_DIR

    embeddings_path = ROOT / PROCESSED_DIR / "embeddings.npy"
    if not embeddings_path.exists():
        print(f"ERROR: {embeddings_path} not found.")
        sys.exit(1)

    print(f"processed_dir (isolated): {PROCESSED_DIR}")
    print(f"checkpoint_dir (isolated): {CHECKPOINT_DIR}")
    print("pca_n_components: 16 (standard)")
    print("force_n_states: None (fresh BIC search)")

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment("marta_regime_hmm")
    with mlflow.start_run(run_name="btc_only_v3_hmm"):
        mlflow.log_param("experiment_tag", "btc_only_v3")
        mlflow.log_param("encoder_checkpoint", "best_encoder_e1_predictive_btc_only_v3.pt")
        result = run_hmm_pipeline(
            config,
            force_n_states=None,
            checkpoint_dir=CHECKPOINT_DIR,
            pca_n_components=16,
        )
        regime_df = result["regime_df"]
        print(f"\nSelected n_states={result['best_n']}")
        print(f"Done: {len(regime_df):,} rows -> "
              f"{ROOT / PROCESSED_DIR / 'regime_stream.parquet'}")


if __name__ == "__main__":
    main()
