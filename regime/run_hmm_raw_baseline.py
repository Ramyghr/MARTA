"""Fits the HMM directly on raw 14-dim features (no encoder, no PCA) — Step 1 of the
raw-feature-baseline pivot (2026-08-12, after the contrastive encoder was rejected for
never beating raw_14 on predictability, see project memory).

Unlike regime/run_hmm_experiment.py (which force-fit n_states=4 to match production for
a controlled single-variable comparison), this runs the FULL BIC search fresh — there's
no "only the encoder changed" comparison to preserve here, this is a genuinely different
model (14-dim raw features per bar, not a 128-dim learned embedding), so the best n_states
for THIS input space is an open question, not a fixed one.

Reads data/processed/raw_baseline/{embeddings,embedding_timestamps,embedding_asset_map}
(built by regime/build_raw_feature_embeddings.py). Writes to fully isolated paths —
never touches production's regime_stream.parquet or regime/checkpoints/{hmm,pca}_model.pkl.

Run: conda run -n marta python regime/run_hmm_raw_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from regime.hmm import run_hmm_pipeline

RAW_PROCESSED_DIR = "data/processed/raw_baseline"
RAW_CHECKPOINT_DIR = ROOT / "regime" / "checkpoints" / "raw_baseline"


def main() -> None:
    with open(ROOT / "configs" / "assets.yaml") as f:
        config = yaml.safe_load(f)
    config["data"]["processed_dir"] = RAW_PROCESSED_DIR

    embeddings_path = ROOT / RAW_PROCESSED_DIR / "embeddings.npy"
    if not embeddings_path.exists():
        print(f"ERROR: {embeddings_path} not found. Run "
              f"regime/build_raw_feature_embeddings.py first.")
        sys.exit(1)

    print(f"processed_dir (isolated): {RAW_PROCESSED_DIR}")
    print(f"checkpoint_dir (isolated): {RAW_CHECKPOINT_DIR}")
    print("pca_n_components: None (raw 14 features fed directly to the HMM)")
    print("force_n_states: None (full BIC search — not a controlled comparison)")

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment("marta_regime_hmm")
    with mlflow.start_run(run_name="raw_baseline_hmm"):
        mlflow.log_param("experiment_tag", "raw_baseline")
        mlflow.log_param("input", "raw_14_features_no_pca")
        result = run_hmm_pipeline(
            config,
            force_n_states=None,
            checkpoint_dir=RAW_CHECKPOINT_DIR,
            pca_n_components=None,
        )
        regime_df = result["regime_df"]
        print(f"\nSelected n_states={result['best_n']}")
        print(f"Done: {len(regime_df):,} rows -> "
              f"{ROOT / RAW_PROCESSED_DIR / 'regime_stream.parquet'}")


if __name__ == "__main__":
    main()
