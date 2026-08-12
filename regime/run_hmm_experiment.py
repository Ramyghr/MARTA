"""Runs the production HMM pipeline against an EXPERIMENTAL encoder's embeddings,
writing to fully isolated output paths — never touches production's regime_stream.parquet
or regime/checkpoints/{hmm,pca}_model.pkl.

Phase D of the E1-predictive downstream-propagation audit (2026-08-12): the encoder is the
only independent variable, so this deliberately forces n_states=4 to match the CURRENTLY
DEPLOYED production HMM (verified via regime/checkpoints/hmm_model.pkl before writing this
script) rather than re-running BIC search, which could pick a different n_states and
confound "different encoder" with "different regime count".

Reads embeddings.npy/embedding_timestamps.parquet/embedding_asset_map.parquet from
data/processed/encoder_experiments/e1_predictive/ (already generated as a side effect of
that encoder's own training run — see encoder/train.py's post-training extract_embeddings
call). Does not regenerate them.

Run: conda run -n marta python regime/run_hmm_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from regime.hmm import run_hmm_pipeline

EXPERIMENT_PROCESSED_DIR = "data/processed/encoder_experiments/e1_predictive"
EXPERIMENT_CHECKPOINT_DIR = ROOT / "regime" / "checkpoints" / "phase0_e1predictive"
FORCE_N_STATES = 4  # matches production's currently-deployed hmm_model.pkl


def main() -> None:
    with open(ROOT / "configs" / "assets.yaml") as f:
        config = yaml.safe_load(f)

    config["data"]["processed_dir"] = EXPERIMENT_PROCESSED_DIR

    embeddings_path = ROOT / EXPERIMENT_PROCESSED_DIR / "embeddings.npy"
    if not embeddings_path.exists():
        print(f"ERROR: {embeddings_path} not found. Run regime/regenerate_embeddings.py "
              f"pointed at the E1-predictive checkpoint first, or verify Phase C's "
              f"finding that this file already exists from the encoder's own training run.")
        sys.exit(1)

    print(f"processed_dir (isolated): {EXPERIMENT_PROCESSED_DIR}")
    print(f"checkpoint_dir (isolated): {EXPERIMENT_CHECKPOINT_DIR}")
    print(f"force_n_states: {FORCE_N_STATES} (matches production, encoder is the only "
          f"independent variable)")

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment("marta_regime_hmm")
    with mlflow.start_run(run_name="phase0_e1predictive_hmm"):
        mlflow.log_param("experiment_tag", "phase0_e1predictive")
        mlflow.log_param("encoder_checkpoint", "best_encoder_e1_predictive.pt")
        result = run_hmm_pipeline(
            config,
            force_n_states=FORCE_N_STATES,
            checkpoint_dir=EXPERIMENT_CHECKPOINT_DIR,
        )
        regime_df = result["regime_df"]
        print(f"Done: {len(regime_df):,} rows -> "
              f"{ROOT / EXPERIMENT_PROCESSED_DIR / 'regime_stream.parquet'}")


if __name__ == "__main__":
    main()
