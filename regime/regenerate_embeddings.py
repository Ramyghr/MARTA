"""Regenerates data/processed/embeddings.npy (+ timestamps/asset-map) from the
CURRENTLY DEPLOYED encoder checkpoint (encoder/checkpoints/best_encoder.pt).

Why this exists: embeddings.npy is normally a side effect of run_encoder.py's
training run, so it's implicitly tied to whatever checkpoint that run produced. If
the deployed checkpoint changes later (e.g. a walk-forward retrain that swaps in a
new best_encoder.pt without a full run_encoder.py pass), embeddings.npy silently
keeps reflecting the OLD encoder — and everything downstream (PCA, HMM, regime_stream)
gets fit on embeddings that no longer match what encoder/checkpoints/best_encoder.pt
actually produces live. Found exactly this: embeddings.npy was dated 2026-07-04,
best_encoder.pt is dated 2026-07-14 (10 days newer, from the walk-forward retrain).

Run this before regime/hmm.py's run_hmm_pipeline() (i.e. before `python run_regime.py`)
whenever best_encoder.pt has changed since embeddings.npy was last generated.

Run: conda run -n marta python regime/regenerate_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from encoder.embed import extract_embeddings
from encoder.model import TFTRegimeEncoder


def main() -> None:
    with open(ROOT / "configs" / "encoder.yaml") as f:
        config = yaml.safe_load(f)

    ckpt_path = ROOT / "encoder" / "checkpoints" / "best_encoder.pt"
    print(f"Loading {ckpt_path} (mtime confirms which retrain this is)")

    model = TFTRegimeEncoder(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        proj_dim=config["model"]["proj_dim"],
        n_lstm_layers=config["model"]["n_lstm_layers"],
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="re_embed_from_best_encoder"):
        mlflow.log_param("checkpoint", str(ckpt_path))
        print(f"Extracting embeddings for assets: {config['data']['assets']}")
        embedding_matrix, timestamps = extract_embeddings(model, config)
        print(f"Done: {embedding_matrix.shape[0]:,} windows x {embedding_matrix.shape[1]}-dim, "
              f"saved to {config['data']['processed_dir']}/embeddings.npy")


if __name__ == "__main__":
    main()
