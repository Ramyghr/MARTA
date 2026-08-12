"""E1 — pooling ablation, INFERENCE ONLY (2026-08-11).

Loads the existing E2 checkpoint (encoder/checkpoints/best_encoder_e2_nojittermask.pt,
epoch 177) into TFTRegimeEncoderLastPool (encoder/model_lastpool.py — last-timestep
pooling instead of mean-pool, all trained weights otherwise identical). No training.

Reuses run_multi_horizon / run_reconstruction / run_collapse_diagnostics from
rl/audit_encoder.py UNCHANGED — this is not a reimplementation, it's the exact same
evaluation code pointed at a differently-pooled read-out of the same trained weights, so
results are directly comparable to every other checkpoint already evaluated with it.

Outputs saved under encoder/eval/E1_lastpool/ (embeddings + result json), as requested,
distinct from data/results/ where the other encoder diagnostics live.

Usage:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        /home/ramy/miniconda3/envs/marta/bin/python -m rl.evaluate_encoder_lastpool
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

from encoder.model_lastpool import TFTRegimeEncoderLastPool
from rl.audit_encoder import run_collapse_diagnostics, run_multi_horizon, run_reconstruction
from rl.minimal_btc_env import _PROCESSED
from rl.supervised_predictability import _valid_indices, build_marta_features
from utils.assets import aligned_path
from utils.logger import logger

_console = Console()
_SEED = 42
_E2_CHECKPOINT = Path("encoder/checkpoints/best_encoder_e2_nojittermask.pt")
_OUT_DIR = Path("encoder/eval/E1_lastpool")


def load_lastpool_shared(device: str = "cpu") -> dict:
    """Same loading contract as rl.minimal_btc_env.load_shared_data, but instantiates
    TFTRegimeEncoderLastPool and loads E2's checkpoint weights into it (state_dict keys
    are identical to TFTRegimeEncoder — only the forward-pass pooling op differs, no new
    parameters, so this load is exact, not a partial/best-effort match)."""
    import pandas as pd

    df = pd.read_parquet(aligned_path("BTC_USDT", _PROCESSED), engine="pyarrow").sort_index()
    encoder = TFTRegimeEncoderLastPool(input_dim=14, hidden_dim=128)
    ckpt = torch.load(_E2_CHECKPOINT, map_location=device, weights_only=True)
    missing, unexpected = encoder.load_state_dict(ckpt["model_state"], strict=True)
    logger.info(f"Loaded E2 checkpoint (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}) into TFTRegimeEncoderLastPool — strict load, no missing/unexpected keys")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return {"df": df, "encoder": encoder, "encoder_path": str(_E2_CHECKPOINT), "pooling": "last_timestep"}


def save_embeddings(shared: dict) -> None:
    """Save the 2024 val-split embeddings for traceability, as requested."""
    df = shared["df"]
    val_idx = _valid_indices(df.index, "val")
    Z = build_marta_features(df, val_idx, shared["encoder"], device="cpu")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_OUT_DIR / "val_2024_embeddings_lastpool.npy", Z)
    logger.success(f"Saved {Z.shape} embeddings -> {_OUT_DIR / 'val_2024_embeddings_lastpool.npy'}")


def main() -> None:
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    _console.rule("[bold]E1 — last-timestep pooling ablation (inference-only, E2-best weights)")
    if not _E2_CHECKPOINT.exists():
        logger.error(f"E2 checkpoint not found: {_E2_CHECKPOINT}")
        raise SystemExit(1)

    shared = load_lastpool_shared(device="cpu")
    save_embeddings(shared)

    horizon_results = run_multi_horizon(shared)
    reconstruction_results = run_reconstruction(shared)
    collapse_results = run_collapse_diagnostics(shared)

    out = {
        "encoder_checkpoint": shared["encoder_path"],
        "pooling": shared["pooling"],
        "item3_multi_horizon": horizon_results,
        "item4_6_reconstruction": reconstruction_results,
        "item5_collapse": collapse_results,
    }
    out_path = _OUT_DIR / "e1_lastpool_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    _console.rule("[bold]E1 evaluation complete")
    logger.success(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
