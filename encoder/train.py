"""
Week 2 encoder training loop — contrastive pretraining with InfoNCE loss.

Data split: train < train_end, val >= val_start (within actual data range 2018-2021).
All 5 assets are concatenated into a shared ConcatDataset so the encoder learns
a single embedding space across crypto, forex, and commodities.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import ConcatDataset, DataLoader

from encoder.dataset import RegimeContrastiveDataset
from encoder.embed import extract_embeddings
from encoder.loss import NTXentLoss
from encoder.model import TFTRegimeEncoder
from encoder.visualize import plot_training_curves, plot_tsne_embeddings
from utils.logger import logger

_ROOT = Path(__file__).resolve().parent.parent


def _load_asset_split(
    asset: str,
    processed_dir: Path,
    train_end: str,
    val_start: str,
    window_size: int,
    step: int,
) -> tuple[RegimeContrastiveDataset, RegimeContrastiveDataset]:
    df = pd.read_parquet(processed_dir / f"{asset}_aligned.parquet", engine="pyarrow")
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_start_ts = pd.Timestamp(val_start, tz="UTC")

    train_df = df[df.index < train_end_ts]
    val_df = df[df.index >= val_start_ts]

    train_ds = RegimeContrastiveDataset(train_df, window_size=window_size, step=step)
    val_ds = RegimeContrastiveDataset(val_df, window_size=window_size, step=step)
    return train_ds, val_ds


def _run_epoch(
    loader: DataLoader,
    model: TFTRegimeEncoder,
    loss_fn: NTXentLoss,
    optimizer: AdamW | None,
    device: torch.device,
    grad_clip: float,
    train: bool,
    temp_bounds: tuple[float, float] | None = None,
) -> tuple[float, float, bool]:
    """Run one full pass over loader.  Returns (mean_loss, mean_acc, nan_grad_hit)."""
    model.train(train)
    nan_grad_hit = False
    losses: list[float] = []
    accs: list[float] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for anchor, positive in loader:
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)

            _, z_a = model(anchor)
            _, z_p = model(positive)
            loss, acc = loss_fn(z_a, z_p)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                all_params = list(model.parameters()) + list(loss_fn.parameters())
                grad_norm = nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
                if torch.isnan(grad_norm):
                    nan_grad_hit = True
                optimizer.step()
                if temp_bounds is not None:
                    loss_fn.clamp_temperature(*temp_bounds)

            losses.append(loss.item())
            accs.append(acc.item())

    return float(np.mean(losses)), float(np.mean(accs)), nan_grad_hit


def train_encoder(config: dict) -> TFTRegimeEncoder:
    """
    Full contrastive pretraining loop.

    Trains TFTRegimeEncoder with NTXentLoss across all 5 aligned asset datasets,
    logs every epoch to MLflow, applies early stopping, saves best checkpoint,
    extracts embeddings, and plots t-SNE — all within the caller's MLflow run.

    Returns the best encoder (lowest val loss).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # ------------------------------------------------------------------ config
    mcfg = config["model"]
    tcfg = config["training"]
    dcfg = config["data"]
    vcfg = config["visualization"]
    conv = config["convergence"]
    ckpt_cfg = config["checkpoints"]

    window_size: int = tcfg["window_size"]
    step: int = tcfg["step"]
    batch_size: int = tcfg["batch_size"]
    lr: float = tcfg["lr"]
    weight_decay: float = tcfg["weight_decay"]
    max_epochs: int = tcfg["max_epochs"]
    patience: int = tcfg["early_stopping_patience"]
    warmup_epochs: int = tcfg["warmup_epochs"]
    grad_clip: float = tcfg["gradient_clip_max_norm"]
    train_end: str = tcfg["train_end"]
    val_start: str = tcfg["val_start"]

    processed_dir = _ROOT / dcfg["processed_dir"]
    ckpt_dir = _ROOT / ckpt_cfg["dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = _ROOT / "encoder" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- data
    train_datasets: list[RegimeContrastiveDataset] = []
    val_datasets: list[RegimeContrastiveDataset] = []
    for asset in dcfg["assets"]:
        tr_ds, va_ds = _load_asset_split(
            asset, processed_dir, train_end, val_start, window_size, step
        )
        if len(tr_ds) > 0:
            train_datasets.append(tr_ds)
        if len(va_ds) > 0:
            val_datasets.append(va_ds)
        logger.info(f"{asset}: {len(tr_ds)} train windows, {len(va_ds)} val windows")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin,
    )

    # -------------------------------------------------------------- model
    model = TFTRegimeEncoder(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        proj_dim=mcfg["proj_dim"],
        n_lstm_layers=mcfg["n_lstm_layers"],
    ).to(device)
    loss_fn = NTXentLoss().to(device)

    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = AdamW(all_params, lr=lr, weight_decay=weight_decay)

    # Linear warmup then cosine annealing
    warmup_sched = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(max_epochs - warmup_epochs, 1)
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    # -------------------------------------------------------------- training
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0
    nan_grad_detected = False
    stopping_epoch = max_epochs

    temp_bounds = (conv["min_temperature"], conv["max_temperature"])

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc, nan_hit = _run_epoch(
            train_loader, model, loss_fn, optimizer, device, grad_clip, train=True,
            temp_bounds=temp_bounds,
        )
        if nan_hit:
            nan_grad_detected = True

        val_loss, val_acc, _ = _run_epoch(
            val_loader, model, loss_fn, None, device, grad_clip, train=False
        )

        scheduler.step()

        current_temp = loss_fn.temperature.item()
        current_lr = optimizer.param_groups[0]["lr"]

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_top1_acc": train_acc,
                "val_loss": val_loss,
                "val_top1_acc": val_acc,
                "temperature": current_temp,
                "learning_rate": current_lr,
            },
            step=epoch,
        )

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train {train_loss:.4f} ({train_acc:.3f}) | "
            f"val {val_loss:.4f} ({val_acc:.3f}) | "
            f"T={current_temp:.4f} | lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_top1_acc": val_acc,
                    "model_state": model.state_dict(),
                    "loss_fn_state": loss_fn.state_dict(),
                },
                ckpt_dir / ckpt_cfg["best"],
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopping_epoch = epoch
                logger.info(f"Early stopping triggered at epoch {epoch}")
                mlflow.log_param("early_stopping_epoch", epoch)
                break

    # Save final checkpoint and log both
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_top1_acc": val_acc,
            "model_state": model.state_dict(),
            "loss_fn_state": loss_fn.state_dict(),
        },
        ckpt_dir / ckpt_cfg["final"],
    )
    mlflow.log_artifact(str(ckpt_dir / ckpt_cfg["best"]))
    mlflow.log_artifact(str(ckpt_dir / ckpt_cfg["final"]))

    mlflow.log_params(
        {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "stopping_epoch": stopping_epoch,
        }
    )

    # -------------------------------------------------------------- load best for inference
    best_ckpt = torch.load(
        ckpt_dir / ckpt_cfg["best"], map_location=device, weights_only=True
    )
    model.load_state_dict(best_ckpt["model_state"])
    loss_fn.load_state_dict(best_ckpt["loss_fn_state"])

    # -------------------------------------------------------------- convergence checks
    checks = _run_convergence_checks(
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        temperature=loss_fn.temperature.item(),
        nan_grad_detected=nan_grad_detected,
        thresholds=conv,
    )

    # Embed the convergence-check results and thresholds INTO the checkpoint file itself —
    # not just MLflow. When training runs on Kaggle (external, no local MLflow), the
    # checkpoint .pt is the only artifact that reliably gets downloaded; if convergence
    # metrics only live in a throwaway remote MLflow run, they're lost the moment the
    # Kaggle session ends. This is exactly how the 2026-07 Kaggle retrain's collapsed
    # temperature (0.0047, 6.4x below the 0.03 floor) went to production unnoticed.
    for path in (ckpt_dir / ckpt_cfg["best"], ckpt_dir / ckpt_cfg["final"]):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved["convergence_checks"] = checks
        saved["convergence_thresholds"] = dict(conv)
        torch.save(saved, path)

    # -------------------------------------------------------------- post-training artefacts
    run_id = mlflow.active_run().info.run_id

    logger.info("Extracting embeddings …")
    embedding_matrix, timestamps = extract_embeddings(model, config)

    logger.info("Rendering t-SNE …")
    tsne_path = plots_dir / "tsne_embeddings.png"
    try:
        plot_tsne_embeddings(
            embedding_matrix,
            timestamps,
            tsne_path,
            n_samples=vcfg["tsne_n_samples"],
        )
        mlflow.log_metric("tsne_saved", 1)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"t-SNE failed: {exc}")
        mlflow.log_metric("tsne_saved", 0)

    logger.info("Rendering training curves …")
    curves_path = plots_dir / "training_curves.png"
    try:
        plot_training_curves(run_id, curves_path)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Training curves plot failed: {exc}")

    return model


def _run_convergence_checks(
    best_val_loss: float,
    best_val_acc: float,
    temperature: float,
    nan_grad_detected: bool,
    thresholds: dict,
) -> dict[str, bool]:
    """Log all convergence checks to MLflow, surface CRITICAL failures, and return the
    pass/fail dict so the caller can embed it in the checkpoint file itself."""
    checks: dict[str, bool] = {
        "val_loss_below_threshold": best_val_loss < thresholds["max_val_loss"],
        "top1_acc_above_threshold": best_val_acc > thresholds["min_top1_acc"],
        "temperature_in_range": thresholds["min_temperature"] <= temperature <= thresholds["max_temperature"],
        "no_nan_gradients": not nan_grad_detected,
    }
    severity = {
        "val_loss_below_threshold": "CRITICAL",
        "top1_acc_above_threshold": "CRITICAL",
        "temperature_in_range": "WARNING",
        "no_nan_gradients": "CRITICAL",
    }
    diagnostics = {
        "val_loss_below_threshold": (
            f"Val loss {best_val_loss:.4f} >= {thresholds['max_val_loss']} "
            f"(random baseline ≈ {__import__('math').log(512):.2f}). "
            "Check: batch size too small, learning rate too high, or augmentation too aggressive."
        ),
        "top1_acc_above_threshold": (
            f"Top-1 acc {best_val_acc:.4f} <= {thresholds['min_top1_acc']}. "
            "Check: projection head collapse, temperature too large, or data too uniform."
        ),
        "temperature_in_range": (
            f"Temperature {temperature:.4f} outside [{thresholds['min_temperature']}, "
            f"{thresholds['max_temperature']}]. Usually harmless but monitor for instability."
        ),
        "no_nan_gradients": (
            "NaN gradients detected during training. "
            "Check: reduce lr, add gradient clipping, or inspect input normalisation."
        ),
    }

    for name, passed in checks.items():
        mlflow.log_metric(f"check_{name}", int(passed))
        level = severity[name]
        if not passed:
            msg = f"[{level}] Convergence check FAILED — {name}: {diagnostics[name]}"
            if level == "CRITICAL":
                logger.error(msg)
            else:
                logger.warning(msg)
        else:
            logger.success(f"[{level}] Convergence check PASSED — {name}")

    return checks
