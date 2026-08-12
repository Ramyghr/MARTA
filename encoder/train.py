"""
Week 2 encoder training loop — contrastive pretraining with InfoNCE loss.

Data split: train < TRAIN_END_EXCLUSIVE, val >= VAL_START (see configs/splits.py, the
canonical source — this module no longer reads train_end/val_start from the YAML config).
All 4 assets are concatenated into a shared ConcatDataset so the encoder learns
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from configs.splits import (
    TEST_START as _TEST_START,
    TRAIN_END_EXCLUSIVE,
    VAL_START as _VAL_START,
    TestSetContaminationError,
)
from encoder.dataset import PredictiveContrastiveDataset, RegimeContrastiveDataset
from encoder.embed import extract_embeddings
from encoder.loss import NTXentLoss, build_temporal_exclusion_mask
from encoder.model import TFTRegimeEncoder
from encoder.model_lastpool import TFTRegimeEncoderLastPool
from encoder.visualize import plot_training_curves, plot_tsne_embeddings
from utils.logger import logger
from utils.reproducibility import set_seed

_ROOT = Path(__file__).resolve().parent.parent


def _load_asset_split(
    asset: str,
    processed_dir: Path,
    train_end: str,
    val_start: str,
    window_size: int,
    step: int,
    augment_kwargs: dict | None = None,
    val_end: str | None = None,
    positive_pair_mode: str = "simclr",
    future_size: int = 24,
    asset_id: int = 0,
    return_metadata: bool = False,
) -> tuple[Dataset, Dataset]:
    df = pd.read_parquet(processed_dir / f"{asset}_aligned.parquet", engine="pyarrow")
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_start_ts = pd.Timestamp(val_start, tz="UTC")

    train_df = df[df.index < train_end_ts]
    # val_end is opt-in (None preserves the original unbounded-above behavior exactly).
    # Production relies on the Kaggle-uploaded parquet already being trimmed to
    # <= 2024-12-31 so "val >= val_start" never reaches 2025 in practice; a LOCAL run
    # against the full local aligned parquet (which extends to 2025-12-31) has no such
    # implicit trim, so val would silently include 2025 ("sacred" test data) without an
    # explicit val_end. Set explicitly whenever training against local data.
    if val_end is not None:
        val_end_ts = pd.Timestamp(val_end, tz="UTC")
        val_df = df[(df.index >= val_start_ts) & (df.index <= val_end_ts)]
    else:
        val_df = df[df.index >= val_start_ts]

    # Contamination guard (P0.2): verify the *resulting* slice actually respects the
    # sacred test boundary, not just that the slicing above was written correctly — this
    # is exactly the local-run footgun described above (missing val_end silently pulling
    # in 2025), turned from a comment into an enforced invariant.
    test_start_ts = pd.Timestamp(_TEST_START, tz="UTC")
    latest = max(
        (train_df.index.max() if not train_df.empty else pd.Timestamp.min.tz_localize("UTC")),
        (val_df.index.max() if not val_df.empty else pd.Timestamp.min.tz_localize("UTC")),
    )
    if latest >= test_start_ts:
        raise TestSetContaminationError(
            f"{asset}: encoder train/val input reaches {latest}, which is >= TEST_START "
            f"({_TEST_START}). The sacred 2025 test set must never reach encoder training "
            f"or validation — pass val_end explicitly (see configs/splits.VAL_END_EXCLUSIVE) "
            f"when training against local data that extends past the val window (the "
            f"Kaggle upload is pre-trimmed and unaffected)."
        )

    # Same augment_kwargs on train AND val — a fair NT-Xent val_loss comparison requires
    # val augmentation to match whatever the model was actually trained against.
    #
    # positive_pair_mode is opt-in ("simclr" default preserves RegimeContrastiveDataset
    # exactly — same class, same call signature as before this option existed). "predictive"
    # selects PredictiveContrastiveDataset (2026-08-12 objective experiment) — see its
    # docstring for the leakage argument (future window is bounded by this same already-
    # split df, so it inherits the guard above without needing a separate check).
    if positive_pair_mode == "predictive":
        train_ds: Dataset = PredictiveContrastiveDataset(
            train_df, window_size=window_size, future_size=future_size, step=step,
            augment_kwargs=augment_kwargs, asset_id=asset_id, return_metadata=return_metadata,
        )
        val_ds: Dataset = PredictiveContrastiveDataset(
            val_df, window_size=window_size, future_size=future_size, step=step,
            augment_kwargs=augment_kwargs, asset_id=asset_id, return_metadata=return_metadata,
        )
    elif positive_pair_mode == "simclr":
        train_ds = RegimeContrastiveDataset(
            train_df, window_size=window_size, step=step, augment_kwargs=augment_kwargs
        )
        val_ds = RegimeContrastiveDataset(
            val_df, window_size=window_size, step=step, augment_kwargs=augment_kwargs
        )
    else:
        raise ValueError(f"Unknown positive_pair_mode: {positive_pair_mode!r} (expected 'simclr' or 'predictive')")
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
    negative_exclusion_radius: int | None = None,
) -> tuple[float, float, bool]:
    """Run one full pass over loader.  Returns (mean_loss, mean_acc, nan_grad_hit).

    negative_exclusion_radius is opt-in (None default preserves the original 2-tuple
    (anchor, positive) unpacking exactly). When set (2026-08-12, E1-NegCtrl), the loader
    is expected to yield 4-tuples (anchor, positive, asset_id, t) — see
    PredictiveContrastiveDataset's return_metadata flag — and a per-batch exclusion mask
    is built and passed to loss_fn, forbidding same-asset temporally-near pairs from
    counting as NT-Xent negatives. See encoder/loss.py's build_temporal_exclusion_mask.
    """
    model.train(train)
    nan_grad_hit = False
    losses: list[float] = []
    accs: list[float] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if negative_exclusion_radius is not None:
                anchor, positive, asset_id, t = batch
                asset_id = asset_id.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)
                exclude_mask = build_temporal_exclusion_mask(asset_id, t, negative_exclusion_radius)
            else:
                anchor, positive = batch
                exclude_mask = None
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)

            _, z_a = model(anchor)
            _, z_p = model(positive)
            loss, acc = loss_fn(z_a, z_p, exclude_mask=exclude_mask)

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
    # Canonical boundary from configs/splits.py — no longer read from tcfg (see P0.1:
    # this and encoder.yaml's own copy were one of four independent places the same
    # boundary was hardcoded, and the yaml copy went stale for a full retrain cycle).
    train_end: str = TRAIN_END_EXCLUSIVE
    val_start: str = _VAL_START
    # Opt-in (None default preserves the original unbounded-above val_df exactly) — see
    # _load_asset_split's docstring-comment for why this matters for local (non-Kaggle) runs.
    val_end: str | None = tcfg.get("val_end")

    # Opt-in only — absent from configs/encoder.yaml, so any run that doesn't set these
    # is byte-for-byte unaffected. seed=None (default) preserves the original fully
    # unseeded behavior. Note: augment.py's per-call np.random.default_rng() draws are
    # OS-entropy-seeded regardless (by design, for multi-worker DataLoader safety — see
    # its module docstring) and are NOT made reproducible by this seed; it controls model
    # init and DataLoader shuffle order only. Not changed here since that would touch
    # augmentation internals beyond the requested on/off toggle.
    seed = tcfg.get("seed")
    if seed is not None:
        set_seed(seed)
        logger.info(
            f"Seeded via utils.reproducibility.set_seed(seed={seed}) "
            f"(DataLoader shuffle order + model init)"
        )

    # Opt-in only — {} default reproduces the exact original augment_window(window) call
    # (no kwargs) for any run that doesn't set training.augment_kwargs.
    augment_kwargs: dict = dict(tcfg.get("augment_kwargs", {}))
    if augment_kwargs:
        logger.info(f"Non-default augment_kwargs: {augment_kwargs}")

    # Opt-in — "simclr" (default, absent from configs/encoder.yaml) preserves
    # RegimeContrastiveDataset exactly. "predictive" selects PredictiveContrastiveDataset
    # (2026-08-12 objective experiment — see encoder/dataset.py docstring).
    positive_pair_mode: str = str(tcfg.get("positive_pair_mode", "simclr"))
    future_size: int = int(tcfg.get("future_size", 24))
    if positive_pair_mode != "simclr":
        logger.info(f"Non-default positive_pair_mode: {positive_pair_mode!r} (future_size={future_size})")

    # Opt-in — None (default, absent from every config until E1-NegCtrl) preserves
    # today's exact behavior: 2-tuple batches, no exclusion mask. An int enables
    # temporal negative exclusion (2026-08-12) — requires positive_pair_mode="predictive"
    # (asset_id/t metadata is only meaningful for context/future pairs).
    negative_exclusion_radius: int | None = tcfg.get("negative_exclusion_radius")
    if negative_exclusion_radius is not None:
        if positive_pair_mode != "predictive":
            raise ValueError("negative_exclusion_radius requires positive_pair_mode='predictive'")
        logger.info(f"Temporal negative exclusion ENABLED, radius={negative_exclusion_radius} bars")

    processed_dir = _ROOT / dcfg["processed_dir"]
    ckpt_dir = _ROOT / ckpt_cfg["dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = _ROOT / "encoder" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # run_tag/save_periodic are opt-in (absent -> "" / False) — when unset, _tagged_name()
    # returns the base filename unchanged, so best_encoder.pt/final_encoder.pt are written
    # exactly as before and can never be overwritten by a tagged experiment run.
    run_tag: str = str(ckpt_cfg.get("run_tag", ""))
    save_periodic: bool = bool(ckpt_cfg.get("save_periodic", False))

    def _tagged_name(base_name: str) -> str:
        if not run_tag:
            return base_name
        return f"{Path(base_name).stem}_{run_tag}.pt"

    best_path = ckpt_dir / _tagged_name(ckpt_cfg["best"])
    final_path = ckpt_dir / _tagged_name(ckpt_cfg["final"])

    # -------------------------------------------------------------- data
    # asset_id assigned from THIS config's own asset list/order (dcfg["assets"]) —
    # deliberately not from rl.env.CANONICAL_ASSETS, see PredictiveContrastiveDataset's
    # asset_id docstring for the GOLD/XAUUSD naming-mismatch reason.
    return_metadata = negative_exclusion_radius is not None
    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []
    for asset_idx, asset in enumerate(dcfg["assets"]):
        tr_ds, va_ds = _load_asset_split(
            asset, processed_dir, train_end, val_start, window_size, step,
            augment_kwargs=augment_kwargs, val_end=val_end,
            positive_pair_mode=positive_pair_mode, future_size=future_size,
            asset_id=asset_idx, return_metadata=return_metadata,
        )
        if len(tr_ds) > 0:
            train_datasets.append(tr_ds)
        if len(va_ds) > 0:
            val_datasets.append(va_ds)
        logger.info(f"{asset}: {len(tr_ds)} train windows, {len(va_ds)} val windows")

    pin = device.type == "cuda"
    shuffle_generator = None
    if seed is not None:
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(seed)
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin,
    )

    # -------------------------------------------------------------- model
    # Opt-in — absent from configs/encoder.yaml (default "mean") preserves TFTRegimeEncoder
    # exactly. "last" selects TFTRegimeEncoderLastPool (encoder/model_lastpool.py, built for
    # the 2026-08-11 E1 inference-only ablation) — identical architecture (VSN, LSTM,
    # gated residual, projector), only the temporal pooling read-out differs.
    pooling: str = mcfg.get("pooling", "mean")
    model_cls = {"mean": TFTRegimeEncoder, "last": TFTRegimeEncoderLastPool}[pooling]
    if pooling != "mean":
        logger.info(f"Non-default pooling: {pooling!r} -> {model_cls.__name__}")
    model = model_cls(
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

    # Periodic checkpoints — early/25%/50%/75% of max_epochs — to reconstruct the training
    # trajectory (val InfoNCE / temperature / downstream metrics vs. epoch) without needing
    # a fresh training run per data point. Opt-in (save_periodic=False by default -> this
    # set is empty and the block below never fires, zero effect on existing runs). If
    # early stopping ends the run before a target epoch is reached, that point is simply
    # never saved — expected, not an error. "Early" = epoch 1.
    periodic_epochs: set[int] = set()
    if save_periodic:
        periodic_epochs = {max(1, round(max_epochs * f)) for f in (0.0, 0.25, 0.50, 0.75)}
        logger.info(f"save_periodic=True, run_tag={run_tag!r} — will save periodic checkpoints at epochs {sorted(periodic_epochs)}")

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc, nan_hit = _run_epoch(
            train_loader, model, loss_fn, optimizer, device, grad_clip, train=True,
            temp_bounds=temp_bounds, negative_exclusion_radius=negative_exclusion_radius,
        )
        if nan_hit:
            nan_grad_detected = True

        val_loss, val_acc, _ = _run_epoch(
            val_loader, model, loss_fn, None, device, grad_clip, train=False,
            negative_exclusion_radius=negative_exclusion_radius,
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

        if epoch in periodic_epochs:
            tag_suffix = f"_{run_tag}" if run_tag else ""
            periodic_path = ckpt_dir / f"periodic_encoder{tag_suffix}_epoch{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_top1_acc": val_acc,
                    "temperature": current_temp,
                    "model_state": model.state_dict(),
                    "loss_fn_state": loss_fn.state_dict(),
                },
                periodic_path,
            )
            logger.info(f"Periodic checkpoint saved -> {periodic_path.name}")

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
                best_path,
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
        final_path,
    )
    mlflow.log_artifact(str(best_path))
    mlflow.log_artifact(str(final_path))

    mlflow.log_params(
        {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "stopping_epoch": stopping_epoch,
        }
    )

    # -------------------------------------------------------------- load best for inference
    best_ckpt = torch.load(
        best_path, map_location=device, weights_only=True
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
    for path in (best_path, final_path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved["convergence_checks"] = checks
        saved["convergence_thresholds"] = dict(conv)
        torch.save(saved, path)

    # -------------------------------------------------------------- post-training artefacts
    run_id = mlflow.active_run().info.run_id

    logger.info("Extracting embeddings …")
    embedding_matrix, timestamps = extract_embeddings(model, config)

    logger.info("Rendering t-SNE …")
    plot_tag_suffix = f"_{run_tag}" if run_tag else ""
    tsne_path = plots_dir / f"tsne_embeddings{plot_tag_suffix}.png"
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
    curves_path = plots_dir / f"training_curves{plot_tag_suffix}.png"
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
