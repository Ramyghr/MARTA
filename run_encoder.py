"""
run_encoder.py
Master script for MARTA Week 2 — contrastive regime encoder pretraining.

Usage:
    conda run -n marta python run_encoder.py
    conda run -n marta python run_encoder.py --config configs/encoder_e2_nojittermask.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import yaml
from rich.console import Console
from rich.table import Table

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from encoder.train import train_encoder  # noqa: E402
from utils.logger import logger  # noqa: E402

console = Console()


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _print_summary(
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    stopping_epoch: int,
    embedding_shape: tuple[int, int],
    temperature: float,
    run_url: str,
) -> None:
    table = Table(
        title="MARTA Encoder Pretraining — Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="cyan", min_width=28)
    table.add_column("Value", justify="right")

    table.add_row("Final train loss", f"{train_loss:.4f}")
    table.add_row("Final val loss", f"{val_loss:.4f}")
    table.add_row("Best val loss", f"{best_val_loss:.4f}")
    table.add_row("Stopping epoch", str(stopping_epoch))
    table.add_row("Embedding matrix shape", f"{embedding_shape[0]:,} × {embedding_shape[1]}")
    table.add_row("Temperature at stop", f"{temperature:.4f}")
    table.add_row("MLflow run URL", run_url)

    console.print()
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="MARTA encoder contrastive pretraining")
    parser.add_argument(
        "--config", default="configs/encoder.yaml",
        help="Path to the encoder config yaml (default: configs/encoder.yaml, unchanged behavior)",
    )
    args = parser.parse_args()

    encoder_cfg_path = _ROOT / args.config
    config = _load_config(encoder_cfg_path)

    aligned_path = _ROOT / "data" / "processed" / "all_assets_aligned.parquet"
    if not aligned_path.exists():
        logger.error(
            "Aligned dataset not found. Run `python run_pipeline.py` first to build it."
        )
        sys.exit(1)

    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(str(_ROOT / "mlruns"))
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]) as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        logger.info(f"MLflow run started: {run_id}")

        mlflow.log_param("encoder_config", str(encoder_cfg_path))
        mlflow.log_params(
            {
                "window_size": config["training"]["window_size"],
                "batch_size": config["training"]["batch_size"],
                "lr": config["training"]["lr"],
                "max_epochs": config["training"]["max_epochs"],
                "hidden_dim": config["model"]["hidden_dim"],
                "proj_dim": config["model"]["proj_dim"],
            }
        )

        best_model = train_encoder(config)

    # ------------------------------------------------------------------ summary
    # Load persisted metrics for the summary table
    client = mlflow.tracking.MlflowClient()

    def _last(metric_name: str) -> float:
        hist = client.get_metric_history(run_id, metric_name)
        return hist[-1].value if hist else float("nan")

    train_loss = _last("train_loss")
    val_loss = _last("val_loss")
    temperature = _last("temperature")

    best_val_loss_param = client.get_run(run_id).data.params.get("best_val_loss", "N/A")
    stopping_epoch_param = client.get_run(run_id).data.params.get("stopping_epoch", "N/A")
    stopping_epoch = int(stopping_epoch_param) if stopping_epoch_param != "N/A" else -1

    embeddings_path = _ROOT / "data" / "processed" / "embeddings.npy"
    emb_shape: tuple[int, int] = (0, 128)
    if embeddings_path.exists():
        import numpy as np  # local import to keep top-level imports minimal
        arr = np.load(embeddings_path, mmap_mode="r")
        emb_shape = (arr.shape[0], arr.shape[1])

    run_url = (
        f"http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run_id}"
    )

    _print_summary(
        train_loss=train_loss,
        val_loss=val_loss,
        best_val_loss=float(best_val_loss_param) if best_val_loss_param != "N/A" else float("nan"),
        stopping_epoch=stopping_epoch,
        embedding_shape=emb_shape,
        temperature=temperature,
        run_url=run_url,
    )

    console.print(f"\n[bold green]MLflow UI:[/bold green] {run_url}")
    console.print(
        "[dim]Start the UI with:[/dim] [cyan]conda run -n marta mlflow ui[/cyan]"
    )


if __name__ == "__main__":
    main()
