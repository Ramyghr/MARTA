"""
run_pipeline.py
Master orchestration script for the MARTA Week 1 data pipeline.
Run with: python run_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlflow
import yaml
from rich.console import Console
from rich.table import Table

# Ensure the project root is on sys.path when run from any working directory
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.align import align_assets
from pipeline.features import compute_features
from pipeline.normalize import normalize_and_save
from pipeline.fetch import fetch_all_assets
from pipeline.validate import validate_pipeline
from utils.logger import logger

console = Console()


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _safe_symbol_name(symbol: str) -> str:
    return symbol.replace("/", "_")


def _print_summary(aligned_assets: dict, validated: bool) -> None:
    status = "[green]PASSED[/green]" if validated else "[red]FAILED[/red]"
    table = Table(title=f"MARTA Pipeline Summary — Validation: {status}", show_header=True)
    table.add_column("Asset", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Start", style="dim")
    table.add_column("End", style="dim")
    table.add_column("Features", justify="right")

    for symbol, df in aligned_assets.items():
        table.add_row(
            symbol,
            f"{len(df):,}",
            str(df.index.min())[:16] if not df.empty else "—",
            str(df.index.max())[:16] if not df.empty else "—",
            str(len(df.columns)),
        )

    console.print(table)


def main() -> None:
    config_path = _ROOT / "configs" / "assets.yaml"
    config = _load_config(config_path)

    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(str(_ROOT / "mlruns"))
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    pipeline_start = time.time()
    completed: dict = {}

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]) as run:
        logger.info(f"MLflow run started: {run.info.run_id}")
        mlflow.log_param("config_path", str(config_path))

        try:
            # ----------------------------------------------------------------
            # Stage 1 — Fetch
            # ----------------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STAGE 1: Fetching raw OHLCV data")
            logger.info("=" * 60)
            t0 = time.time()
            raw_assets = fetch_all_assets(config)
            mlflow.log_metric("fetch_duration_s", time.time() - t0)
            mlflow.log_metric("assets_fetched", len(raw_assets))
            completed["raw"] = raw_assets

            if not raw_assets:
                logger.error("No assets fetched. Aborting.")
                sys.exit(1)

            # ----------------------------------------------------------------
            # Stage 2 — Feature engineering
            # ----------------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STAGE 2: Computing technical indicators")
            logger.info("=" * 60)
            t0 = time.time()
            featured_assets: dict = {}
            for symbol, df in raw_assets.items():
                try:
                    featured_assets[symbol] = compute_features(df, symbol)
                except Exception as exc:
                    logger.error(f"[{symbol}] Feature computation failed: {exc}")
            mlflow.log_metric("feature_duration_s", time.time() - t0)
            mlflow.log_metric("assets_featured", len(featured_assets))
            completed["featured"] = featured_assets

            # ----------------------------------------------------------------
            # Stage 3 — Normalization
            # ----------------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STAGE 3: Rolling z-score normalization")
            logger.info("=" * 60)
            t0 = time.time()
            normalized_assets: dict = {}
            for symbol, df in featured_assets.items():
                try:
                    normalized_assets[symbol] = normalize_and_save(df, symbol, config)
                except Exception as exc:
                    logger.error(f"[{symbol}] Normalization failed: {exc}")
            mlflow.log_metric("normalize_duration_s", time.time() - t0)
            mlflow.log_metric("assets_normalized", len(normalized_assets))
            completed["normalized"] = normalized_assets

            # ----------------------------------------------------------------
            # Stage 4 — Alignment
            # ----------------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STAGE 4: Aligning assets to common index")
            logger.info("=" * 60)
            t0 = time.time()
            aligned_assets = align_assets(normalized_assets, config)
            mlflow.log_metric("align_duration_s", time.time() - t0)
            mlflow.log_metric("assets_aligned", len(aligned_assets))
            completed["aligned"] = aligned_assets

            # ----------------------------------------------------------------
            # Stage 5 — Validation
            # ----------------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STAGE 5: Validating pipeline output")
            logger.info("=" * 60)
            t0 = time.time()
            validated = validate_pipeline(aligned_assets, raw_assets, config)
            mlflow.log_metric("validate_duration_s", time.time() - t0)
            mlflow.log_param("validation_passed", str(validated))

            # ----------------------------------------------------------------
            # Final summary
            # ----------------------------------------------------------------
            total_s = time.time() - pipeline_start
            mlflow.log_metric("total_pipeline_duration_s", total_s)

            _print_summary(aligned_assets, validated)
            logger.info(f"Total pipeline time: {total_s:.1f}s")

            # Save summary JSON as artifact
            summary = {
                "run_id": run.info.run_id,
                "validation_passed": validated,
                "total_duration_s": round(total_s, 2),
                "assets": {
                    sym: {
                        "rows": len(df),
                        "features": len(df.columns),
                        "start": str(df.index.min()),
                        "end": str(df.index.max()),
                    }
                    for sym, df in aligned_assets.items()
                },
            }
            summary_path = Path("data/validated/pipeline_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            mlflow.log_artifact(str(summary_path))

            if not validated:
                report_path = Path("data/validated/validation_report.json")
                if report_path.exists():
                    with open(report_path) as f:
                        report = json.load(f)
                    console.rule("[red]CRITICAL FAILURES[/red]")
                    for fail in report.get("critical_failures", []):
                        console.print(f"  [red]✗[/red] {fail['check']}: {fail.get('detail', '')}")
                sys.exit(1)

        except KeyboardInterrupt:
            elapsed = time.time() - pipeline_start
            logger.warning(f"Pipeline interrupted after {elapsed:.1f}s. Partial results saved.")
            mlflow.log_metric("total_pipeline_duration_s", elapsed)
            mlflow.log_param("interrupted", "true")
            completed_stages = list(completed.keys())
            logger.info(f"Completed stages before interrupt: {completed_stages}")
            sys.exit(130)


if __name__ == "__main__":
    main()
