"""Closes the Phase 3 gap from the regime-layer audit (2026-08-19): the crypto-only
panels (crypto5, btc_only — built by regime/build_crypto5_features.py) were fetched,
aligned, and sanity-checked ad hoc, but pipeline/validate.py's actual structural/range/
gap/distribution/cross-asset validation suite was never run against them, as the
original phase plan specified.

Runs pipeline.validate.validate_pipeline() against each panel's RAW (data/raw/) and
ALIGNED 14-col (data/processed/{tag}/) data, writing to isolated
data/validated/validation_report_{tag}.json files — never touches the sacred
production data/validated/validation_report.json.

Run: conda run -n marta python regime/validate_crypto_panels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.validate import validate_pipeline
from utils.logger import logger

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PANELS = {
    "crypto5": ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"],
    "btc_only": ["BTC_USDT"],
}


def main() -> None:
    with open(ROOT / "configs" / "assets.yaml") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment("marta_data_pipeline")

    results: dict[str, bool] = {}
    for tag, assets in PANELS.items():
        raw_assets = {a: pd.read_parquet(RAW_DIR / f"{a}_1h_raw.parquet", engine="pyarrow") for a in assets}
        aligned_assets = {
            a: pd.read_parquet(PROCESSED_DIR / tag / f"{a}_aligned.parquet", engine="pyarrow") for a in assets
        }

        print(f"\n=== validating panel: {tag} ({assets}) ===")
        with mlflow.start_run(run_name=f"validate_{tag}"):
            mlflow.log_param("panel", tag)
            passed = validate_pipeline(
                aligned_assets, raw_assets, config,
                output_filename=f"validation_report_{tag}.json",
            )
        results[tag] = passed
        logger.success(f"[{tag}] validate_pipeline: {'PASSED' if passed else 'FAILED'}")

    print("\n\n=== SUMMARY ===")
    for tag, passed in results.items():
        print(f"{tag}: {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
