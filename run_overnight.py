#!/usr/bin/env python
"""
MARTA Overnight Pipeline — Full retrain with 2024 data extension.

Run:  nohup /home/ramy/miniconda3/envs/marta/bin/python run_overnight.py \
          > /tmp/overnight.log 2>&1 &

Expected runtime: 6-8 hours (encoder retrains on Kaggle separately — see
kaggle_retrain_instructions.md).

Dependency note (IMPORTANT): Steps 3+ embed the aligned data with
encoder/checkpoints/best_encoder.pt. If the Kaggle retrain has not finished and the
checkpoint has not been replaced, the regime stream / RL policy are built on the OLD
encoder and Steps 3-9 must be re-run after swapping the checkpoint in (the checklist
covers this). The script prints the checkpoint's mtime before Step 3 so the morning
log shows which encoder was used.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON = "/home/ramy/miniconda3/envs/marta/bin/python"
ENCODER_CKPT = Path("encoder/checkpoints/best_encoder.pt")


def run_step(name: str, cmd: list[str], timeout: int = 3600) -> None:
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 60}", flush=True)

    try:
        # stdout/stderr stream directly to the log — no capture.
        result = subprocess.run(cmd, capture_output=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"\n❌ TIMED OUT: {name} after {timeout}s")
        print("Stopping pipeline. Fix the error and resume manually.")
        sys.exit(1)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {name} (exit code {result.returncode})")
        print("Stopping pipeline. Fix the error and resume manually.")
        sys.exit(1)

    print(f"✓ COMPLETED: {name} at {datetime.now().strftime('%H:%M:%S')}", flush=True)
    time.sleep(2)


steps: list[tuple[str, list[str], int]] = [
    # Step 1: Fetch 2024 data extension (idempotent — skips assets already extended)
    ("Fetch 2024 data", [PYTHON, "pipeline/fetch_2024.py"], 1800),

    # Step 2: Re-run full pipeline (features + normalize + align + validate)
    ("Data pipeline", [PYTHON, "run_pipeline.py"], 1800),

    # Step 3: Refit HMM on new data (encoder retraining on Kaggle separately)
    # NOTE: encoder must be retrained on Kaggle first and best_encoder.pt replaced.
    # This step uses whatever checkpoint is on disk (its mtime is printed below).
    ("HMM regime clustering", [PYTHON, "run_regime.py"], 1800),

    # Step 4: Regenerate weekly debates on the new regime stream (2023 val window)
    ("Weekly debates", [PYTHON, "run_weekly_debates.py"], 7200),

    # Step 5: Regenerate clean no-regime debates (re-runs pairs from the main cache)
    ("Clean no-regime debates", [PYTHON, "run_clean_ablation_debates.py"], 7200),

    # Step 6: RL training on new data (train ≤2022, val 2023 — rl/env.py).
    # 6h cap: 900k curriculum steps + GARCH cache rebuild for the extended data;
    # a mid-training kill wastes the night, so this is deliberately generous.
    ("RL training", [PYTHON, "run_rl.py"], 21600),

    # Step 7: Test set evaluation on 2024
    ("Test set evaluation", [PYTHON, "run_test_set.py"], 600),

    # Step 8: Final ablation (7 variants incl. RSI / MA crossover)
    ("Final ablation", [PYTHON, "run_rl.py", "--eval-only"], 1800),
]


def main() -> None:
    print("MARTA Overnight Pipeline")
    print(f"Started: {datetime.now()}")
    print(f"Steps: {len(steps)}")
    print("Estimated time: 6-8 hours (plus Kaggle encoder retrain)")
    print()
    print("⚠ IMPORTANT: Encoder retraining on Kaggle must complete")
    print("  and best_encoder.pt must be replaced BEFORE Step 3.")
    print("  The script will use the existing encoder if not replaced.")
    if ENCODER_CKPT.exists():
        mtime = datetime.fromtimestamp(ENCODER_CKPT.stat().st_mtime)
        print(f"  Current best_encoder.pt last modified: {mtime}")
    else:
        print("  ⚠ best_encoder.pt NOT FOUND — Steps 3+ will fail without it.")
    print(flush=True)

    for name, cmd, timeout in steps:
        run_step(name, cmd, timeout)

    print(f"\n{'=' * 60}")
    print(f"✓ PIPELINE COMPLETE: {datetime.now()}")
    print("Check data/results/ for final metrics")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
