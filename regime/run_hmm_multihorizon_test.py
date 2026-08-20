"""Phase 2 of the regime-layer audit (2026-08-19): does adding 90d/180d rolling
z-score horizons (on top of production's 720h/30d one) lengthen HMM regime dwell
times, i.e. was the 720h window silently capping how long a "regime" can persist?

Controlled, single-variable comparison — BOTH fits below use the Phase-1-fixed engine
(lengths=, no BIC subsampling, best-of-5-seed refit) and the SAME force_n_states, so
the only thing that differs between them is the extra 8 multi-horizon columns:

  A. "raw_baseline_fixed" — the original 14-dim raw-feature set
     (data/processed/raw_baseline/), refit through the fixed engine. The existing
     regime/checkpoints/raw_baseline/hmm_model.pkl was fit with the OLD buggy engine
     (2026-08-12, subsampling + no lengths=) — comparing a NEW multihorizon fit against
     that STALE artifact would confound "horizon effect" with "engine fix effect", so
     this refit exists purely to give the multihorizon run a same-engine baseline.
  B. "multihorizon" — the 22-dim set (data/processed/multihorizon/, see
     regime/build_multihorizon_features.py), same asset panel (BTC/ETH/EURUSD/GOLD),
     same force_n_states.

Both use pca_n_components=None (raw features straight to the HMM — no PCA/whitening),
matching the raw-feature baseline's own design, and n_states is FORCED at 6 to match
raw_baseline's original BIC-selected value rather than re-running BIC search, so
n_states itself isn't a second free variable in this comparison.

Run: conda run -n marta python regime/run_hmm_multihorizon_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from regime.hmm import run_hmm_pipeline
from utils.logger import logger

FORCE_N_STATES = 6  # matches raw_baseline's original BIC-selected n_states

VARIANTS = {
    "raw_baseline_fixed": {
        "processed_dir": "data/processed/raw_baseline",
        "checkpoint_dir": ROOT / "regime" / "checkpoints" / "raw_baseline_fixed_engine",
    },
    "multihorizon": {
        "processed_dir": "data/processed/multihorizon",
        "checkpoint_dir": ROOT / "regime" / "checkpoints" / "multihorizon",
    },
}


def _dwell_and_crosstab(regime_df: pd.DataFrame, n_states: int) -> dict:
    """Per-asset mean dwell time (run-length) + asset-crosstab occupancy, same
    methodology as the regime-layer audit's earlier ad hoc checks — reused here so
    variant A and B are scored identically."""
    out: dict = {}
    for asset in regime_df["asset"].unique():
        sub = regime_df[regime_df["asset"] == asset].sort_values("timestamp")
        labels = sub["regime_label"].to_numpy()
        dwell: dict[int, list[int]] = {s: [] for s in range(n_states)}
        cur, cur_len = labels[0], 1
        for lab in labels[1:]:
            if lab == cur:
                cur_len += 1
            else:
                dwell[int(cur)].append(cur_len)
                cur, cur_len = lab, 1
        dwell[int(cur)].append(cur_len)
        out[asset] = {s: (round(float(np.mean(v)), 1) if v else 0.0) for s, v in dwell.items()}

    crosstab = pd.crosstab(regime_df["asset"], regime_df["regime_label"], normalize="columns")
    return {"dwell_by_asset": out, "asset_share_per_state": crosstab.round(3).to_dict()}


def main() -> None:
    results: dict[str, dict] = {}

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment("marta_regime_hmm")

    for name, paths in VARIANTS.items():
        embeddings_path = ROOT / paths["processed_dir"] / "embeddings.npy"
        if not embeddings_path.exists():
            print(f"ERROR: {embeddings_path} not found — build it first "
                  f"(regime/build_raw_feature_embeddings.py or regime/build_multihorizon_features.py).")
            sys.exit(1)

        with open(ROOT / "configs" / "assets.yaml") as f:
            config = yaml.safe_load(f)
        config["data"]["processed_dir"] = paths["processed_dir"]

        print(f"\n=== {name}: force_n_states={FORCE_N_STATES}, pca=None, 5-seed best-of-N ===")
        with mlflow.start_run(run_name=f"phase2_{name}_n{FORCE_N_STATES}"):
            mlflow.log_param("experiment_tag", "phase2_multihorizon")
            mlflow.log_param("variant", name)
            result = run_hmm_pipeline(
                config,
                force_n_states=FORCE_N_STATES,
                checkpoint_dir=paths["checkpoint_dir"],
                pca_n_components=None,
                n_seeds=5,
            )
        diag = _dwell_and_crosstab(result["regime_df"], result["best_n"])
        results[name] = {
            "seed_sensitivity": result["seed_sensitivity"],
            **diag,
        }
        logger.success(
            f"[{name}] mean_ari={result['seed_sensitivity']['mean_ari']:.3f}  "
            f"min_ari={result['seed_sensitivity']['min_ari']:.3f}"
        )

    print("\n\n=== COMPARISON ===")
    for name, r in results.items():
        print(f"\n--- {name} ---")
        print(f"seed sensitivity: mean_ari={r['seed_sensitivity']['mean_ari']:.3f}  "
              f"min_ari={r['seed_sensitivity']['min_ari']:.3f}  "
              f"mean_nmi={r['seed_sensitivity']['mean_nmi']:.3f}")
        print("dwell time (hours) by asset x state:")
        for asset, dwell in r["dwell_by_asset"].items():
            print(f"  {asset}: {dwell}")
        print("asset share per state (column-normalized):")
        for state, shares in r["asset_share_per_state"].items():
            print(f"  state {state}: {shares}")


if __name__ == "__main__":
    main()
