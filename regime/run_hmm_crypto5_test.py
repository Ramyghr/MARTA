"""Regime-layer audit (2026-08-19) — crypto-only 5-asset panel (BTC/ETH/BNB/LTC/XRP),
replacing the earlier BTC/ETH/EURUSD/GOLD-panel comparison
(regime/run_hmm_multihorizon_test.py — kept as a historical record of why that panel
was abandoned: EURUSD/GOLD's degenerate zero-volume features made the raw-feature HMM
key on asset identity, confirmed both via crosstab and a literal `ValueError: Invalid
input, x is constant` from the stationarity check).

Same controlled, single-variable design: BOTH fits use the Phase-1-fixed engine
(lengths=, no BIC subsampling, best-of-5-seed refit), pca_n_components=None (raw
features, no PCA/whitening), and the SAME force_n_states=6 — only the 8 extra
multi-horizon columns differ between them.

  A. "crypto5"              — 14-col (data/processed/crypto5/)
  B. "crypto5_multihorizon" — 22-col (data/processed/crypto5_multihorizon/)

Also runs the asset-collapse check (per-state asset composition) that caught the
EURUSD/GOLD panel's failure — the primary thing this new panel needs to clear before
either variant is trusted. (For a single-asset panel — e.g. --tag btc_only — this
check is trivially 1.0 and not meaningful; it's about whether an HMM state locks onto
one asset out of several, which can't happen with only one asset in the first place.)

--tag generalizes VARIANTS to any panel built by regime/build_crypto5_features.py's
--tag option, e.g. --tag btc_only for the BTC-only comparison run alongside this one.

Run:
  conda run -n marta python regime/run_hmm_crypto5_test.py
  conda run -n marta python regime/run_hmm_crypto5_test.py --tag btc_only
"""

from __future__ import annotations

import argparse
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

FORCE_N_STATES = 6  # matches the (now-abandoned) EURUSD/GOLD-panel raw_baseline's BIC pick, kept fixed for control


def _diagnostics(regime_df: pd.DataFrame, n_states: int) -> dict:
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
    max_single_asset_share = float(crosstab.max().max())
    return {
        "dwell_by_asset": out,
        "asset_share_per_state": crosstab.round(3).to_dict(),
        "max_single_asset_share_of_any_state": max_single_asset_share,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag", type=str, default="crypto5",
        help="Base tag matching regime/build_crypto5_features.py's --tag "
             "(reads data/processed/{tag}/ and {tag}_multihorizon/). Default: crypto5.",
    )
    args = parser.parse_args()
    tag = args.tag

    variants = {
        tag: {
            "processed_dir": f"data/processed/{tag}",
            "checkpoint_dir": ROOT / "regime" / "checkpoints" / tag,
        },
        f"{tag}_multihorizon": {
            "processed_dir": f"data/processed/{tag}_multihorizon",
            "checkpoint_dir": ROOT / "regime" / "checkpoints" / f"{tag}_multihorizon",
        },
    }

    results: dict[str, dict] = {}

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment("marta_regime_hmm")

    for name, paths in variants.items():
        embeddings_path = ROOT / paths["processed_dir"] / "embeddings.npy"
        if not embeddings_path.exists():
            print(f"ERROR: {embeddings_path} not found — run regime/build_crypto5_features.py first.")
            sys.exit(1)

        with open(ROOT / "configs" / "assets.yaml") as f:
            config = yaml.safe_load(f)
        config["data"]["processed_dir"] = paths["processed_dir"]

        print(f"\n=== {name}: force_n_states={FORCE_N_STATES}, pca=None, 5-seed best-of-N ===")
        with mlflow.start_run(run_name=f"phase3_{name}_n{FORCE_N_STATES}"):
            mlflow.log_param("experiment_tag", f"phase3_{tag}")
            mlflow.log_param("variant", name)
            result = run_hmm_pipeline(
                config,
                force_n_states=FORCE_N_STATES,
                checkpoint_dir=paths["checkpoint_dir"],
                pca_n_components=None,
                n_seeds=5,
            )
            diag = _diagnostics(result["regime_df"], result["best_n"])
            mlflow.log_metric("max_single_asset_share_of_any_state", diag["max_single_asset_share_of_any_state"])
        results[name] = {"seed_sensitivity": result["seed_sensitivity"], **diag}
        logger.success(
            f"[{name}] mean_ari={result['seed_sensitivity']['mean_ari']:.3f}  "
            f"max_single_asset_share={diag['max_single_asset_share_of_any_state']:.3f}"
        )

    print("\n\n=== COMPARISON ===")
    for name, r in results.items():
        print(f"\n--- {name} ---")
        print(f"seed sensitivity: mean_ari={r['seed_sensitivity']['mean_ari']:.3f}  "
              f"min_ari={r['seed_sensitivity']['min_ari']:.3f}  "
              f"mean_nmi={r['seed_sensitivity']['mean_nmi']:.3f}")
        print(f"max single-asset share of any state: {r['max_single_asset_share_of_any_state']:.3f} "
              f"(>0.7 would be an asset-collapse red flag, cf. EURUSD/GOLD panel's ~0.97-0.985)")
        print("dwell time (hours) by asset x state:")
        for asset, dwell in r["dwell_by_asset"].items():
            print(f"  {asset}: {dwell}")
        print("asset share per state (column-normalized):")
        for state, shares in r["asset_share_per_state"].items():
            print(f"  state {state}: {shares}")


if __name__ == "__main__":
    main()
