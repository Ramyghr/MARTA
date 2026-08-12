"""regime/n_states_sweep.py

Empirical n_states selection for the regime HMM. BIC alone is unreliable here — at this
training-set scale (136K rows) the log-likelihood term dwarfs the complexity penalty, so
any relative-percentage flat-curve threshold degenerates to picking the smallest n in the
search grid (see regime/hmm.py's fit_hmm_bic fix, 2026-08-08). This script instead fits
each candidate n_states on the real data and reports the two structural symptoms found by
regime/regime_quality_audit.py to be decisive: near-duplicate states (state-mean distance)
and single-bar-dwell "flicker" states — the fast, cheap proxies for the full 5-part audit.

Run: python regime/n_states_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM
from scipy.spatial.distance import pdist, squareform

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.splits import TRAIN_END_EXCLUSIVE  # noqa: E402
from regime.hmm import fit_pca  # noqa: E402
from utils.logger import logger  # noqa: E402

_CANDIDATES = [3, 4, 5, 6]


def main() -> None:
    config = yaml.safe_load(open(_ROOT / "configs" / "assets.yaml"))
    processed_dir = _ROOT / config["data"]["processed_dir"]

    embeddings = np.load(processed_dir / "embeddings.npy")
    timestamps = pd.read_parquet(processed_dir / "embedding_timestamps.parquet")
    if "asset" not in timestamps.columns:
        asset_map = pd.read_parquet(processed_dir / "embedding_asset_map.parquet")
        alias = {"XAUUSD": "GOLD"}
        asset_col = np.empty(len(timestamps), dtype=object)
        for _, row in asset_map.iterrows():
            name = alias.get(row["asset_name"], row["asset_name"])
            asset_col[int(row["start_idx"]):int(row["end_idx"])] = name
        timestamps = timestamps.copy()
        timestamps["asset"] = asset_col

    train_cutoff = pd.Timestamp(TRAIN_END_EXCLUSIVE, tz="UTC")
    ts_series = pd.to_datetime(timestamps["timestamp"], utc=True)
    train_mask = (ts_series < train_cutoff).values

    mlflow.set_tracking_uri(str(_ROOT / "mlruns"))
    mlflow.set_experiment("marta_regime_hmm")
    mlflow.start_run(run_name="n_states_sweep")

    pca, emb_reduced = fit_pca(embeddings, train_mask, n_components=16)

    for n in _CANDIDATES:
        model = GaussianHMM(
            n_components=n, covariance_type="diag", n_iter=200, random_state=42, tol=1e-4,
        )
        model.fit(emb_reduced[train_mask])

        dist = squareform(pdist(model.means_))
        np.fill_diagonal(dist, np.inf)
        min_dist = float(dist.min())
        min_pair = np.unravel_index(np.argmin(dist), dist.shape)

        labels = model.predict(emb_reduced)
        # dwell time on the full stream (one asset at a time so we don't blend
        # cross-asset boundaries into the run-length encoding)
        dwell_by_state: dict[int, list[int]] = {s: [] for s in range(n)}
        for asset in timestamps["asset"].unique():
            asset_mask = (timestamps["asset"] == asset).values
            asset_labels = labels[asset_mask]
            if len(asset_labels) == 0:
                continue
            cur, cur_len = asset_labels[0], 1
            for lab in asset_labels[1:]:
                if lab == cur:
                    cur_len += 1
                else:
                    dwell_by_state[int(cur)].append(cur_len)
                    cur, cur_len = lab, 1
            dwell_by_state[int(cur)].append(cur_len)

        mean_dwell = {s: (float(np.mean(v)) if v else 0.0) for s, v in dwell_by_state.items()}
        flicker_states = [s for s, d in mean_dwell.items() if d < 2.0]
        dist_pct = model.transmat_.diagonal()

        logger.info(
            f"n_states={n}: min inter-state distance={min_dist:.4f} (states "
            f"{min_pair[0]}/{min_pair[1]}) | mean dwell/state={ {s: round(d,1) for s,d in mean_dwell.items()} } "
            f"| flicker states (dwell<2h)={flicker_states} | self-trans prob={np.round(dist_pct,3).tolist()}"
        )

    mlflow.end_run()


if __name__ == "__main__":
    main()
