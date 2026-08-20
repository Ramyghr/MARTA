"""Builds an asset-panel's aligned features + raw-feature embeddings, in TWO variants
sharing one alignment pass:
  - data/processed/{tag}/              14-col (production feature set, 720h only)
  - data/processed/{tag}_multihorizon/ 22-col (14 + 90d/180d on vol-shape features)

Default (no args): the crypto-only 5-asset panel (BTC/ETH/BNB/LTC/XRP), which replaced
the BTC/ETH/EURUSD/GOLD panel for regime-representation testing (2026-08-19) after
finding EURUSD/GOLD's degenerate zero-volume features made the raw-feature HMM key on
asset identity instead of market regime. See pipeline/fetch_crypto5.py for the asset-
selection/data-availability rationale (XRP starts 2018-05-04, not 2018-01-01; SOL
excluded entirely — no data before 2020-08-11).

--assets/--tag generalize this to other panels, e.g. the BTC-only comparison
(--assets BTC_USDT --tag btc_only) requested to run alongside the 5-asset panel.

No encoder, no PCA — raw features straight to the HMM, matching the project's
raw-feature pivot (2026-08-12) and Phase 2's multihorizon design
(regime/build_multihorizon_features.py — same extra-horizon columns, same rationale).

Run:
  conda run -n marta python regime/build_crypto5_features.py
  conda run -n marta python regime/build_crypto5_features.py --assets BTC_USDT --tag btc_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.features import compute_features
from pipeline.normalize import FEATURE_COLS, normalize_features_multihorizon
from utils.logger import logger

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

_DEFAULT_ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]
_DEFAULT_TAG = "crypto5"
# BTC/ETH already have an 18-col features.parquet cached from the production pipeline —
# reuse it rather than recomputing from raw (identical result either way).
_HAS_CACHED_FEATURES = {"BTC_USDT", "ETH_USDT"}

# Set by main() from CLI args before any function below runs.
ASSETS = list(_DEFAULT_ASSETS)
BTC_KEY = "BTC_USDT"
OUT_14 = PROCESSED_DIR / _DEFAULT_TAG
OUT_22 = PROCESSED_DIR / f"{_DEFAULT_TAG}_multihorizon"


def _load_features(asset: str) -> pd.DataFrame:
    cached_path = PROCESSED_DIR / f"{asset}_features.parquet"
    if asset in _HAS_CACHED_FEATURES and cached_path.exists():
        logger.info(f"[{asset}] reusing cached {cached_path.name}")
        return pd.read_parquet(cached_path, engine="pyarrow")
    raw_path = RAW_DIR / f"{asset}_1h_raw.parquet"
    raw_df = pd.read_parquet(raw_path, engine="pyarrow")
    return compute_features(raw_df, asset)


def _build_and_save(normalized: dict[str, pd.DataFrame], cols: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    btc_index = normalized[BTC_KEY][cols].index
    common_index = btc_index
    for df in normalized.values():
        common_index = common_index.intersection(df[cols].index)
    overlap_pct = len(common_index) / len(btc_index) * 100
    logger.info(
        f"[{out_dir.name}] alignment: {len(common_index):,} common timestamps "
        f"({overlap_pct:.1f}% of BTC range)"
    )

    blocks: list[np.ndarray] = []
    ts_rows: list[pd.Timestamp] = []
    asset_map_rows: list[dict] = []
    cursor = 0
    for asset in ASSETS:
        df_aligned = normalized[asset][cols].reindex(common_index)
        assert not df_aligned.isnull().any().any(), f"[{asset}] NaN after alignment"
        df_aligned.to_parquet(out_dir / f"{asset}_aligned.parquet", engine="pyarrow")

        n = len(df_aligned)
        blocks.append(df_aligned.values.astype(np.float32))
        ts_rows.extend(df_aligned.index.tolist())
        asset_map_rows.append({"asset_name": asset, "start_idx": cursor, "end_idx": cursor + n})
        cursor += n
        logger.success(f"[{out_dir.name}/{asset}] aligned ({n:,} rows)")

    embeddings = np.concatenate(blocks, axis=0)
    assert embeddings.shape[1] == len(cols), f"expected {len(cols)} cols, got {embeddings.shape[1]}"
    assert not np.isnan(embeddings).any(), "NaN in feature matrix"

    np.save(out_dir / "embeddings.npy", embeddings)
    pd.DataFrame({"timestamp": ts_rows}).to_parquet(
        out_dir / "embedding_timestamps.parquet", engine="pyarrow", index=False
    )
    pd.DataFrame(asset_map_rows).to_parquet(
        out_dir / "embedding_asset_map.parquet", engine="pyarrow", index=False
    )
    logger.success(
        f"[{out_dir.name}] done: {embeddings.shape[0]:,} rows x {embeddings.shape[1]} cols -> {out_dir}"
    )


def main() -> None:
    global ASSETS, BTC_KEY, OUT_14, OUT_22

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets", type=str, default=None,
        help="Comma-separated logical asset ids (default: BTC_USDT,ETH_USDT,BNB_USDT,LTC_USDT,XRP_USDT)",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Output dir tag: data/processed/{tag}/ and {tag}_multihorizon/ (default: crypto5)",
    )
    args = parser.parse_args()

    if args.assets:
        ASSETS = args.assets.split(",")
        BTC_KEY = ASSETS[0]  # alignment backbone; named for the common (BTC=4-asset) case
    tag = args.tag or _DEFAULT_TAG
    OUT_14 = PROCESSED_DIR / tag
    OUT_22 = PROCESSED_DIR / f"{tag}_multihorizon"

    logger.info(f"assets={ASSETS}  tag={tag}  backbone={BTC_KEY}")

    with open(ROOT / "configs" / "assets.yaml") as f:
        config = yaml.safe_load(f)

    normalized: dict[str, pd.DataFrame] = {}
    for asset in ASSETS:
        feats = _load_features(asset)
        out = normalize_features_multihorizon(feats, config)  # 22-col; includes the 14 production cols
        normalized[asset] = out
        logger.info(
            f"[{asset}] {len(out):,} rows x {len(out.columns)} cols "
            f"[{out.index.min()} -> {out.index.max()}]"
        )

    ref_cols = list(normalized[BTC_KEY].columns)
    for asset, df in normalized.items():
        assert list(df.columns) == ref_cols, f"[{asset}] column mismatch: {df.columns.tolist()}"

    extra_cols = [c for c in ref_cols if c not in FEATURE_COLS]
    logger.info(f"14-col set: {FEATURE_COLS}")
    logger.info(f"extra multi-horizon cols: {extra_cols}")

    _build_and_save(normalized, FEATURE_COLS, OUT_14)
    _build_and_save(normalized, ref_cols, OUT_22)


if __name__ == "__main__":
    main()
