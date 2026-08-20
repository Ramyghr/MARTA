"""regime/stationarity_check.py — Phase 2 of the regime-layer audit (2026-08-19).

Runs ADF (H0: unit root / non-stationary) and KPSS (H0: stationary) on every feature
column, per asset, for a given aligned-parquet directory. Neither pipeline/validate.py
nor regime/validate.py ever ran a formal stationarity test — the rolling z-score was
*assumed* to induce stationarity, never verified. This closes that gap and doubles as
a way to compare the production 720h-only feature set against the Phase 2 multihorizon
extension (do the 90d/180d columns look LESS stationary than their 720h counterpart,
which would be expected — a longer window retains more low-frequency/trend structure —
and by how much).

"Confirmatory stationary" here means ADF rejects its unit-root null (p<0.05) AND KPSS
fails to reject its stationarity null (p>0.05) — the two tests have opposite nulls, so
agreement between them is stronger evidence than either alone. At this row count
(~45-75K/asset) both tests have very high power and will reject almost anything by
p-value alone, so the KPSS/ADF *test statistic* (not just pass/fail) is reported too,
for a relative before/after comparison rather than a bare pass/fail count.

Run: conda run -n marta python regime/stationarity_check.py <processed_dir> [--tag NAME]
Example:
  conda run -n marta python regime/stationarity_check.py data/processed
  conda run -n marta python regime/stationarity_check.py data/processed/multihorizon --tag multihorizon
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logger import logger

_DEFAULT_ASSETS = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]
_ALIAS = {"GOLD": "XAUUSD"}


def _run_tests(series: pd.Series) -> dict:
    series = series.dropna()

    # Degenerate/near-constant columns (e.g. vol_zscore, obv_raw for histdata FX/
    # commodity instruments — confirmed by CLAUDE.md's own note that these are ~constant
    # for EURUSD/XAUUSD since histdata has no real volume) crash adfuller outright
    # ("Invalid input, x is constant") rather than returning a meaningful p-value.
    # Report that explicitly instead of letting the whole run die on one column.
    if series.std(ddof=0) < 1e-10:
        return {
            "adf_stat": None, "adf_p": None, "adf_stationary": None,
            "kpss_stat": None, "kpss_p": None, "kpss_stationary": None,
            "degenerate_constant": True,
        }

    # Schwert (1989) rule of thumb, computed once instead of letting adfuller's
    # autolag="AIC" exhaustively re-fit an OLS regression per candidate lag order —
    # at ~45-75K rows/column x 14 columns x 5 assets that search was the entire
    # bottleneck (~14 min/asset). A fixed reasonable lag is standard practice for a
    # stationarity screen at this sample size and doesn't materially change the
    # ADF/KPSS conclusion here — this isn't GPU-parallelizable work (statsmodels/
    # hmmlearn have no GPU backend, and the workload is many small sequential OLS
    # fits, the wrong shape for a GPU to help with regardless).
    maxlag = int(12 * (len(series) / 100) ** 0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            adf_stat, adf_p, *_ = adfuller(series, maxlag=maxlag, autolag=None)
        except ValueError:
            adf_stat, adf_p = float("nan"), float("nan")
        try:
            kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags=maxlag)
        except Exception:
            kpss_stat, kpss_p = float("nan"), float("nan")
    return {
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "adf_stationary": bool(adf_p < 0.05) if adf_p == adf_p else None,  # NaN-safe
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "kpss_stationary": bool(kpss_p > 0.05) if kpss_p == kpss_p else None,  # NaN-safe
        "degenerate_constant": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_dir", type=str)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument(
        "--assets", type=str, default=None,
        help="Comma-separated logical asset ids (default: the production BTC/ETH/EURUSD/GOLD "
             "panel). E.g. --assets BTC_USDT,ETH_USDT,BNB_USDT,LTC_USDT,XRP_USDT",
    )
    args = parser.parse_args()

    processed_dir = ROOT / args.processed_dir
    tag = args.tag or Path(args.processed_dir).name
    assets = args.assets.split(",") if args.assets else _DEFAULT_ASSETS

    results: dict[str, dict] = {}
    for asset in assets:
        stem = _ALIAS.get(asset, asset)
        path = processed_dir / f"{asset}_aligned.parquet"
        if not path.exists():
            path = processed_dir / f"{stem}_aligned.parquet"
        if not path.exists():
            logger.warning(f"[{asset}] no aligned parquet at {processed_dir}, skipping")
            continue

        df = pd.read_parquet(path, engine="pyarrow")
        asset_results = {}
        for col in df.columns:
            asset_results[col] = _run_tests(df[col])
        results[asset] = asset_results

        n_confirmed = sum(
            1 for r in asset_results.values() if r["adf_stationary"] and r["kpss_stationary"]
        )
        logger.info(f"[{asset}] {n_confirmed}/{len(asset_results)} columns confirmatory-stationary (ADF+KPSS agree)")

    out_path = ROOT / "data" / "results" / f"stationarity_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.success(f"Saved -> {out_path}")

    # compact table: per-column ADF/KPSS test statistic, pooled across assets (mean)
    all_cols = sorted({c for a in results.values() for c in a})
    header = f"confirmatory(n/{len(results)})"
    print(f"\n{'column':<22} {'adf_stat(mean)':>15} {'kpss_stat(mean)':>16} {header:>18}")
    for col in all_cols:
        adf_vals = [results[a][col]["adf_stat"] for a in results if col in results[a]]
        kpss_vals = [results[a][col]["kpss_stat"] for a in results if col in results[a]]
        n_conf = sum(
            1 for a in results if col in results[a]
            and results[a][col]["adf_stationary"] and results[a][col]["kpss_stationary"]
        )
        adf_mean = sum(adf_vals) / len(adf_vals) if adf_vals else float("nan")
        kpss_mean = sum(kpss_vals) / len(kpss_vals) if kpss_vals else float("nan")
        print(f"{col:<22} {adf_mean:>15.3f} {kpss_mean:>16.3f} {n_conf:>18d}")


if __name__ == "__main__":
    main()
