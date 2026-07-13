"""
pipeline/validate.py
Comprehensive validation of the pipeline output. Logs every check as PASS/FAIL.
Logs metrics and artifacts to MLflow. Returns True iff all critical checks pass.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from utils.logger import logger

VALIDATED_DIR = Path("data/validated")

# Expected minimum row counts after full processing.
# Crypto: ~52k raw - warmup - weekend intersection loss ≈ 30k after alignment with forex.
# Traditional: daily-upsampled to hourly, same intersection, minus warmup ≈ 15k.
_MIN_ROWS_CRYPTO = 30_000
_MIN_ROWS_TRADITIONAL = 15_000
_CRYPTO_SYMBOLS = {"BTC_USDT", "ETH_USDT"}

INDICATOR_COLS = [
    "atr",
    "adx",
    "rsi",
    "bb_width",
    "macd_histogram",
    "obv_raw",
    "vol_zscore",
    "realized_vol",
    "vol_ratio",
    "return_skew",
    "return_kurtosis",
    "autocorr_lag1",
    "realized_vol_raw",
]
FEATURE_COLS = INDICATOR_COLS + ["close_return"]  # 14 total


def _pass(check: str, detail: str = "") -> dict:
    msg = f"[PASS] {check}" + (f" — {detail}" if detail else "")
    logger.success(msg)
    return {"check": check, "status": "PASS", "detail": detail}


def _fail(check: str, detail: str = "", critical: bool = True) -> dict:
    level = "CRITICAL" if critical else "WARNING"
    msg = f"[FAIL/{level}] {check}" + (f" — {detail}" if detail else "")
    if critical:
        logger.error(msg)
    else:
        logger.warning(msg)
    return {"check": check, "status": "FAIL", "detail": detail, "critical": critical}


def validate_pipeline(
    aligned_assets: dict[str, pd.DataFrame],
    raw_assets: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> bool:
    """
    Run all validation checks on aligned and raw asset data.
    Logs results to MLflow and saves a JSON report.
    Returns True iff all CRITICAL checks pass.
    """
    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)

    max_gap_crypto: float = config["validation"]["max_gap_hours_crypto"]
    max_gap_trad: float = config["validation"]["max_gap_hours_traditional"]
    zscore_thresh: float = config["validation"]["zscore_flag_threshold"]
    min_overlap: float = config["validation"]["min_timestamp_overlap_pct"]

    results: list[dict] = []
    mlflow_metrics: dict[str, float] = {}
    mlflow_params: dict[str, str] = {}

    # -----------------------------------------------------------------------
    # 1. Structural checks
    # -----------------------------------------------------------------------
    for symbol, df in aligned_assets.items():
        min_rows = _MIN_ROWS_CRYPTO if symbol in _CRYPTO_SYMBOLS else _MIN_ROWS_TRADITIONAL
        row_count = len(df)
        mlflow_metrics[f"{symbol}_row_count"] = float(row_count)

        if row_count >= min_rows:
            results.append(_pass(f"{symbol} row count", f"{row_count:,} >= {min_rows:,}"))
        else:
            results.append(_fail(f"{symbol} row count", f"{row_count:,} < {min_rows:,} required"))

        expected_cols = set(FEATURE_COLS)
        actual_cols = set(df.columns)
        if actual_cols == expected_cols:
            results.append(_pass(f"{symbol} column count", f"{len(actual_cols)} columns"))
        else:
            extra = actual_cols - expected_cols
            missing = expected_cols - actual_cols
            results.append(_fail(
                f"{symbol} columns",
                f"extra={extra or 'none'}, missing={missing or 'none'}"
            ))

        if df.index.is_monotonic_increasing:
            results.append(_pass(f"{symbol} monotonic index"))
        else:
            results.append(_fail(f"{symbol} monotonic index", "timestamps not monotonically increasing"))

        dup_count = df.index.duplicated().sum()
        if dup_count == 0:
            results.append(_pass(f"{symbol} no duplicate timestamps"))
        else:
            results.append(_fail(f"{symbol} duplicate timestamps", f"{dup_count} duplicates found"))

    # All assets share the same index
    if len(aligned_assets) > 1:
        symbols = list(aligned_assets.keys())
        ref_idx = aligned_assets[symbols[0]].index
        all_same = all(
            aligned_assets[s].index.equals(ref_idx) for s in symbols[1:]
        )
        if all_same:
            results.append(_pass("all assets share identical timestamp index"))
        else:
            results.append(_fail("all assets share identical timestamp index", "indices diverge"))

    # -----------------------------------------------------------------------
    # 2. Range checks (CRITICAL)
    # -----------------------------------------------------------------------
    # We check raw-derived indicators that were carried through normalization.
    # After rolling z-score, raw range constraints no longer hold — so we check
    # structural properties instead: no NaN, no Inf.

    for symbol, df in aligned_assets.items():
        nan_count = int(df.isnull().sum().sum())
        mlflow_metrics[f"{symbol}_nan_count"] = float(nan_count)
        if nan_count == 0:
            results.append(_pass(f"{symbol} no NaN values"))
        else:
            results.append(_fail(f"{symbol} no NaN values", f"{nan_count} NaNs found"))

        inf_count = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
        mlflow_metrics[f"{symbol}_inf_count"] = float(inf_count)
        if inf_count == 0:
            results.append(_pass(f"{symbol} no Inf values"))
        else:
            results.append(_fail(f"{symbol} no Inf values", f"{inf_count} Inf values found"))

    # -----------------------------------------------------------------------
    # 2b. realized_vol_raw range + crash-detection sanity (CRITICAL / INFO)
    # -----------------------------------------------------------------------
    # realized_vol_raw is the only unnormalized feature — it must stay non-negative
    # and within a sane annualized range, and it is what makes the March 2020 crash
    # detectable again (the z-scored realized_vol compresses that signal away).
    for symbol, df in aligned_assets.items():
        if "realized_vol_raw" not in df.columns:
            results.append(_fail(f"{symbol} realized_vol_raw present", "column missing"))
            continue

        rvr = df["realized_vol_raw"]
        if (rvr >= 0).all():
            results.append(_pass(f"{symbol} realized_vol_raw non-negative"))
        else:
            n_neg = int((rvr < 0).sum())
            results.append(_fail(f"{symbol} realized_vol_raw non-negative", f"{n_neg} negative values"))

        if (rvr < 50).all():
            results.append(_pass(f"{symbol} realized_vol_raw in range", "< 50 annualized"))
        else:
            n_oor = int((rvr >= 50).sum())
            results.append(_fail(
                f"{symbol} realized_vol_raw in range",
                f"{n_oor} values >= 50 annualized",
            ))

        # Crash-detection probe: March 2020 should show elevated raw vol for BTC.
        if symbol == "BTC_USDT":
            btc_mar2020 = df[(df.index >= "2020-03-10") & (df.index <= "2020-03-14")]
            if len(btc_mar2020) > 0:
                crash_vol = float(btc_mar2020["realized_vol_raw"].mean())
                normal_vol = float(df["realized_vol_raw"].median())
                ratio = crash_vol / (normal_vol + 1e-8)
                logger.info(f"BTC March 2020 vol ratio: {ratio:.1f}x normal (should be >>1)")
                mlflow_metrics["btc_march2020_vol_ratio"] = ratio
                if ratio > 1.0:
                    results.append(_pass(
                        "BTC March 2020 crash detectable",
                        f"realized_vol_raw {ratio:.1f}x median",
                    ))
                else:
                    results.append(_fail(
                        "BTC March 2020 crash detectable",
                        f"realized_vol_raw only {ratio:.1f}x median",
                        critical=False,
                    ))

    # Raw OHLCV sanity checks
    for symbol, raw_df in raw_assets.items():
        checks = {
            "high >= close": (raw_df["high"] >= raw_df["close"]).all(),
            "high >= open": (raw_df["high"] >= raw_df["open"]).all(),
            "low <= close": (raw_df["low"] <= raw_df["close"]).all(),
            "low <= open": (raw_df["low"] <= raw_df["open"]).all(),
            "volume >= 0": (raw_df["volume"] >= 0).all(),
            "no zero prices": (raw_df["close"] > 0).all(),
        }
        for check_name, passed in checks.items():
            if passed:
                results.append(_pass(f"{symbol} OHLCV: {check_name}"))
            else:
                n_violations = int((~_ohlcv_check_series(raw_df, check_name)).sum())
                results.append(_fail(
                    f"{symbol} OHLCV: {check_name}",
                    f"{n_violations} violations"
                ))

    # -----------------------------------------------------------------------
    # 3. Gap checks (WARNING)
    # -----------------------------------------------------------------------
    for symbol, df in aligned_assets.items():
        max_gap = max_gap_crypto if symbol in _CRYPTO_SYMBOLS else max_gap_trad
        diffs = df.index.to_series().diff().dropna()
        threshold = pd.Timedelta(hours=max_gap)
        gap_series = diffs[diffs > threshold]
        gap_count = len(gap_series)
        mlflow_metrics[f"{symbol}_gap_count"] = float(gap_count)

        if gap_count == 0:
            results.append(_pass(f"{symbol} gap check (>{max_gap}h)", "no gaps"))
        else:
            top5 = gap_series.nlargest(5)
            detail = f"{gap_count} gaps. Top 5: " + ", ".join(
                f"{ts}: {gap}" for ts, gap in top5.items()
            )
            results.append(_fail(f"{symbol} gap check (>{max_gap}h)", detail, critical=False))

    # -----------------------------------------------------------------------
    # 4. Distribution checks (WARNING)
    # -----------------------------------------------------------------------
    for symbol, df in aligned_assets.items():
        numeric = df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            col_mean = numeric[col].mean()
            col_std = numeric[col].std()
            if col_std == 0:
                continue
            zscores = ((numeric[col] - col_mean) / col_std).abs()
            flagged = int((zscores > zscore_thresh).sum())
            flagged_pct = flagged / len(numeric) * 100
            mlflow_metrics[f"{symbol}_{col}_flagged_pct"] = flagged_pct
            if flagged > 0:
                results.append(_fail(
                    f"{symbol}/{col} distribution",
                    f"{flagged:,} values ({flagged_pct:.2f}%) beyond {zscore_thresh}σ",
                    critical=False,
                ))

    # -----------------------------------------------------------------------
    # 5. Cross-asset checks (CRITICAL)
    # -----------------------------------------------------------------------
    if "BTC_USDT" in aligned_assets and "BTC_USDT" in raw_assets:
        btc_aligned_len = len(aligned_assets["BTC_USDT"])
        btc_raw_len = len(raw_assets["BTC_USDT"])
        overlap = btc_aligned_len / btc_raw_len if btc_raw_len > 0 else 0.0
        mlflow_metrics["btc_timestamp_overlap_pct"] = overlap * 100

        if overlap >= min_overlap:
            results.append(_pass(
                "cross-asset timestamp overlap",
                f"{overlap * 100:.1f}% >= {min_overlap * 100:.0f}%"
            ))
        else:
            results.append(_fail(
                "cross-asset timestamp overlap",
                f"{overlap * 100:.1f}% < {min_overlap * 100:.0f}%"
            ))

    # All assets have same columns in same order
    if len(aligned_assets) > 1:
        symbols = list(aligned_assets.keys())
        ref_cols = list(aligned_assets[symbols[0]].columns)
        all_same_cols = all(
            list(aligned_assets[s].columns) == ref_cols for s in symbols[1:]
        )
        if all_same_cols:
            results.append(_pass("all assets have identical column order"))
        else:
            results.append(_fail("all assets have identical column order"))

    # -----------------------------------------------------------------------
    # 6. MLflow logging
    # -----------------------------------------------------------------------
    for key, val in mlflow_metrics.items():
        try:
            mlflow.log_metric(key, val)
        except Exception:
            pass

    for r in results:
        param_key = r["check"].replace(" ", "_").replace("/", "_")[:250]
        mlflow_params[param_key] = r["status"]

    try:
        mlflow.log_params({k: v for k, v in list(mlflow_params.items())[:100]})
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # 7. Save JSON report
    # -----------------------------------------------------------------------
    critical_failures = [r for r in results if r.get("status") == "FAIL" and r.get("critical", True)]
    warnings = [r for r in results if r.get("status") == "FAIL" and not r.get("critical", True)]
    passed = [r for r in results if r.get("status") == "PASS"]

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_checks": len(results),
            "passed": len(passed),
            "critical_failures": len(critical_failures),
            "warnings": len(warnings),
            "pipeline_ok": len(critical_failures) == 0,
        },
        "critical_failures": critical_failures,
        "warnings": warnings,
        "passed": passed,
        "metrics": mlflow_metrics,
    }

    report_path = VALIDATED_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    try:
        mlflow.log_artifact(str(report_path))
    except Exception:
        pass

    logger.info(f"Validation report saved → {report_path}")

    if critical_failures:
        logger.error(
            f"Validation FAILED: {len(critical_failures)} critical failures, "
            f"{len(warnings)} warnings, {len(passed)} passed."
        )
        return False

    logger.success(
        f"Validation PASSED: {len(passed)} checks passed, "
        f"{len(warnings)} warnings (non-critical)."
    )
    return True


def _ohlcv_check_series(df: pd.DataFrame, check_name: str) -> pd.Series:
    """Return boolean Series for a named OHLCV sanity check."""
    mapping = {
        "high >= close": df["high"] >= df["close"],
        "high >= open": df["high"] >= df["open"],
        "low <= close": df["low"] <= df["close"],
        "low <= open": df["low"] <= df["open"],
        "volume >= 0": df["volume"] >= 0,
        "no zero prices": df["close"] > 0,
    }
    return mapping.get(check_name, pd.Series(dtype=bool))
