"""Phase 6 of the regime-layer audit (2026-08-20) — the question everything before this
was in service of: does conditioning on the regime label improve risk-adjusted returns?
Minimal, self-contained backtest — no RL/PPO, no LLM/agent layer, no encoder. Just price
data + the regime label already sitting in regime_stream.parquet.

Every predictive-signal test in this audit found regime_only carries ~chance-level
next-bar DIRECTIONAL AUC (0.49-0.52) across all 6 models tested (production, crypto5,
crypto5_multihorizon, btc_only, btc_only_multihorizon). But realized_vol_raw distinctness
was p~0 everywhere — these are real volatility regimes. So the honest question isn't
"can regime predict direction" (already answered: no) — it's "does discretizing
volatility into regime buckets, for POSITION SIZING, beat just using continuous
volatility directly." That's the fair, hard comparison: if regime-conditioned sizing
doesn't beat continuous vol-targeting (which uses very similar underlying information,
just not regime-bucketed), regime isn't adding anything a much simpler rule wouldn't.

Three long-only strategies per asset, all sharing the same directional signal (always
long — isolates the SIZING question, doesn't conflate it with a directional edge we
already know doesn't exist):
  1. buy_hold             — weight=1.0 constant.
  2. vol_target_continuous — weight_t = clip(target_vol / realized_vol_raw_{t-1}, 0.1, 2.0).
                              Regime-free: uses the asset's own trailing realized vol
                              directly, no bucketing.
  3. regime_conditioned    — weight_t = clip(target_vol / regime_vol_map[regime_label_{t-1}],
                              0.1, 2.0). regime_vol_map = mean realized_vol_raw per
                              regime, computed on TRAIN data ONLY (< 2024) — leakage-safe,
                              matches the HMM's own train/val/test discipline
                              (configs/splits.py). Decisions use t-1 information, applied
                              to the t-1->t return — no lookahead.

target_vol = train-period mean realized_vol_raw, so both sized strategies run at the same
average exposure as buy_hold over the period the mapping was learned on — a fair
comparison, not just "trade smaller."

Evaluated on val(2024) and test(2025) separately (test is the walk-forward split the rest
of this project treats as sacred). Reports annualized Sharpe, max drawdown, turnover per
strategy per asset, plus an equal-weight portfolio aggregate for crypto5 (for a fair
single-number comparison against the necessarily single-asset btc_only). Block-bootstrap
(1000 resamples, 168h/1-week blocks, seed 42) on the Sharpe delta between
regime_conditioned and vol_target_continuous — same methodology rl/ablation.py's
bootstrap_sharpe_ci uses, reimplemented standalone here rather than imported, so this
script has zero dependency on the RL layer.

Run: conda run -n marta python regime/phase6_regime_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.splits import TEST_END_EXCLUSIVE, TEST_START, TRAIN_END_EXCLUSIVE, VAL_END_EXCLUSIVE, VAL_START
from utils.logger import logger

PROCESSED_DIR = ROOT / "data" / "processed"
TXN_COST = 0.0005  # 5bps per unit of weight change — conservative, not over-engineered
WEIGHT_CLIP = (0.1, 2.0)

PANELS = {
    "crypto5_multihorizon": ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"],
    "btc_only_multihorizon": ["BTC_USDT"],
}


def _annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or returns.std() < 1e-12:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(24 * 365))


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def _bootstrap_sharpe_delta(
    returns_a: np.ndarray, returns_b: np.ndarray, n_bootstrap: int = 1000, block_size: int = 168,
) -> tuple[float, float, float]:
    """Block bootstrap CI on Sharpe(a) - Sharpe(b). Same methodology as rl/ablation.py's
    bootstrap_sharpe_ci (168h blocks, 1000 resamples, seed 42) — reimplemented standalone
    so this script has no dependency on the RL layer."""
    n = min(len(returns_a), len(returns_b))
    n_blocks = n // block_size
    if n_blocks < 2:
        delta = _annualized_sharpe(returns_a[:n]) - _annualized_sharpe(returns_b[:n])
        return delta, delta, delta

    a = returns_a[: n_blocks * block_size].reshape(n_blocks, block_size)
    b = returns_b[: n_blocks * block_size].reshape(n_blocks, block_size)

    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        sa, sb = a[idx].ravel(), b[idx].ravel()
        deltas.append(_annualized_sharpe(sa) - _annualized_sharpe(sb))
    deltas = np.array(deltas)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _load_asset(panel_dir: Path, asset: str) -> pd.DataFrame:
    aligned = pd.read_parquet(panel_dir / f"{asset}_aligned.parquet", engine="pyarrow")
    rs = pd.read_parquet(panel_dir / "regime_stream.parquet", engine="pyarrow")
    rs_asset = rs[rs["asset"] == asset].set_index("timestamp")[["regime_label"]]
    rs_asset.index = pd.to_datetime(rs_asset.index, utc=True)
    df = aligned[["close_return", "realized_vol_raw"]].join(rs_asset, how="inner")
    return df.dropna()


def _run_strategies(df: pd.DataFrame, regime_vol_map: dict, target_vol: float) -> dict[str, pd.DataFrame]:
    """Weight decided from t-1 info, applied to the t-1->t return (df['close_return']
    at row t already IS that return) — no lookahead."""
    realized_vol_prev = df["realized_vol_raw"].shift(1)
    regime_prev = df["regime_label"].shift(1)

    weights = {
        "buy_hold": pd.Series(1.0, index=df.index),
        "vol_target_continuous": (target_vol / realized_vol_prev.clip(lower=1e-6)).clip(*WEIGHT_CLIP),
        "regime_conditioned": regime_prev.map(
            lambda r: target_vol / regime_vol_map.get(r, target_vol) if pd.notna(r) else 1.0
        ).clip(*WEIGHT_CLIP),
    }

    out = {}
    for name, w in weights.items():
        w = w.fillna(1.0)
        turnover = w.diff().abs().fillna(0.0)
        strat_return = w * df["close_return"] - TXN_COST * turnover
        out[name] = pd.DataFrame({"return": strat_return, "weight": w})
    return out


def main() -> None:
    train_cutoff = pd.Timestamp(TRAIN_END_EXCLUSIVE, tz="UTC")
    val_start, val_end = pd.Timestamp(VAL_START, tz="UTC"), pd.Timestamp(VAL_END_EXCLUSIVE, tz="UTC")
    test_start, test_end = pd.Timestamp(TEST_START, tz="UTC"), pd.Timestamp(TEST_END_EXCLUSIVE, tz="UTC")

    all_results: dict[str, dict] = {}

    for panel, assets in PANELS.items():
        panel_dir = PROCESSED_DIR / panel
        logger.info(f"=== panel: {panel} ({assets}) ===")

        panel_returns: dict[str, dict[str, list[pd.Series]]] = {
            "val": {"buy_hold": [], "vol_target_continuous": [], "regime_conditioned": []},
            "test": {"buy_hold": [], "vol_target_continuous": [], "regime_conditioned": []},
        }
        per_asset: dict[str, dict] = {}

        for asset in assets:
            df = _load_asset(panel_dir, asset)
            train_mask = df.index < train_cutoff
            regime_vol_map = df.loc[train_mask].groupby("regime_label")["realized_vol_raw"].mean().to_dict()
            target_vol = float(df.loc[train_mask, "realized_vol_raw"].mean())

            strategies = _run_strategies(df, regime_vol_map, target_vol)

            asset_out = {}
            for split_name, (start, end) in [("val", (val_start, val_end)), ("test", (test_start, test_end))]:
                split_mask = (df.index >= start) & (df.index < end)
                split_out = {}
                for strat_name, strat_df in strategies.items():
                    r = strat_df.loc[split_mask, "return"]
                    w = strat_df.loc[split_mask, "weight"]
                    split_out[strat_name] = {
                        "sharpe": _annualized_sharpe(r.to_numpy()),
                        "max_drawdown": _max_drawdown(r),
                        "total_return": float((1 + r).prod() - 1),
                        "mean_weight": float(w.mean()),
                        "n_steps": int(len(r)),
                    }
                    panel_returns[split_name][strat_name].append(r)
                asset_out[split_name] = split_out
            per_asset[asset] = asset_out

            logger.success(
                f"[{asset}] test Sharpe — buy_hold={asset_out['test']['buy_hold']['sharpe']:.3f}  "
                f"vol_target={asset_out['test']['vol_target_continuous']['sharpe']:.3f}  "
                f"regime={asset_out['test']['regime_conditioned']['sharpe']:.3f}"
            )

        # equal-weight portfolio aggregate
        portfolio_out = {}
        for split_name in ("val", "test"):
            split_out = {}
            for strat_name, series_list in panel_returns[split_name].items():
                aligned_series = pd.concat(series_list, axis=1).mean(axis=1).dropna()
                split_out[strat_name] = {
                    "sharpe": _annualized_sharpe(aligned_series.to_numpy()),
                    "max_drawdown": _max_drawdown(aligned_series),
                    "total_return": float((1 + aligned_series).prod() - 1),
                }
            # bootstrap: regime_conditioned vs vol_target_continuous, on this split's portfolio returns
            reg_r = pd.concat(panel_returns[split_name]["regime_conditioned"], axis=1).mean(axis=1).dropna().to_numpy()
            vol_r = pd.concat(panel_returns[split_name]["vol_target_continuous"], axis=1).mean(axis=1).dropna().to_numpy()
            delta, ci_lo, ci_hi = _bootstrap_sharpe_delta(reg_r, vol_r)
            split_out["bootstrap_regime_vs_voltarget"] = {"delta_sharpe": delta, "ci_95": [ci_lo, ci_hi]}
            portfolio_out[split_name] = split_out

        logger.success(
            f"[{panel} PORTFOLIO] test Sharpe — buy_hold={portfolio_out['test']['buy_hold']['sharpe']:.3f}  "
            f"vol_target={portfolio_out['test']['vol_target_continuous']['sharpe']:.3f}  "
            f"regime={portfolio_out['test']['regime_conditioned']['sharpe']:.3f}  "
            f"| delta(regime-voltarget)={portfolio_out['test']['bootstrap_regime_vs_voltarget']['delta_sharpe']:.3f} "
            f"95% CI={portfolio_out['test']['bootstrap_regime_vs_voltarget']['ci_95']}"
        )

        all_results[panel] = {"per_asset": per_asset, "portfolio": portfolio_out}

    out_path = ROOT / "data" / "results" / "phase6_regime_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    out_path.write_text(json.dumps(all_results, indent=2, default=float))
    logger.success(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
