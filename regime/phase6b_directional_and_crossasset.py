"""Phase 6b — closes two gaps explicitly left open by phase6_regime_backtest.py's own
caveat: that test was long-only sizing only. It never tested (a) regime combined with a
DIRECTIONAL signal, or (b) cross-asset correlation-aware allocation. Both below, still
self-contained (no RL/PPO/LLM) and reusing phase6_regime_backtest.py's helpers rather
than duplicating them.

A. Directional conditioning: does regime help decide WHEN TO TRUST a momentum signal?
   momentum_only:        direction = sign(trailing 24h return), size=1.0 always.
   momentum_regime_gated: same direction, but flat (size=0) unless the current regime
                          had a positive train-period Sharpe for this exact momentum
                          rule (leakage-safe — "good regime" is determined from train
                          data only, per asset).
   This is different from Phase 6a's vol-target sizing (which never changed direction,
   always long) — here regime's job is to gate a real directional bet.

B. Cross-asset correlation-aware allocation: does knowing how many OTHER assets are
   simultaneously in the worst (highest train-period realized_vol_raw) regime add
   value beyond each asset's own signal?
   portfolio_equal_weight:   1.0 weight per asset, always (same as Phase 6a's buy_hold
                             portfolio).
   portfolio_systemic_scaled: portfolio-wide weight multiplier = 1 - crisis_frac_{t-1},
                             where crisis_frac = fraction of the 5 assets currently in
                             the empirically-worst regime (determined from train data).
                             Applied uniformly to all 5 assets' weights, not per-asset —
                             this is specifically a cross-asset signal, unavailable to
                             any single-asset strategy.

Same walk-forward val(2024)/test(2025) split, same 168h block-bootstrap methodology
(1000 resamples, seed 42) as Phase 6a.

Run: conda run -n marta python regime/phase6b_directional_and_crossasset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.splits import TEST_END_EXCLUSIVE, TEST_START, TRAIN_END_EXCLUSIVE, VAL_END_EXCLUSIVE, VAL_START
from regime.phase6_regime_backtest import _annualized_sharpe, _bootstrap_sharpe_delta, _max_drawdown
from utils.logger import logger

PANEL_DIR = ROOT / "data" / "processed" / "crypto5_multihorizon"
TXN_COST = 0.0005
MOMENTUM_WINDOW = 24  # hours
ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]


def _load_asset(asset: str, regime_stream: pd.DataFrame) -> pd.DataFrame:
    aligned = pd.read_parquet(PANEL_DIR / f"{asset}_aligned.parquet", engine="pyarrow")
    rs_asset = regime_stream[regime_stream["asset"] == asset].set_index("timestamp")[["regime_label"]]
    rs_asset.index = pd.to_datetime(rs_asset.index, utc=True)
    df = aligned[["close_return", "realized_vol_raw"]].join(rs_asset, how="inner")
    df["momentum_24h"] = df["close_return"].rolling(MOMENTUM_WINDOW).sum()
    return df.dropna()


# --------------------------------------------------------------------- A. directional
def _run_directional(df: pd.DataFrame, good_regimes: set[int]) -> dict[str, pd.DataFrame]:
    direction = np.sign(df["momentum_24h"]).shift(1).fillna(0.0)
    regime_prev = df["regime_label"].shift(1)
    gate = regime_prev.map(lambda r: 1.0 if r in good_regimes else 0.0).fillna(0.0)

    weights = {
        "momentum_only": direction,
        "momentum_regime_gated": direction * gate,
    }
    out = {}
    for name, w in weights.items():
        turnover = w.diff().abs().fillna(0.0)
        strat_return = w * df["close_return"] - TXN_COST * turnover
        out[name] = pd.DataFrame({"return": strat_return, "weight": w})
    return out


def _train_good_regimes(df: pd.DataFrame, train_mask: pd.Series) -> set[int]:
    """Per-regime train-period Sharpe of the momentum rule alone — regimes with a
    positive train Sharpe are 'good' (gate=1), everything else is 'bad' (gate=0).
    Leakage-safe: uses only train-period rows to decide, same discipline as Phase 6a's
    regime_vol_map."""
    train = df.loc[train_mask].copy()
    direction = np.sign(train["momentum_24h"]).shift(1).fillna(0.0)
    strat_return = direction * train["close_return"]
    good = set()
    for regime, idx in train.groupby("regime_label").groups.items():
        r = strat_return.loc[idx]
        if _annualized_sharpe(r.to_numpy()) > 0:
            good.add(int(regime))
    return good


# --------------------------------------------------------------------- B. cross-asset
def _worst_regime_per_asset(data: dict[str, pd.DataFrame], train_mask_fn) -> dict[str, int]:
    """The empirically-worst (highest train-period mean realized_vol_raw) regime per
    asset — the 'crisis' regime, determined from train data only."""
    worst = {}
    for asset, df in data.items():
        train_mask = train_mask_fn(df)
        vol_by_regime = df.loc[train_mask].groupby("regime_label")["realized_vol_raw"].mean()
        worst[asset] = int(vol_by_regime.idxmax())
    return worst


def main() -> None:
    train_cutoff = pd.Timestamp(TRAIN_END_EXCLUSIVE, tz="UTC")
    val_start, val_end = pd.Timestamp(VAL_START, tz="UTC"), pd.Timestamp(VAL_END_EXCLUSIVE, tz="UTC")
    test_start, test_end = pd.Timestamp(TEST_START, tz="UTC"), pd.Timestamp(TEST_END_EXCLUSIVE, tz="UTC")
    train_mask_fn = lambda df: df.index < train_cutoff  # noqa: E731

    regime_stream = pd.read_parquet(PANEL_DIR / "regime_stream.parquet", engine="pyarrow")
    data = {a: _load_asset(a, regime_stream) for a in ASSETS}

    results: dict = {"A_directional": {}, "B_cross_asset": {}}

    # ---------------- A. directional conditioning ----------------
    logger.info("=== A. Directional conditioning: momentum gated by regime ===")
    panel_returns = {
        "val": {"momentum_only": [], "momentum_regime_gated": []},
        "test": {"momentum_only": [], "momentum_regime_gated": []},
    }
    for asset in ASSETS:
        df = data[asset]
        train_mask = train_mask_fn(df)
        good_regimes = _train_good_regimes(df, train_mask)
        strategies = _run_directional(df, good_regimes)

        asset_out = {"good_regimes_from_train": sorted(good_regimes)}
        for split_name, (start, end) in [("val", (val_start, val_end)), ("test", (test_start, test_end))]:
            split_mask = (df.index >= start) & (df.index < end)
            split_out = {}
            for strat_name, strat_df in strategies.items():
                r = strat_df.loc[split_mask, "return"]
                split_out[strat_name] = {
                    "sharpe": _annualized_sharpe(r.to_numpy()),
                    "max_drawdown": _max_drawdown(r),
                    "total_return": float((1 + r).prod() - 1),
                }
                panel_returns[split_name][strat_name].append(r)
            asset_out[split_name] = split_out
        results["A_directional"][asset] = asset_out
        logger.success(
            f"[{asset}] good_regimes(train)={sorted(good_regimes)}  "
            f"test Sharpe — momentum_only={asset_out['test']['momentum_only']['sharpe']:.3f}  "
            f"regime_gated={asset_out['test']['momentum_regime_gated']['sharpe']:.3f}"
        )

    portfolio_a = {}
    for split_name in ("val", "test"):
        mom_r = pd.concat(panel_returns[split_name]["momentum_only"], axis=1).mean(axis=1).dropna()
        gated_r = pd.concat(panel_returns[split_name]["momentum_regime_gated"], axis=1).mean(axis=1).dropna()
        delta, ci_lo, ci_hi = _bootstrap_sharpe_delta(gated_r.to_numpy(), mom_r.to_numpy())
        portfolio_a[split_name] = {
            "momentum_only_sharpe": _annualized_sharpe(mom_r.to_numpy()),
            "momentum_regime_gated_sharpe": _annualized_sharpe(gated_r.to_numpy()),
            "delta_sharpe_gated_minus_momentum": delta,
            "ci_95": [ci_lo, ci_hi],
        }
    results["A_directional"]["portfolio"] = portfolio_a
    logger.success(f"[PORTFOLIO A] test: {portfolio_a['test']}")

    # ---------------- B. cross-asset correlation-aware allocation ----------------
    logger.info("=== B. Cross-asset: portfolio-wide scaling by systemic crisis-regime co-occurrence ===")
    worst_regime = _worst_regime_per_asset(data, train_mask_fn)
    logger.info(f"empirically-worst (crisis) regime per asset (train-derived): {worst_regime}")

    # shared index across all 5 assets (crypto5 is aligned, so this is the common timestamp set)
    common_index = data[ASSETS[0]].index
    for a in ASSETS[1:]:
        common_index = common_index.intersection(data[a].index)

    in_crisis = pd.DataFrame(
        {a: (data[a].loc[common_index, "regime_label"] == worst_regime[a]).astype(float) for a in ASSETS},
        index=common_index,
    )
    crisis_frac = in_crisis.mean(axis=1).shift(1).fillna(0.0)  # t-1 info only
    systemic_weight = (1.0 - crisis_frac).clip(0.0, 1.0)

    asset_returns = pd.DataFrame({a: data[a].loc[common_index, "close_return"] for a in ASSETS}, index=common_index)
    equal_weight_portfolio_return = asset_returns.mean(axis=1)  # weight=1.0 per asset, equal-weight average
    systemic_turnover = systemic_weight.diff().abs().fillna(0.0)
    systemic_scaled_return = systemic_weight * equal_weight_portfolio_return - TXN_COST * systemic_turnover

    portfolio_b = {}
    for split_name, (start, end) in [("val", (val_start, val_end)), ("test", (test_start, test_end))]:
        split_mask = (common_index >= start) & (common_index < end)
        eq = equal_weight_portfolio_return[split_mask]
        sys_r = systemic_scaled_return[split_mask]
        delta, ci_lo, ci_hi = _bootstrap_sharpe_delta(sys_r.to_numpy(), eq.to_numpy())
        portfolio_b[split_name] = {
            "equal_weight_sharpe": _annualized_sharpe(eq.to_numpy()),
            "equal_weight_max_dd": _max_drawdown(eq),
            "systemic_scaled_sharpe": _annualized_sharpe(sys_r.to_numpy()),
            "systemic_scaled_max_dd": _max_drawdown(sys_r),
            "mean_systemic_weight": float(systemic_weight[split_mask].mean()),
            "delta_sharpe_systemic_minus_equal": delta,
            "ci_95": [ci_lo, ci_hi],
        }
    results["B_cross_asset"] = {"worst_regime_per_asset": worst_regime, **portfolio_b}
    logger.success(f"[PORTFOLIO B] test: {portfolio_b['test']}")

    out_path = ROOT / "data" / "results" / "phase6b_directional_and_crossasset.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    logger.success(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
