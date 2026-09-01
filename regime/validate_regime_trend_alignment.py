"""regime/validate_regime_trend_alignment.py

Objective test of a specific concern raised after reviewing regime_history_*.png:
"sometimes in a crash I see a strong bull [regime], in an uptrend I see mild bear
chop" — i.e. does the crypto5_multihorizon HMM's regime label actually track price
DIRECTION, or is that impression right?

Deliberately does NOT use hand-picked event windows (regime/analyze_regime_vs_history.py
did, and picking the windows yourself risks confirming what you expect to see). Instead:
  1. Runs a standard zigzag/peak-trough algorithm on daily close price to segment the
     year into objective up-legs and down-legs (a >=15% move, found from the data, not
     from memory of "when the bull run was").
  2. For each leg, takes the dominant (mode) regime_label over that leg's hours and
     classifies it via the regime's OWN aggregate Sharpe sign (from
     data/validated/regime_semantics_crypto5_multihorizon.json): R1/R4 = "bull-labeled"
     (positive Sharpe), R0/R2/R3 = "bear-labeled" (negative Sharpe), R5 = "crisis"
     (negative Sharpe, highest vol — kept separate since it's plausible during both
     violent selloffs and violent rallies, not purely directional).
  3. Reports a match rate: up-leg -> bull-labeled dominant, down-leg -> bear-labeled
     dominant. Every mismatch is printed explicitly with dates and % move so it can be
     checked by eye against the corresponding regime_history_*.png plot.

Run:
  python regime/validate_regime_trend_alignment.py --year 2021
  python regime/validate_regime_trend_alignment.py --year 2021 --threshold 0.15

Outputs:
  regime/plots/regime_trend_alignment_{ASSET}_{year}.png — price with zigzag legs
    marked + regime shading, so mismatches are visible at a glance.
  data/results/regime_trend_alignment_{year}.json — full leg table + match rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

MARTA_ROOT = Path(__file__).resolve().parent.parent
REGIME_STREAM = MARTA_ROOT / "data" / "processed" / "crypto5_multihorizon" / "regime_stream_causal.parquet"
RAW_DIR = MARTA_ROOT / "data" / "raw"
PLOTS_DIR = MARTA_ROOT / "regime" / "plots"
RESULTS_DIR = MARTA_ROOT / "data" / "results"

ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]

PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
REGIME_NAMES = {
    0: "R0 bear/drawdown (Sharpe -1.04)",
    1: "R1 mild bull (Sharpe +1.11)",
    2: "R2 mild bear/chop (Sharpe -0.22)",
    3: "R3 low-vol bear (Sharpe -0.26)",
    4: "R4 strong bull (Sharpe +1.39)",
    5: "R5 crisis/high-vol (Sharpe -0.44)",
}
BULL_LABELED = {1, 4}
BEAR_LABELED = {0, 2, 3}
CRISIS = {5}


def zigzag(prices: pd.Series, threshold: float) -> list[tuple[pd.Timestamp, float]]:
    """Standard peak/trough zigzag. Returns list of (timestamp, price) pivots."""
    idx = prices.index
    vals = prices.values
    pivots = [(idx[0], vals[0])]
    trend = None
    last_pivot_i = 0

    for i in range(1, len(vals)):
        change = (vals[i] - pivots[-1][1]) / pivots[-1][1]
        if trend is None:
            if change >= threshold:
                trend = "up"
                last_pivot_i = i
            elif change <= -threshold:
                trend = "down"
                last_pivot_i = i
        elif trend == "up":
            if vals[i] >= vals[last_pivot_i]:
                last_pivot_i = i
            elif (vals[i] - vals[last_pivot_i]) / vals[last_pivot_i] <= -threshold:
                pivots.append((idx[last_pivot_i], vals[last_pivot_i]))
                trend = "down"
                last_pivot_i = i
        elif trend == "down":
            if vals[i] <= vals[last_pivot_i]:
                last_pivot_i = i
            elif (vals[i] - vals[last_pivot_i]) / vals[last_pivot_i] >= threshold:
                pivots.append((idx[last_pivot_i], vals[last_pivot_i]))
                trend = "up"
                last_pivot_i = i

    pivots.append((idx[last_pivot_i], vals[last_pivot_i]))
    return pivots


def classify_regime(regime: int) -> str:
    if regime in BULL_LABELED:
        return "bull-labeled"
    if regime in BEAR_LABELED:
        return "bear-labeled"
    return "crisis"


def build_legs(pivots: list[tuple[pd.Timestamp, float]]) -> list[dict]:
    legs = []
    for i in range(len(pivots) - 1):
        t0, p0 = pivots[i]
        t1, p1 = pivots[i + 1]
        ret = (p1 - p0) / p0
        legs.append({
            "start": t0, "end": t1,
            "start_price": p0, "end_price": p1,
            "return_pct": ret * 100,
            "direction": "up" if ret > 0 else "down",
        })
    return legs


def evaluate_legs(legs: list[dict], regime_df: pd.DataFrame, year: int) -> list[dict]:
    out = []
    for leg in legs:
        # only report legs whose midpoint falls inside the target year (a leg can
        # start in Dec of the prior year — that's fine, we still want the full leg
        # for regime lookup, just don't double count it under two years)
        mid = leg["start"] + (leg["end"] - leg["start"]) / 2
        if mid.year != year:
            continue
        sub = regime_df[(regime_df["timestamp"] >= leg["start"]) & (regime_df["timestamp"] <= leg["end"])]
        if sub.empty:
            continue
        vc = sub["regime_label"].value_counts(normalize=True)
        dominant = int(vc.idxmax())
        dominant_class = classify_regime(dominant)
        match = (
            (leg["direction"] == "up" and dominant_class == "bull-labeled")
            or (leg["direction"] == "down" and dominant_class == "bear-labeled")
            or (leg["direction"] == "down" and dominant_class == "crisis")
        )
        out.append({
            **{k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in leg.items()},
            "n_days": (leg["end"] - leg["start"]).days,
            "dominant_regime": dominant,
            "dominant_regime_name": REGIME_NAMES[dominant],
            "dominant_regime_class": dominant_class,
            "regime_pct_breakdown": {str(k): round(v, 3) for k, v in vc.items()},
            "match": match,
        })
    return out


def plot_asset(asset: str, price: pd.Series, legs: list[dict], regime_df: pd.DataFrame,
               year: int, save_path: Path) -> None:
    sub_regime = regime_df.set_index("timestamp")["regime_label"]
    daily_regime = sub_regime.resample("1D").agg(lambda s: s.mode().iloc[0] if len(s) else np.nan).dropna()

    fig, ax = plt.subplots(figsize=(15, 6))

    # shade by daily regime
    idx = daily_regime.index
    vals = daily_regime.values
    run_start = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[run_start]:
            ax.axvspan(idx[run_start], idx[min(i, len(vals) - 1)],
                       color=PALETTE[int(vals[run_start])], alpha=0.20, linewidth=0)
            run_start = i

    ax.plot(price.index, price.values, color="#111111", linewidth=1.1)

    for leg in legs:
        color = "green" if leg["direction"] == "up" else "red"
        marker_ok = "o" if leg.get("match") else "x"
        ax.plot([leg["start"]], [leg["start_price"]], marker="^", color=color, markersize=6, zorder=5)
        ax.plot([leg["end"]], [leg["end_price"]], marker="v", color=color, markersize=6, zorder=5)
        mid_t = leg["start"] + (leg["end"] - leg["start"]) / 2
        mid_p = max(leg["start_price"], leg["end_price"]) * 1.03
        tag = "OK" if leg.get("match") else "MISMATCH"
        ax.annotate(f"{leg['return_pct']:+.0f}% {tag}", xy=(mid_t, mid_p), fontsize=7,
                    color=("black" if leg.get("match") else "red"),
                    fontweight=("normal" if leg.get("match") else "bold"),
                    ha="center")

    ax.set_title(f"{asset} — {year} zigzag legs vs. dominant regime (crypto5_multihorizon, causal)")
    ax.set_ylabel("Close price, USD")
    ax.grid(True, alpha=0.2)
    handles = [mpatches.Patch(color=PALETTE[i], alpha=0.5, label=REGIME_NAMES[i]) for i in range(6)]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=7)

    fig.autofmt_xdate()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--threshold", type=float, default=0.15)
    args = ap.parse_args()

    regime_df = pd.read_parquet(REGIME_STREAM)
    regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"], utc=True)

    all_legs = {}
    total_matches = 0
    total_legs = 0

    for asset in ASSETS:
        close = pd.read_parquet(RAW_DIR / f"{asset}_1h_raw.parquet")["close"]
        close.index = pd.to_datetime(close.index, utc=True)
        # pad with a few months of context on either side so a leg spanning the
        # year boundary is still detected whole, not artificially cut
        window = close.loc[f"{args.year - 1}-09-01": f"{args.year + 1}-03-31"]
        daily_price = window.resample("1D").last().dropna()

        pivots = zigzag(daily_price, args.threshold)
        legs = build_legs(pivots)
        asset_regime = regime_df[regime_df["asset"] == asset]
        evaluated = evaluate_legs(legs, asset_regime, args.year)

        all_legs[asset] = evaluated
        total_matches += sum(1 for l in evaluated if l["match"])
        total_legs += len(evaluated)

        plot_asset(asset, close.loc[f"{args.year}-01-01":f"{args.year}-12-31"],
                   [l for l in legs if (l["start"] + (l["end"] - l["start"]) / 2).year == args.year],
                   asset_regime, args.year,
                   PLOTS_DIR / f"regime_trend_alignment_{asset}_{args.year}.png")

        print(f"\n=== {asset} {args.year}: {len(evaluated)} legs, threshold={args.threshold:.0%} ===")
        for l in evaluated:
            tag = "OK" if l["match"] else "MISMATCH"
            print(f"  [{tag:8s}] {l['start'][:10]} -> {l['end'][:10]} ({l['n_days']:3d}d) "
                  f"{l['direction']:4s} {l['return_pct']:+7.1f}%  dominant={l['dominant_regime_name']}")

    results = {
        "year": args.year,
        "threshold": args.threshold,
        "overall_match_rate": round(total_matches / total_legs, 3) if total_legs else None,
        "total_legs": total_legs,
        "total_matches": total_matches,
        "per_asset_legs": all_legs,
    }
    out_path = RESULTS_DIR / f"regime_trend_alignment_{args.year}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\n\n=== OVERALL: {total_matches}/{total_legs} legs match "
          f"({results['overall_match_rate']:.0%}) across {len(ASSETS)} assets, {args.year} ===")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
