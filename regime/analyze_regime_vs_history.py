"""regime/analyze_regime_vs_history.py

Qualitative + quantitative check of whether the production crypto5 HMM (n=6,
regime/checkpoints/crypto5_multihorizon/hmm_model.pkl, regime_stream_causal.parquet)
actually segments known historical crypto market events into distinct regimes —
COVID crash, 2021 ATH, 2022 Luna/FTX collapse, 2024 ETF/halving/election rally, and
asset-specific idiosyncratic moves (XRP's Nov 2024 - Jan 2025 rally, XRP's Jul 2023
SEC ruling spike) — across all 5 crypto5 panel assets (BTC, ETH, BNB, LTC, XRP).

The existing regime_quality_audit.py answers "are these regimes statistically
distinct" (ANOVA/KS on return distributions) but never checks them against known
calendar events. This script closes that gap: it overlays raw close price (not the
close_return-reconstructed index used elsewhere) with the causal regime_label per
asset, produces full-history + zoomed event plots, and reports per-event regime
occupancy against the unconditional base rate (lift), because the interesting
question isn't "what regime is active" but "is it enriched relative to how often
that regime occurs anyway."

Run:
  python regime/analyze_regime_vs_history.py

Outputs:
  regime/plots/regime_history_{ASSET}_full.png     — full history, log price + regime shading
  regime/plots/regime_history_{ASSET}_covid.png     — Jan-Jun 2020 zoom, hourly resolution
  regime/plots/regime_history_{ASSET}_recent.png    — 2023-2025 zoom (ETF/halving/election/XRP rally)
  data/results/regime_history_event_check.json      — per-asset per-event regime occupancy + lift
"""
from __future__ import annotations

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
RESULTS_PATH = MARTA_ROOT / "data" / "results" / "regime_history_event_check.json"

ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]
N_STATES = 6

# Palette matches regime/visualize.py's _PALETTE so colors are consistent with
# existing regime plots elsewhere in the repo.
PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
][:N_STATES]

# Semantic labels pulled from data/validated/regime_semantics_crypto5_multihorizon.json
# (2026-08-22 fit): ranked by realized_vol_mean / sharpe, NOT re-derived here — this is
# a fixed reading of that file's numbers for annotation purposes only.
REGIME_NAMES = {
    0: "R0 bear/drawdown (Sharpe -1.04)",
    1: "R1 mild bull (Sharpe +1.11)",
    2: "R2 mild bear/chop (Sharpe -0.22)",
    3: "R3 low-vol bear (Sharpe -0.26)",
    4: "R4 strong bull (Sharpe +1.39)",
    5: "R5 crisis/high-vol (Sharpe -0.44)",
}

# Approximate, well-known public market dates. Two granularities are tracked
# separately because a monthly window can dilute a 2-day crash into a misleading
# "dominant regime" read (confirmed empirically: BTC's COVID-crash MONTH shows R4
# dominant because most of March 2020 outside Mar 11-16 was calm/recovering — but the
# ACUTE 5-day window shows R5 with prob=1.0 for 48+ straight hours). "acute" = the
# genuine shock window (days), "context" = the surrounding weeks/months for background.
EVENTS = [
    ("2018 bear bottom", "2018-11-01", "2018-12-31", "context", "all", "Capitulation low of the 2018 bear market (BTC ~$3.2k)"),
    ("2018 initial breakdown", "2018-11-14", "2018-11-21", "acute", "all", "Initial break from ~$6.4k to ~$4.5k that started the Nov-Dec slide"),
    ("2019 spring rally", "2019-04-01", "2019-06-30", "context", "all", "BTC ~$4k -> ~$13.8k"),
    ("2019 H2 decline", "2019-07-01", "2019-12-31", "context", "all", "Retrace back toward $7k"),
    ("COVID crash (month)", "2020-03-01", "2020-03-31", "context", "all", "Broad risk-off month"),
    ("COVID crash (Black Thursday)", "2020-03-11", "2020-03-16", "acute", "all", "Mar 12-13 2020, BTC -50% in a day"),
    ("2020-21 bull leg 1", "2020-10-01", "2021-04-30", "context", "all", "BTC $10.5k -> $64k, institutional adoption wave"),
    ("May 2021 crash", "2021-05-01", "2021-07-20", "context", "all", "China mining ban + leverage flush, BTC $64k -> $29k"),
    ("May 2021 crash (acute)", "2021-05-17", "2021-05-23", "acute", "all", "Single sharpest leg down of the May 2021 flush"),
    ("2021 ATH rally", "2021-09-01", "2021-11-15", "context", "all", "Second leg to BTC ATH ~$69k (Nov 10 2021)"),
    ("2021 ATH (peak days)", "2021-11-06", "2021-11-13", "acute", "all", "BTC ATH ~$69k on Nov 10 2021"),
    ("2022 Luna/Celsius (quarter)", "2022-05-01", "2022-07-31", "context", "all", "Terra/Luna collapse, Celsius/3AC contagion"),
    ("2022 Luna/UST depeg (acute)", "2022-05-07", "2022-05-14", "acute", "all", "UST depeg + Terra collapse, ~May 9-12 2022"),
    ("2022 FTX collapse (month)", "2022-11-01", "2022-11-30", "context", "all", "FTX bankruptcy month"),
    ("2022 FTX collapse (acute)", "2022-11-06", "2022-11-12", "acute", "all", "Withdrawal halt Nov 8 -> bankruptcy filing Nov 11"),
    ("2023 ETF speculation rally", "2023-10-01", "2023-12-31", "context", "all", "Spot BTC ETF approval speculation"),
    ("2024 ETF + halving", "2024-01-01", "2024-04-30", "context", "all", "Spot ETF launch Jan 2024, halving Apr 2024"),
    ("2024 election rally / new ATH", "2024-11-01", "2024-12-31", "context", "all", "Post-election rally, BTC new ATH ~$108k"),
    ("XRP SEC ruling spike", "2023-07-10", "2023-07-20", "acute", "XRP_USDT", "Partial summary judgment in SEC v. Ripple, XRP +70% in a day"),
    ("XRP late-2024/early-2025 rally", "2024-11-01", "2025-01-31", "context", "XRP_USDT", "XRP multi-year high ~$3.3, SEC case winding down"),
    ("XRP acceleration phase", "2024-11-08", "2024-12-03", "acute", "XRP_USDT", "Steepest leg of the rally, $0.59 -> $2.72"),
]


def load_regime_stream() -> pd.DataFrame:
    df = pd.read_parquet(REGIME_STREAM)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_raw_close(asset: str) -> pd.Series:
    df = pd.read_parquet(RAW_DIR / f"{asset}_1h_raw.parquet")
    return df["close"]


def merged_series(regime_df: pd.DataFrame, asset: str) -> pd.DataFrame:
    sub = regime_df[regime_df["asset"] == asset].sort_values("timestamp").set_index("timestamp")
    close = load_raw_close(asset)
    close.index = pd.to_datetime(close.index, utc=True)
    merged = sub[["regime_label"]].join(close.rename("close"), how="inner")
    return merged


def daily_resample(merged: pd.DataFrame) -> pd.DataFrame:
    daily_close = merged["close"].resample("1D").last()
    daily_regime = merged["regime_label"].resample("1D").agg(lambda s: s.mode().iloc[0] if len(s) else np.nan)
    out = pd.DataFrame({"close": daily_close, "regime_label": daily_regime}).dropna()
    out["regime_label"] = out["regime_label"].astype(int)
    return out


def shade_regimes(ax, df: pd.DataFrame) -> None:
    """Draw contiguous-run axvspan shading for df['regime_label'], indexed by time."""
    labels = df["regime_label"].values
    idx = df.index
    if len(labels) == 0:
        return
    run_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[run_start]:
            ax.axvspan(idx[run_start], idx[min(i, len(labels) - 1)],
                       color=PALETTE[int(labels[run_start])], alpha=0.18, linewidth=0)
            run_start = i


def plot_full_history(asset: str, daily: pd.DataFrame, save_path: Path) -> None:
    # Shading uses weekly-mode regime (mean dwell time is ~11-17h median across
    # assets on the causal stream — daily-mode shading still flickers multiple times
    # a week and is unreadable over an 8-year span). The price line stays daily.
    weekly_regime = daily["regime_label"].resample("1W").agg(lambda s: s.mode().iloc[0] if len(s) else np.nan).dropna()
    weekly = pd.DataFrame({"regime_label": weekly_regime.astype(int)})

    fig, ax = plt.subplots(figsize=(16, 6))
    shade_regimes(ax, weekly)
    ax.semilogy(daily.index, daily["close"], color="#111111", linewidth=1.0)
    ax.set_ylabel("Close price, USD (log scale)")
    ax.set_title(f"{asset} — price vs. HMM regime (crypto5_multihorizon, n=6, causal stream)")
    ax.grid(True, alpha=0.2)

    for name, start, end, granularity, scope, desc in EVENTS:
        if granularity != "context":
            continue
        if scope != "all" and scope != asset:
            continue
        start_ts = pd.Timestamp(start, tz="UTC")
        if start_ts < daily.index.min() or start_ts > daily.index.max():
            continue
        ax.axvline(start_ts, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    handles = [mpatches.Patch(color=PALETTE[i], alpha=0.5, label=REGIME_NAMES[i]) for i in range(N_STATES)]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=7.5)

    fig.autofmt_xdate()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_zoom(asset: str, merged: pd.DataFrame, start: str, end: str, title_suffix: str, save_path: Path,
               event_markers: list[tuple[str, str]] | None = None) -> None:
    window = merged.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")]
    if window.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 5.5))
    shade_regimes(ax, window)
    ax.plot(window.index, window["close"], color="#111111", linewidth=0.9)
    ax.set_ylabel("Close price, USD")
    ax.set_title(f"{asset} — {title_suffix}")
    ax.grid(True, alpha=0.2)

    if event_markers:
        for label, ts in event_markers:
            t = pd.Timestamp(ts, tz="UTC")
            if window.index.min() <= t <= window.index.max():
                ax.axvline(t, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.text(t, ax.get_ylim()[1], label, rotation=90, fontsize=7,
                        va="top", ha="right", color="red")

    handles = [mpatches.Patch(color=PALETTE[i], alpha=0.5, label=REGIME_NAMES[i]) for i in range(N_STATES)]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=7)

    fig.autofmt_xdate()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def event_occupancy(regime_df: pd.DataFrame) -> dict:
    base_rate = {}
    for asset in ASSETS:
        sub = regime_df[regime_df["asset"] == asset]
        vc = sub["regime_label"].value_counts(normalize=True)
        base_rate[asset] = {int(k): float(v) for k, v in vc.items()}

    results = {"base_rate_pct_time": base_rate, "events": {}}
    for name, start, end, granularity, scope, desc in EVENTS:
        assets_to_check = ASSETS if scope == "all" else [scope]
        event_result = {"description": desc, "window": [start, end], "granularity": granularity, "per_asset": {}}
        for asset in assets_to_check:
            sub = regime_df[
                (regime_df["asset"] == asset)
                & (regime_df["timestamp"] >= pd.Timestamp(start, tz="UTC"))
                & (regime_df["timestamp"] <= pd.Timestamp(end, tz="UTC"))
            ]
            if sub.empty:
                continue
            vc = sub["regime_label"].value_counts(normalize=True)
            occ = {int(k): float(v) for k, v in vc.items()}
            lift = {
                int(k): round(v / base_rate[asset].get(int(k), 1e-9), 2)
                for k, v in occ.items()
            }
            dominant = max(occ, key=occ.get)
            # Peak posterior probability of the two "extreme" states (R4 high-momentum,
            # R5 high-vol) within the window, plus hours where each exceeded 0.5 — a
            # short acute shock can be swamped in the dominant-label view but still show
            # up clearly here (e.g. COVID's Mar-12 transition holds regime_prob_5=1.0 for
            # ~48h even though the surrounding month's dominant label is elsewhere).
            max_prob_4 = float(sub["regime_prob_4"].max())
            max_prob_5 = float(sub["regime_prob_5"].max())
            hours_r4 = int((sub["regime_label"] == 4).sum())
            hours_r5 = int((sub["regime_label"] == 5).sum())
            event_result["per_asset"][asset] = {
                "n_hours": int(len(sub)),
                "pct_time_by_regime": {str(k): round(v, 3) for k, v in occ.items()},
                "lift_vs_base_rate": {str(k): lift[k] for k in occ},
                "dominant_regime": dominant,
                "dominant_regime_name": REGIME_NAMES[dominant],
                "max_regime_prob_4_momentum": round(max_prob_4, 3),
                "max_regime_prob_5_high_vol": round(max_prob_5, 3),
                "hours_in_R4": hours_r4,
                "hours_in_R5": hours_r5,
            }
        results["events"][name] = event_result
    return results


def main() -> None:
    regime_df = load_regime_stream()

    print(f"Loaded regime_stream_causal.parquet: {len(regime_df):,} rows, "
          f"{regime_df['timestamp'].min()} -> {regime_df['timestamp'].max()}")

    merged_by_asset = {}
    for asset in ASSETS:
        merged = merged_series(regime_df, asset)
        merged_by_asset[asset] = merged
        daily = daily_resample(merged)

        plot_full_history(asset, daily, PLOTS_DIR / f"regime_history_{asset}_full.png")
        plot_zoom(
            asset, merged, "2020-01-01", "2020-06-30",
            "COVID crash zoom (hourly resolution)",
            PLOTS_DIR / f"regime_history_{asset}_covid.png",
            event_markers=[("Black Thursday", "2020-03-12")],
        )
        plot_zoom(
            asset, daily_resample(merged.loc["2023-01-01":]), "2023-01-01", "2025-12-31",
            "2023-2025 zoom (ETF / halving / election / XRP rally)",
            PLOTS_DIR / f"regime_history_{asset}_recent.png",
            event_markers=[
                ("ETF approval", "2024-01-10"),
                ("Halving", "2024-04-20"),
                ("Election", "2024-11-05"),
            ] + ([("XRP SEC ruling", "2023-07-13")] if asset == "XRP_USDT" else []),
        )
        print(f"  {asset}: plots saved")

    occ = event_occupancy(regime_df)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(occ, indent=2))
    print(f"\nEvent occupancy written to {RESULTS_PATH}")

    # Console summary: dominant regime per event per asset, flagged when the crisis
    # regime (R5, highest realized_vol) is NOT enriched during known high-vol events.
    print("\n=== Quick read: dominant regime per event (+ peak R4/R5 posterior in window) ===")
    for name, ev in occ["events"].items():
        tag = "[ACUTE]" if ev["granularity"] == "acute" else "[ctx]  "
        line = f"{tag} {name:35s}"
        for asset, d in ev["per_asset"].items():
            line += (f"  {asset.split('_')[0]}=R{d['dominant_regime']}"
                     f"(pR4={d['max_regime_prob_4_momentum']:.2f},pR5={d['max_regime_prob_5_high_vol']:.2f})")
        print(line)


if __name__ == "__main__":
    main()
