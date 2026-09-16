# debate_index values below were re-verified against the actual contents of
# data/debates/debate_results.json (2026-07 audit) — the previous indices were
# essentially arbitrary and pointed at debates with the wrong asset/date/regime
# entirely (e.g. "BTC Calm Bull" pointed at a short/R2 debate). Every debate_index
# here is now the real nearest same-asset (+ same-regime, where one exists within a
# reasonable window) debate, and `timestamp` has been snapped to that debate's actual
# timestamp so the regime badge and the replayed transcript are for the same moment.
#
# STALE as of the 2026-07 audit: the debate cache only had 33 stored debates (32
# BTC_USDT, 1 GOLD) at the time these debate_index values were assigned — there were NO
# ETH_USDT or EURUSD debates, and nothing near March 2020. Scenarios for those omitted
# `debate_index` entirely (rather than pointing at an unrelated asset's debate); the
# frontend already handles a missing debate_index gracefully (no "Replay Debate" data,
# no crash). The cache has since grown to 1,861 debates across all 5 crypto5 assets
# (BTC/ETH/BNB/LTC/XRP) via the Kaggle debate-coverage runs — the scenarios below have
# NOT been re-audited against that larger cache, so gaps noted here may no longer be
# real gaps. Re-run the same re-verification pass described above if this file is
# touched again.
DEMO_SCENARIOS = [
    # ── BTC scenarios ──────────────────────────────────────────────────────────
    {
        "id": "btc_calm_bull_2021",
        "label": "BTC Calm Bull — Jan 2021",
        "asset": "BTC_USDT",
        "timestamp": "2021-01-31 20:00:00+00:00",
        "regime": "R1",
        "regime_label": 1,
        "description": "Bitcoin calm bull regime — agents reach bullish consensus",
        "debate_index": 7,
        "group": "BTC",
    },
    {
        "id": "btc_high_conviction_2021",
        "label": "BTC High Conviction — Jul 2021",
        "asset": "BTC_USDT",
        "timestamp": "2021-07-25 20:00:00+00:00",
        "regime": "R1",
        "regime_label": 1,
        "description": "High consensus triggers Contrarian devil's advocate",
        "debate_index": 23,
        "group": "BTC",
    },
    {
        "id": "btc_calm_bear_2021",
        "label": "BTC Calm Bear — Feb 2021",
        "asset": "BTC_USDT",
        "timestamp": "2021-02-15 00:00:00+00:00",
        "regime": "R4",
        "regime_label": 4,
        "description": "Calm bear — regime override reduces conviction",
        "debate_index": 8,
        "group": "BTC",
    },
    {
        "id": "btc_bear_2018",
        "label": "BTC Crypto Winter — Sep 2018",
        "asset": "BTC_USDT",
        "timestamp": "2018-09-26 16:00:00+00:00",
        "regime": "R4",
        "regime_label": 4,
        "description": "Crypto winter — bearish regime, reduced positions",
        "debate_index": 0,
        "group": "BTC",
    },
    {
        "id": "btc_march_2020_crash",
        "label": "BTC March 2020 Crash",
        "asset": "BTC_USDT",
        "timestamp": "2020-03-12 18:00:00+00:00",
        "regime": "R1",
        "regime_label": 1,
        "description": (
            "Black Thursday COVID crash — stress test of risk management. "
            "No debate was ever run for this date; replay is unavailable until one is generated."
        ),
        "group": "BTC",
    },
    # ── ETH scenarios ──────────────────────────────────────────────────────────
    {
        "id": "eth_bull_2021",
        "label": "ETH Bull Run — May 2021",
        "asset": "ETH_USDT",
        "timestamp": "2021-05-10 12:00:00+00:00",
        "regime": "R1",
        "regime_label": 1,
        "description": (
            "ETH bull regime before the May 2021 correction. "
            "No ETH_USDT debate exists in the cache; replay is unavailable until one is generated."
        ),
        "group": "ETH",
    },
    {
        "id": "eth_bear_2022",
        "label": "ETH Bear Market — Jun 2022",
        "asset": "ETH_USDT",
        "timestamp": "2022-06-15 08:00:00+00:00",
        "regime": "R3",
        "regime_label": 3,
        "description": (
            "Deep bear trend during crypto market rout. "
            "No ETH_USDT debate exists in the cache; replay is unavailable until one is generated."
        ),
        "group": "ETH",
    },
    # ── GOLD scenarios ─────────────────────────────────────────────────────────
    {
        "id": "gold_crisis_2018",
        "label": "GOLD Stress/Crisis — Nov 2018",
        "asset": "GOLD",
        "timestamp": "2018-11-08 17:00:00+00:00",
        "regime": "R2",
        "regime_label": 2,
        "description": "Trade war stress regime — Contrarian activates",
        "debate_index": 2,
        "group": "GOLD",
    },
    {
        "id": "gold_safe_haven_2020",
        "label": "GOLD Safe Haven — Aug 2020",
        "asset": "GOLD",
        "timestamp": "2020-08-04 14:00:00+00:00",
        "regime": "R2",
        "regime_label": 2,
        "description": (
            "Gold all-time high — COVID safe haven demand, stress regime. "
            "Only one GOLD debate exists in the cache (Nov 2018) — replayed here as the closest "
            "available example rather than a debate actually run on this date."
        ),
        "debate_index": 2,
        "group": "GOLD",
    },
    # ── EURUSD scenarios ───────────────────────────────────────────────────────
    {
        "id": "eurusd_trending_2020",
        "label": "EUR/USD Trending — Oct 2020",
        "asset": "EURUSD",
        "timestamp": "2020-10-15 10:00:00+00:00",
        "regime": "R5",
        "regime_label": 5,
        "description": (
            "EUR trending forex regime — momentum signals active. "
            "No EURUSD debate exists in the cache; replay is unavailable until one is generated."
        ),
        "group": "FX",
    },
    {
        "id": "eurusd_bear_2022",
        "label": "EUR/USD Parity — Jul 2022",
        "asset": "EURUSD",
        "timestamp": "2022-07-12 12:00:00+00:00",
        "regime": "R3",
        "regime_label": 3,
        "description": (
            "EUR/USD at parity for first time since 2002 — bear trend. "
            "No EURUSD debate exists in the cache; replay is unavailable until one is generated."
        ),
        "group": "FX",
    },
]

SCENARIO_BY_ID: dict = {s["id"]: s for s in DEMO_SCENARIOS}
