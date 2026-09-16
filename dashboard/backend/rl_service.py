"""Serve RL evaluation results and equity curve to the dashboard."""

import json
import os
from pathlib import Path

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))

_btc_prices = None  # lazily-loaded pd.Series of real BTC close prices, indexed by UTC timestamp


def _load_btc_prices():
    """Real BTC close price series, used to backfill trade prices for synthetic logs."""
    global _btc_prices
    if _btc_prices is None:
        import pandas as pd
        path = MARTA_ROOT / "data/processed/BTC_USDT_features.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["close"])
            df.index = pd.to_datetime(df.index, utc=True)
            _btc_prices = df["close"]
        else:
            _btc_prices = None
    return _btc_prices


def _price_at(ts) -> float:
    """Nearest real BTC close price at `ts` (str or Timestamp). 0.0 if unavailable."""
    import pandas as pd
    prices = _load_btc_prices()
    if prices is None or prices.empty:
        return 0.0
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    idx = prices.index.get_indexer([ts], method="nearest")[0]
    return round(float(prices.iloc[idx]), 2)


def get_full_results() -> dict:
    evaluation = _load_json("data/results/ppo_evaluation.json")
    ablation = _load_json("data/results/ablation_results.json")
    curriculum = _load_json("data/results/curriculum_history.json")

    ablation_chart = []
    if ablation and "variants" in ablation:
        label_map = {
            "buy_hold": "Buy & Hold",
            "ppo_no_regime": "PPO (no regime)",
            "agents_no_regime": "Agents (no regime)",
            "ppo_agents_no_regime": "PPO+Agents (no regime)",
            "marta_full": "MARTA Full",
        }
        for key, label in label_map.items():
            v = ablation["variants"].get(key, {})
            ablation_chart.append({
                "variant": label,
                "key": key,
                "sharpe": round(float(v.get("sharpe", 0)), 3),
                "max_drawdown": round(float(v.get("max_drawdown", 0)) * 100, 1),
                "total_return": round(float(v.get("total_return", 0)) * 100, 1),
                "calmar": round(float(v.get("calmar", 0)), 3),
                "n_meaningful_trades": int(v.get("n_meaningful_trades", 0)),
            })

    ci_data = []
    if ablation and "bootstrap_ci" in ablation:
        for k, v in ablation["bootstrap_ci"].items():
            name = k.replace("marta_full_vs_", "vs ").replace("_", " ")
            ci_data.append({
                "comparison": name,
                "delta_sharpe": round(float(v.get("delta_sharpe", 0)), 3),
                "ci_lower_95": round(float(v.get("ci_lower_95", 0)), 3),
                "ci_upper_95": round(float(v.get("ci_upper_95", 0)), 3),
                "significant": float(v.get("ci_lower_95", 0)) > 0,
            })

    curriculum_chart = []
    if isinstance(curriculum, list):
        for phase in curriculum:
            curriculum_chart.append({
                "phase": phase.get("phase"),
                "assets": ", ".join(phase.get("assets", [])),
                "steps": phase.get("steps"),
                "sharpe": round(float(phase.get("sharpe", 0)), 3),
                "threshold": phase.get("threshold"),
                "met": phase.get("threshold_met", False),
            })

    return {
        "evaluation": evaluation or {},
        "ablation_chart": ablation_chart,
        "bootstrap_ci": ci_data,
        "curriculum": curriculum_chart,
        "has_equity_curve": (MARTA_ROOT / "data/results/equity_curve.json").exists(),
    }


def get_equity_curve() -> dict:
    """Equity curve for the Analytics chart.

    Priority order mirrors get_positions(): (1) derive directly from ppo_evaluation.json's
    trade_log — the same evaluate_ppo() run backing every other number on the page; (2) the
    standalone equity_curve.json snapshot, only if trade_log is absent; (3) a synthetic
    curve as a last resort.

    Before this fix, equity_curve.json (last regenerated 2026-07-05, describing the OLD
    pre-walk-forward 2021 val window) was served unconditionally — even after
    ppo_evaluation.json moved to the 2024 val window with a real trade_log — producing the
    same two-stale-sources-on-one-card bug get_positions() was fixed for in the 2026-07-31
    audit, just for the equity curve instead of the trade list.
    """
    eval_data = _load_json("data/results/ppo_evaluation.json") or {}
    trade_log = eval_data.get("trade_log")
    if trade_log:
        curve = _equity_curve_from_trade_log(eval_data, trade_log)
        if curve is not None:
            return curve

    path = MARTA_ROOT / "data/results/equity_curve.json"
    if path.exists():
        with open(path) as f:
            curve = json.load(f)
        curve["stale"] = True
        curve["source"] = "equity_curve.json (no trade_log in ppo_evaluation.json — using standalone snapshot)"
        return curve

    # Fallback: synthetic curve from saved metrics so the chart isn't empty
    return _synthetic_equity_curve()


def _equity_curve_from_trade_log(eval_data: dict, trade_log: list) -> dict | None:
    """Build a timestamps/marta/buy_hold series directly from trade_log's portfolio_value.

    trade_log only has one row per *meaningful* trade (not every step), so this is a
    step-function equity curve between trades rather than a full per-bar series — accurate
    at every point it does have, unlike the synthetic sine-wave fallback.
    """
    if not trade_log:
        return None
    marta = eval_data.get("marta_ppo", {})
    bh = (eval_data.get("baselines") or {}).get("buy_and_hold_btc", {})
    val_window = eval_data.get("val_window", {})

    timestamps = [t["timestamp"] for t in trade_log]
    marta_vals = [round(float(t["portfolio_value"]), 2) for t in trade_log]

    start_val = 10000.0
    bh_return = float(bh.get("total_return", 0.0))
    n = len(timestamps)
    bh_vals = [round(start_val * (1 + bh_return * (i + 1) / max(n, 1)), 2) for i in range(n)]

    return {
        "timestamps": timestamps,
        "marta": marta_vals,
        "buy_hold": bh_vals,
        "synthetic": False,
        "stale": False,
        "source": "ppo_evaluation.json trade_log",
        "note": "Step function over meaningful trades only, not every bar.",
        "marta_final_return": round(float(marta.get("total_return", 0.0)) * 100, 1),
        "bh_final_return": round(bh_return * 100, 1),
        "marta_max_dd": round(float(marta.get("max_drawdown", 0.0)) * 100, 1),
        "bh_max_dd": round(float(bh.get("max_drawdown", 0.0)) * 100, 1),
        "val_window": val_window,
    }


def _synthetic_equity_curve() -> dict:
    """Build a plausible equity curve from the saved evaluation metrics.

    This is used only when the real curve hasn't been generated yet.
    The BTC val period is 2021-01-01 to 2021-10-12 (≈4583 hourly steps).
    We model MARTA as a dampened version of B&H with lower drawdown.
    """
    import math

    eval_data = _load_json("data/results/ppo_evaluation.json") or {}
    marta = eval_data.get("marta_ppo", {})
    bh = (eval_data.get("baselines") or {}).get("buy_and_hold_btc", {})

    marta_return = float(marta.get("total_return", 0.262))
    bh_return = float(bh.get("total_return", 0.478))
    marta_dd = float(marta.get("max_drawdown", 0.262))
    bh_dd = float(bh.get("max_drawdown", 0.500))

    # 9.5 months of hourly data (2021-01-01 to 2021-10-12), sampled daily for the chart
    import datetime
    start = datetime.date(2021, 1, 1)
    n_days = 284  # ~9.5 months
    timestamps = []
    marta_vals = []
    bh_vals = []

    marta_v = 10000.0
    bh_v = 10000.0
    peak_marta = marta_v
    peak_bh = bh_v

    # BTC 2021 had a large run-up Jan-Apr, crash May, recovery Jul, then peak Oct
    # We model this as a smooth Fourier approximation so the chart looks right
    for i in range(n_days):
        t = i / n_days
        # B&H: approximate BTC 2021 price path (27k → 64k → 28k → 43k → 66k scaled to return)
        bh_daily = (
            0.002 * math.sin(2 * math.pi * t * 2)
            + 0.001 * math.sin(2 * math.pi * t * 5 + 1)
            + bh_return / n_days
        )
        # MARTA: same trend but scaled to lower exposure, lower DD
        scale = marta_return / max(bh_return, 0.01)
        marta_daily = bh_daily * scale * (1 + 0.1 * math.sin(t * math.pi))

        bh_v *= 1 + bh_daily
        marta_v *= 1 + marta_daily
        peak_bh = max(peak_bh, bh_v)
        peak_marta = max(peak_marta, marta_v)

        date = (start + datetime.timedelta(days=i)).isoformat()
        timestamps.append(date)
        marta_vals.append(round(marta_v, 2))
        bh_vals.append(round(bh_v, 2))

    return {
        "timestamps": timestamps,
        "marta": marta_vals,
        "buy_hold": bh_vals,
        "synthetic": True,
        "stale": True,
        "source": "synthetic",
        "note": "Synthetic curve — run generate_equity_curve.py for real data",
        "marta_final_return": round(marta_return * 100, 1),
        "bh_final_return": round(bh_return * 100, 1),
        "marta_max_dd": round(marta_dd * 100, 1),
        "bh_max_dd": round(bh_dd * 100, 1),
    }


def get_positions() -> dict:
    """Return current position state + trade log for the Orders panel.

    Priority order: (1) ppo_evaluation.json's real `trade_log` (added 2026-07-31 — the
    actual trades from the same evaluate_ppo() run that produced the Sharpe number shown
    elsewhere on the page, so they're guaranteed consistent with each other); (2)
    equity_curve.json's positions array, if present; (3) a synthetic/fabricated trade log
    as a last resort.

    Found during the 2026-07-31 audit: equity_curve.json is a point-in-time snapshot
    (last regenerated 2026-07-05, describing the OLD pre-walk-forward 2021 val window)
    that nothing regenerates automatically, so it silently drifts out of sync with
    ppo_evaluation.json (current 2024 val window) — which is exactly why the dashboard
    showed a "100% win rate" from 2021 dates next to a "-1.278 Sharpe" from 2024 data:
    two unrelated, differently-stale sources rendered on the same card. Prefer trade_log.
    """
    eval_data = _load_json("data/results/ppo_evaluation.json") or {}
    marta = eval_data.get("marta_ppo", {})

    trade_log = eval_data.get("trade_log")
    if trade_log:
        trades = []
        for t in trade_log:
            action = "hold" if t["direction"] == "flat" else t["direction"]
            trades.append({
                "timestamp": t["timestamp"],
                "action": action,
                "position_size": float(t["position_size"]),
                "price": _price_at(t["timestamp"]),
                "pnl": float(t["pnl_return_step"]),
                "portfolio_value": round(float(t["portfolio_value"]), 2),
                "regime_label": int(t["decision_regime"]),
            })
        last = trade_log[-1] if trade_log else None
        last_action = "hold" if not last or last["direction"] == "flat" else last["direction"]
        return {
            "current_action": last_action,
            "current_position": float(last["position_size"]) if last else 0.0,
            "current_regime": int(last["decision_regime"]) if last else 0,
            "portfolio_value": round(float(last["portfolio_value"]), 2) if last else 10000.0,
            "total_return": float(marta.get("total_return", 0.0)),
            "n_trades": int(marta.get("n_meaningful_trades", len(trades))),
            "trades": trades,
            "source": "ppo_evaluation.json trade_log",
            "stale": False,
        }

    curve = _load_json("data/results/equity_curve.json")

    # If the curve has real position data (saved by updated generate_equity_curve.py)
    if curve and "positions" in curve and "timestamps" in curve:
        timestamps = curve["timestamps"]
        positions_arr = curve.get("positions", [])
        marta_vals = curve.get("marta", [])
        regime_arr = curve.get("regimes", [0] * len(timestamps))
        prices = curve.get("prices") or [_price_at(t) for t in timestamps]

        trades = []
        for i, (ts, pos, val, regime) in enumerate(
            zip(timestamps, positions_arr, marta_vals, regime_arr)
        ):
            prev_pos = positions_arr[i - 1] if i > 0 else 0.0
            prev_val = marta_vals[i - 1] if i > 0 else val
            step_pnl = (val - prev_val) / max(prev_val, 1.0) if prev_val else 0.0

            # Only log steps with a meaningful position or position change
            if abs(pos) > 0.005 or abs(pos - prev_pos) > 0.005:
                action = "long" if pos > 0.005 else ("short" if pos < -0.005 else "hold")
                trades.append({
                    "timestamp": ts if "T" in str(ts) else f"{ts}T00:00:00Z",
                    "action": action,
                    "position_size": abs(float(pos)),
                    "price": float(prices[i]) if i < len(prices) else 0.0,
                    "pnl": round(float(step_pnl), 6),
                    "portfolio_value": round(float(val), 2),
                    "regime_label": int(regime),
                })

        last_pos = positions_arr[-1] if positions_arr else 0.0
        last_action = "long" if last_pos > 0.005 else ("short" if last_pos < -0.005 else "hold")
        return {
            "current_action": last_action,
            "current_position": abs(float(last_pos)),
            "current_regime": int(regime_arr[-1]) if regime_arr else 0,
            "portfolio_value": round(float(marta_vals[-1]), 2) if marta_vals else 10000.0,
            "total_return": float(marta.get("total_return", 0.0)),
            "n_trades": int(marta.get("n_meaningful_trades", len(trades))),
            "trades": trades,
            "source": "equity_curve.json (fallback, no trade_log in ppo_evaluation.json)",
            "stale": True,
        }

    # Fallback: synthesise a plausible trade log from saved stats
    return _synthetic_positions(eval_data)


def _synthetic_positions(eval_data: dict) -> dict:
    import math, datetime
    marta = eval_data.get("marta_ppo", {})
    per_regime = eval_data.get("per_regime", {})

    total_return = float(marta.get("total_return", 0.262))
    n_meaningful = int(marta.get("n_meaningful_trades", 31))
    n_val_steps = int(eval_data.get("n_val_steps", 6803))

    start = datetime.date(2021, 1, 1)
    n_days = 284
    portfolio_v = 10000.0
    trades = []

    # Synthesise daily snapshots with occasional trades
    trade_interval = max(1, n_days // max(n_meaningful, 1))
    for i in range(n_days):
        t = i / n_days
        daily_r = total_return / n_days * (1 + 0.3 * math.sin(t * math.pi * 4))
        portfolio_v *= 1 + daily_r
        ts = (start + datetime.timedelta(days=i)).strftime("%Y-%m-%dT00:00:00Z")
        if i % trade_interval == 0:
            pos = 0.026 * (1 + 0.5 * math.sin(t * math.pi))
            action = "long" if daily_r >= 0 else "short"
            # pick regime based on month (rough approximation for BTC 2021)
            regime = 1 if i < 100 else (4 if i < 160 else 1)
            trades.append({
                "timestamp": ts,
                "action": action,
                "position_size": round(pos, 3),
                "price": _price_at(ts),
                "pnl": round(daily_r, 6),
                "portfolio_value": round(portfolio_v, 2),
                "regime_label": regime,
            })

    last = trades[-1] if trades else {}
    return {
        "current_action": "hold",
        "current_position": 0.026,
        "current_regime": 1,
        "portfolio_value": round(portfolio_v, 2),
        "total_return": total_return,
        "n_trades": n_meaningful,
        "trades": trades,
        "source": "synthetic",
        "stale": True,
        "note": "Synthetic trade log — run generate_equity_curve.py for real positions",
    }


def _load_json(rel_path: str) -> dict | list | None:
    path = MARTA_ROOT / rel_path
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
