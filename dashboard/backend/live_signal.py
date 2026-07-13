"""Live paper trading signal.

Signal priority:
  1. PPO model (ppo_best.zip) via ppo_inference.predict() — when the checkpoint is loaded.
  2. Rule-based fallback (_REGIME_POLICY) — when the checkpoint is absent (first run
     before training, or intentionally disabled).
  3. Explicit caller-supplied debate override (`last_debate_action` param) — direction
     only (not the regime-forced flat in stress/crisis regimes). Currently unused by the
     frontend; kept for callers that want to pass a specific action directly.
  4. Automatic high-conviction debate override (_get_debate_override) — used whenever no
     explicit override is supplied. Reads the most recent debate_results.json entry for
     this asset; if it's <=24h old and conviction > 0.7, its action AND position_size
     both drive the signal. This is the path "Run Debate & Execute" on the Trading page
     relies on, since trigger-signal calls get_live_signal() with no explicit override.

Also tracks a running paper P&L against the live CCXT feed so the Execution panel
can show a live equity line.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).parent.parent.parent)))
_DEBATE_RESULTS_PATH = MARTA_ROOT / "data" / "debates" / "debate_results.json"
_DEBATE_MATCH_WINDOW_DAYS = 7

# Regime → (action, position_size)  — mirrors trained PPO behaviour
_REGIME_POLICY: dict[int, tuple[str, float]] = {
    0: ("hold", 0.010),   # Mixed/Transitional — cautious
    1: ("long", 0.040),   # Calm Bull — full send
    2: ("hold", 0.000),   # Stress/Crisis — flat
    3: ("short", 0.020),  # Bear Trend — small short
    4: ("hold", 0.010),   # Calm Bear — stay flat
    5: ("long", 0.015),   # Trending Forex A
    6: ("long", 0.015),   # Trending Forex B
}

# Paper portfolio state is persisted to SQLite via paper_db — survives restarts.


def _load_most_recent_debate(asset: str, now) -> Optional[dict]:
    """Most recent debate for `asset` within `_DEBATE_MATCH_WINDOW_DAYS` of `now`, if any.

    Debates are matched by list index, since /api/debates/{id} already treats the
    position in debate_results.json as the debate's id.
    """
    import pandas as pd

    if not _DEBATE_RESULTS_PATH.exists():
        return None
    try:
        debates = json.loads(_DEBATE_RESULTS_PATH.read_text())
    except Exception:
        return None

    best_idx, best_ts = None, None
    for idx, d in enumerate(debates):
        if d.get("asset") != asset:
            continue
        try:
            ts = pd.Timestamp(d["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
        except Exception:
            continue
        age_days = abs((now - ts).total_seconds()) / 86400
        if age_days > _DEBATE_MATCH_WINDOW_DAYS:
            continue
        if best_ts is None or ts > best_ts:
            best_idx, best_ts = idx, ts

    if best_idx is None:
        return None
    d = debates[best_idx]
    fm = d.get("fund_manager_decision", {})
    return {
        "debate_id": str(best_idx),
        "agent_theses": [
            {
                "agent": t.get("agent_name"),
                "bias": t.get("directional_bias"),
                "confidence": t.get("confidence"),
                "reasoning": t.get("reasoning"),
            }
            for t in d.get("round1_theses", [])
        ],
        "fund_manager_rationale": fm.get("rationale"),
    }


def _get_debate_override(asset: str, now, min_conviction: float = 0.7, max_age_hours: float = 24.0) -> Optional[dict]:
    """Most recent debate decision for `asset`, if recent and high-conviction enough to
    drive the live signal directly — both direction AND position size, unlike the
    lighter-weight explicit `last_debate_action` param below (direction only).

    This is what makes "Run Debate & Execute" on the Trading page actually change the
    resulting trade: /api/rl/trigger-signal calls get_live_signal() with no explicit
    debate_action, so this automatic lookup is the only path that can pick up a
    just-completed live debate (saved to debate_results.json by debate_stream.py).
    """
    import pandas as pd

    if not _DEBATE_RESULTS_PATH.exists():
        return None
    try:
        debates = json.loads(_DEBATE_RESULTS_PATH.read_text())
    except Exception:
        return None

    asset_debates = [d for d in debates if d.get("asset") == asset]
    if not asset_debates:
        return None

    def _ts(d: dict) -> pd.Timestamp:
        try:
            ts = pd.Timestamp(d.get("timestamp"))
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            return pd.Timestamp.min.tz_localize("UTC")

    latest = max(asset_debates, key=_ts)
    debate_time = _ts(latest)
    # abs(), not a plain subtraction — a debate timestamped a few hundred ms ahead of
    # `now` (e.g. saved just before this call, with clocks not perfectly aligned) would
    # otherwise produce a negative age that always passes the "> max_age_hours" check.
    age_hours = abs((now - debate_time).total_seconds()) / 3600
    if age_hours > max_age_hours:
        return None

    fm = latest.get("fund_manager_decision", {})
    conviction = float(fm.get("conviction", 0.0))
    if conviction < min_conviction:
        return None

    return {
        "action": fm.get("final_action", "hold"),
        "position_size": float(fm.get("position_size", 0.01)),
        "conviction": conviction,
        "debate_id": latest.get("timestamp"),
        "source": "debate_override",
    }


def get_live_signal(asset: str, last_debate_action: Optional[str] = None) -> dict:
    """Return the best available action signal + running paper P&L for `asset`."""
    import pandas as pd
    import regime_service
    import ppo_inference

    now = pd.Timestamp("now", tz="UTC")
    regime_info = regime_service.get_regime(asset, now)
    regime_label = regime_info.get("regime_label", 0)
    regime_prob = regime_info.get("probabilities", {})

    # --- Signal priority 1: trained PPO ---
    ppo_result = ppo_inference.predict(asset, now)
    if ppo_result is not None:
        action = ppo_result["action"]
        size = ppo_result["position_size"]
        source = "ppo"
    else:
        # --- Signal priority 2: rule-based fallback ---
        action, size = _REGIME_POLICY.get(regime_label, ("hold", 0.01))
        source = "regime_policy"

    # --- Signal priority 3: explicit caller-supplied debate override (direction only;
    # stress/crisis stays flat) ---
    ppo_override = False
    if last_debate_action and last_debate_action in ("long", "short", "hold"):
        if regime_label != 2:  # never override flat in stress/crisis
            if source == "ppo" and last_debate_action != action:
                ppo_override = True
            action = last_debate_action
            source = f"{source}+debate"
    else:
        # --- Signal priority 4: automatic high-conviction debate override ---
        # No explicit override was supplied — fall back to checking debate_results.json
        # directly for a recent, high-conviction call for this asset. Overrides BOTH
        # direction and size (the explicit path above only touches direction).
        debate_override = _get_debate_override(asset, now)
        if debate_override and regime_label != 2:  # never override flat in stress/crisis
            if source == "ppo" and debate_override["action"] != action:
                ppo_override = True
            action = debate_override["action"]
            size = debate_override["position_size"]
            source = f"{source}+debate_override"

    price = _fetch_latest_price(asset)

    debate_info = _load_most_recent_debate(asset, now) or {}

    import paper_db
    import app_settings
    regime_at_decision = regime_label

    # Approval mode: a non-hold signal is parked as a pending trade instead of being
    # executed — no capital moves, no position opens, until a human approves it via
    # POST /api/trades/pending/{id}/approve. A "hold" never risks capital, so it still
    # runs straight through update() even in approval mode (keeps mark-to-market PnL
    # on any already-open position flowing every tick).
    if app_settings.get_approval_mode() and action != "hold":
        pending_reasoning = {
            "asset": asset,
            "timestamp": now.isoformat(),
            "regime_label": regime_at_decision,
            "regime_description": regime_info.get("regime_description", ""),
            "source": "debate_override" if "+debate" in source else source,
            "ppo_confidence": size if ppo_result is not None else None,
            "debate_id": debate_info.get("debate_id"),
            "agent_theses": debate_info.get("agent_theses", []),
            "fund_manager_rationale": debate_info.get("fund_manager_rationale"),
            "ppo_override": ppo_override,
        }
        pending = paper_db.create_pending_trade(
            asset, action, size, price, regime_label=regime_at_decision, source=source,
            reasoning=pending_reasoning,
        )
        current = paper_db.get_portfolio(asset)
        return {
            "asset": asset,
            "regime_label": regime_label,
            "regime_description": regime_info.get("regime_description", ""),
            "regime_prob": regime_prob,
            "action": action,
            "position_size": size,
            "price": price,
            "paper_portfolio_value": current["portfolio_value"],
            "paper_total_return": current["total_return"],
            "paper_n_trades": current["n_trades"],
            "paper_trades": paper_db.get_trades(asset, limit=50),
            "source": source,
            "ppo_loaded": ppo_inference.is_loaded(),
            "pending_approval": True,
            "pending_trade_id": pending["id"],
        }

    paper = paper_db.update(asset, action, size, price, regime_label=regime_at_decision)
    trades = paper_db.get_trades(asset, limit=50)

    trade_id = paper.get("trade_id")
    if trade_id is not None:
        reasoning = {
            "asset": asset,
            "timestamp": now.isoformat(),
            "regime_label": regime_at_decision,
            "regime_description": regime_info.get("regime_description", ""),
            "source": "debate_override" if "+debate" in source else source,
            "ppo_confidence": size if ppo_result is not None else None,
            "debate_id": debate_info.get("debate_id"),
            "agent_theses": debate_info.get("agent_theses", []),
            "fund_manager_rationale": debate_info.get("fund_manager_rationale"),
            "ppo_override": ppo_override,
        }
        paper_db.save_trade_reasoning(trade_id, reasoning)

    return {
        "asset": asset,
        "regime_label": regime_label,
        "regime_description": regime_info.get("regime_description", ""),
        "regime_prob": regime_prob,
        "action": action,
        "position_size": size,
        "price": price,
        "paper_portfolio_value": paper["portfolio_value"],
        "paper_total_return": paper["total_return"],
        "paper_n_trades": paper["n_trades"],
        "paper_trades": trades,
        "source": source,
        "ppo_loaded": ppo_inference.is_loaded(),
        "pending_approval": False,
    }


def _fetch_latest_price(asset: str) -> float:
    """Best-effort latest close price — tries CCXT REST, falls back to last parquet row."""
    symbol_map = {"BTC_USDT": "BTC/USDT", "ETH_USDT": "ETH/USDT"}
    symbol = symbol_map.get(asset)
    if symbol:
        try:
            import ccxt  # type: ignore
            ex = ccxt.binance({"enableRateLimit": True})
            ticker = ex.fetch_ticker(symbol)
            return float(ticker.get("last", 0.0))
        except Exception:
            pass
    # Fallback to historical parquet
    try:
        import pandas as pd
        from historical import _load
        df = _load(asset)
        return float(df.iloc[-1]["close"]) if not df.empty else 0.0
    except Exception:
        return 0.0


