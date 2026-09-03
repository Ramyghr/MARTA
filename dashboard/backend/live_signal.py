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

# Regime -> (action, position_size), generated from data/validated/regime_semantics.json
# instead of hand-maintained by label number. A hardcoded-by-index table like this one
# used to be went stale silently every time run_regime.py refits the HMM — label 3 can
# mean "severe crisis" after one refit and "mild bull" after the next, since regime
# label assignment isn't stable across fits. Generating it from each regime's actual
# Sharpe (the same statistic already used for the human-readable descriptions in
# agents/context.py and regime_service.py) keeps this table correct automatically.
_REGIME_POLICY_PATH = MARTA_ROOT / "data" / "validated" / "regime_semantics.json"
_REGIME_POLICY_FALLBACK: dict[int, tuple[str, float]] = {
    0: ("hold", 0.010),
}  # used only if regime_semantics.json is missing (pipeline never run)

_regime_policy_cache: dict[int, tuple[str, float]] | None = None


def _generate_regime_policy() -> dict[int, tuple[str, float]]:
    """Sharpe-banded action/size per regime, thresholds chosen to mirror how the old
    hand-written table reasoned about its regimes: stay flat in a severe-negative
    regime rather than short into it (crisis regimes are exactly the high-variance,
    hard-to-model conditions you don't want to lever into on a fallback rule), scale
    conviction (position size) with Sharpe magnitude otherwise.
    """
    global _regime_policy_cache
    if _regime_policy_cache is not None:
        return _regime_policy_cache

    if not _REGIME_POLICY_PATH.exists():
        _regime_policy_cache = dict(_REGIME_POLICY_FALLBACK)
        return _regime_policy_cache

    try:
        sem = json.loads(_REGIME_POLICY_PATH.read_text())
    except Exception:
        _regime_policy_cache = dict(_REGIME_POLICY_FALLBACK)
        return _regime_policy_cache

    policy: dict[int, tuple[str, float]] = {}
    for label, stats in sem.get("regimes", {}).items():
        sharpe = float(stats.get("sharpe", 0.0))
        if sharpe <= -1.0:
            policy[int(label)] = ("hold", 0.000)      # severe crisis/bear — stay flat
        elif sharpe <= -0.15:
            policy[int(label)] = ("short", 0.015)     # real negative edge
        elif sharpe < 0.15:
            policy[int(label)] = ("hold", 0.010)      # too weak to act on — cautious
        elif sharpe < 0.5:
            policy[int(label)] = ("long", 0.015)      # modest positive edge
        elif sharpe < 1.0:
            policy[int(label)] = ("long", 0.020)      # solid positive edge
        else:
            policy[int(label)] = ("long", 0.040)      # strong edge

    _regime_policy_cache = policy or dict(_REGIME_POLICY_FALLBACK)
    return _regime_policy_cache


def invalidate_regime_policy_cache() -> None:
    """Call after regime_semantics.json is regenerated so the next lookup re-reads it."""
    global _regime_policy_cache
    _regime_policy_cache = None

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
    # Live, not the historical nearest-lookup — this drives the crisis-override guard
    # below, so it needs to reflect right now, not regime_stream.parquet's last
    # precomputed row (frozen at 2024-12-31 as of this fix). log=True here because
    # get_live_signal() is the natural once-per-hour-per-asset cadence (the scheduler
    # tick) to grow the live regime audit log at — see regime_service.get_live_regime.
    regime_info = regime_service.get_live_regime(asset, now, log=True)
    regime_label = regime_info.get("regime_label", 0)
    regime_prob = regime_info.get("probabilities", {})

    import app_settings

    # --- Signal priority 0: weekly-cadence PPO — opt-in, BTC_USDT only, takes over
    # completely for this asset (see weekly_ppo_live.py's module docstring for why the
    # debate-override priorities 3-4 below are deliberately skipped when this is
    # active — they'd fight a policy that's only supposed to decide once a week). A
    # different checkpoint from priority-1 "ppo" below (native decision_interval=168
    # training, validated with a positive Sharpe, vs. the hourly checkpoint which
    # isn't) — the two toggles are independent but only one can drive BTC_USDT at once.
    ppo_override = False
    ppo_result = None  # only ever set (possibly) in the priority-1 branch below; stays
    # None on the weekly-PPO path too — "ppo_confidence" downstream specifically means
    # the hourly checkpoint's confidence, not this one's position size.
    if asset == "BTC_USDT" and app_settings.get_weekly_ppo_enabled():
        import weekly_ppo_live
        weekly = weekly_ppo_live.get_weekly_signal(app_settings.get_weekly_ppo_obs_mode(), now)
        action = weekly["action"]
        size = weekly["position_size"]
        source = weekly["source"]
    else:
        # --- Signal priority 1: trained PPO — opt-in only (see app_settings.py) ---
        # Default off per docs/RESULTS_AND_METHODOLOGY.md Section 9: a complete, controlled
        # ablation found no crypto5 RL configuration with a defensible edge over
        # buy-and-hold. Skipping the predict() call entirely when disabled (not just
        # ignoring its result) so this behaves exactly like "no checkpoint loaded" already
        # does below — no new code path, just gating the existing fallback earlier.
        ppo_result = ppo_inference.predict(asset, now) if app_settings.get_rl_execution_enabled() else None
        if ppo_result is not None:
            action = ppo_result["action"]
            size = ppo_result["position_size"]
            source = "ppo"
        else:
            # --- Signal priority 2: rule-based fallback ---
            action, size = _generate_regime_policy().get(regime_label, ("hold", 0.01))
            source = "regime_policy"

        # --- Signal priority 3: explicit caller-supplied debate override (direction only;
        # stress/crisis stays flat) ---
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
        import telegram_service
        telegram_service.fire_and_forget(telegram_service.notify_pending_trade(
            pending_id=pending["id"], asset=asset, action=action, position_size=size,
            price=price, regime_label=regime_at_decision, source=source,
        ))
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


_binance_client = None  # module-level cache — see _get_binance_client()


def _get_binance_client():
    """Cached ccxt.binance() instance, constructed once per process.

    Confirmed 2026-09-03: a fresh ccxt.binance() instance lazily calls load_markets()
    on its first API call (fetch_ticker included), fetching Binance's full exchange
    metadata — measured ~4s on this network. _fetch_latest_price is on the hot path of
    every get_live_signal() call (every Trading-page mount, every manual trigger, every
    hourly scheduler tick), and previously constructed a brand-new client per call, so
    every single call paid that ~4s cold-start cost again instead of once. Restricting
    to spot markets (historical.py's _fetch_extension_binance found this already, same
    root cause) trims the one-time cold start further (~2s vs ~8s unrestricted).
    """
    global _binance_client
    if _binance_client is None:
        import ccxt  # type: ignore
        _binance_client = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot", "fetchMarkets": ["spot"]},
        })
    return _binance_client


def _fetch_latest_price(asset: str) -> float:
    """Best-effort latest close price — tries CCXT REST, falls back to last parquet row."""
    from live_feed import SYMBOL_MAP
    symbol = SYMBOL_MAP.get(asset)
    if symbol:
        try:
            ticker = _get_binance_client().fetch_ticker(symbol)
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


