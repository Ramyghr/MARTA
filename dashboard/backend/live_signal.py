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
     relies on, since trigger-signal calls compute_signal() with no explicit override.

Split into two phases (per-user paper trading migration) since the decision itself is
asset-level (computed once per tick) but booking it is per-user:
  - compute_signal(asset, ...)      — resolves the decision once, no paper_db writes.
  - book_signal(user_id, asset, ...) — books that decision into one user's account.
  - get_live_signal(user_id, asset, ...) — convenience wrapper doing both, for callers
    that only ever need one user's view (e.g. a single browser session's manual trigger).

The one exception: when the trained PPO checkpoint is the active source, its observation
includes *this user's own* portfolio state (signed position, hours since last trade) —
genuinely per-user input, not asset-level. compute_signal() can't resolve that branch, so
it returns a `needs_per_user_ppo` marker instead, and book_signal() calls
ppo_inference.predict(..., user_id=...) itself, once per user, before booking.

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
    resulting trade: /api/rl/trigger-signal calls compute_signal() with no explicit
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

    def _ts(d: dict) -> "pd.Timestamp":
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


def compute_signal(asset: str, last_debate_action: Optional[str] = None) -> dict:
    """Resolve the trading decision for `asset` right now — once per asset per tick,
    independent of any user. No paper_db reads/writes here, EXCEPT the PPO branch
    (priority 1) is deferred: its own signal depends on per-user portfolio state, so
    this returns a `needs_per_user_ppo` marker instead of a resolved action/size for
    that one case, and the caller-supplied/automatic debate override (priorities 3/4,
    both asset-level) is pre-resolved into `override` so book_signal() can apply it
    identically after resolving PPO per user.
    """
    import pandas as pd
    import regime_service
    import app_settings

    now = pd.Timestamp("now", tz="UTC")
    # Live, not the historical nearest-lookup — this drives the crisis-override guard
    # below, so it needs to reflect right now, not regime_stream.parquet's last
    # precomputed row (frozen at 2024-12-31 as of this fix). log=True here because
    # compute_signal() is the natural once-per-hour-per-asset cadence (the scheduler
    # tick) to grow the live regime audit log at — see regime_service.get_live_regime.
    regime_info = regime_service.get_live_regime(asset, now, log=True)
    regime_label = regime_info.get("regime_label", 0)
    regime_prob = regime_info.get("probabilities", {})

    # --- Signal priority 0: weekly-cadence PPO — opt-in, BTC_USDT only, takes over
    # completely for this asset (see weekly_ppo_live.py's module docstring for why the
    # debate-override priorities 3-4 below are deliberately skipped when this is
    # active — they'd fight a policy that's only supposed to decide once a week).
    weekly_active = asset == "BTC_USDT" and app_settings.get_weekly_ppo_enabled()
    needs_per_user_ppo = False

    if weekly_active:
        import weekly_ppo_live
        weekly = weekly_ppo_live.get_weekly_signal(app_settings.get_weekly_ppo_obs_mode(), now)
        action = weekly["action"]
        size = weekly["position_size"]
        source = weekly["source"]
    elif app_settings.get_rl_execution_enabled():
        # --- Signal priority 1: trained PPO — opt-in only (see app_settings.py) ---
        # Default off per docs/RESULTS_AND_METHODOLOGY.md Section 9: a complete,
        # controlled ablation found no crypto5 RL configuration with a defensible edge
        # over buy-and-hold. The actual predict() call is deferred to book_signal()
        # since it needs per-user portfolio state — action/size here are placeholders,
        # never booked as-is.
        needs_per_user_ppo = True
        action, size, source = "hold", 0.0, "ppo"
    else:
        # --- Signal priority 2: rule-based fallback ---
        action, size = _generate_regime_policy().get(regime_label, ("hold", 0.01))
        source = "regime_policy"

    # --- Signal priorities 3/4: debate override — asset-level, resolved once here
    # regardless of the PPO branch's per-user deferral. Never applies in stress/crisis
    # (regime_label == 2) or when weekly-cadence PPO owns this asset.
    override: Optional[dict] = None
    if not weekly_active and regime_label != 2:
        if last_debate_action and last_debate_action in ("long", "short", "hold"):
            # Priority 3: explicit caller-supplied override (direction only).
            override = {"action": last_debate_action, "position_size": None, "tag": "+debate"}
        else:
            # Priority 4: automatic high-conviction debate override (direction + size).
            debate_override = _get_debate_override(asset, now)
            if debate_override:
                override = {
                    "action": debate_override["action"],
                    "position_size": debate_override["position_size"],
                    "tag": "+debate_override",
                }

    ppo_override_flag = False
    if not needs_per_user_ppo and override is not None:
        if source == "ppo" and override["action"] != action:
            ppo_override_flag = True
        action = override["action"]
        if override["position_size"] is not None:
            size = override["position_size"]
        source = f"{source}{override['tag']}"

    price = _fetch_latest_price(asset)
    debate_info = _load_most_recent_debate(asset, now) or {}

    return {
        "asset": asset,
        "now": now,
        "regime_label": regime_label,
        "regime_description": regime_info.get("regime_description", ""),
        "regime_prob": regime_prob,
        "action": action,
        "position_size": size,
        "price": price,
        "source": source,
        "ppo_override": ppo_override_flag,
        "needs_per_user_ppo": needs_per_user_ppo,
        "override": override,
        "debate_info": debate_info,
    }


def book_signal(user_id: int, asset: str, signal: dict) -> dict:
    """Apply an already-computed signal (from compute_signal) to one user's paper
    account. Called once per user per tick by signal_scheduler's fan-out loop, or once
    for a single user by the get_live_signal() convenience wrapper below."""
    import paper_db
    import app_settings
    import ppo_inference

    action = signal["action"]
    size = signal["position_size"]
    source = signal["source"]
    ppo_override = signal["ppo_override"]
    price = signal["price"]
    now = signal["now"]
    regime_label = signal["regime_label"]
    regime_description = signal["regime_description"]
    regime_prob = signal["regime_prob"]
    debate_info = signal["debate_info"]
    regime_at_decision = regime_label
    ppo_ran = False

    if signal["needs_per_user_ppo"]:
        ppo_result = ppo_inference.predict(asset, now, user_id=user_id)
        ppo_ran = ppo_result is not None
        if ppo_ran:
            action = ppo_result["action"]
            size = ppo_result["position_size"]
        else:
            action, size = _generate_regime_policy().get(regime_at_decision, ("hold", 0.01))
            source = "regime_policy"
        override = signal["override"]
        if override is not None:
            if action != override["action"]:
                ppo_override = True
            action = override["action"]
            if override["position_size"] is not None:
                size = override["position_size"]
            source = f"{source}{override['tag']}"

    ppo_confidence = size if ppo_ran else None

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
            "regime_description": regime_description,
            "source": "debate_override" if "+debate" in source else source,
            "ppo_confidence": ppo_confidence,
            "debate_id": debate_info.get("debate_id"),
            "agent_theses": debate_info.get("agent_theses", []),
            "fund_manager_rationale": debate_info.get("fund_manager_rationale"),
            "ppo_override": ppo_override,
        }
        pending = paper_db.create_pending_trade(
            user_id, asset, action, size, price, regime_label=regime_at_decision, source=source,
            reasoning=pending_reasoning,
        )
        # Telegram is scoped to the primary account only (see telegram_bot.py) — only
        # notify for that one user's pending trade, not once per registered user, since
        # every user's pending trade this tick shares the same underlying decision and
        # there's only one Telegram chat to notify regardless of how many accounts exist.
        import auth_db
        if user_id == auth_db.get_primary_user_id():
            import telegram_service
            telegram_service.fire_and_forget(telegram_service.notify_pending_trade(
                pending_id=pending["id"], asset=asset, action=action, position_size=size,
                price=price, regime_label=regime_at_decision, source=source,
            ))
        current = paper_db.get_portfolio(user_id, asset)
        return {
            "asset": asset,
            "regime_label": regime_label,
            "regime_description": regime_description,
            "regime_prob": regime_prob,
            "action": action,
            "position_size": size,
            "price": price,
            "paper_portfolio_value": current["portfolio_value"],
            "paper_total_return": current["total_return"],
            "paper_n_trades": current["n_trades"],
            "paper_trades": paper_db.get_trades(user_id, asset, limit=50),
            "source": source,
            "ppo_loaded": ppo_inference.is_loaded(),
            "pending_approval": True,
            "pending_trade_id": pending["id"],
        }

    paper = paper_db.update(user_id, asset, action, size, price, regime_label=regime_at_decision)
    trades = paper_db.get_trades(user_id, asset, limit=50)

    trade_id = paper.get("trade_id")
    if trade_id is not None:
        reasoning = {
            "asset": asset,
            "timestamp": now.isoformat(),
            "regime_label": regime_at_decision,
            "regime_description": regime_description,
            "source": "debate_override" if "+debate" in source else source,
            "ppo_confidence": ppo_confidence,
            "debate_id": debate_info.get("debate_id"),
            "agent_theses": debate_info.get("agent_theses", []),
            "fund_manager_rationale": debate_info.get("fund_manager_rationale"),
            "ppo_override": ppo_override,
        }
        paper_db.save_trade_reasoning(trade_id, user_id, reasoning)

    return {
        "asset": asset,
        "regime_label": regime_label,
        "regime_description": regime_description,
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


def get_live_signal(user_id: int, asset: str, last_debate_action: Optional[str] = None) -> dict:
    """Convenience wrapper for a single user: compute the signal then book it for just
    this one account. The scheduler's per-user fan-out calls compute_signal()/
    book_signal() directly instead (one shared computed signal, booked N times)."""
    signal = compute_signal(asset, last_debate_action=last_debate_action)
    return book_signal(user_id, asset, signal)


_binance_client = None  # module-level cache — see _get_binance_client()


def _get_binance_client():
    """Cached ccxt.binance() instance, constructed once per process.

    Confirmed 2026-09-03: a fresh ccxt.binance() instance lazily calls load_markets()
    on its first API call (fetch_ticker included), fetching Binance's full exchange
    metadata — measured ~4s on this network. _fetch_latest_price is on the hot path of
    every signal computation (every Trading-page mount, every manual trigger, every
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
