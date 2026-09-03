"""Live (not backtest) weekly-cadence PPO signal generation for BTC_USDT.

Opt-in (app_settings.get_weekly_ppo_enabled(), default False). When enabled, this
becomes THE signal source for BTC_USDT in live_signal.get_live_signal() — it replaces
both ppo_inference's hourly PPO path and the regime-policy fallback, and the debate
override logic is skipped for this asset too. That's deliberate, not an oversight:
layering an hourly debate override on top of a policy specifically trained and
validated at 168h decision cadence would undermine the reason it works — see
rl/train_weekly_smoketest.py, and CLAUDE.md's "Dense BTC debate coverage" section,
which already measured that denser debate coverage doesn't change this class of
checkpoint's behavior at all.

The real-time scheduler ticks hourly (signal_scheduler.py), but this policy should only
actually decide once a week — so unlike every other signal source here, this one is
stateful across calls. State (data/processed/weekly_ppo_state.json): the last real
decision's timestamp, direction and target_signed, plus a small synthetic portfolio
value/peak/bars-in-position — kept separate from paper_db.py's shared-capital
bookkeeping (which only tracks realized/marked-to-market value+position, not "is it
time to decide again," and mixing this policy's own training-consistent portfolio
trajectory into the shared pool that other signals/assets also write to would put its
observation off the distribution it was validated on). Between decisions, the held
position's return is caught up in one batch (not hour-by-hour) using the same
close_return series the backtest compounds — same result, computed at the next decision
instead of on every hourly tick, since nothing except that catch-up sum depends on the
intermediate hours.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import ppo_strategy_backtest as backtest

MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
_STATE_PATH = MARTA_ROOT / "data" / "processed" / "weekly_ppo_state.json"

_DEFAULT_STATE = {
    "last_decision_ts": None,
    "target_signed": 0.0,
    "bars_in_position": 0,
    "portfolio_value": backtest.INITIAL_CAPITAL,
    "peak_value": backtest.INITIAL_CAPITAL,
}


def _read_state() -> dict:
    if not _STATE_PATH.exists():
        return dict(_DEFAULT_STATE)
    try:
        return {**_DEFAULT_STATE, **json.loads(_STATE_PATH.read_text())}
    except Exception:
        return dict(_DEFAULT_STATE)


def _write_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2, default=float))


def _action_from_signed(target_signed: float) -> str:
    if target_signed > 1e-9:
        return "long"
    if target_signed < -1e-9:
        return "short"
    return "hold"


def reset_state() -> dict:
    """Used by the settings toggle when weekly PPO is turned off then back on later —
    starting from a stale multi-week-old state would make the very next tick catch up
    a long, possibly-irrelevant return history under the wrong regime context."""
    state = dict(_DEFAULT_STATE)
    _write_state(state)
    return state


def get_weekly_signal(obs_mode: str, now: pd.Timestamp) -> dict:
    """Returns {"action", "position_size", "source", "next_decision_at",
    "is_new_decision"} for BTC_USDT at `now` — computes a fresh weekly decision only if
    >= DECISION_INTERVAL hours have passed since the last one; otherwise holds the
    previously decided position untouched (still reports a real next_decision_at so the
    dashboard can show a countdown instead of looking stuck).
    """
    state = _read_state()
    last_ts = pd.Timestamp(state["last_decision_ts"], tz="UTC") if state["last_decision_ts"] else None
    hours_since = (now - last_ts).total_seconds() / 3600 if last_ts is not None else None

    if last_ts is not None and hours_since is not None and hours_since < backtest.DECISION_INTERVAL:
        next_at = last_ts + pd.Timedelta(hours=backtest.DECISION_INTERVAL)
        return {
            "action": _action_from_signed(state["target_signed"]),
            "position_size": float(abs(state["target_signed"])),
            "source": f"weekly_ppo_{obs_mode}",
            "next_decision_at": next_at.isoformat(),
            "is_new_decision": False,
        }

    from rl import obs_builder

    model = backtest._load_model(obs_mode)
    full_df = backtest._load_feature_df()
    end_pos = full_df.index.searchsorted(now, side="right")
    t = end_pos - 1
    if t < backtest.WINDOW_SIZE - 1:
        raise ValueError("Not enough live feature history for a weekly PPO decision yet.")

    arr = full_df[backtest.FEATURE_COLS].to_numpy(dtype=np.float32)
    returns = full_df["close_return"].to_numpy(dtype=np.float64)
    vol = backtest._vol_forecast(full_df)
    regime_probs = backtest._load_regime_probs(full_df.index) if obs_mode == "raw_regime" else None

    prev_signed = float(state["target_signed"])
    portfolio_value = float(state["portfolio_value"])
    peak_value = float(state["peak_value"])
    bars_in_position = int(state["bars_in_position"])

    # Catch up the held position's realized return over the elapsed interval since the
    # last decision (hour-by-hour compounding, same as rl/env.py's step() and the
    # backtest's own loop) — first decision ever (last_ts is None) starts flat/fresh,
    # nothing to catch up.
    if last_ts is not None:
        start_pos = full_df.index.searchsorted(last_ts, side="right")
        for idx in range(start_pos, end_pos):
            r = returns[idx]
            if np.isnan(r):
                r = 0.0
            portfolio_value += prev_signed * r * portfolio_value
            peak_value = max(peak_value, portfolio_value)

    asset_id_vec = np.zeros(len(backtest.CANONICAL_ASSETS), dtype=np.float32)
    asset_id_vec[backtest.CANONICAL_ASSETS.index(backtest.ASSET)] = 1.0
    consensus_vec = np.zeros(backtest.CONSENSUS_DIM, dtype=np.float32)  # no debate cache — see module docstring

    portfolio_dim = len(backtest.CANONICAL_ASSETS) + 4
    port_vec = np.zeros(portfolio_dim, dtype=np.float32)
    port_vec[backtest.CANONICAL_ASSETS.index(backtest.ASSET)] = prev_signed
    base = len(backtest.CANONICAL_ASSETS)
    port_vec[base] = portfolio_value / backtest.INITIAL_CAPITAL - 1.0
    current_dd = max(0.0, (peak_value - portfolio_value) / max(peak_value, 1e-8))
    port_vec[base + 1] = current_dd
    port_vec[base + 2] = 0.0  # steps_since_trade — not meaningfully trackable across restarts; neutral default
    port_vec[base + 3] = float(vol[t])

    market_block = obs_builder.market_block(obs_mode, arr, t, backtest.WINDOW_SIZE, regime_probs_arr=regime_probs)
    obs = np.concatenate([market_block, consensus_vec, port_vec, asset_id_vec]).astype(np.float32)
    obs = np.clip(obs, -10.0, 10.0)

    action_arr, _ = model.predict(obs, deterministic=True)
    arr_action = np.asarray(action_arr, dtype=np.float64).reshape(-1)
    direction = int(round(float(np.clip(arr_action[0], 0, 2))))
    frac = float(np.clip(arr_action[1], 0.01, 0.99))

    risk_cap = backtest._risk_cap_from_vol(float(vol[t])) * backtest.MAX_LEVERAGE
    target_signed = backtest._direction_sign(direction) * frac * risk_cap

    if (
        abs(prev_signed) > 1e-9
        and bars_in_position < backtest.MIN_HOLD_BARS
        and current_dd < backtest.MIN_HOLD_STOP_LOSS_DD
        and np.sign(target_signed) != np.sign(prev_signed)
    ):
        target_signed = prev_signed

    turnover = abs(target_signed - prev_signed)
    cost = backtest.TRANSACTION_COST * turnover * portfolio_value
    portfolio_value -= cost
    peak_value = max(peak_value, portfolio_value)

    if abs(target_signed) < 1e-9:
        bars_in_position = 0
    elif np.sign(target_signed) != np.sign(prev_signed):
        bars_in_position = 1
    else:
        bars_in_position += 1

    _write_state({
        "last_decision_ts": now.isoformat(),
        "target_signed": target_signed,
        "bars_in_position": bars_in_position,
        "portfolio_value": portfolio_value,
        "peak_value": peak_value,
    })

    return {
        "action": _action_from_signed(target_signed),
        "position_size": float(abs(target_signed)),
        "source": f"weekly_ppo_{obs_mode}",
        "next_decision_at": (now + pd.Timedelta(hours=backtest.DECISION_INTERVAL)).isoformat(),
        "is_new_decision": True,
    }
