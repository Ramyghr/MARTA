"""Native weekly-cadence PPO checkpoints as selectable Strategy Tester entries.

strategy_backtest.py deliberately avoids torch/stable-baselines3 (every other strategy
is a `close: pd.Series -> pd.Series` rule) — this module holds the one exception, kept
separate so its heavy imports (torch, stable_baselines3, rl.env) only happen when a PPO
strategy is actually requested, not on every dashboard backend import.

Faithfully reproduces rl/env.py's MARTATradingEnv step() logic for decision_interval=168
(CVaR-based position sizing, transaction cost charged once per decision, the
min_hold_bars reversal lock, hour-by-hour compounding within a held decision) as a
plain walk-forward loop — this is inference + bookkeeping over already-computed data,
not training, so the full Gymnasium env (reset/reward/termination machinery) isn't
needed. See rl/train_weekly_smoketest.py for how these two checkpoints were trained
(decision_interval=168, BTC_USDT-only, crypto5 5-asset canonical/portfolio slots) and
CLAUDE.md's "Dense BTC debate coverage" section for their validated Sharpe (~2.04 raw,
~1.74 raw_regime).

Output position values feed back into strategy_backtest.py's existing turnover/returns/
trade-extraction/metrics pipeline unchanged (same [-1, 1]-scaled position convention,
same 0.001 transaction cost) — this module only produces the position series.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Path(__file__).parent.parent.parent assumes dashboard/backend/'s on-disk nesting
# under the repo root — true on the host, but the docker container's bind mount is
# flat (/dashboard_backend, not /app/dashboard/backend), so that walk overshoots to
# "/" there. Every other module in this directory (main.py, live_signal.py,
# regime_service.py, ...) already routes through the MARTA_ROOT env var for exactly
# this reason (set to /app in the container) — this one needs the same fix, found
# 2026-09-03 when it silently reported both checkpoints as missing/locked.
MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
_CRYPTO5_PROCESSED = MARTA_ROOT / "data" / "processed" / "crypto5_multihorizon"
_CHECKPOINT_DIR = MARTA_ROOT / "rl" / "checkpoints"
_LIVE_REGIME_LOG = MARTA_ROOT / "data" / "processed" / "regime_live_log.parquet"

# rl/train.py's CRYPTO5_CANONICAL_ASSETS — portfolio/asset-id one-hot slot order the
# checkpoints were trained with. Only BTC_USDT (index 0) is exposed as a strategy here;
# the checkpoints were only ever trained/evaluated on BTC_USDT (all_assets=["BTC_USDT"]
# in build_weekly_config), so the other 4 slots exist in the observation shape but are
# always zero — not a limitation specific to this backtest.
CANONICAL_ASSETS = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]
ASSET = "BTC_USDT"

# regime_service.py's _CRYPTO5_FEATURE_COLS — the exact 22-column order
# normalize_features_multihorizon() produces and the crypto5 HMM/PPO checkpoints were
# both fit on. Order matters: this is fed straight into `.values`, not selected by name
# at inference time inside the model.
FEATURE_COLS = [
    "atr", "adx", "rsi", "bb_width", "macd_histogram", "obv_raw", "vol_zscore",
    "realized_vol", "vol_ratio", "return_skew", "return_kurtosis", "autocorr_lag1",
    "realized_vol_raw", "close_return", "realized_vol_z90d", "realized_vol_z180d",
    "atr_z90d", "atr_z180d", "bb_width_z90d", "bb_width_z180d", "vol_ratio_z90d",
    "vol_ratio_z180d",
]

WINDOW_SIZE = 60
DECISION_INTERVAL = 168
CONSENSUS_DIM = 16
N_REGIME_STATES = 6
TRANSACTION_COST = 0.001  # matches rl/env.py's default and strategy_backtest.py's own
MAX_LEVERAGE = 1.0
MIN_HOLD_BARS = 4       # decisions, not hours — 4 weekly decisions at this cadence
MIN_HOLD_STOP_LOSS_DD = 0.05
INITIAL_CAPITAL = 10_000.0
MIN_ROWS_REQUIRED = WINDOW_SIZE + DECISION_INTERVAL  # need >=1 full decision interval

_CHECKPOINTS = {
    "raw": _CHECKPOINT_DIR / "ppo_best_crypto5_weekly_BTC_USDT_raw_seed0.zip",
    "raw_regime": _CHECKPOINT_DIR / "ppo_best_crypto5_weekly_BTC_USDT_raw_regime_seed0.zip",
}

_model_cache: dict[str, object] = {}


def available() -> dict[str, bool]:
    """Which obs_mode checkpoints actually exist on disk right now."""
    return {mode: path.exists() for mode, path in _CHECKPOINTS.items()}


def _load_model(obs_mode: str):
    if obs_mode not in _CHECKPOINTS:
        raise ValueError(f"Unknown PPO obs_mode: {obs_mode!r}")
    if obs_mode not in _model_cache:
        from stable_baselines3 import PPO
        from rl.policy import MARTAActorCriticPolicy
        path = _CHECKPOINTS[obs_mode]
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        _model_cache[obs_mode] = PPO.load(
            str(path), device="cpu", custom_objects={"policy_class": MARTAActorCriticPolicy}
        )
    return _model_cache[obs_mode]


def _load_feature_df() -> pd.DataFrame:
    """{ASSET}_normalized_live.parquet — the same real, live-bridge-extended 22-column
    panel the crypto5 live regime HMM reads (dashboard/backend/live_feature_bridge.py),
    reaching today rather than stopping at the training-time static file's boundary.
    """
    path = _CRYPTO5_PROCESSED / f"{ASSET}_normalized_live.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No live-bridge feature file for {ASSET}: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df[FEATURE_COLS].astype(np.float32)


def _load_regime_probs(index: pd.DatetimeIndex) -> np.ndarray:
    """Regime probability vector per row of `index`, forward-fill-only (never a future
    value) — regime_stream_causal.parquet for history up to its frozen 2025-12-31
    boundary, regime_live_log.parquet for the gap since (see CLAUDE.md's "EURUSD/GOLD
    live regime" section for why crypto5's static stream is frozen and this log exists).
    Missing regime_prob_4/5 in a minority of recent live-log rows (a separate, already
    flagged data-quality issue) fall back to 0.0 via build_regime_prob_series's own
    fillna — degraded, not broken, for the handful of decision points that might land on
    one of those rows.
    """
    from rl import obs_builder

    frames = []
    causal_path = _CRYPTO5_PROCESSED / "regime_stream_causal.parquet"
    if causal_path.exists():
        frames.append(pd.read_parquet(causal_path, engine="pyarrow"))
    if _LIVE_REGIME_LOG.exists():
        live = pd.read_parquet(_LIVE_REGIME_LOG, engine="pyarrow").reset_index()
        frames.append(live)
    prob_cols = [f"regime_prob_{i}" for i in range(N_REGIME_STATES)]
    if not frames:
        return np.zeros((len(index), N_REGIME_STATES), dtype=np.float32)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["asset"] == ASSET]
    return obs_builder.build_regime_prob_series(combined, prob_cols, ASSET, index)


def _vol_forecast(df: pd.DataFrame) -> np.ndarray:
    """GARCH(1,1) volatility forecast, same formula/cache as training/eval
    (rl.env.precompute_vol_forecast) — reused directly rather than reimplemented, so a
    live-reaching window transparently extends (not bypasses) the existing cache."""
    from rl.env import precompute_vol_forecast
    return precompute_vol_forecast(df, ASSET)


def _direction_sign(direction: int) -> float:
    return {0: 1.0, 1: -1.0, 2: 0.0}.get(direction, 0.0)


def _risk_cap_from_vol(vol: float) -> float:
    from rl.env import risk_cap_from_vol
    return risk_cap_from_vol(vol)


def run(obs_mode: str, index: pd.DatetimeIndex) -> pd.Series:
    """Walk-forward PPO position series aligned to `index` (an hourly DatetimeIndex,
    already restricted to the requested backtest window by strategy_backtest.py).

    Needs WINDOW_SIZE bars of lookback before `index[0]` for the raw feature window —
    those are pulled from the full feature file (not just `index`'s own range), exactly
    as a real walk-forward evaluation would have context before the window starts.
    Returns one position value per hour in `index`, held constant (a step function)
    between decision points every DECISION_INTERVAL hours — turnover is naturally zero
    except at the bar a new decision actually changes it, so strategy_backtest.py's
    existing per-bar transaction-cost computation already charges cost once per
    decision "for free", with no special-casing needed there.
    """
    from rl import obs_builder

    model = _load_model(obs_mode)
    full_df = _load_feature_df()

    start_pos = full_df.index.searchsorted(index[0])
    lookback_start = max(0, start_pos - WINDOW_SIZE + 1)
    end_pos = full_df.index.searchsorted(index[-1], side="right")
    window_df = full_df.iloc[lookback_start:end_pos]

    if len(window_df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"Only {len(window_df)} feature rows available for the PPO strategy "
            f"(need >= {MIN_ROWS_REQUIRED} = window_size + one decision interval) — "
            f"pick a longer date range."
        )

    arr = window_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    returns = window_df["close_return"].to_numpy(dtype=np.float64)
    vol = _vol_forecast(window_df)
    regime_probs = _load_regime_probs(window_df.index) if obs_mode == "raw_regime" else None

    asset_id_vec = np.zeros(len(CANONICAL_ASSETS), dtype=np.float32)
    asset_id_vec[CANONICAL_ASSETS.index(ASSET)] = 1.0
    consensus_vec = np.zeros(CONSENSUS_DIM, dtype=np.float32)  # no debate cache for arbitrary backtest dates

    n = len(window_df)
    positions = np.zeros(n, dtype=np.float64)

    portfolio_value = INITIAL_CAPITAL
    peak_value = INITIAL_CAPITAL
    signed_position = 0.0
    steps_since_trade = 0
    bars_in_position = 0

    # First decision point needs >=WINDOW_SIZE-1 bars of lookback already available.
    t = WINDOW_SIZE - 1
    while t < n:
        portfolio_dim = len(CANONICAL_ASSETS) + 4
        port_vec = np.zeros(portfolio_dim, dtype=np.float32)
        port_vec[CANONICAL_ASSETS.index(ASSET)] = signed_position
        base = len(CANONICAL_ASSETS)
        port_vec[base] = portfolio_value / INITIAL_CAPITAL - 1.0
        port_vec[base + 1] = max(0.0, (peak_value - portfolio_value) / max(peak_value, 1e-8))
        port_vec[base + 2] = min(steps_since_trade / 100.0, 1.0)
        port_vec[base + 3] = float(vol[t])

        market_block = obs_builder.market_block(
            obs_mode, arr, t, WINDOW_SIZE,
            regime_probs_arr=regime_probs,
        )
        obs = np.concatenate([market_block, consensus_vec, port_vec, asset_id_vec]).astype(np.float32)
        obs = np.clip(obs, -10.0, 10.0)

        action, _ = model.predict(obs, deterministic=True)
        arr_action = np.asarray(action, dtype=np.float64).reshape(-1)
        direction = int(round(float(np.clip(arr_action[0], 0, 2))))
        frac = float(np.clip(arr_action[1], 0.01, 0.99))

        risk_cap = _risk_cap_from_vol(float(vol[t])) * MAX_LEVERAGE
        target_signed = _direction_sign(direction) * frac * risk_cap

        prev_signed = signed_position
        # Min-hold lock: block flattening/reversing before MIN_HOLD_BARS decisions have
        # elapsed, unless drawdown already exceeds the stop-loss escape hatch — mirrors
        # rl/env.py's step() exactly (see its docstring for why this exists: without it
        # PPO's own Beta-head noise between adjacent decisions can flap a real position).
        current_dd = max(0.0, (peak_value - portfolio_value) / max(peak_value, 1e-8))
        if (
            abs(prev_signed) > 1e-9
            and bars_in_position < MIN_HOLD_BARS
            and current_dd < MIN_HOLD_STOP_LOSS_DD
            and np.sign(target_signed) != np.sign(prev_signed)
        ):
            target_signed = prev_signed

        turnover = abs(target_signed - prev_signed)
        cost = TRANSACTION_COST * turnover * portfolio_value
        traded = turnover > 1e-9
        portfolio_value -= cost
        peak_value = max(peak_value, portfolio_value)

        span = min(DECISION_INTERVAL, n - t - 1)
        for step_i in range(1, span + 1):
            idx = t + step_i
            r = returns[idx]
            if np.isnan(r):
                r = 0.0
            portfolio_value += target_signed * r * portfolio_value
            peak_value = max(peak_value, portfolio_value)
            positions[idx] = target_signed

        signed_position = target_signed
        steps_since_trade = 0 if traded else steps_since_trade + 1
        if abs(target_signed) < 1e-9:
            bars_in_position = 0
        elif np.sign(target_signed) != np.sign(prev_signed):
            bars_in_position = 1
        else:
            bars_in_position += 1

        t += DECISION_INTERVAL

    pos_series = pd.Series(positions, index=window_df.index)
    return pos_series.reindex(index).fillna(0.0)
