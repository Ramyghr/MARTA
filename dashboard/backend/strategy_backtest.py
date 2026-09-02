"""User-configurable strategy backtest for the dashboard's Backtest page.

Standalone from rl/ablation.py's backtest harness (which is wired to MARTATradingEnv and
pulls in torch/stable-baselines3) — this only needs numpy/pandas plus the same OHLCV
`close` series `historical.py` already loads (raw + live-extended), so the dashboard
backend doesn't have to load the RL stack just to answer a backtest request.

Every strategy is a function `close: pd.Series -> signal: pd.Series` producing a raw
(same-bar) position in [-1, 1]. `run_backtest` shifts the signal by one bar before
applying it to that bar's return, so every strategy is decided from data available
*before* the bar it trades — no lookahead.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

import historical

TRANSACTION_COST = 0.001  # 10 bps per unit of position turnover — matches rl/env.py's default
ANNUALIZE_HOURLY = np.sqrt(24 * 365)
MIN_BARS = 60
_DEFAULT_LOOKBACK_DAYS = 365
_MAX_TRADES_RETURNED = 200
_MAX_CHART_POINTS = 1500


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return sma - n_std * std, sma, sma + n_std * std


def _sticky(enter: np.ndarray, exit_: np.ndarray, index: pd.Index) -> pd.Series:
    """Turn enter/exit boolean signals into a sticky 0/1 position: enters when flat and
    ``enter`` fires, exits when in position and ``exit_`` fires, otherwise holds state."""
    n = len(enter)
    pos = np.zeros(n)
    in_pos = False
    for i in range(n):
        if not in_pos and enter[i]:
            in_pos = True
        elif in_pos and exit_[i]:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=index)


# ── Strategies ──────────────────────────────────────────────────────────────────

def _sig_buy_hold(close: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=close.index)


def _sig_ma_crossover(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)


def _sig_trend_following(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    return _sig_ma_crossover(close, fast, slow)


def _sig_rsi_oversold(close: pd.Series, period: int = 14, oversold: float = 30.0, exit_level: float = 50.0) -> pd.Series:
    rsi = _rsi(close, period)
    return _sticky((rsi < oversold).to_numpy(), (rsi > exit_level).to_numpy(), close.index)


def _sig_macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line, signal_line = _macd(close, fast, slow, signal)
    return pd.Series(np.sign((macd_line - signal_line).to_numpy()), index=close.index)


def _sig_resistance_breakout(close: pd.Series, lookback: int = 20) -> pd.Series:
    prior_high = close.shift(1).rolling(lookback).max()
    sma = close.rolling(lookback).mean()
    enter = (close > prior_high).to_numpy()
    exit_ = (close < sma).to_numpy()
    return _sticky(enter, exit_, close.index)


def _sig_bollinger_band(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    lower, mid, _upper = _bollinger(close, window, n_std)
    enter = (close <= lower).to_numpy()
    exit_ = (close >= mid).to_numpy()
    return _sticky(enter, exit_, close.index)


def _sig_donchian_channel(close: pd.Series, window: int = 20) -> pd.Series:
    prior_high = close.shift(1).rolling(window).max()
    prior_low = close.shift(1).rolling(window).min()
    enter = (close > prior_high).to_numpy()
    exit_ = (close < prior_low).to_numpy()
    return _sticky(enter, exit_, close.index)


def _sig_mean_reversion(close: pd.Series, window: int = 20, z_entry: float = -2.0, increment: float = 1.0 / 3) -> pd.Series:
    """Adds `increment` of position on each sharp-dip bar (return z-score < z_entry) while
    below full size, and exits fully once price rebounds back above its rolling mean."""
    ret = close.pct_change()
    mu = ret.rolling(window).mean()
    sigma = ret.rolling(window).std()
    z = ((ret - mu) / sigma.replace(0.0, np.nan)).fillna(0.0).to_numpy()
    sma = close.rolling(window).mean().to_numpy()
    c = close.to_numpy()
    n = len(c)
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if cur > 0.0 and c[i] > sma[i]:
            cur = 0.0
        elif z[i] < z_entry and cur < 1.0 - 1e-9:
            cur = min(1.0, cur + increment)
        pos[i] = cur
    return pd.Series(pos, index=close.index)


def _sig_alpha_accelerator(
    close: pd.Series,
    trend_fast: int = 20, trend_slow: int = 50,
    rsi_period: int = 14, rsi_bull: float = 55.0, rsi_bear: float = 45.0,
    vol_window: int = 20, vol_lookback: int = 90, vol_percentile: float = 0.75,
) -> pd.Series:
    """Three independently-computed filters, AND-gated: trades only when trend, momentum,
    and volatility all agree — long/short otherwise stays flat rather than acting on a
    partial signal.

    - Trend: fast/slow SMA alignment (same mechanic as ma_crossover/trend_following).
    - Momentum: RSI confirming the same direction (>55 bull / <45 bear — a dead zone
      around 50 avoids whipsawing on a trend-only signal with no momentum behind it).
    - Volatility: realized vol (20h rolling std of returns) must sit at or below its own
      trailing 90h percentile — mutes entries during vol spikes/crashes, where trend and
      momentum are least reliable.

    Re-evaluated fresh every bar (no sticky enter/exit state) — a filter dropping out
    flattens the position immediately, which is the point of a risk-layered strategy.
    """
    fast_ma = close.rolling(trend_fast).mean()
    slow_ma = close.rolling(trend_slow).mean()
    trend_bull = fast_ma > slow_ma
    trend_bear = fast_ma < slow_ma

    rsi = _rsi(close, rsi_period)
    momentum_bull = rsi > rsi_bull
    momentum_bear = rsi < rsi_bear

    realized_vol = close.pct_change().rolling(vol_window).std()
    vol_threshold = realized_vol.rolling(vol_lookback).quantile(vol_percentile)
    vol_ok = realized_vol <= vol_threshold

    signal = pd.Series(0.0, index=close.index)
    signal[(trend_bull & momentum_bull & vol_ok).to_numpy()] = 1.0
    signal[(trend_bear & momentum_bear & vol_ok).to_numpy()] = -1.0
    return signal


_STRATEGIES: dict[str, dict[str, object]] = {
    "buy_hold": {
        "label": "Buy and hold", "category": "Strategies", "fn": _sig_buy_hold,
        "description": "Buys at the start and holds to the end. The benchmark for every other strategy.",
    },
    "ma_crossover": {
        "label": "Moving average crossover", "category": "Strategies", "fn": _sig_ma_crossover,
        "description": "Buys when the short average crosses above the long one, and exits when it crosses below.",
    },
    "rsi_oversold": {
        "label": "RSI oversold", "category": "Strategies", "fn": _sig_rsi_oversold,
        "description": "Buys when RSI is low and sells once it returns to normal. Works in a sideways market.",
    },
    "macd_signal": {
        "label": "MACD signal", "category": "Strategies", "fn": _sig_macd_signal,
        "description": "Changes direction as the MACD line crosses the signal line.",
    },
    "resistance_breakout": {
        "label": "Resistance breakout", "category": "Strategies", "fn": _sig_resistance_breakout,
        "description": "Enters when price breaks the last high; exits if the breakout turns out to be false.",
    },
    "bollinger_band": {
        "label": "Bollinger band", "category": "Strategies", "fn": _sig_bollinger_band,
        "description": "Buys when price touches the lower band and exits when it returns to the middle band.",
    },
    "trend_following": {
        "label": "Trend following", "category": "Popular", "fn": _sig_trend_following,
        "description": "Follows the trend while the 20- and 50-day averages align, and moves to cash once they break.",
    },
    "mean_reversion": {
        "label": "Mean reversion", "category": "Popular", "fn": _sig_mean_reversion,
        "description": "Buys incrementally on sharp dips and exits on the rebound rally.",
    },
    "donchian_channel": {
        "label": "Donchian channel", "category": "Popular", "fn": _sig_donchian_channel,
        "description": "Enters when it breaks the N-day high, and exits when it breaks the N-day low.",
    },
    "alpha_accelerator": {
        "label": "Alpha Accelerator", "category": "Advanced", "fn": _sig_alpha_accelerator,
        "description": "Multi-layer setup combining trend, momentum, and volatility filters.",
    },
}

# Not implemented — shown in the UI as disabled/"coming soon" cards. Unlike
# alpha_accelerator (built entirely from primitives already computed from `close`),
# these two need data this pipeline doesn't have for an arbitrary backtest date range:
# news_verified_trend needs a historical sentiment score (the news cache only covers
# static 2018-2021 headlines — agents/sentiment_agent.py), and qqq_sharpe_core needs
# US-equity price data entirely outside the crypto5-only pipeline.
_LOCKED_STRATEGIES: list[dict[str, str]] = [
    {
        "id": "news_verified_trend", "label": "News-verified trend", "category": "Advanced",
        "description": "Verifies the technical signal against the news score; doesn't take unconfirmed signals.",
    },
    {
        "id": "qqq_sharpe_core", "label": "QQQ Sharpe core", "category": "Advanced",
        "description": "Coming soon.",
    },
]


def list_strategies() -> dict:
    strategies = [
        {"id": sid, "label": s["label"], "category": s["category"], "description": s["description"], "locked": False}
        for sid, s in _STRATEGIES.items()
    ]
    strategies += [{**s, "locked": True} for s in _LOCKED_STRATEGIES]
    return {"strategies": strategies}


# ── Backtest engine ───────────────────────────────────────────────────────────

def _max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    dd = (peak - values) / np.maximum(peak, 1e-8)
    return float(np.max(dd))


def _extract_trades(index: pd.Index, positions: np.ndarray, strat_returns: np.ndarray) -> list[dict]:
    """Group contiguous same-direction bars into trades and compound each trade's own
    realized per-bar returns (already position-sized and net of transaction cost) for its
    PnL — robust to strategies like mean_reversion that vary position size mid-trade."""
    trades = []
    n = len(positions)
    i = 0
    while i < n:
        sign = np.sign(positions[i])
        if sign == 0:
            i += 1
            continue
        start_i = i
        while i < n and np.sign(positions[i]) == sign:
            i += 1
        end_i = i - 1
        block_return = float(np.prod(1.0 + strat_returns[start_i:end_i + 1]) - 1.0)
        trades.append({
            "entry_time": index[start_i].isoformat(),
            "exit_time": index[end_i].isoformat(),
            "direction": "long" if sign > 0 else "short",
            "pnl_pct": block_return,
        })
    return trades


def _downsample_equity_curve(index: pd.Index, strat_values: np.ndarray, bh_values: np.ndarray) -> list[dict]:
    n = len(index)
    step = max(1, n // _MAX_CHART_POINTS)
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return [
        {"timestamp": index[i].isoformat(), "strategy": float(strat_values[i]), "buy_hold": float(bh_values[i])}
        for i in idxs
    ]


def run_backtest(asset: str, strategy: str, start: str | None, end: str | None) -> dict:
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown or locked strategy: '{strategy}'")

    df = historical._load(asset)
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    start_ts = pd.Timestamp(start, tz="UTC") if start else end_ts - pd.Timedelta(days=_DEFAULT_LOOKBACK_DAYS)
    window = df[(df.index >= start_ts) & (df.index <= end_ts)]

    if len(window) < MIN_BARS:
        raise ValueError(
            f"Only {len(window)} bars available for {asset} in "
            f"[{start_ts.date()}, {end_ts.date()}] — need at least {MIN_BARS}"
        )

    close = window["close"].astype(float)
    spec = _STRATEGIES[strategy]
    signal_fn: Callable[[pd.Series], pd.Series] = spec["fn"]  # type: ignore[assignment]
    signal = signal_fn(close)
    positions = signal.shift(1).fillna(0.0).to_numpy()
    returns = close.pct_change().fillna(0.0).to_numpy()

    turnover = np.abs(np.diff(np.concatenate([[0.0], positions])))
    strat_returns = positions * returns - TRANSACTION_COST * turnover
    strat_values = np.cumprod(1.0 + strat_returns)

    bh_values = np.cumprod(1.0 + returns)  # frictionless buy & hold benchmark

    trades = _extract_trades(window.index, positions, strat_returns)
    n_trades = len(trades)
    win_rate = float(np.mean([t["pnl_pct"] > 0 for t in trades])) if trades else 0.0

    total_return = float(strat_values[-1] - 1.0)
    bh_total_return = float(bh_values[-1] - 1.0)
    sharpe = float(strat_returns.mean() / (strat_returns.std() + 1e-8) * ANNUALIZE_HOURLY)

    return {
        "asset": asset,
        "strategy": strategy,
        "strategy_label": spec["label"],
        "start": window.index[0].isoformat(),
        "end": window.index[-1].isoformat(),
        "n_bars": len(window),
        "metrics": {
            "total_return": total_return,
            "buy_hold_total_return": bh_total_return,
            "vs_buy_hold": total_return - bh_total_return,
            "sharpe": sharpe,
            "max_drawdown": _max_drawdown(strat_values),
            "win_rate": win_rate,
            "n_trades": n_trades,
        },
        "equity_curve": _downsample_equity_curve(window.index, strat_values, bh_values),
        "trades": trades[-_MAX_TRADES_RETURNED:],
    }
