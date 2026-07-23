"""MARTATradingEnv — custom Gymnasium environment for the MARTA PPO execution agent.

State (152-dim):
    [128-dim regime embedding | 16-dim agent consensus | 8-dim portfolio state]

Action (hybrid, exposed to SB3 as a flat 2-dim Box — the "flattened" form):
    action[0] = direction  (0=long, 1=short, 2=hold), sampled from a Categorical head
    action[1] = position_fraction in (0, 1), sampled from a Beta head

The semantic Dict action space is exposed as ``self.action_space_dict`` for reference;
SB3's PPO does not support Dict action spaces, so the flat Box is the training interface
and :meth:`unflatten_action` recovers the {"direction", "position_fraction"} dict.

Reward:
    r_t = ΔPnL / CVaR_t − λ × drawdown_penalty

CVaR_t is computed inline from a GARCH(1,1) volatility forecast using the exact same
formula as ``agents.risk_agent.RiskAgent`` (Cornish-Fisher normal CVaR at 95%). The GARCH
fits are precomputed on a coarse weekly stride and cached to disk — the arch model is never
refit per-step (which would make training intractable) and the LLM debate system is never
invoked during RL.

Walk-forward split (strict — PPO never trains on validation timestamps):
    train: 2018-01-10 .. 2023-12-31
    val:   2024-01-01 .. 2024-12-31
    test:  2025-01-01 .. 2025-12-31  (never touched here — see run_test_set.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from scipy.stats import norm

from encoder.model import TFTRegimeEncoder
from utils.logger import logger

# Canonical asset ordering — fixes the portfolio position slots so the 8-dim portfolio
# vector is stable across curriculum phases.
CANONICAL_ASSETS: list[str] = ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"]

REGIME_DIM = 128
CONSENSUS_DIM = 16
PORTFOLIO_DIM = 8
STATE_DIM = REGIME_DIM + CONSENSUS_DIM + PORTFOLIO_DIM  # 152
N_POSITION_SLOTS = 4  # first 4 portfolio dims hold per-asset signed positions

# 2026-07-13 walk-forward split (was train ≤2022-12-31 / val 2023 for the
# checkpoints trained before this extension — see git history). Test is now 2025
# (see run_test_set.py); encoder/HMM train through the same 2023-12-31 cutoff.
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")
VAL_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
VAL_END   = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
VAL_END_EXTENDED = VAL_END  # legacy alias (pre-2024-extension --extended-val); now a no-op

# 95% normal CVaR multiplier — matches RiskAgent: -pdf(ppf(0.05))/0.05
_Z05 = float(norm.ppf(0.05))
_CVAR_MULT = float(-norm.pdf(_Z05) / 0.05)  # ~ -2.063 (applied to vol → negative CVaR)

_GARCH_WINDOW = 90 * 24       # 90 days hourly (same lookback as RiskAgent)
_GARCH_STRIDE = 168           # refit weekly, forward-fill between refits
_CACHE_DIR = Path("data/processed/rl_cache")

_DIRECTION_SIGN = {0: 1.0, 1: -1.0, 2: 0.0}  # long / short / hold

# _REGIME_POSITION_SCALE — DISABLED (kept commented out as a historical reference).
# This was calibrated for an earlier HMM fit. When the HMM is refit with new data, regime
# labels change semantics completely, making any hardcoded scale wrong — hard multipliers
# suppress positions in regimes that are actually profitable under the current fit and
# collapse the policy toward always-hold. The regime embedding is in the 152-dim state
# vector — PPO learns regime-conditional position sizing from the reward signal.
# Do not re-enable without recalibrating against the current regime semantics in
# data/validated/regime_semantics.json.
# _REGIME_POSITION_SCALE: dict[int, float] = {
#     0: 0.7,   # mixed — reduce exposure
#     1: 1.0,   # calm bull — full exposure
#     2: 0.2,   # stress/crisis — minimal exposure
#     3: 0.3,   # bear trend — minimal exposure
#     4: 0.5,   # calm bear — half exposure
#     5: 0.9,   # trending forex A
#     6: 0.9,   # trending forex B
# }


def _fit_garch_vol(returns: np.ndarray) -> float:
    """Fit GARCH(1,1) and return the 1-step volatility forecast (decimal).

    Identical formulation to ``RiskAgent._fit_garch`` — percentage-scaled returns,
    falls back to rolling std on failure.
    """
    try:
        from arch import arch_model

        scaled = returns * 100.0
        am = arch_model(scaled, vol="Garch", p=1, q=1, rescale=False)
        res = am.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=1, reindex=False)
        vol_pct = float(np.sqrt(forecast.variance.values[-1, 0]))
        return max(vol_pct / 100.0, 1e-6)
    except Exception as exc:  # pragma: no cover - numerical edge cases
        logger.warning(f"GARCH fit failed ({exc}); using rolling std fallback")
        return float(np.std(returns)) if len(returns) > 1 else 0.02


def precompute_vol_forecast(
    df: pd.DataFrame,
    asset: str,
    window: int = _GARCH_WINDOW,
    stride: int = _GARCH_STRIDE,
    cache_dir: Path = _CACHE_DIR,
) -> np.ndarray:
    """Precompute a per-row GARCH(1,1) volatility forecast series, cached to disk.

    The arch model is refit every ``stride`` rows and forward-filled in between — this
    keeps the same GARCH/CVaR formula as the live RiskAgent while staying tractable for
    100k+ training steps. Returns an array aligned 1:1 with ``df`` rows.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"vol_{asset}_{window}_{stride}.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path, engine="pyarrow")
        if len(cached) == len(df):
            return cached["vol_forecast"].to_numpy(dtype=np.float64)

    returns = df["close_return"].to_numpy(dtype=np.float64)
    n = len(returns)
    vol = np.full(n, 0.02, dtype=np.float64)

    last_vol = 0.02
    for t in range(n):
        if t >= 20 and (t % stride == 0 or t == n - 1):
            start = max(0, t - window + 1)
            r = returns[start : t + 1]
            r = r[~np.isnan(r)]
            if len(r) >= 20:
                last_vol = _fit_garch_vol(r)
        vol[t] = last_vol

    pd.DataFrame({"vol_forecast": vol}).to_parquet(cache_path, engine="pyarrow")
    logger.info(f"Precomputed GARCH vol forecast for {asset} ({n} rows) → {cache_path.name}")
    return vol


def cvar_from_vol(vol: float) -> float:
    """95% normal CVaR from a volatility forecast (negative, same as RiskAgent.cvar_95)."""
    return _CVAR_MULT * vol


def risk_cap_from_vol(vol: float) -> float:
    """CVaR-based max position fraction — RiskAgent's 2% risk-budget rule."""
    cvar = abs(cvar_from_vol(vol))
    return float(min(1.0, 0.02 / max(cvar, 1e-6)))


class MARTATradingEnv(gym.Env):
    """Single-asset-per-episode trading environment for the MARTA PPO agent.

    One asset is sampled from the active curriculum list at each ``reset`` (or stepped
    sequentially during evaluation). Curriculum phases are advanced externally by
    mutating :attr:`assets` via :meth:`set_active_assets`.
    """

    metadata = {"render_modes": []}

    # Asset classes — used to apply asset-specific reward shaping.
    _CRYPTO_ASSETS     = frozenset({"BTC_USDT", "ETH_USDT"})
    _FOREX_ASSETS      = frozenset({"EURUSD"})
    _COMMODITY_ASSETS  = frozenset({"GOLD", "XAUUSD"})

    def __init__(
        self,
        assets: list[str],
        aligned_dfs: dict[str, pd.DataFrame],
        regime_stream_df: pd.DataFrame,
        encoder: TFTRegimeEncoder,
        debate_results: list[dict] | None,
        initial_capital: float = 10_000.0,
        window_size: int = 60,
        transaction_cost: float = 0.001,
        lambda_drawdown: float = 0.01,
        max_leverage: float = 1.0,
        device: str = "cpu",
        # --- additive options (not part of the core spec interface) ---
        split: Literal["train", "val", "all"] = "train",
        sequential: bool = False,
        max_steps: int = 2000,
        zero_regime: bool = False,
        zero_consensus: bool = False,
        # Idle-step penalty: small negative reward per step where position ≈ 0.
        # Addresses the always-flat collapse on low-signal (forex/commodity) assets.
        idle_penalty: float = 0.0,
    ) -> None:
        super().__init__()

        self.assets = list(assets)
        self.aligned_dfs = aligned_dfs
        self.regime_stream_df = regime_stream_df
        self.encoder = encoder.to(device).eval()
        self.initial_capital = float(initial_capital)
        self.window_size = int(window_size)
        self.transaction_cost = float(transaction_cost)
        self.lambda_drawdown = float(lambda_drawdown)
        self.max_leverage = float(max_leverage)
        self.device = device
        self.split = split
        self.sequential = sequential
        self.max_steps = int(max_steps)
        self.zero_regime = zero_regime
        self.zero_consensus = zero_consensus
        self.idle_penalty = float(idle_penalty)

        # Per-asset cached arrays: raw feature matrix, returns, vol, cvar, valid starts.
        self._arrays: dict[str, np.ndarray] = {}
        self._returns: dict[str, np.ndarray] = {}
        self._vol: dict[str, np.ndarray] = {}
        self._index: dict[str, pd.DatetimeIndex] = {}
        self._valid_starts: dict[str, np.ndarray] = {}
        self._regime_labels: dict[str, np.ndarray] = {}
        for asset in CANONICAL_ASSETS:
            if asset in aligned_dfs:
                self._prepare_asset(asset)

        # Consensus lookup keyed by (asset, timestamp) → 16-dim vector.
        self._consensus_lookup = self._build_consensus_lookup(debate_results)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(STATE_DIM,), dtype=np.float32
        )
        # Flattened Box action (training interface for SB3).
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.01], dtype=np.float32),
            high=np.array([2.0, 0.99], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        # Semantic Dict equivalent (reference only).
        self.action_space_dict = spaces.Dict(
            {
                "direction": spaces.Discrete(3),
                "position_fraction": spaces.Box(
                    low=0.01, high=0.99, shape=(1,), dtype=np.float32
                ),
            }
        )

        # Episode state (initialised in reset).
        self._asset: str = self.assets[0]
        self._t: int = self.window_size
        self._step_count: int = 0
        self._portfolio_value: float = self.initial_capital
        self._peak_value: float = self.initial_capital
        self._signed_position: float = 0.0
        self._steps_since_trade: int = 0
        self._decision_regime: int = -1  # regime that drove the most recent position decision
        self._returns_history: list[float] = []
        self._turnover_history: list[float] = []
        self._seq_cursor: int = 0  # for sequential evaluation

    # ------------------------------------------------------------------ setup
    def _prepare_asset(self, asset: str) -> None:
        df = self.aligned_dfs[asset].sort_index()
        self._arrays[asset] = df.values.astype(np.float32)
        self._returns[asset] = df["close_return"].to_numpy(dtype=np.float64)
        self._index[asset] = df.index
        self._vol[asset] = precompute_vol_forecast(df, asset)
        self._regime_labels[asset] = self._build_regime_label_series(asset, df.index)

        ts = df.index
        n = len(df)
        # Valid start = enough history for a window AND a next-bar return AND in-split.
        in_split = self._split_mask(ts)
        valid = np.zeros(n, dtype=bool)
        for t in range(self.window_size, n - 1):
            if in_split[t] and in_split[t + 1]:
                valid[t] = True
        self._valid_starts[asset] = np.flatnonzero(valid)

    def _split_mask(self, ts: pd.DatetimeIndex) -> np.ndarray:
        if self.split == "train":
            return np.asarray(ts <= TRAIN_END)
        if self.split == "val":
            val_end = getattr(self, "_val_end_override", VAL_END)
            return np.asarray((ts >= VAL_START) & (ts <= val_end))
        return np.ones(len(ts), dtype=bool)

    def _build_regime_label_series(
        self, asset: str, index: pd.DatetimeIndex
    ) -> np.ndarray:
        sub = self.regime_stream_df[self.regime_stream_df["asset"] == asset]
        if sub.empty:
            return np.full(len(index), -1, dtype=np.int64)
        sub = sub.set_index("timestamp")["regime_label"].sort_index()
        aligned = sub.reindex(index, method="pad")  # forward-fill only — never use future labels
        return aligned.fillna(-1).to_numpy(dtype=np.int64)

    def _build_consensus_lookup(
        self, debate_results: list[dict] | None
    ) -> dict[tuple[str, pd.Timestamp], np.ndarray]:
        lookup: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
        if not debate_results:
            return lookup
        bias_sign = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
        action_sign = {"long": 1.0, "short": -1.0, "hold": 0.0}
        for d in debate_results:
            asset = d.get("asset")
            ts_raw = d.get("timestamp")
            if asset is None or ts_raw is None:
                continue
            ts = pd.Timestamp(ts_raw)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            vec = np.zeros(CONSENSUS_DIM, dtype=np.float32)
            theses = d.get("round1_theses", []) or []
            for i, th in enumerate(theses[:5]):
                conf = float(th.get("confidence", 0.0))
                vec[i] = bias_sign.get(th.get("directional_bias", "neutral"), 0.0) * conf
            fmd = d.get("fund_manager_decision", {}) or {}
            vec[5] = action_sign.get(fmd.get("final_action", "hold"), 0.0) * float(
                fmd.get("conviction", 0.0)
            )
            vec[6] = float(fmd.get("position_size", 0.0))
            vec[7] = 1.0 if fmd.get("regime_override", False) else 0.0
            vec[8] = float(len(fmd.get("dissenting_agents", [])))
            lookup[(asset, ts)] = vec
        return lookup

    # ----------------------------------------------------------- observations
    def _encode_window(self, asset: str, t: int) -> np.ndarray:
        if self.zero_regime:
            return np.zeros(REGIME_DIM, dtype=np.float32)
        window = self._arrays[asset][t - self.window_size + 1 : t + 1]  # (60, 14)
        tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)  # (1, 60, 13)
        with torch.no_grad():
            emb = self.encoder.encode(tensor)  # (1, 128)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def _consensus_vec(self, asset: str, t: int) -> np.ndarray:
        if self.zero_consensus:
            return np.zeros(CONSENSUS_DIM, dtype=np.float32)
        ts = self._index[asset][t]
        return self._consensus_lookup.get(
            (asset, ts), np.zeros(CONSENSUS_DIM, dtype=np.float32)
        )

    def _portfolio_vec(self, asset: str, t: int) -> np.ndarray:
        vec = np.zeros(PORTFOLIO_DIM, dtype=np.float32)
        slot = min(CANONICAL_ASSETS.index(asset), N_POSITION_SLOTS - 1)
        vec[slot] = self._signed_position
        vec[4] = self._portfolio_value / self.initial_capital - 1.0
        vec[5] = self._current_drawdown()
        vec[6] = min(self._steps_since_trade / 100.0, 1.0)
        vec[7] = float(self._vol[asset][t])
        return vec

    def _build_observation(self) -> np.ndarray:
        regime = self._encode_window(self._asset, self._t)
        consensus = self._consensus_vec(self._asset, self._t)
        portfolio = self._portfolio_vec(self._asset, self._t)
        obs = np.concatenate([regime, consensus, portfolio]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _current_drawdown(self) -> float:
        if self._peak_value <= 0:
            return 0.0
        return float(max(0.0, (self._peak_value - self._portfolio_value) / self._peak_value))

    # ---------------------------------------------------------------- actions
    @staticmethod
    def unflatten_action(action: np.ndarray | dict) -> dict[str, Any]:
        """Recover the semantic {"direction", "position_fraction"} dict from a flat action."""
        if isinstance(action, dict):
            direction = int(action["direction"])
            frac = float(np.asarray(action["position_fraction"]).reshape(-1)[0])
        else:
            arr = np.asarray(action, dtype=np.float64).reshape(-1)
            direction = int(round(float(np.clip(arr[0], 0, 2))))
            frac = float(np.clip(arr[1], 0.01, 0.99))
        return {"direction": direction, "position_fraction": frac}

    # ------------------------------------------------------------------ reset
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        active = [a for a in self.assets if a in self._valid_starts and len(self._valid_starts[a]) > 0]
        if not active:
            raise RuntimeError(f"No active asset has valid start timestamps: {self.assets}")

        if self.sequential:
            # Deterministic walk-forward: single asset, start at first valid index.
            self._asset = active[0]
            starts = self._valid_starts[self._asset]
            self._seq_cursor = 0
            self._t = int(starts[0])
        else:
            self._asset = active[self.np_random.integers(len(active))]
            starts = self._valid_starts[self._asset]
            self._t = int(starts[self.np_random.integers(len(starts))])

        self._step_count = 0
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._signed_position = 0.0
        self._steps_since_trade = 0
        self._decision_regime = int(self._regime_labels[self._asset][min(self._t, len(self._regime_labels[self._asset]) - 1)])
        self._returns_history = []
        self._turnover_history = []

        obs = self._build_observation()
        return obs, self._info(pnl_delta=0.0, cvar_t=cvar_from_vol(self._vol[self._asset][self._t]))

    # ------------------------------------------------------------------- step
    def step(
        self, action: np.ndarray | dict
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        decoded = self.unflatten_action(action)
        direction = decoded["direction"]
        frac = decoded["position_fraction"]

        asset = self._asset
        vol = float(self._vol[asset][self._t])
        cvar_t = cvar_from_vol(vol)
        risk_cap = risk_cap_from_vol(vol) * self.max_leverage

        # Record the decision-time regime so per-regime breakdowns credit the position to
        # the regime at decision time (not the post-step regime, which can differ when labels
        # interleave bar-to-bar — e.g. BTC R1/R4 in the 2021 val window).
        regime_idx = min(self._t, len(self._regime_labels[asset]) - 1)
        current_regime_label = int(self._regime_labels[asset][regime_idx])
        self._decision_regime = current_regime_label

        # Use the Beta head's fraction directly — no regime multiplier (see the disabled
        # _REGIME_POSITION_SCALE note above). PPO learns regime-conditional sizing from
        # the regime embedding in the state vector.
        target_signed = _DIRECTION_SIGN[direction] * frac * risk_cap

        prev_value = self._portfolio_value
        # Transaction cost only on a change in position (hold → no cost).
        turnover = abs(target_signed - self._signed_position)
        cost = self.transaction_cost * turnover * prev_value
        traded = turnover > 1e-9
        self._turnover_history.append(float(turnover))

        # Advance one bar and realise the held position's return.
        self._t += 1
        ret = float(self._returns[asset][self._t])
        if np.isnan(ret):
            ret = 0.0
        gross_pnl = target_signed * ret * prev_value
        pnl_delta_cash = gross_pnl - cost
        self._portfolio_value = prev_value + pnl_delta_cash
        self._peak_value = max(self._peak_value, self._portfolio_value)

        self._signed_position = target_signed
        self._steps_since_trade = 0 if traded else self._steps_since_trade + 1
        self._step_count += 1

        pnl_return = pnl_delta_cash / max(prev_value, 1e-8)
        self._returns_history.append(pnl_return)

        drawdown = self._current_drawdown()
        reward = self._compute_reward(pnl_return, cvar_t, drawdown)

        terminated = self._portfolio_value < 0.2 * self.initial_capital

        if self.sequential:
            # Walk-forward: traverse the whole split, ignore the per-episode max_steps cap.
            starts = self._valid_starts[asset]
            truncated = (
                self._t >= int(starts[-1]) + 1 or self._t >= len(self._returns[asset]) - 1
            )
        else:
            truncated = self._step_count >= self.max_steps

        obs = self._build_observation()
        info = self._info(pnl_delta=pnl_return, cvar_t=cvar_t)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _compute_reward(self, pnl_delta: float, cvar_t: float, drawdown: float) -> float:
        # pnl_delta is the fractional portfolio return for the step (see step()), not dollars.
        # Floor |CVaR| at 0.001 so low-vol regimes can't blow up the ratio (division by ~0).
        cvar_t = max(abs(cvar_t), 0.001)
        risk_adjusted = pnl_delta / cvar_t
        penalty = self.lambda_drawdown * max(0.0, drawdown - 0.05)

        # Idle-step penalty: deduct when the agent holds a near-zero position.
        # This addresses always-flat collapse on forex/commodity assets where near-zero
        # intraday returns make "hold everything" the minimum-variance policy.
        idle = self.idle_penalty if abs(self._signed_position) < 0.005 else 0.0

        # Clip to [-10, 10] — keeps reward magnitude in a range PPO's value net can learn.
        return float(np.clip(risk_adjusted - penalty - idle, -10.0, 10.0))

    # -------------------------------------------------------------------- info
    def _sharpe_so_far(self) -> float:
        if len(self._returns_history) < 2:
            return 0.0
        r = np.asarray(self._returns_history)
        return float(r.mean() / (r.std() + 1e-8) * np.sqrt(24 * 365))

    def _info(self, pnl_delta: float, cvar_t: float) -> dict:
        asset = self._asset
        regime = int(self._regime_labels[asset][min(self._t, len(self._regime_labels[asset]) - 1)])
        return {
            "pnl": self._portfolio_value - self.initial_capital,
            "pnl_return": pnl_delta,
            "portfolio_value": self._portfolio_value,
            "drawdown": self._current_drawdown(),
            "sharpe_so_far": self._sharpe_so_far(),
            "current_regime": regime,
            "decision_regime": self._decision_regime,
            "position_size": abs(self._signed_position),
            "signed_position": self._signed_position,
            "cvar_t": cvar_t,
            "asset": asset,
        }

    # ------------------------------------------------------------- curriculum
    def set_active_assets(self, assets: list[str]) -> None:
        """Update the active curriculum asset list (used by CurriculumCallback)."""
        for a in assets:
            if a not in self._arrays:
                raise ValueError(f"Asset {a} not loaded into env")
        self.assets = list(assets)
        logger.info(f"Env active assets → {self.assets}")
