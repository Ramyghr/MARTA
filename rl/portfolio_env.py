"""MultiAssetPortfolioEnv — joint N-asset(+cash) portfolio allocation environment.

Step 3 of the pre-RL audit's remediation plan (Step 1: regime/causal_regime_stream.py,
Step 2: rl/obs_builder.py). Unlike rl/env.py's MARTATradingEnv — single-asset-per-episode,
"multi-asset" meaning sequential curriculum, never a joint decision (see that class's own
docstring) — this environment exposes ALL assets simultaneously in one observation and
takes ONE portfolio-allocation action per step across all of them: the literal "Shared
Portfolio Agent" design (one policy managing N assets + cash) that MARTATradingEnv cannot
structurally provide.

Shares, not duplicates, the per-asset observation-block logic with MARTATradingEnv via
rl/obs_builder.py (market_block / regime series / consensus lookup — Step 2). Reuses the
GARCH/CVaR primitives directly from rl/env.py (precompute_vol_forecast, cvar_from_vol are
already free functions there, not methods on MARTATradingEnv — no duplication needed).
Reuses the exact same 10bps transaction-cost default and TestSetContaminationError split
guard as MARTATradingEnv, so a portfolio-vs-single-asset comparison isn't confounded by
also silently changing cost assumptions or leaking test data differently.

Data contract: aligned_dfs[asset] for every asset in `assets` must share an IDENTICAL
DatetimeIndex — true by construction for crypto5_multihorizon data (regime/build_crypto5_
features.py aligns all 5 assets onto one common intersected index before writing
{asset}_aligned.parquet), asserted in __init__ rather than assumed silently.

Action: Box(N+1,) raw logits -> softmax -> portfolio weights summing to 1.0 (N assets +
cash, at index N). Logits->softmax in the env (not a custom Dirichlet action head) is the
simpler, PPO-compatible first cut — SB3's default MlpPolicy already samples a Gaussian in
logit space with no custom distribution class required (see rl/policy_portfolio.py). A
native Dirichlet head is a documented, NOT-yet-built future upgrade — subclassing SB3's
ActorCriticPolicy for a new distribution has real log-prob/entropy/rollout-buffer
consistency subtleties (see rl/policy.py's MARTAActorCriticPolicy for how much custom
machinery that requires even for a two-headed Categorical+Beta case) and isn't required to
get a first portfolio training run working.

Observation, per step: concat of, for each asset in canonical order,
    [market_block(asset) | consensus_block(asset)]
followed by [previous portfolio weights (N+1,)] and [portfolio scalar state (2,):
value_norm, drawdown]. Per-asset blocks are grouped together (not interleaved) so a future
per-asset feature extractor (rl/policy_portfolio.py's PortfolioFeatureExtractor) can slice
fixed-size chunks straight out of the flat vector.

Reward: portfolio log-return net of total transaction cost, i.e.
    reward = log(1 + gross_simple_return - cost_fraction)
gross_simple_return is the weight-averaged simple return of the N assets (cash
contributes exactly 0). cost_fraction = transaction_cost * sum_i |w_i_new - w_i_old| over
ASSETS ONLY (not cash — cash's weight change is fully determined by
-sum(asset weight changes), so counting it separately would double the charged cost),
matching rl/env.py's own turnover convention (turnover = |target_signed - prev_signed|,
not halved) rather than introducing a different halved-L1 convention.

Walk-forward split: identical TestSetContaminationError guard to rl/env.py, checked once
against the shared index (not per-asset, since the index is shared by construction here).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from configs.splits import (
    TEST_START,
    TRAIN_END_EXCLUSIVE,
    VAL_END_EXCLUSIVE,
    VAL_START as _VAL_START,
    TestSetContaminationError,
)
from encoder.model import TFTRegimeEncoder
from rl import obs_builder
from rl.env import cvar_from_vol, precompute_vol_forecast  # reused, not duplicated
from utils.logger import logger

# Default panel for this module — the crypto5 5-asset raw-feature-HMM panel this env was
# built for. Deliberately a LOCAL constant, not a new repo-wide shared one: the pre-RL
# audit found the crypto5 asset list already independently hardcoded in ~8 other files
# (regime/build_crypto5_features.py, rl/train.py, dashboard/backend/regime_service.py,
# etc.) — consolidating all of those is a separate, wider refactor outside this module's
# scope; this constant only serves as MultiAssetPortfolioEnv's own default.
CRYPTO5_ASSETS: list[str] = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "LTC_USDT", "XRP_USDT"]

TRAIN_END = pd.Timestamp(TRAIN_END_EXCLUSIVE, tz="UTC") - pd.Timedelta(seconds=1)
VAL_START = pd.Timestamp(_VAL_START, tz="UTC")
VAL_END = pd.Timestamp(VAL_END_EXCLUSIVE, tz="UTC") - pd.Timedelta(seconds=1)

_CACHE_DIR = Path("data/processed/rl_cache")


class MultiAssetPortfolioEnv(gym.Env):
    """Joint N-asset(+cash) portfolio allocation environment — one action per step across
    all assets simultaneously, not one asset per episode. See module docstring.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        assets: list[str],
        aligned_dfs: dict[str, pd.DataFrame],
        regime_stream_df: pd.DataFrame,
        encoder: TFTRegimeEncoder | None,
        debate_results: list[dict] | None = None,
        initial_capital: float = 10_000.0,
        window_size: int = 60,
        transaction_cost: float = 0.001,  # 10bps — matches rl/env.py's MARTATradingEnv exactly
        lambda_drawdown: float = 0.01,
        device: str = "cpu",
        split: Literal["train", "val", "all"] = "train",
        obs_mode: Literal["encoder", "raw", "raw_regime"] = "raw",
        sequential: bool = False,
        max_steps: int = 2000,
        zero_regime: bool = False,
        zero_consensus: bool = False,
        max_drawdown_termination: float = 0.20,
    ) -> None:
        super().__init__()
        if len(assets) < 2:
            raise ValueError("MultiAssetPortfolioEnv needs at least 2 assets — use MARTATradingEnv for one")
        for a in assets:
            if a not in aligned_dfs:
                raise ValueError(f"asset {a!r} missing from aligned_dfs")

        self.assets = list(assets)
        self.n_assets = len(self.assets)
        self.n_actions = self.n_assets + 1  # + cash
        self.cash_index = self.n_assets

        self.aligned_dfs = aligned_dfs
        self.regime_stream_df = regime_stream_df
        self.obs_mode = obs_mode
        if obs_mode == "encoder" and encoder is None:
            raise ValueError("obs_mode='encoder' requires a non-None encoder")
        self.encoder = encoder.to(device).eval() if encoder is not None else None
        self.initial_capital = float(initial_capital)
        self.window_size = int(window_size)
        self.transaction_cost = float(transaction_cost)
        self.lambda_drawdown = float(lambda_drawdown)
        self.device = device
        self.split = split
        self.sequential = sequential
        self.max_steps = int(max_steps)
        self.zero_regime = zero_regime
        self.zero_consensus = zero_consensus
        self.max_drawdown_termination = float(max_drawdown_termination)

        # --- data contract: every asset must share one index (see module docstring) ---
        ref_index = aligned_dfs[self.assets[0]].sort_index().index
        self._index = ref_index
        for a in self.assets:
            df = aligned_dfs[a].sort_index()
            if not df.index.equals(ref_index):
                raise ValueError(
                    f"asset {a!r}'s index differs from {self.assets[0]!r}'s — "
                    "MultiAssetPortfolioEnv requires all assets pre-aligned onto one shared "
                    "index (see regime/build_crypto5_features.py)."
                )

        n = len(ref_index)
        self.n_raw_features = aligned_dfs[self.assets[0]].shape[1]

        self._regime_prob_cols = [c for c in regime_stream_df.columns if c.startswith("regime_prob_")]
        self._n_regime_prob_dims = len(self._regime_prob_cols)

        self._arrays: dict[str, np.ndarray] = {}
        self._returns: dict[str, np.ndarray] = {}
        self._vol: dict[str, np.ndarray] = {}
        self._regime_labels: dict[str, np.ndarray] = {}
        self._regime_probs: dict[str, np.ndarray] = {}
        for a in self.assets:
            df = aligned_dfs[a].sort_index()
            self._arrays[a] = df.values.astype(np.float32)
            self._returns[a] = df["close_return"].to_numpy(dtype=np.float64)
            self._vol[a] = precompute_vol_forecast(df, a, cache_dir=_CACHE_DIR)
            self._regime_labels[a] = obs_builder.build_regime_label_series(regime_stream_df, a, ref_index)
            if self.obs_mode == "raw_regime":
                self._regime_probs[a] = obs_builder.build_regime_prob_series(
                    regime_stream_df, self._regime_prob_cols, a, ref_index
                )

        self._consensus_lookup = obs_builder.build_consensus_lookup(debate_results)

        # --- shared valid-start mask (joint, not per-asset — the index is shared) ---
        in_split = self._split_mask(ref_index)
        valid = np.zeros(n, dtype=bool)
        for t in range(self.window_size, n - 1):
            if in_split[t] and in_split[t + 1]:
                valid[t] = True
        self._valid_starts = np.flatnonzero(valid)

        if self.split in ("train", "val") and len(self._valid_starts):
            max_used_ts = ref_index[self._valid_starts].max()
            if max_used_ts >= pd.Timestamp(TEST_START, tz="UTC"):
                raise TestSetContaminationError(
                    f"({self.split} split): env includes timestamps up to {max_used_ts}, "
                    f"which is >= TEST_START ({TEST_START}). Sacred 2025 test data must "
                    "never reach RL training or validation."
                )

        # --- spaces ---
        per_asset_market_dim = {
            "encoder": obs_builder.REGIME_DIM,
            "raw": self.window_size * self.n_raw_features,
            "raw_regime": self.window_size * self.n_raw_features + self._n_regime_prob_dims,
        }[self.obs_mode]
        self.per_asset_block_dim = per_asset_market_dim + obs_builder.CONSENSUS_DIM
        self.tail_dim = self.n_actions + 2  # prev_weights (N+1) + [value_norm, drawdown]
        obs_dim = self.n_assets * self.per_asset_block_dim + self.tail_dim
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.n_actions,), dtype=np.float32
        )

        # Episode state (initialised in reset).
        self._t: int = self.window_size
        self._step_count: int = 0
        self._portfolio_value: float = self.initial_capital
        self._peak_value: float = self.initial_capital
        self._weights: np.ndarray = self._all_cash_weights()
        self._seq_cursor: int = 0

    # ------------------------------------------------------------------ setup
    def _split_mask(self, ts: pd.DatetimeIndex) -> np.ndarray:
        if self.split == "train":
            return np.asarray(ts <= TRAIN_END)
        if self.split == "val":
            return np.asarray((ts >= VAL_START) & (ts <= VAL_END))
        return np.ones(len(ts), dtype=bool)

    def _all_cash_weights(self) -> np.ndarray:
        w = np.zeros(self.n_actions, dtype=np.float64)
        w[self.cash_index] = 1.0
        return w

    # ----------------------------------------------------------- observations
    def _build_observation(self) -> np.ndarray:
        blocks = []
        for a in self.assets:
            regime_probs_arr = self._regime_probs.get(a) if self.obs_mode == "raw_regime" else None
            market = obs_builder.market_block(
                self.obs_mode, self._arrays[a], self._t, self.window_size,
                encoder=self.encoder, device=self.device,
                regime_probs_arr=regime_probs_arr, zero_regime=self.zero_regime,
            )
            if self.zero_consensus:
                consensus = np.zeros(obs_builder.CONSENSUS_DIM, dtype=np.float32)
            else:
                ts = self._index[self._t]
                consensus = obs_builder.consensus_vec(self._consensus_lookup, a, ts)
            blocks.append(market)
            blocks.append(consensus)

        tail = np.concatenate([
            self._weights.astype(np.float32),
            np.array([
                self._portfolio_value / self.initial_capital - 1.0,
                self._current_drawdown(),
            ], dtype=np.float32),
        ])
        obs = np.concatenate([*blocks, tail]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _current_drawdown(self) -> float:
        if self._peak_value <= 0:
            return 0.0
        return float(max(0.0, (self._peak_value - self._portfolio_value) / self._peak_value))

    @staticmethod
    def weights_from_action(action: np.ndarray) -> np.ndarray:
        """Softmax over raw logits -> portfolio weights summing to 1.0. Exposed as a
        staticmethod so callers (e.g. a diagnostic script, or the info dict below) can
        recover the actual applied weights from a raw action without re-deriving softmax."""
        logits = np.asarray(action, dtype=np.float64).reshape(-1)
        logits = logits - logits.max()  # numerically stable softmax
        exp = np.exp(logits)
        return exp / exp.sum()

    # ------------------------------------------------------------------ reset
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if len(self._valid_starts) == 0:
            raise RuntimeError("No valid start timestamps for this split")

        if self.sequential:
            self._seq_cursor = 0
            self._t = int(self._valid_starts[0])
        else:
            self._t = int(self._valid_starts[self.np_random.integers(len(self._valid_starts))])

        self._step_count = 0
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._weights = self._all_cash_weights()  # episode starts fully in cash

        obs = self._build_observation()
        return obs, self._info(step_return=0.0, cost_fraction=0.0)

    # ------------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Bounds guard: a training episode's start is sampled from _valid_starts (max
        # entry n-2, see __init__), so ONE step is always safe — but nothing stops a
        # training episode from taking a SECOND step past that if it hasn't yet hit
        # max_steps/drawdown termination (unlike sequential eval, which truncates at the
        # data boundary explicitly). Truncate here instead of indexing self._returns[...]
        # out of bounds. No trade is considered "applied" this call (weights/portfolio
        # value unchanged) since there is no new bar's return to realize it against.
        if self._t + 1 >= len(self._index):
            obs = self._build_observation()
            info = self._info(step_return=0.0, cost_fraction=0.0)
            return obs, 0.0, False, True, info

        target_weights = self.weights_from_action(action)

        prev_weights = self._weights
        turnover = float(np.abs(target_weights[: self.n_assets] - prev_weights[: self.n_assets]).sum())
        cost_fraction = self.transaction_cost * turnover

        self._t += 1
        asset_returns = np.array(
            [float(self._returns[a][self._t]) for a in self.assets], dtype=np.float64
        )
        asset_returns = np.nan_to_num(asset_returns, nan=0.0)
        gross_return = float(np.dot(target_weights[: self.n_assets], asset_returns))
        net_return = gross_return - cost_fraction

        prev_value = self._portfolio_value
        self._portfolio_value = prev_value * (1.0 + net_return)
        self._peak_value = max(self._peak_value, self._portfolio_value)
        self._weights = target_weights
        self._step_count += 1

        drawdown = self._current_drawdown()
        reward = self._compute_reward(net_return, drawdown)

        if self.sequential:
            terminated = self._portfolio_value < 0.2 * self.initial_capital
            truncated = (
                self._t >= int(self._valid_starts[-1]) + 1
                or self._t >= len(self._index) - 1
            )
        else:
            terminated = drawdown > self.max_drawdown_termination
            truncated = self._step_count >= self.max_steps

        obs = self._build_observation()
        info = self._info(step_return=net_return, cost_fraction=cost_fraction)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _compute_reward(self, net_return: float, drawdown: float) -> float:
        # log1p is numerically safer than log(1+x) for small x and matches "portfolio
        # log-return" directly — net_return already has the turnover cost subtracted
        # (see module docstring for why cost is folded in before the log rather than
        # subtracted as a separate additive term after it).
        log_return = float(np.log1p(np.clip(net_return, -0.99, None)))
        penalty = self.lambda_drawdown * drawdown
        return float(np.clip(log_return - penalty, -10.0, 10.0))

    # -------------------------------------------------------------------- info
    def _info(self, step_return: float, cost_fraction: float) -> dict:
        return {
            "pnl": self._portfolio_value - self.initial_capital,
            "portfolio_value": self._portfolio_value,
            "drawdown": self._current_drawdown(),
            "weights": {**{a: float(self._weights[i]) for i, a in enumerate(self.assets)},
                        "cash": float(self._weights[self.cash_index])},
            "turnover_cost_fraction": cost_fraction,
            "step_return": step_return,
        }
