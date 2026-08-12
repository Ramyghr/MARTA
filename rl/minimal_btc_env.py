"""MinimalBTCEnv — TEST 3A/3B environment for the RL learning-pipeline audit.

The SAME class, reward, and action space serve both experiments; only the
observation differs (``observation_mode``):

    "simple" (TEST 3A): the current bar's 14 already-computed normalized
        indicator columns (atr, adx, rsi, bb_width, macd_histogram, obv_raw,
        vol_zscore, realized_vol, vol_ratio, return_skew, return_kurtosis,
        autocorr_lag1, realized_vol_raw, close_return) — no temporal encoder,
        no regime/consensus/portfolio context. Real data, zero MARTA machinery.

    "marta" (TEST 3B): the exact 156-dim MARTA observation layout
        (128-dim frozen TFTRegimeEncoder embedding of the trailing 60-bar
        window | 16-dim consensus, always zero — no debate cache is used |
        8-dim portfolio state | 4-dim asset one-hot, always BTC). Built the
        same way ``rl.env.MARTATradingEnv._build_observation`` builds it, so
        this is genuinely "the existing MARTA observation," not a
        reimplementation of it.

Deliberately stripped out (per the 2026-08-11 RL audit plan), present in the
production ``MARTATradingEnv`` but NOT here:
    - Beta position sizing (action is Discrete(3), position size is FIXED)
    - CVaR reward normalization
    - drawdown penalty
    - idle penalty
    - minimum holding period
    - curriculum / multi-asset / asset switching
    - regime label / consensus features (folded into "marta" mode's embedding
      only insofar as the frozen encoder itself was trained on regime-relevant
      features — the explicit consensus block is zeroed, not read from a cache)

Reward (identical in both modes): ``position * next_period_return - transaction_cost
* turnover`` — no normalization, no shaping. position is FIXED magnitude
(``FIXED_POSITION_SIZE``), signed by direction.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import deque
from pathlib import Path
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from encoder.model import TFTRegimeEncoder
from rl.env import TRAIN_END, VAL_END, VAL_START, precompute_vol_forecast
from rl.evaluate import _compute_trade_durations, count_position_changes
from utils.assets import aligned_path
from utils.logger import logger

_console = Console()
_PROCESSED = Path("data/processed")
_ENCODER_PATH = Path("encoder/checkpoints/best_encoder.pt")
_RESULTS_DIR = Path("data/results")
_SEED = 42

# Copied verbatim from rl/synthetic_ppo_test.py's already-PASSing TEST 2 config — not
# tuned for this experiment (per audit instruction: no hyperparameter changes in TEST 3).
_PPO_KWARGS = dict(
    learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
    gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
    max_grad_norm=0.5, verbose=0,
)

FIXED_POSITION_SIZE = 0.3
_DIRECTION_SIGN = {0: 1.0, 1: -1.0, 2: 0.0}  # long / short / hold — same convention as rl.env
SIMPLE_FEATURE_COLS = [
    "atr", "adx", "rsi", "bb_width", "macd_histogram", "obv_raw", "vol_zscore",
    "realized_vol", "vol_ratio", "return_skew", "return_kurtosis", "autocorr_lag1",
    "realized_vol_raw", "close_return",
]
SIMPLE_DIM = len(SIMPLE_FEATURE_COLS)  # 14
REGIME_DIM = 128
CONSENSUS_DIM = 16
PORTFOLIO_DIM = 8
ASSET_ID_DIM = 4
MARTA_DIM = REGIME_DIM + CONSENSUS_DIM + PORTFOLIO_DIM + ASSET_ID_DIM  # 156


class MinimalBTCEnv(gym.Env):
    """Single-asset (BTC) minimal trading env — see module docstring."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        aligned_df: pd.DataFrame,
        encoder: TFTRegimeEncoder | None,
        observation_mode: Literal["simple", "marta"] = "simple",
        window_size: int = 60,
        transaction_cost: float = 0.001,
        initial_capital: float = 10_000.0,
        split: Literal["train", "val", "all"] = "train",
        sequential: bool = False,
        max_steps: int = 500,
        device: str = "cpu",
        # Controls the 8-dim self-referential portfolio block (own position, P&L,
        # drawdown, steps_since_trade, vol_forecast) — see module docstring's
        # portfolio-state-confound note. None preserves the ORIGINAL TEST 3A/3B
        # defaults (False for "simple", True for "marta"); pass True/False explicitly
        # to build the 3A+P (simple+portfolio) / 3B-P (marta-portfolio) controls.
        include_portfolio: bool | None = None,
    ) -> None:
        super().__init__()
        if observation_mode == "marta" and encoder is None:
            raise ValueError("observation_mode='marta' requires a frozen encoder")

        self.observation_mode = observation_mode
        self.window_size = int(window_size)
        self.transaction_cost = float(transaction_cost)
        self.initial_capital = float(initial_capital)
        self.split = split
        self.sequential = sequential
        self.max_steps = int(max_steps)
        self.device = device
        self.include_portfolio = (
            include_portfolio if include_portfolio is not None else (observation_mode == "marta")
        )

        df = aligned_df.sort_index()
        self._simple_arr = df[SIMPLE_FEATURE_COLS].to_numpy(dtype=np.float32)
        self._full_arr = df.to_numpy(dtype=np.float32)  # 14-col, feeds the encoder window
        self._returns = df["close_return"].to_numpy(dtype=np.float64)
        self._index = df.index
        self.encoder = encoder.to(device).eval() if encoder is not None else None
        if self.include_portfolio:
            self._vol = precompute_vol_forecast(df, "BTC_USDT_test3")

        n = len(df)
        in_split = self._split_mask(self._index)
        valid = np.zeros(n, dtype=bool)
        for t in range(self.window_size, n - 1):
            if in_split[t] and in_split[t + 1]:
                valid[t] = True
        self._valid_starts = np.flatnonzero(valid)
        if len(self._valid_starts) == 0:
            raise RuntimeError(f"No valid start indices for split={split!r}")

        self.action_space = spaces.Discrete(3)
        portfolio_dim = PORTFOLIO_DIM if self.include_portfolio else 0
        if observation_mode == "simple":
            obs_dim = SIMPLE_DIM + portfolio_dim
        else:
            obs_dim = REGIME_DIM + CONSENSUS_DIM + portfolio_dim + ASSET_ID_DIM
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._t = self.window_size
        self._step_count = 0
        self._seq_cursor = 0
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._signed_position = 0.0
        self._steps_since_trade = 0

    def _split_mask(self, ts: pd.DatetimeIndex) -> np.ndarray:
        if self.split == "train":
            return np.asarray(ts <= TRAIN_END)
        if self.split == "val":
            return np.asarray((ts >= VAL_START) & (ts <= VAL_END))
        return np.ones(len(ts), dtype=bool)

    # ------------------------------------------------------------ observation
    def _current_drawdown(self) -> float:
        if self._peak_value <= 0:
            return 0.0
        return float(max(0.0, (self._peak_value - self._portfolio_value) / self._peak_value))

    def _portfolio_vec(self) -> np.ndarray:
        """8-dim self-referential block (own position/P&L/drawdown) — SAME construction
        for both modes when included, so 3A+P and 3B share byte-identical portfolio dims."""
        portfolio = np.zeros(PORTFOLIO_DIM, dtype=np.float32)
        portfolio[0] = self._signed_position  # single-asset slot 0 (matches CANONICAL_ASSETS[0]=BTC)
        portfolio[4] = self._portfolio_value / self.initial_capital - 1.0
        portfolio[5] = self._current_drawdown()
        portfolio[6] = min(self._steps_since_trade / 100.0, 1.0)
        portfolio[7] = float(self._vol[self._t])
        return portfolio

    def _build_observation(self) -> np.ndarray:
        if self.observation_mode == "simple":
            parts = [self._simple_arr[self._t]]
            if self.include_portfolio:
                parts.append(self._portfolio_vec())
            obs = np.concatenate(parts).astype(np.float32)
            return np.clip(obs, -10.0, 10.0)

        window = self._full_arr[self._t - self.window_size + 1 : self._t + 1]
        tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.encoder.encode(tensor).squeeze(0).cpu().numpy().astype(np.float32)

        consensus = np.zeros(CONSENSUS_DIM, dtype=np.float32)  # no debate cache used (item 6/11)

        asset_id = np.zeros(ASSET_ID_DIM, dtype=np.float32)
        asset_id[0] = 1.0  # BTC always

        parts = [emb, consensus]
        if self.include_portfolio:
            parts.append(self._portfolio_vec())
        parts.append(asset_id)
        obs = np.concatenate(parts).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.sequential:
            self._seq_cursor = 0
            self._t = int(self._valid_starts[0])
        else:
            self._t = int(self._valid_starts[self.np_random.integers(len(self._valid_starts))])

        self._step_count = 0
        self._portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._signed_position = 0.0
        self._steps_since_trade = 0

        obs = self._build_observation()
        return obs, {"signed_position": self._signed_position, "portfolio_value": self._portfolio_value}

    # ------------------------------------------------------------------- step
    def step(self, action: int):
        direction = int(action)
        target_signed = _DIRECTION_SIGN[direction] * FIXED_POSITION_SIZE

        prev_signed = self._signed_position
        turnover = abs(target_signed - prev_signed)
        traded = turnover > 1e-9

        self._t += 1
        next_ret = float(self._returns[self._t])
        if np.isnan(next_ret):
            next_ret = 0.0

        # reward = position * next_period_return - transaction_cost * turnover (exact spec, no shaping)
        pnl_return = target_signed * next_ret - self.transaction_cost * turnover
        reward = pnl_return

        self._portfolio_value *= (1.0 + pnl_return)
        self._peak_value = max(self._peak_value, self._portfolio_value)
        self._signed_position = target_signed
        self._steps_since_trade = 0 if traded else self._steps_since_trade + 1
        self._step_count += 1

        if self.sequential:
            terminated = False
            truncated = (
                self._t >= int(self._valid_starts[-1]) + 1 or self._t >= len(self._returns) - 1
            )
        else:
            terminated = False  # no drawdown termination in the minimal env (item 6)
            truncated = self._step_count >= self.max_steps

        obs = self._build_observation()
        info = {
            "signed_position": self._signed_position,
            "portfolio_value": self._portfolio_value,
            "pnl_return": pnl_return,
            "next_return": next_ret,
            "direction": direction,
            "timestamp": self._index[self._t],
        }
        return obs, float(reward), bool(terminated), bool(truncated), info


def build_env(
    aligned_df: pd.DataFrame,
    encoder: TFTRegimeEncoder | None,
    observation_mode: Literal["simple", "marta"],
    split: Literal["train", "val", "all"],
    sequential: bool = False,
    max_steps: int = 500,
    device: str = "cpu",
    include_portfolio: bool | None = None,
) -> MinimalBTCEnv:
    return MinimalBTCEnv(
        aligned_df=aligned_df,
        encoder=encoder,
        observation_mode=observation_mode,
        split=split,
        sequential=sequential,
        max_steps=max_steps,
        device=device,
        include_portfolio=include_portfolio,
    )


# --------------------------------------------------------------------------- data loading
def load_shared_data(device: str, encoder_path: Path | None = None) -> dict[str, Any]:
    """Load BTC_USDT aligned data + a frozen encoder checkpoint once.

    encoder_path defaults to the production _ENCODER_PATH (best_encoder.pt) — pass an
    explicit path (e.g. an E2 periodic/best/final checkpoint) to evaluate a different
    checkpoint with the exact same diagnostic protocol, without touching production.
    """
    path = encoder_path if encoder_path is not None else _ENCODER_PATH
    df = pd.read_parquet(aligned_path("BTC_USDT", _PROCESSED), engine="pyarrow").sort_index()
    encoder = TFTRegimeEncoder(input_dim=14, hidden_dim=128)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["model_state"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return {"df": df, "encoder": encoder, "encoder_path": str(path)}


# --------------------------------------------------------------------------- PPO diagnostics
class DiagnosticsCallback(BaseCallback):
    """Records PPO training diagnostics every rollout (log_freq == n_steps)."""

    def __init__(self, log_freq: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self.recent_rewards: deque[float] = deque(maxlen=log_freq)
        self.history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.recent_rewards.extend(float(r) for r in np.asarray(rewards).reshape(-1))
        if self.num_timesteps % self.log_freq != 0:
            return True
        entry: dict[str, Any] = {
            "timesteps": self.num_timesteps,
            "mean_recent_reward": float(np.mean(self.recent_rewards)) if self.recent_rewards else None,
        }
        for key, short in [
            ("train/entropy_loss", "entropy_loss"),
            ("train/policy_gradient_loss", "policy_loss"),
            ("train/value_loss", "value_loss"),
            ("train/explained_variance", "explained_variance"),
            ("train/approx_kl", "approx_kl"),
            ("train/clip_fraction", "clip_fraction"),
        ]:
            val = self.model.logger.name_to_value.get(key)
            entry[short] = float(val) if val is not None else None
        ep_buffer = list(self.model.ep_info_buffer or [])
        entry["mean_episode_reward"] = (
            float(np.mean([ep["r"] for ep in ep_buffer])) if ep_buffer else None
        )
        self.history.append(entry)
        return True


# --------------------------------------------------------------------------- policies (eval)
Policy = Callable[[int], int]


def policy_fixed(direction: int) -> Policy:
    return lambda step_idx: direction


def make_random_policy(rng: np.random.Generator) -> Policy:
    return lambda step_idx: int(rng.integers(0, 3))


def run_trajectory(
    env: MinimalBTCEnv,
    actor: Policy | PPO,
    is_model: bool,
    seed: int = _SEED,
    deterministic: bool = True,
) -> dict[str, np.ndarray]:
    """Run one policy/model through env sequentially, collecting the full trajectory.

    ``actor`` is either a plain ``Policy`` callable (fixed/random baselines, takes the
    step index and returns a direction int) or a trained SB3 ``PPO`` model (``is_model=True``,
    predicts from the observation via ``model.predict``).
    """
    obs, info = env.reset(seed=seed)
    positions = [info["signed_position"]]
    rewards: list[float] = []
    pnl_returns: list[float] = []
    next_returns: list[float] = []
    directions: list[int] = []

    step_idx = 0
    done = False
    while not done:
        if is_model:
            action, _ = actor.predict(obs, deterministic=deterministic)
            action = int(action)
        else:
            action = actor(step_idx)
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(info["signed_position"])
        rewards.append(reward)
        pnl_returns.append(info["pnl_return"])
        next_returns.append(info["next_return"])
        directions.append(info["direction"])
        step_idx += 1
        done = terminated or truncated

    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "pnl_returns": np.asarray(pnl_returns, dtype=np.float64),
        "next_returns": np.asarray(next_returns, dtype=np.float64),
        "directions": np.asarray(directions, dtype=np.int64),
    }


def summarize(traj: dict[str, np.ndarray]) -> dict[str, Any]:
    positions = traj["positions"]
    rewards = traj["rewards"]
    pnl = traj["pnl_returns"]
    next_ret = traj["next_returns"]
    directions = traj["directions"]
    n = len(rewards)
    if n == 0:
        return {}

    total_return = float(np.prod(1.0 + pnl) - 1.0)
    nonzero_mask = directions != 2
    if nonzero_mask.sum() > 0:
        signed_dir = np.where(directions[nonzero_mask] == 0, 1.0, -1.0)
        direction_accuracy = float(np.mean(signed_dir == np.sign(next_ret[nonzero_mask])))
    else:
        direction_accuracy = None

    n_trades = count_position_changes(positions, threshold=1e-9)
    trade_durations = _compute_trade_durations(positions, 1e-9)
    dir_counts = {
        "long": int(np.sum(directions == 0)),
        "short": int(np.sum(directions == 1)),
        "hold": int(np.sum(directions == 2)),
    }
    return {
        "n_samples": n,
        "mean_reward": float(rewards.mean()),
        "eval_reward": float(rewards.mean()),
        "total_return": total_return,
        "direction_accuracy": direction_accuracy,
        "n_trades": n_trades,
        "mean_holding_duration_hours": float(np.mean(trade_durations)) if trade_durations else 0.0,
        "median_holding_duration_hours": float(np.median(trade_durations)) if trade_durations else 0.0,
        "direction_distribution": dir_counts,
        "direction_distribution_pct": {k: v / n for k, v in dir_counts.items()},
    }


# --------------------------------------------------------------------------- one full mode run
def run_mode(
    shared: dict[str, Any],
    observation_mode: Literal["simple", "marta"],
    steps: int,
    train_device: str,
    seed: int = _SEED,
    include_portfolio: bool | None = None,
) -> dict[str, Any]:
    resolved_portfolio = include_portfolio if include_portfolio is not None else (observation_mode == "marta")
    if observation_mode == "simple":
        label = "3A+P" if resolved_portfolio else "3A"
    else:
        label = "3B" if resolved_portfolio else "3B-P"
    variant_id = label.replace("+", "plus").replace("-", "minus")
    _console.rule(f"[bold]TEST {label} — observation_mode={observation_mode} include_portfolio={resolved_portfolio}")

    base_encoder = shared["encoder"] if observation_mode == "marta" else None

    def _make_env(split: str, sequential: bool, device: str) -> MinimalBTCEnv:
        # MinimalBTCEnv.__init__ does encoder.to(device) IN PLACE (same as
        # MARTATradingEnv) — sharing one encoder object across a CUDA train_env and a
        # CPU eval_env would silently drag whichever was built more recently onto the
        # other's device. Deep-copy whenever this env's device differs from train_device
        # (identical guard to rl/train.py's build_env_factory).
        if base_encoder is None:
            env_encoder = None
        elif device == train_device:
            env_encoder = base_encoder
        else:
            env_encoder = copy.deepcopy(base_encoder)
        return build_env(
            shared["df"], env_encoder, observation_mode, split,
            sequential=sequential, device=device, include_portfolio=resolved_portfolio,
        )

    train_env = Monitor(_make_env("train", sequential=False, device=train_device))
    # Plain SB3 MlpPolicy + Discrete(3) action — TEST 2 already validated the custom
    # MARTAActorCriticPolicy's Categorical+Beta PPO integration on synthetic data; TEST 3
    # has no Beta sizing at all (see module docstring), so there's nothing hybrid left to
    # test here. Using the stock policy isolates "does PPO learn from this reward/data",
    # independent of our custom policy code.
    model = PPO(policy="MlpPolicy", env=train_env, seed=seed, **_PPO_KWARGS)

    # Pre-training eval (untrained policy) — "before" reference, full val split, CPU
    # (matches production's non-train-split-always-CPU determinism convention).
    pre_env = _make_env("val", sequential=True, device="cpu")
    pre_summary = summarize(run_trajectory(pre_env, model, is_model=True, seed=seed))
    logger.info(f"[{label}] PRE-TRAINING eval: {pre_summary}")

    diag_cb = DiagnosticsCallback(log_freq=_PPO_KWARGS["n_steps"])
    logger.info(f"[{label}] Training PPO for {steps} steps on device={train_device}")
    model.learn(total_timesteps=steps, callback=diag_cb, progress_bar=False)

    # Post-training eval — sequential walk over the FULL val split, deterministic, CPU.
    post_env = _make_env("val", sequential=True, device="cpu")
    post_summary = summarize(run_trajectory(post_env, model, is_model=True, seed=seed))
    logger.success(f"[{label}] POST-TRAINING eval: {post_summary}")

    # Fixed-policy baselines on the SAME val split.
    rng = np.random.default_rng(seed)
    baseline_policies: dict[str, Policy] = {
        "ALWAYS_LONG": policy_fixed(0),
        "ALWAYS_SHORT": policy_fixed(1),
        "ALWAYS_HOLD": policy_fixed(2),
        "RANDOM": make_random_policy(rng),
    }
    baseline_summaries: dict[str, Any] = {}
    for name, pol in baseline_policies.items():
        env_i = _make_env("val", sequential=True, device="cpu")
        baseline_summaries[name] = summarize(run_trajectory(env_i, pol, is_model=False, seed=seed))

    # ABSTENTION is reported SEPARATELY from PASS — a policy that collapses to ~always-hold
    # trivially beats pre-training/random on mean_reward (holding earns ~0, which beats any
    # negative baseline) without having learned any directional signal. Conflating that with
    # "PASS" was flagged as a reporting mistake after the original TEST 3B run — do not repeat it.
    post_hold_pct = post_summary["direction_distribution_pct"]["hold"]
    is_abstention = post_hold_pct > 0.95
    beats_baselines = (
        post_summary["mean_reward"] > pre_summary["mean_reward"]
        and post_summary["mean_reward"] > baseline_summaries["RANDOM"]["mean_reward"]
    )
    if is_abstention:
        verdict = "ABSTENTION"
    elif beats_baselines:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    result = {
        "observation_mode": observation_mode,
        "include_portfolio": resolved_portfolio,
        "variant_label": label,
        "steps": steps,
        "pre_training_eval": pre_summary,
        "post_training_eval": post_summary,
        "baselines": baseline_summaries,
        "training_diagnostics": diag_cb.history,
        "verdict": verdict,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"minimal_btc_env_{variant_id}.json"
    out_path.write_text(json.dumps(result, indent=2, default=float))
    logger.success(f"[{label}] results -> {out_path}")

    # ---------------------------------------------------------------- console report
    table = Table(title=f"TEST {label} ({observation_mode}) — {post_summary['n_samples']} eval steps")
    table.add_column("Policy")
    table.add_column("Mean Reward", justify="right")
    table.add_column("Total Return", justify="right")
    table.add_column("Dir. Acc.", justify="right")
    table.add_column("N Trades", justify="right")
    table.add_column("Avg Hold (h)", justify="right")
    table.add_column("L/S/H %", justify="right")

    def _row(name: str, s: dict[str, Any]) -> None:
        acc = "n/a" if s["direction_accuracy"] is None else f"{s['direction_accuracy']:.1%}"
        dd = s["direction_distribution_pct"]
        lsh = f"{dd['long']:.0%}/{dd['short']:.0%}/{dd['hold']:.0%}"
        table.add_row(
            name, f"{s['mean_reward']:.6f}", f"{s['total_return']:.4%}", acc,
            str(s["n_trades"]), f"{s['mean_holding_duration_hours']:.1f}", lsh,
        )

    _row("PPO (pre-training)", pre_summary)
    _row("PPO (post-training)", post_summary)
    for name, s in baseline_summaries.items():
        _row(name, s)
    _console.print(table)
    _console.rule(f"[bold]VERDICT ({label}): {verdict}")

    return result


# (observation_mode, include_portfolio) for each of the 4 matrix cells.
VARIANTS: dict[str, tuple[Literal["simple", "marta"], bool]] = {
    "3A": ("simple", False),
    "3A+P": ("simple", True),
    "3B-P": ("marta", False),
    "3B": ("marta", True),
}


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="TEST 3A/3B/3A+P/3B-P — minimal real-BTC PPO env, 2x2 matrix")
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS) + ["controls", "all"], default=["3A", "3B"],
        help="'controls' = 3A+P and 3B-P only (the portfolio-confound isolation runs); 'all' = all 4",
    )
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--train-device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if "all" in args.variants:
        variant_names = list(VARIANTS)
    elif "controls" in args.variants:
        variant_names = ["3A+P", "3B-P"]
    else:
        variant_names = args.variants

    logger.info("Loading shared BTC_USDT data (aligned parquet + frozen encoder)")
    shared = load_shared_data(device=args.train_device)

    results: dict[str, Any] = {}
    for name in variant_names:
        obs_mode, include_p = VARIANTS[name]
        results[name] = run_mode(
            shared, obs_mode, args.steps, args.train_device, seed=args.seed, include_portfolio=include_p,
        )

    if len(results) > 1:
        _console.rule(f"[bold]TEST {' vs '.join(variant_names)}")
        for name, r in results.items():
            _console.print(f"{name:6s} verdict={r['verdict']:10s} post_eval_mean_reward={r['post_training_eval']['mean_reward']:.6f}")
        (_RESULTS_DIR / "test3_matrix_comparison.json").write_text(json.dumps(results, indent=2, default=float))
        logger.success(f"Combined comparison -> {_RESULTS_DIR / 'test3_matrix_comparison.json'}")


if __name__ == "__main__":
    main()
