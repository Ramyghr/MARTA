"""PPO training with curriculum learning for the MARTA execution agent."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import mlflow
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from encoder.model import TFTRegimeEncoder
from rl.callbacks import CurriculumCallback, EntropyDecayCallback, MLflowCallback
from rl.env import CANONICAL_ASSETS, MARTATradingEnv
from rl.policy import MARTAActorCriticPolicy
from utils.assets import aligned_path
from utils.freshness import (
    StaleRegimeStreamError,
    check_encoder_regime_stream_freshness,
)
from utils.logger import logger

_PROCESSED = Path("data/processed")
_CHECKPOINT_DIR = Path("rl/checkpoints")
_DEBATE_PATH = Path("data/debates/debate_results.json")
_HISTORY_PATH = Path("data/results/curriculum_history.json")
_ENCODER_PATH = Path("encoder/checkpoints/best_encoder.pt")
_REGIME_STREAM_PATH = _PROCESSED / "regime_stream.parquet"

# Curriculum: each phase adds an asset and raises the Sharpe bar.
# idle_penalty=0.002 in the env reward is a small constant deduction each step the
# agent holds a near-zero position, breaking the "always-flat is cheapest" local
# optimum. Confirmed empirically (2026-07-05 retrain) that it also breaks BTC-only
# abstention in Phase 1, so it is now enabled from Phase 1 onward, not just Phases 3-4.
# Sharpe thresholds for Phases 3-4 are deliberately low (0.2 / 0.1) to allow the
# policy to transfer even if cross-asset generalisation is imperfect. The key metric
# is that the Phase-2 BTC/ETH policy is NOT eroded: Phase 2 must be gated at ≥0.8
# before Phase 3 is unlocked.
#
# To fall back to the original 2-phase curriculum, pass config["phases"] = CURRICULUM_PHASES[:2].
CURRICULUM_PHASES: list[dict] = [
    {
        "assets": ["BTC_USDT"],
        "sharpe_threshold": 0.5,
        "max_steps": 200_000,
        "label": "Phase1_BTC",
        "idle_penalty": 0.002,
    },
    {
        "assets": ["BTC_USDT", "ETH_USDT"],
        "sharpe_threshold": 0.8,
        "max_steps": 300_000,
        "label": "Phase2_BTC_ETH",
        "idle_penalty": 0.002,
    },
    {
        "assets": ["BTC_USDT", "ETH_USDT", "EURUSD"],
        "sharpe_threshold": 0.2,
        "max_steps": 200_000,
        "label": "Phase3_add_EURUSD",
        "idle_penalty": 0.002,   # break the always-flat collapse on near-zero intraday returns
    },
    {
        "assets": ["BTC_USDT", "ETH_USDT", "EURUSD", "GOLD"],
        "sharpe_threshold": 0.1,
        "max_steps": 200_000,
        "label": "Phase4_add_GOLD",
        "idle_penalty": 0.002,
    },
]


def load_data(config: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, TFTRegimeEncoder]:
    """Load aligned parquets, the regime stream, and the frozen encoder.

    StaleRegimeStreamError/check_encoder_regime_stream_freshness moved to
    utils/freshness.py (P0.4) so the debate-generation scripts can reuse the same guard
    without importing this module's SB3/torch training stack. Re-imported above for
    backward compatibility (nothing currently does `from rl.train import
    StaleRegimeStreamError`, but this keeps that working if it ever did).
    """
    check_encoder_regime_stream_freshness()
    assets = config.get("all_assets", CANONICAL_ASSETS)
    aligned_dfs = {
        a: pd.read_parquet(aligned_path(a, _PROCESSED), engine="pyarrow")
        for a in assets
    }
    regime_stream_df = pd.read_parquet(_REGIME_STREAM_PATH, engine="pyarrow")

    device = config["device"]
    encoder = TFTRegimeEncoder(input_dim=14, hidden_dim=128)
    ckpt = torch.load(_ENCODER_PATH, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["model_state"])
    encoder.eval()
    for param in encoder.parameters():  # FROZEN — no gradients through the encoder
        param.requires_grad = False
    logger.success(f"Loaded encoder (epoch {ckpt.get('epoch')}, frozen) on {device}")
    return aligned_dfs, regime_stream_df, encoder


def load_debate_results() -> list[dict] | None:
    if _DEBATE_PATH.exists():
        return json.loads(_DEBATE_PATH.read_text())
    return None


def build_env_factory(
    aligned_dfs: dict[str, pd.DataFrame],
    regime_stream_df: pd.DataFrame,
    encoder: TFTRegimeEncoder,
    debate_results: list[dict] | None,
    config: dict,
) -> Callable[..., MARTATradingEnv]:
    """Return a factory that builds fresh :class:`MARTATradingEnv` instances.

    Keyword overrides (split, sequential, assets, zero_regime, zero_consensus) let callers
    construct training, validation, and ablation variants from one shared data set.
    """

    def factory(
        assets: list[str] | None = None,
        split: str = "train",
        sequential: bool = False,
        zero_regime: bool = False,
        zero_consensus: bool = False,
        debate: list[dict] | None | bool = True,
        idle_penalty: float = 0.0,
        device: str | None = None,
        min_hold_bars: int | None = None,
    ) -> MARTATradingEnv:
        used_debate = debate_results if debate is True else (None if debate is False else debate)
        # Non-train splits (val/all/test) always run the frozen encoder on CPU: GPU
        # forward passes (cuDNN algorithm selection, non-deterministic reductions) are
        # unseeded and were confirmed to flip the sign of val Sharpe run-to-run on an
        # otherwise-identical checkpoint. CPU float ops are deterministic by default and
        # the encoder is tiny (60-step window, hidden_dim 128), so the cost is negligible.
        # Training keeps config["device"] (GPU) since determinism doesn't matter there and
        # speed does. Callers may still force a device explicitly via the `device` kwarg.
        resolved_device = device if device is not None else (
            config["device"] if split == "train" else "cpu"
        )
        # MARTATradingEnv.__init__ does `self.encoder = encoder.to(device).eval()` —
        # nn.Module.to() moves parameters IN PLACE and returns self, so envs sharing the
        # same `encoder` object would fight over its device (e.g. a CPU val_env built
        # after a CUDA train_env would silently drag the train_env's encoder to CPU too,
        # crashing the next GPU step). Deep-copy whenever this env's resolved device
        # differs from config["device"] so eval envs never mutate the training encoder.
        env_encoder = encoder if resolved_device == config["device"] else copy.deepcopy(encoder)
        return MARTATradingEnv(
            assets=assets or ["BTC_USDT"],
            aligned_dfs=aligned_dfs,
            regime_stream_df=regime_stream_df,
            encoder=env_encoder,
            debate_results=used_debate,
            initial_capital=config["initial_capital"],
            window_size=config["window_size"],
            transaction_cost=config["transaction_cost"],
            lambda_drawdown=config["lambda_drawdown"],
            max_leverage=config["max_leverage"],
            device=resolved_device,
            split=split,
            sequential=sequential,
            max_steps=config["max_episode_steps"],
            zero_regime=zero_regime,
            zero_consensus=zero_consensus,
            idle_penalty=idle_penalty,
            # Unlike idle_penalty, min_hold applies uniformly at train AND eval time —
            # it's a market-structure rule (a position, once opened, must be held for a
            # minimum number of bars), not reward shaping. Defaults from config so every
            # env the factory builds (train/val/all/test) is consistent unless a caller
            # explicitly overrides it per-call.
            min_hold_bars=(
                min_hold_bars if min_hold_bars is not None else config.get("min_hold_bars", 4)
            ),
            min_hold_stop_loss_dd=config.get("min_hold_stop_loss_dd", 0.05),
            max_drawdown_termination=config.get("max_drawdown_termination", 0.20),
            benchmark_relative=config.get("benchmark_relative", False),
        )

    return factory


def train_ppo(
    config: dict,
    pre_train_check: Callable[[MARTATradingEnv], Any] | None = None,
) -> dict[str, Any]:
    """Train PPO across the curriculum and return model + artefacts.

    ``pre_train_check`` (if given) is called with the training env before ``model.learn``
    — used for the random-policy environment sanity check (raises on a degenerate env).

    Returns a dict with ``model``, ``curriculum_history``, ``env_factory``, ``best_path``,
    ``final_path`` and ``early_stopped``. (Superset of the bare ``PPO`` for the master script.)
    """
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    aligned_dfs, regime_stream_df, encoder = load_data(config)
    debate_results = load_debate_results()
    env_factory = build_env_factory(
        aligned_dfs, regime_stream_df, encoder, debate_results, config
    )

    phases = config.get("phases", CURRICULUM_PHASES)
    phase1_assets = phases[0]["assets"]

    # Phase 1 starts BTC-only with NO debate conditioning (consensus = zeros).
    # idle_penalty must be wired in here too — CurriculumCallback only applies a phase's
    # idle_penalty on transition (_advance_phase), so without this the Phase-1 env silently
    # trains with idle_penalty=0.0 regardless of what CURRICULUM_PHASES[0] specifies.
    phase1_idle_penalty = phases[0].get("idle_penalty", 0.0)

    def _make_train_env() -> MARTATradingEnv:
        return env_factory(
            assets=phase1_assets, split="train", debate=False, idle_penalty=phase1_idle_penalty,
        )

    # idle_penalty=0.0 always on val_env — it is a training signal only.
    # Applying it during validation inflates Sharpe by penalizing abstention.
    # Bug confirmed: idle_penalty=0.002 on val_env inflated 0.449 → 1.413.
    val_env = env_factory(
        assets=[phase1_assets[0]], split="val", sequential=True, debate=False,
        idle_penalty=0.0,
    )

    # Environment sanity check BEFORE any training — fail fast on a degenerate env
    # (all-zero rewards/positions) instead of after hours of PPO steps. Runs on one
    # throwaway raw env (gymnasium's 5-tuple step API) — independent of the n_envs>1
    # vectorized fleet below, which speaks SB3 VecEnv's differently-shaped step API.
    if pre_train_check is not None:
        pre_train_check(_make_train_env())

    # n_envs parallel training environments. A 2026-08 multi-seed audit found training
    # with a single env produced high seed-to-seed Sharpe variance (std=0.861 on
    # mean=0.798 across 3 seeds) — each rollout buffer was filled from one long
    # autocorrelated trajectory. n_envs>1 fills each buffer from n_envs independent
    # random start points instead, which is the standard fix in the RL-trading
    # literature (see FinRL's vectorized-environment writeups) for exactly this kind of
    # instability. DummyVecEnv (single-process), not SubprocVecEnv, is deliberate: the
    # frozen TFTRegimeEncoder lives on config["device"] (often CUDA), and SubprocVecEnv
    # would require re-initializing a CUDA context per subprocess (fork-unsafe); the
    # MLflowCallback below also reads ``self.training_env.envs[0]`` directly, which only
    # DummyVecEnv exposes. total_timesteps and n_steps are unchanged by this — see
    # run_rl.py's config comment on how n_steps was rebalanced against n_envs to keep
    # the per-update rollout buffer size (n_steps × n_envs) roughly constant.
    n_envs = int(config.get("n_envs", 1))
    if n_envs > 1:
        train_env: Any = DummyVecEnv([lambda: Monitor(_make_train_env()) for _ in range(n_envs)])
        monitored = train_env  # each sub-env is already Monitor-wrapped
    else:
        train_env = _make_train_env()
        monitored = Monitor(train_env)

    device = config["device"]
    model = PPO(
        policy=MARTAActorCriticPolicy,
        env=monitored,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        verbose=config.get("verbose", 1),
        seed=config.get("seed", 0),
        device=device,
    )

    # checkpoint_suffix isolates this run's checkpoints from any other seed's — see
    # CurriculumCallback's docstring on checkpoint_suffix for the clobbering bug this
    # fixes. run_rl.py's _run_single_seed() sets config["checkpoint_suffix"] = f"_seed{seed}".
    checkpoint_suffix = config.get("checkpoint_suffix", "")
    curriculum_cb = CurriculumCallback(
        phases=phases,
        train_env=train_env,
        val_env=val_env,
        eval_freq=config["eval_freq"],
        early_stop_sharpe=config.get("early_stop_sharpe", -0.5),
        early_stop_patience=config.get("early_stop_patience", 3),
        checkpoint_suffix=checkpoint_suffix,
        min_trades_for_best=config.get("min_trades_for_best", 5),
    )
    mlflow_cb = MLflowCallback(log_freq=config["n_steps"])
    entropy_cb = EntropyDecayCallback(
        start=config["ent_coef"],
        end=config.get("ent_coef_final", 0.005),
        decay_steps=config.get("ent_coef_decay_steps", 100_000),
    )

    total_timesteps = sum(p["max_steps"] for p in phases)
    logger.info(f"Starting PPO curriculum training: {total_timesteps:,} total steps, {len(phases)} phases")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_cb, mlflow_cb, entropy_cb],
        progress_bar=config.get("progress_bar", False),
    )

    final_path = _CHECKPOINT_DIR / f"ppo_final{checkpoint_suffix}.zip"
    best_path = _CHECKPOINT_DIR / f"ppo_best{checkpoint_suffix}.zip"
    model.save(final_path)
    if not best_path.exists():  # best is saved on first *trade-count-qualifying* eval;
        model.save(best_path)  # guard tiny runs / runs that never cleared min_trades_for_best

    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_PATH.write_text(json.dumps(curriculum_cb.phase_history, indent=2, default=float))

    logger.success(
        f"Training complete. best_sharpe={curriculum_cb.best_sharpe:.3f} "
        f"early_stopped={curriculum_cb.early_stopped}"
    )

    return {
        "model": model,
        "curriculum_history": curriculum_cb.phase_history,
        "best_sharpe": curriculum_cb.best_sharpe,
        "early_stopped": curriculum_cb.early_stopped,
        "env_factory": env_factory,
        "best_path": best_path,
        "final_path": final_path,
    }
