"""SB3 callbacks: curriculum advancement and MLflow logging for MARTA PPO training."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Union

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from rl.env import MARTATradingEnv
from rl.evaluate import validation_sharpe_and_trades
from utils.logger import logger

TrainEnv = Union[MARTATradingEnv, VecEnv]

# Same MARTA_ROOT anchoring as rl/train.py's identical constant — see its comment.
_MARTA_ROOT = Path(os.environ.get("MARTA_ROOT", str(Path(__file__).resolve().parent.parent)))
_CHECKPOINT_DIR = _MARTA_ROOT / "rl/checkpoints"


class CurriculumCallback(BaseCallback):
    """Advance curriculum phases when the validation Sharpe threshold is met.

    Each phase adds the next asset to the (training and validation) env active list. PPO
    weights are *not* reset between phases — the whole point of the curriculum is transfer.
    Also implements early stopping when validation Sharpe stays very negative.
    """

    def __init__(
        self,
        phases: list[dict],
        train_env: TrainEnv,
        val_env: MARTATradingEnv,
        eval_freq: int = 10_000,
        early_stop_sharpe: float = -0.5,
        early_stop_patience: int = 3,
        verbose: int = 1,
        # Appended to every checkpoint filename this callback saves (e.g. "_seed0" ->
        # ppo_best_seed0.zip). A 2026-08 audit found every seed in a --seeds sweep was
        # saving to the SAME bare "ppo_best.zip"/"ppo_phase_N.zip" — each seed clobbered
        # the previous one's checkpoint, so only the last seed's weights survived on disk
        # even though the per-seed *numbers* (evaluated immediately after each seed's own
        # training, before the next seed could clobber anything) were already correct.
        # run_rl.py's _run_single_seed() already looked for "ppo_best_seed{seed}.zip" —
        # nothing ever wrote that file until this fix. Default "" preserves the original
        # bare filenames for the single-seed/production path (dashboard/ppo_inference.py
        # reads "ppo_best.zip" directly).
        checkpoint_suffix: str = "",
        # Minimum meaningful-trade count (see rl.evaluate._MEANINGFUL_TRADE_THRESHOLD) an
        # eval must have before it's eligible to become the new best_sharpe / overwrite
        # ppo_best.zip. A 2026-08 audit found a checkpoint crowned "best" off a single eval
        # with only 3 trades and Sortino=4728 — a near-zero-variance statistical fluke, not
        # a validated result. This does NOT affect phase-advancement threshold checks
        # (whether the phase's target behavior was reached) — only best-checkpoint tracking.
        # Raised 5->30 in a follow-up audit after 5 still let a 9-trade eval get crowned
        # "best" (Sharpe 1.893) — not enough trade outcomes to trust the ratio.
        min_trades_for_best: int = 30,
    ) -> None:
        super().__init__(verbose)
        self.phases = phases
        self.train_env = train_env
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.early_stop_sharpe = early_stop_sharpe
        self.early_stop_patience = early_stop_patience
        self.checkpoint_suffix = checkpoint_suffix
        self.min_trades_for_best = int(min_trades_for_best)

        self.phase_idx = 0
        self.phase_start_step = 0
        self.best_sharpe = -np.inf
        self.last_eval_sharpe = 0.0
        self._neg_streak = 0
        self.early_stopped = False
        self.done_curriculum = False
        self.phase_history: list[dict] = []
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def _current_phase(self) -> dict:
        return self.phases[self.phase_idx]

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        sharpe, n_trades = validation_sharpe_and_trades(self.model, self.val_env)
        self.last_eval_sharpe = sharpe
        phase = self._current_phase()
        steps_in_phase = self.num_timesteps - self.phase_start_step

        logger.info(
            f"[curriculum] phase {self.phase_idx + 1}/{len(self.phases)} "
            f"({'+'.join(phase['assets'])}) step={self.num_timesteps} "
            f"val_sharpe={sharpe:.3f} thresh={phase['sharpe_threshold']:.2f} n_trades={n_trades}"
        )
        mlflow.log_metric("val/sharpe", sharpe, step=self.num_timesteps)
        mlflow.log_metric("val/n_trades", n_trades, step=self.num_timesteps)
        mlflow.log_metric("train/current_phase", self.phase_idx + 1, step=self.num_timesteps)

        if sharpe > self.best_sharpe:
            if n_trades < self.min_trades_for_best:
                logger.info(
                    f"[curriculum] sharpe={sharpe:.3f} is a new high but only {n_trades} "
                    f"trades (< min_trades_for_best={self.min_trades_for_best}) — not "
                    f"trusting it as best_sharpe, skipping checkpoint save"
                )
            else:
                self.best_sharpe = sharpe
                self.model.save(_CHECKPOINT_DIR / f"ppo_best{self.checkpoint_suffix}.zip")

        # Early stopping on a persistent negative Sharpe.
        if sharpe < self.early_stop_sharpe:
            self._neg_streak += 1
            if self._neg_streak >= self.early_stop_patience:
                logger.warning(
                    f"[curriculum] early stop: val_sharpe < {self.early_stop_sharpe} "
                    f"for {self.early_stop_patience} evals"
                )
                self.early_stopped = True
                return False
        else:
            self._neg_streak = 0

        if not self.done_curriculum:
            threshold_met = sharpe >= phase["sharpe_threshold"]
            max_steps_hit = steps_in_phase >= phase["max_steps"]
            if threshold_met:
                self._advance_phase(threshold_met, sharpe, steps_in_phase)
            elif max_steps_hit:
                # Hard gate (2026-08-24) — previously this branch still called
                # _advance_phase() and moved on to the next phase regardless, logging only
                # a WARNING. That let a policy that never proved itself on Phase 1 keep
                # training on Phase 2's harder, multi-asset problem — propagating a broken
                # policy forward instead of stopping to fix it. _record_phase_result still
                # saves the per-phase checkpoint and phase_history entry (diagnostic value
                # unchanged) but the curriculum does not advance and training stops here.
                self._record_phase_result(threshold_met, sharpe, steps_in_phase)
                logger.warning(
                    f"[curriculum] phase {self.phase_idx + 1} FAILED: max_steps reached "
                    f"without meeting threshold ({sharpe:.3f} < {phase['sharpe_threshold']:.2f}). "
                    f"Stopping training — policy is not ready to advance."
                )
                self.early_stopped = True
                return False

        return True

    def _record_phase_result(self, threshold_met: bool, sharpe: float, steps_in_phase: int) -> Path:
        """Save the per-phase checkpoint + phase_history entry. Called on both a
        successful (threshold met) and failed (max_steps hit, gated — see _on_step)
        phase outcome, so either way there's a checkpoint and a record to inspect."""
        phase = self._current_phase()
        n = self.phase_idx + 1
        ckpt = _CHECKPOINT_DIR / f"ppo_phase_{n}{self.checkpoint_suffix}.zip"
        self.model.save(ckpt)

        mlflow.log_param(f"phase_{n}_completed_at_step", self.num_timesteps)
        mlflow.log_metric(f"phase_{n}_final_sharpe", sharpe, step=self.num_timesteps)
        self.phase_history.append(
            {
                "phase": n,
                "assets": phase["assets"],
                "steps": steps_in_phase,
                "sharpe": sharpe,
                "threshold": phase["sharpe_threshold"],
                "threshold_met": bool(threshold_met),
            }
        )
        status = "threshold met" if threshold_met else "max_steps reached (gated, not advancing)"
        logger.success(
            f"[curriculum] phase {n} complete ({status}) sharpe={sharpe:.3f} → {ckpt.name}"
        )
        return ckpt

    def _advance_phase(self, threshold_met: bool, sharpe: float, steps_in_phase: int) -> None:
        self._record_phase_result(threshold_met, sharpe, steps_in_phase)

        if self.phase_idx < len(self.phases) - 1:
            self.phase_idx += 1
            self.phase_start_step = self.num_timesteps
            new_phase = self._current_phase()
            new_assets = new_phase["assets"]
            # Apply the new phase's idle penalty so the reward shaping is correct
            # for forex/commodity phases even though PPO weights persist.
            # idle_penalty=0.0 always on val_env — it is a training signal only.
            # Applying it during validation inflates Sharpe by penalizing abstention.
            # Bug confirmed: idle_penalty=0.002 on val_env inflated 0.449 → 1.413.
            new_penalty = float(new_phase.get("idle_penalty", 0.0))
            # train_env is a bare VecEnv.env_method (n_envs>1, see rl/train.py) or a raw
            # MARTATradingEnv (n_envs=1) — VecEnv has no set_active_assets/idle_penalty
            # attribute of its own, so broadcast through env_method to every sub-env
            # instead. set_idle_penalty is a method (not a bare attribute set) precisely
            # so this works — see its docstring in rl/env.py for why set_attr wouldn't.
            if isinstance(self.train_env, VecEnv):
                self.train_env.env_method("set_active_assets", new_assets)
                self.train_env.env_method("set_idle_penalty", new_penalty)
            else:
                self.train_env.set_active_assets(new_assets)
                self.train_env.set_idle_penalty(new_penalty)
            self.val_env.set_active_assets([new_assets[0]])  # eval on primary asset
            self.val_env.idle_penalty = 0.0
            if new_penalty > 0:
                logger.info(f"[curriculum] idle_penalty set to {new_penalty} for {'+'.join(new_assets)}")
        else:
            # Last phase finished — keep training to total_timesteps but stop advancing.
            self.done_curriculum = True
            logger.info("[curriculum] final phase complete; continuing to total_timesteps")


class EntropyDecayCallback(BaseCallback):
    """Linearly decay PPO's entropy coefficient over the course of training.

    ``ent_coef`` starts high (0.05, vs SB3's default 0.0) specifically to stop the policy
    collapsing to always-hold early in training (see run_rl.py's config comment). But
    holding it that high for the *entire* run works against the policy ever settling into
    a confident, sustained directional stance later on: a 2026-08 multi-seed audit found
    the trained policy satisfying idle_penalty with single-bar open-then-immediately-close
    "pokes" (every completed position in the eval trade log lasted exactly 1 hour) rather
    than genuine multi-bar conviction — consistent with entropy_loss staying flat around
    -0.7 to -0.9 for the whole run instead of decaying as training progressed. Decaying
    ent_coef lets the policy explore early (breaking hold-collapse) and commit later
    (once it has something worth committing to).

    ``model.ent_coef`` is a plain mutable attribute on SB3's ``PPO`` (read fresh in its
    loss computation each update), so overwriting it here takes effect immediately —
    no need to touch the optimizer or reconstruct the model.
    """

    def __init__(
        self,
        start: float = 0.05,
        end: float = 0.005,
        decay_steps: int = 100_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = int(decay_steps)

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / max(self.decay_steps, 1))
        new_coef = self.start + frac * (self.end - self.start)
        self.model.ent_coef = float(new_coef)
        if self.num_timesteps % 10_000 == 0:
            mlflow.log_metric("train/ent_coef", new_coef, step=self.num_timesteps)
        return True


class MLflowCallback(BaseCallback):
    """Log rollout and optimisation metrics to MLflow at a fixed step cadence."""

    def __init__(self, log_freq: int = 2048, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        # Rolling window of |position| per env step — used for the policy-collapse check.
        self.recent_positions: deque[float] = deque(maxlen=log_freq)

    def _safe(self, key: str) -> float | None:
        val = self.model.logger.name_to_value.get(key)
        return float(val) if val is not None else None

    def _on_step(self) -> bool:
        # Track positions every step (info["position_size"] is already |signed_position|).
        for info in self.locals.get("infos", []):
            if "position_size" in info:
                self.recent_positions.append(float(info["position_size"]))

        if self.num_timesteps % self.log_freq != 0:
            return True

        # Policy-collapse detector: if the policy holds ~zero position across the whole
        # recent window after warm-up, it has collapsed to always-hold — warn loudly so
        # a dead run is visible immediately instead of after hours of training.
        avg_pos = float(np.mean(self.recent_positions)) if self.recent_positions else 0.0
        mlflow.log_metric("train/avg_abs_position", avg_pos, step=self.num_timesteps)
        if avg_pos < 0.001 and self.num_timesteps > 10_000:
            logger.warning(
                f"[POLICY COLLAPSE] avg_position={avg_pos:.6f} at step {self.num_timesteps}. "
                f"Policy has collapsed to always-hold. Consider: "
                f"1) Check ent_coef (currently {self.model.ent_coef}), "
                f"2) Check reward scale (ep_rew_mean should be in [-5, +5]), "
                f"3) Verify regime_stream.parquet matches current HMM n_states."
            )
            mlflow.log_metric("train/policy_collapsed", 1, step=self.num_timesteps)
        else:
            mlflow.log_metric("train/policy_collapsed", 0, step=self.num_timesteps)

        ep_buffer = list(self.model.ep_info_buffer or [])
        if ep_buffer:
            rewards = np.array([ep["r"] for ep in ep_buffer], dtype=np.float64)
            lengths = np.array([ep["l"] for ep in ep_buffer], dtype=np.float64)
            mlflow.log_metric("train/mean_reward", float(rewards.mean()), step=self.num_timesteps)
            mlflow.log_metric(
                "train/mean_episode_length", float(lengths.mean()), step=self.num_timesteps
            )
            sharpe = float(rewards.mean() / (rewards.std() + 1e-8) * np.sqrt(24 * 365))
            mlflow.log_metric("train/sharpe_ratio", sharpe, step=self.num_timesteps)

        # Trajectory stats pulled from the underlying env via the Monitor wrapper.
        # With n_envs>1 (rl/train.py), this is just one of several parallel envs — a
        # representative sample for logging, not an aggregate across all of them.
        env = self.training_env.envs[0]
        base = getattr(env, "unwrapped", env)
        if getattr(base, "_returns_history", None):
            pv = base._portfolio_value
            dd = base._current_drawdown()
            mlflow.log_metric("train/max_drawdown", float(dd), step=self.num_timesteps)
            mlflow.log_metric(
                "train/mean_position_size", float(abs(base._signed_position)),
                step=self.num_timesteps,
            )
            mlflow.log_metric("train/portfolio_value", float(pv), step=self.num_timesteps)
            turnover_hist = getattr(base, "_turnover_history", None)
            if turnover_hist:
                mlflow.log_metric(
                    "train/mean_turnover_per_step",
                    float(np.mean(turnover_hist)),
                    step=self.num_timesteps,
                )

        for key, mlflow_key in [
            ("train/entropy_loss", "train/entropy_loss"),
            ("train/policy_gradient_loss", "train/policy_loss"),
            ("train/value_loss", "train/value_loss"),
        ]:
            val = self._safe(key)
            if val is not None:
                mlflow.log_metric(mlflow_key, val, step=self.num_timesteps)

        return True
