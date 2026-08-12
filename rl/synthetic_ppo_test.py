"""Synthetic PPO sanity test — TEST 2 of the RL audit.

Deliberately trivial directional task: observation is a single scalar
``feature = true_return + small_noise``. The optimal policy is obvious:
    feature > 0  -> LONG
    feature < 0  -> SHORT

This uses the REAL ``MARTAActorCriticPolicy`` (rl/policy.py) — not a reimplementation
— and the REAL ``MARTATradingEnv.unflatten_action`` decoding convention, so a failure
here isolates the problem to PPO/the custom Categorical+Beta policy integration,
independent of MARTA's env complexity (regime encoder, CVaR, curriculum, etc).

We do NOT care about Sharpe. We care whether PPO learns the obvious relationship.

Usage:
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.synthetic_ppo_test
    /home/ramy/miniconda3/envs/marta/bin/python -m rl.synthetic_ppo_test --steps 30000
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rl.env import MARTATradingEnv
from rl.policy import MARTAActorCriticPolicy
from utils.logger import logger

_SEED = 42
_NOISE_STD = 0.05
_RETURN_STD = 1.0
_EPISODE_LEN = 200
_TRANSACTION_COST = 0.0  # per item #6: strip costs out for the pure directional test


class SyntheticDirectionEnv(gym.Env):
    """obs = [feature]; feature = true_return + noise. reward = position * true_return."""

    metadata = {"render_modes": []}

    def __init__(self, episode_len: int = _EPISODE_LEN, seed: int | None = None):
        super().__init__()
        self.episode_len = episode_len
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        # Same flattened action-space convention as rl.env.MARTATradingEnv.
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.01], dtype=np.float32),
            high=np.array([2.0, 0.99], dtype=np.float32),
            shape=(2,), dtype=np.float32,
        )
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._true_return = 0.0

    def _sample(self) -> tuple[float, float]:
        true_return = self._rng.normal(0.0, _RETURN_STD)
        feature = true_return + self._rng.normal(0.0, _NOISE_STD)
        return true_return, feature

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._true_return, feature = self._sample()
        obs = np.array([feature], dtype=np.float32)
        return obs, {}

    def step(self, action):
        decoded = MARTATradingEnv.unflatten_action(action)
        direction = decoded["direction"]
        frac = decoded["position_fraction"]
        sign = {0: 1.0, 1: -1.0, 2: 0.0}[direction]
        position = sign * frac

        reward = float(position * self._true_return)
        self._step_count += 1

        self._true_return, feature = self._sample()
        obs = np.array([feature], dtype=np.float32)
        terminated = False
        truncated = self._step_count >= self.episode_len
        return obs, reward, terminated, truncated, {"position": position}


def evaluate_direction_accuracy(model: PPO, n_samples: int = 2000) -> dict[str, float]:
    """Deterministic policy pass: does sign(position) match sign(feature)?"""
    env = SyntheticDirectionEnv(episode_len=n_samples, seed=_SEED + 1)
    obs, _ = env.reset(seed=_SEED + 1)
    correct = 0
    total = 0
    rewards = []
    positions = []
    features = []
    for _ in range(n_samples):
        action, _ = model.predict(obs, deterministic=True)
        feature = float(obs[0])
        decoded = MARTATradingEnv.unflatten_action(action)
        sign = {0: 1.0, 1: -1.0, 2: 0.0}[decoded["direction"]]
        obs, reward, terminated, truncated, info = env.step(action)
        expected_sign = np.sign(feature)
        if sign != 0 and np.sign(sign) == expected_sign:
            correct += 1
        total += 1
        rewards.append(reward)
        positions.append(info["position"])
        features.append(feature)
        if terminated or truncated:
            obs, _ = env.reset()

    positions_arr = np.asarray(positions)
    features_arr = np.asarray(features)
    corr = float(np.corrcoef(positions_arr, features_arr)[0, 1])
    return {
        "direction_accuracy": correct / total,
        "mean_reward": float(np.mean(rewards)),
        "position_feature_correlation": corr,
        "mean_abs_position": float(np.mean(np.abs(positions_arr))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic PPO sanity test")
    parser.add_argument("--steps", type=int, default=20_000)
    args = parser.parse_args()

    logger.info(f"Training PPO ({args.steps} steps) on SyntheticDirectionEnv with MARTAActorCriticPolicy")

    def _make_env():
        return SyntheticDirectionEnv(seed=_SEED)

    train_env = Monitor(_make_env())

    # Pre-training baseline (random init policy).
    baseline_model = PPO(
        policy=MARTAActorCriticPolicy, env=train_env,
        learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
        gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5, verbose=0, seed=_SEED,
    )
    pre = evaluate_direction_accuracy(baseline_model, n_samples=2000)
    logger.info(f"BEFORE training: {pre}")

    baseline_model.learn(total_timesteps=args.steps, progress_bar=False)
    post = evaluate_direction_accuracy(baseline_model, n_samples=2000)
    logger.info(f"AFTER training ({args.steps} steps): {post}")

    print("\n" + "=" * 70)
    print(f"{'metric':35s} {'before':>15s} {'after':>15s}")
    print("=" * 70)
    for key in pre:
        print(f"{key:35s} {pre[key]:>15.4f} {post[key]:>15.4f}")
    print("=" * 70)

    verdict = "PASS" if post["direction_accuracy"] > 0.75 and post["mean_reward"] > pre["mean_reward"] else "FAIL"
    print(f"\nVERDICT: {verdict}")
    print(
        "PASS threshold: direction_accuracy > 0.75 AND mean_reward improved over the "
        "untrained baseline. This is a trivially learnable task (feature IS the signal "
        "plus small noise) — PPO/the custom policy should clear this easily if the "
        "action/log-prob/gradient plumbing is sound."
    )
    if verdict == "FAIL":
        print(
            "FAIL means there is a bug in PPO/MARTAActorCriticPolicy's integration "
            "(Categorical+Beta sampling, log_prob computation, or gradient flow) — fix "
            "this BEFORE testing on real BTC data, since real data adds noise and "
            "non-stationarity on top of whatever is broken here."
        )


if __name__ == "__main__":
    main()
