"""Behavior-cloning warm-start for MARTAActorCriticPolicy (2026-08-24).

Motivation: PPO trained from a random init on a 1300+ dim raw observation has been
exploring badly (rl/env.py's obs_mode="raw"/"raw_regime" pivot) — best val Sharpe across
several runs never exceeded roughly -0.5, no upward trend. The original plan was to
warm-start from agents_no_regime's rule (rl/ablation.py's _run_agents_rule), but that
rule's actions are gated by _make_nearest_consensus_fn, which only has real signal within
7 days of a cached debate — and the debate cache has exactly ONE entry anywhere in the
training split (2018-2023): a single 2019-06-15 debate. Rolling that rule out over the
training split would produce a BC dataset that is ~99% "hold", teaching the network the
exact always-flat collapse this project has fought elsewhere (idle_penalty exists for the
same reason).

buy_hold is used as the teacher instead: it is the single best-performing baseline in
every ablation run this session (Sharpe 1.1-1.5, consistently ahead of PPO, RSI, MA
crossover, and momentum), and — unlike agents_no_regime — it is defined at every
timestep with no debate-cache dependency, so it has real, dense signal across the whole
6-year training split. rl/ablation.py's own _run_buy_hold bypasses env.step() entirely
(computes portfolio value directly from env._returns), so it can't be reused as-is for
generating (obs, action) pairs — generate_buy_hold_trajectories below rolls a constant
"long, near-max fraction" action through the real env.step() loop instead, so the
observations seen are exactly what a trained policy would see, and the realized position
after CVaR risk-capping is whatever risk_cap_from_vol allows (not literally 100% notional
exposure — see module docstring on why that's the correct comparison to make).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rl.env import MARTATradingEnv
from rl.policy import MARTAActorCriticPolicy
from utils.logger import logger

# direction=0 (long), position_fraction=0.99 (near-max — actual realized position is
# still CVaR risk-capped by the environment, same as any other action).
_BUY_HOLD_ACTION = np.array([0.0, 0.99], dtype=np.float32)


def generate_buy_hold_trajectories(
    env: MARTATradingEnv, asset: str = "BTC_USDT"
) -> tuple[np.ndarray, np.ndarray]:
    """Roll a constant long/near-max-fraction action through env.step() over the env's
    active split (sequential walk-forward), recording (obs_before_step, action) pairs.

    Returns (obs_array, action_array) — obs_array has shape (n_steps, obs_dim),
    action_array has shape (n_steps, 2) and is literally _BUY_HOLD_ACTION repeated
    (recorded per-step for a uniform, simple training loop below — its value never
    varies, only the observation does).
    """
    if list(env.assets) != [asset]:
        env.set_active_assets([asset])
    obs, _ = env.reset()

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    done = False
    n_steps = 0
    while not done:
        observations.append(obs)
        actions.append(_BUY_HOLD_ACTION.copy())
        obs, _, terminated, truncated, _ = env.step(_BUY_HOLD_ACTION)
        done = terminated or truncated
        n_steps += 1

    logger.info(f"[bc] generated {n_steps:,} buy-hold teacher (obs, action) pairs from {asset}")
    return np.stack(observations).astype(np.float32), np.stack(actions).astype(np.float32)


def pretrain_bc(
    policy: MARTAActorCriticPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> None:
    """Supervised pretraining of the shared trunk + direction/position heads on
    (obs, action) pairs, before PPO fine-tuning ever starts.

    Direction: cross-entropy against the teacher's discrete choice (always "long" for
    the buy-hold teacher, but the loop is generic over whatever actions are passed in).
    Position: MSE between the Beta distribution's mean (alpha / (alpha+beta) — the same
    quantity policy.forward() returns for deterministic actions) and the teacher's
    position_fraction. This does not touch value_head — PPO's own rollouts establish the
    value function from scratch; BC only seeds the actor.
    """
    device = policy.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = obs.shape[0]
    indices = np.arange(n)

    policy.train()
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            obs_batch = obs[batch_idx]
            act_batch = actions[batch_idx]

            obs_tensor, _ = policy.obs_to_tensor(obs_batch)
            target_dir = torch.as_tensor(act_batch[:, 0], dtype=torch.long, device=device)
            target_pos = torch.as_tensor(act_batch[:, 1], dtype=torch.float32, device=device)

            latent = policy._latent(obs_tensor)
            dir_logits = policy.direction_head(latent)
            ab = F.softplus(policy.position_head(latent)) + 1.0
            pos_mean = ab[:, 0] / (ab[:, 0] + ab[:, 1])

            dir_loss = F.cross_entropy(dir_logits, target_dir)
            pos_loss = F.mse_loss(pos_mean, target_pos)
            loss = dir_loss + pos_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        logger.info(f"[bc] epoch {epoch + 1}/{epochs} mean_loss={epoch_loss / max(n_batches, 1):.4f}")

    policy.eval()
    logger.success(f"[bc] pretraining complete — {n:,} samples, {epochs} epochs")
