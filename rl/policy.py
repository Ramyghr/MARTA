"""MARTAActorCriticPolicy — custom SB3 policy with a hybrid Categorical+Beta action head.

The action is hybrid:
    direction       ~ Categorical(logits)         over {0=long, 1=short, 2=hold}
    position_frac   ~ Beta(α, β)  ∈ (0, 1)         α, β = softplus(·) + 1  (α,β > 1)

Why Beta over a clipped Gaussian: the Beta distribution is natively bounded in (0, 1), so
the position fraction needs no clipping and carries no boundary gradient bias. Forcing
α, β > 1 (via ``softplus + 1``) keeps the density unimodal — it cannot collapse into the
U-shaped Beta(<1, <1) regime that would push mass to the 0/1 extremes.

The policy bypasses SB3's built-in distribution machinery (which only supports a single
homogeneous action distribution) by overriding ``_build``, ``forward``,
``evaluate_actions``, ``_predict`` and ``predict_values`` directly.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch.distributions import Beta, Categorical

# Must match MARTATradingEnv.action_space's position_fraction bounds exactly (rl/env.py,
# also duplicated in SyntheticDirectionEnv/MinimalBTCEnv). SB3's rollout collection clips
# the sampled action to the action_space bounds before calling env.step() (see
# on_policy_algorithm.py), but stores the UNCLIPPED sample for log_prob/gradient purposes.
# Clamping here to the action_space bounds (not an arbitrary [1e-6, 1-1e-6]) keeps the
# stored log_prob consistent with what the env actually executes — previously a sample
# near the Beta's support boundary (e.g. 0.9998) would be clipped to 0.99 by SB3 before
# reaching env.step(), but log_prob was computed at 0.9998, crediting/blaming the wrong
# value. Confirmed via rl/diagnose_env.py-style repro (2026-08-11 RL audit).
_POS_LOW = 0.01
_POS_HIGH = 0.99
N_DIRECTIONS = 3


class MARTAActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy with a shared MLP trunk and split direction/position/value heads."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        lr_schedule: Schedule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # net_arch / activation are unused (we build the trunk by hand) but accepted.
        kwargs.setdefault("ortho_init", False)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    # ------------------------------------------------------------------ build
    def _build(self, lr_schedule: Schedule) -> None:
        """Construct the shared trunk, the three heads, and the optimizer.

        Overrides SB3's default ``_build`` entirely so no DiagGaussian action net is created.
        """
        feature_dim = int(self.features_dim)  # 156 (FlattenExtractor on a 156-d Box, see rl/env.py STATE_DIM)

        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.direction_head = nn.Linear(64, N_DIRECTIONS)  # Categorical logits
        self.position_head = nn.Linear(64, 2)              # (α, β) pre-softplus
        self.value_head = nn.Linear(64, 1)                 # state value

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ---------------------------------------------------------------- helpers
    def _latent(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs)
        return self.shared_net(features)

    def _distributions(
        self, latent: torch.Tensor
    ) -> tuple[Categorical, Beta]:
        dir_dist = Categorical(logits=self.direction_head(latent))
        ab = F.softplus(self.position_head(latent)) + 1.0  # α, β > 1 → unimodal Beta
        pos_dist = Beta(ab[:, 0], ab[:, 1])
        return dir_dist, pos_dist

    # ---------------------------------------------------------------- forward
    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self._latent(obs)
        dir_dist, pos_dist = self._distributions(latent)

        if deterministic:
            direction = torch.argmax(dir_dist.probs, dim=1)
            # Beta mean = α / (α + β)
            position = pos_dist.mean
        else:
            direction = dir_dist.sample()
            position = pos_dist.sample()

        position = position.clamp(_POS_LOW, _POS_HIGH)
        log_prob = dir_dist.log_prob(direction) + pos_dist.log_prob(position)

        actions = torch.stack([direction.float(), position], dim=1)  # (B, 2)
        values = self.value_head(latent)                              # (B, 1)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self._latent(obs)
        dir_dist, pos_dist = self._distributions(latent)

        direction = actions[:, 0].long().clamp(0, N_DIRECTIONS - 1)
        position = actions[:, 1].clamp(_POS_LOW, _POS_HIGH)

        log_prob = dir_dist.log_prob(direction) + pos_dist.log_prob(position)
        entropy = dir_dist.entropy() + pos_dist.entropy()
        values = self.value_head(latent)
        return values, log_prob, entropy

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self._latent(obs))

    def get_distribution(self, obs: torch.Tensor):  # type: ignore[override]
        """Return the direction distribution (SB3 introspection hook; rarely used here)."""
        latent = self._latent(obs)
        dir_dist, _ = self._distributions(latent)
        return dir_dist


# Discrete action index -> signed position fraction. 5 levels: flat, small/full long,
# small/full short. Index order matches N_DISCRETE_ACTIONS below.
DISCRETE_POSITION_MAP: list[float] = [0.0, 0.25, 0.50, -0.25, -0.50]
N_DISCRETE_ACTIONS = len(DISCRETE_POSITION_MAP)


class MARTADiscretePolicy(ActorCriticPolicy):
    """Actor-critic policy with a single Categorical head over N_DISCRETE_ACTIONS position
    levels (2026-08-24 discrete-action experiment).

    Motivation (see rl/crypto5_feature_predictability.py): the crypto5 panel shows a real
    but small directional edge (best single feature ~54.3% accuracy) and ~zero R² on
    return magnitude. MARTAActorCriticPolicy's Beta head is trying to learn a
    magnitude/position-fraction signal that isn't there, and micro-jitters around its mean
    as a result (6,019 trades in one crypto5 eval). This policy removes the Beta head
    entirely — it only ever picks one of N_DISCRETE_ACTIONS fixed position levels, so there
    is no continuous fraction to jitter.

    Requires action_space=Discrete(N_DISCRETE_ACTIONS) (MARTATradingEnv's action_mode=
    "discrete" — see rl/env.py). Entirely separate from MARTAActorCriticPolicy — that
    class, its Box(2) action space, and every existing checkpoint/caller (including
    dashboard/backend/ppo_inference.py) are untouched by this addition.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        lr_schedule: Schedule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("ortho_init", False)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        feature_dim = int(self.features_dim)

        # Same 3-layer trunk shape as MARTAActorCriticPolicy (256->128->64) — deliberately
        # unchanged from the hybrid policy's architecture, so a discrete-vs-continuous
        # action-space comparison isn't confounded by also changing the trunk.
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, N_DISCRETE_ACTIONS)
        self.value_head = nn.Linear(64, 1)

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _latent(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs)
        return self.shared_net(features)

    def _distribution(self, latent: torch.Tensor) -> Categorical:
        return Categorical(logits=self.action_head(latent))

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self._latent(obs)
        dist = self._distribution(latent)

        # Return the discrete ACTION INDEX, not the mapped position — SB3 stores whatever
        # forward() returns as the rollout-buffer action and later feeds it straight back
        # into evaluate_actions(). Returning the mapped continuous position here instead
        # would silently corrupt every log_prob/entropy computation once evaluate_actions
        # tries to .long() a float position back into an action index.
        action = torch.argmax(dist.probs, dim=1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        values = self.value_head(latent)
        return action.unsqueeze(-1).float(), values, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self._latent(obs)
        dist = self._distribution(latent)

        action_idx = actions.long().reshape(-1).clamp(0, N_DISCRETE_ACTIONS - 1)
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()
        values = self.value_head(latent)
        return values, log_prob, entropy

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action, _, _ = self.forward(observation, deterministic=deterministic)
        return action

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self._latent(obs))

    def get_distribution(self, obs: torch.Tensor):  # type: ignore[override]
        return self._distribution(self._latent(obs))
