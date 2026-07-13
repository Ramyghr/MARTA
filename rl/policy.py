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

_EPS = 1e-6
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
        feature_dim = int(self.features_dim)  # 152 (FlattenExtractor on a 152-d Box)

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

        position = position.clamp(_EPS, 1.0 - _EPS)
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
        position = actions[:, 1].clamp(_EPS, 1.0 - _EPS)

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
