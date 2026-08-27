"""rl/policy_portfolio.py — PPO construction for MultiAssetPortfolioEnv.

Step 4 of the pre-RL audit's remediation plan (Step 3: rl/portfolio_env.py).

Unlike rl/policy.py's MARTAActorCriticPolicy (a hand-built Categorical+Beta hybrid head,
required because MARTATradingEnv's action space is a hybrid direction+fraction pair with
no native SB3 distribution), MultiAssetPortfolioEnv's action space is a plain
Box(n_actions,) of raw logits — the softmax-to-weights step happens inside the env's
step(), not the policy. SB3's default ActorCriticPolicy already samples a DiagGaussian
over a Box action space with no custom distribution class required, so
``make_ppo_portfolio`` below is a plain PPO("MlpPolicy", ...) factory, not a new policy
class — there is nothing non-standard about the ACTION side of this problem.

The OBSERVATION side is non-standard: MultiAssetPortfolioEnv's obs is N per-asset blocks
of *identical* structure concatenated together, plus a small tail (previous weights +
portfolio scalars). Flattening that into one monolithic MLP (SB3's default FlattenExtractor
+ net_arch) works but throws away the fact that the same per-asset transformation should
apply to every asset's block — a weight-tied encoder is both a better inductive bias and
matches the "one policy reasoning about N assets uniformly" framing the portfolio design
is meant to embody. PortfolioFeatureExtractor below implements that: slices the flat obs
back into its N per-asset chunks + tail (using the env's own per_asset_block_dim/n_assets/
tail_dim, not hardcoded), runs each chunk through one SHARED (weight-tied) small MLP,
concatenates the N resulting per-asset embeddings with the (separately-encoded) tail, and
hands that off to SB3's standard downstream net_arch. This is optional — make_ppo_portfolio
defaults to plain SB3 behavior (features_extractor=None) since a first training run doesn't
require it; pass extractor="portfolio" to opt in.

Documented, NOT-yet-built future upgrade: replacing the Gaussian-on-logits action space
with a native Dirichlet head. This would need a custom ActorCriticPolicy subclass (see
rl/policy.py's MARTAActorCriticPolicy for how much machinery a custom distribution
requires even for a two-headed case — forward/evaluate_actions/predict_values/
get_distribution all need overriding, plus care that stored log_probs stay consistent
with what SB3's rollout buffer replays). Not required to get a first portfolio run
working; the logits->softmax approach in the env already produces valid simplex actions.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PortfolioFeatureExtractor(BaseFeaturesExtractor):
    """Weight-tied per-asset encoder for MultiAssetPortfolioEnv's observation.

    Slices the flat (n_assets * per_asset_block_dim + tail_dim,) observation into its
    n_assets per-asset chunks (in the exact order MultiAssetPortfolioEnv._build_observation
    concatenates them — see that method) plus the trailing tail block, runs every per-asset
    chunk through the SAME small MLP (weight sharing across assets — the point: the network
    should not need n_assets times the parameters to learn "how to read one asset's block",
    only how to combine what it reads), and concatenates the n_assets resulting embeddings
    with a small tail encoding.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        n_assets: int,
        per_asset_block_dim: int,
        tail_dim: int,
        per_asset_embed_dim: int = 32,
        tail_embed_dim: int = 16,
    ) -> None:
        expected = n_assets * per_asset_block_dim + tail_dim
        obs_dim = int(observation_space.shape[0])  # type: ignore[union-attr]
        if obs_dim != expected:
            raise ValueError(
                f"observation_space dim {obs_dim} != n_assets*per_asset_block_dim+tail_dim "
                f"({n_assets}*{per_asset_block_dim}+{tail_dim}={expected})"
            )
        features_dim = n_assets * per_asset_embed_dim + tail_embed_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.n_assets = n_assets
        self.per_asset_block_dim = per_asset_block_dim
        self.tail_dim = tail_dim

        self.per_asset_encoder = nn.Sequential(
            nn.Linear(per_asset_block_dim, 64),
            nn.ReLU(),
            nn.Linear(64, per_asset_embed_dim),
            nn.ReLU(),
        )
        self.tail_encoder = nn.Sequential(
            nn.Linear(tail_dim, tail_embed_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        per_asset_len = self.n_assets * self.per_asset_block_dim
        asset_part = observations[:, :per_asset_len].reshape(
            batch, self.n_assets, self.per_asset_block_dim
        )
        tail_part = observations[:, per_asset_len:]

        # Shared weights across the asset dimension: fold (batch, n_assets) into one
        # batch dim, apply the encoder once, unfold back.
        asset_flat = asset_part.reshape(batch * self.n_assets, self.per_asset_block_dim)
        asset_embed = self.per_asset_encoder(asset_flat).reshape(batch, self.n_assets, -1)
        asset_embed = asset_embed.reshape(batch, -1)  # concat per-asset embeddings

        tail_embed = self.tail_encoder(tail_part)
        return torch.cat([asset_embed, tail_embed], dim=1)


def make_ppo_portfolio(
    env: gym.Env,
    extractor: Optional[str] = None,
    net_arch: Optional[list[int]] = None,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    ent_coef: float = 0.0,
    device: str = "cpu",
    tensorboard_log: Optional[str] = None,
    seed: Optional[int] = None,
    **ppo_kwargs: Any,
) -> PPO:
    """Construct a PPO agent for MultiAssetPortfolioEnv.

    extractor=None (default): plain SB3 FlattenExtractor + net_arch MLP over the whole
    flat observation — the simplest path, no assumptions about env internals.
    extractor="portfolio": PortfolioFeatureExtractor (weight-tied per-asset encoder, see
    class docstring) — requires `env` to be (or wrap) a MultiAssetPortfolioEnv exposing
    n_assets/per_asset_block_dim/tail_dim, since the extractor needs those to slice the
    flat observation correctly.
    """
    policy_kwargs: dict[str, Any] = {}
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    if extractor == "portfolio":
        base_env = env
        while hasattr(base_env, "env") and not hasattr(base_env, "n_assets"):
            base_env = base_env.env  # unwrap VecEnv/Monitor-style wrappers
        if not hasattr(base_env, "n_assets"):
            raise ValueError(
                "extractor='portfolio' requires env to be/wrap a MultiAssetPortfolioEnv "
                "(no n_assets attribute found after unwrapping)"
            )
        policy_kwargs["features_extractor_class"] = PortfolioFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "n_assets": base_env.n_assets,
            "per_asset_block_dim": base_env.per_asset_block_dim,
            "tail_dim": base_env.tail_dim,
        }
    elif extractor is not None:
        raise ValueError(f"Unknown extractor: {extractor!r} (expected None or 'portfolio')")

    return PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        device=device,
        tensorboard_log=tensorboard_log,
        seed=seed,
        policy_kwargs=policy_kwargs or None,
        **ppo_kwargs,
    )
