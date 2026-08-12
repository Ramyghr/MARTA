from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def infonce_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute InfoNCE loss and top-1 accuracy for a batch of contrastive pairs.

    Returns (loss, top1_accuracy) — both scalar tensors.
    Random-chance baseline: loss = log(batch_size), accuracy = 1/batch_size.
    """
    z_a = F.normalize(z_anchor, dim=1)   # (B, D)
    z_p = F.normalize(z_positive, dim=1) # (B, D)

    batch_size = z_a.shape[0]
    sim_matrix = torch.mm(z_a, z_p.T) / temperature  # (B, B)

    labels = torch.arange(batch_size, device=z_anchor.device)
    loss = F.cross_entropy(sim_matrix, labels)

    with torch.no_grad():
        preds = sim_matrix.argmax(dim=1)
        top1_acc = (preds == labels).float().mean()

    return loss, top1_acc


class NTXentLoss(nn.Module):
    """
    NT-Xent loss with a learnable temperature parameter.

    Temperature is parameterised as exp(-log_temp) so that gradient descent
    can anneal it during training without going negative.
    Initialised to log(1/0.07) ≈ 2.66 → effective starting temperature = 0.07.
    """

    def __init__(self) -> None:
        super().__init__()
        # log(1/0.07) initialisation → temperature = exp(-log_temp) = 0.07
        self.log_temp = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(-self.log_temp)

    def clamp_temperature(self, min_temperature: float, max_temperature: float) -> None:
        """Hard-bound ``log_temp`` in place so ``temperature = exp(-log_temp)`` cannot
        wander outside ``[min_temperature, max_temperature]``.

        Call after each optimizer step. Gradient descent on an unconstrained log_temp was
        observed to drive temperature to 0.0047 (6.4x below the 0.03 floor) during the
        2026-07 Kaggle walk-forward retrain — the convergence check in encoder/train.py
        only warns on this, it doesn't constrain it, so it went to production unnoticed.
        """
        with torch.no_grad():
            # temperature <= max_temperature  <=>  log_temp >= -ln(max_temperature)
            # temperature >= min_temperature  <=>  log_temp <= -ln(min_temperature)
            low = -math.log(max_temperature)
            high = -math.log(min_temperature)
            self.log_temp.clamp_(min=low, max=high)

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        exclude_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """exclude_mask: optional (B,B) bool tensor, True = forbid this (anchor_i,
        positive_j) pair from counting as a negative (2026-08-12, E1-NegCtrl). None
        (default) reproduces the original behavior exactly — every existing caller that
        doesn't pass this is unaffected. The diagonal must never be True in exclude_mask
        (that's the real positive) — see build_temporal_exclusion_mask, which enforces
        this by construction; this method does not re-check it, callers are responsible.
        """
        z_a = F.normalize(z_anchor, dim=1)
        z_p = F.normalize(z_positive, dim=1)

        batch_size = z_a.shape[0]
        # Divide by self.temperature tensor so gradient flows to log_temp
        sim_matrix = torch.mm(z_a, z_p.T) / self.temperature  # (B, B)
        if exclude_mask is not None:
            sim_matrix = sim_matrix.masked_fill(exclude_mask, -1e9)

        labels = torch.arange(batch_size, device=z_anchor.device)
        loss = F.cross_entropy(sim_matrix, labels)

        with torch.no_grad():
            preds = sim_matrix.argmax(dim=1)
            top1_acc = (preds == labels).float().mean()

        return loss, top1_acc


def build_temporal_exclusion_mask(
    asset_ids: torch.Tensor,
    times: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """(B,B) bool mask, True = same asset AND |t_i - t_j| < radius — i.e. forbid this
    pair from counting as an NT-Xent negative (2026-08-12, E1-NegCtrl). Cross-asset pairs
    are never masked regardless of timestamp (asset_ids must match too). The diagonal
    (i==i) is always False (a sample is never excluded from being its own positive) —
    enforced here explicitly, not left to the caller.
    """
    same_asset = asset_ids.unsqueeze(1) == asset_ids.unsqueeze(0)         # (B, B)
    time_close = (times.unsqueeze(1) - times.unsqueeze(0)).abs() < radius  # (B, B)
    mask = same_asset & time_close
    mask.fill_diagonal_(False)
    return mask
