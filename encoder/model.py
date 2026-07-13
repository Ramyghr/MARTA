"""
TFT-inspired regime encoder for MARTA.

Architecture replicates the encoder portion of a Temporal Fusion Transformer:
  VSN (feature gating) → 2-layer LSTM → mean pool → gated residual → projection head.

Month 2 refinement note:
  Replace the VSN + LSTM block with pytorch-forecasting's TemporalFusionTransformer
  by passing windows through TFT.predict() and extracting the internal encoder hidden
  states (access via monkey-patching TFT._get_encoder_output or a forward hook on
  the lstm_encoder sub-module). The projection head and gated residual remain unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TFTRegimeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 128,
        proj_dim: int = 64,
        n_lstm_layers: int = 2,
    ) -> None:
        super().__init__()

        # Variable Selection Network: learned per-timestep feature importance
        self.vsn_linear = nn.Linear(input_dim, input_dim)

        # Sequence encoder (mirrors TFT's internal LSTM encoder)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=0.1 if n_lstm_layers > 1 else 0.0,
        )

        # Gated residual connection applied after mean pooling
        self.gate = nn.Linear(hidden_dim, hidden_dim)

        # Projection head — discarded after pretraining; only used by NTXentLoss
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def _vsn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply VSN: per-timestep softmax gating over features."""
        weights = torch.softmax(self.vsn_linear(x), dim=-1)  # (B, T, 14)
        return x * weights

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder path used by both forward() and encode()."""
        x_weighted = self._vsn(x)                        # (B, T, 14)
        lstm_out, _ = self.lstm(x_weighted)               # (B, T, 128)
        h = lstm_out.mean(dim=1)                          # (B, 128) — robust to regime length
        gate = torch.sigmoid(self.gate(h))                # (B, 128)
        return h * gate                                   # (B, 128)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding, projection) — projection used only during contrastive training."""
        embedding = self._encode(x)                       # (B, 128)
        projection = self.projector(embedding)            # (B, 64)
        return embedding, projection

    def get_vsn_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-timestep VSN feature importance weights (B, T, 14).

        Passed to the Technical Agent in Month 2 as the attention map.
        """
        return torch.softmax(self.vsn_linear(x), dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the 128-dim embedding — used for inference after pretraining."""
        return self._encode(x)
