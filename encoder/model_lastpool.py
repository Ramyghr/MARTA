"""E1 pooling ablation — INFERENCE-ONLY variant of TFTRegimeEncoder (2026-08-11).

Not used for training. Loads existing (already-trained) checkpoint weights and swaps
only the temporal pooling step: `lstm_out[:, -1, :]` (last-timestep hidden state) instead
of `lstm_out.mean(dim=1)` (unweighted mean over all 60 timesteps). Everything else — VSN,
LSTM, gated residual, projection head, and all trained weights — is byte-identical to
encoder/model.py; only the aggregation of the LSTM's already-computed per-timestep hidden
states differs.

Why this is a valid test without retraining: for a non-bidirectional nn.LSTM,
`lstm_out[:, -1, :]` is mathematically identical to `h_n[-1]` (the last layer's final
hidden state) — a well-known PyTorch LSTM invariant, verified earlier in this
investigation. The LSTM already computes a hidden state at every timestep regardless of
how the caller pools them; extracting the last one instead of averaging doesn't require
different weights, just a different read-out of already-computed activations. This tests
whether mean-pooling was discarding recency-relevant information that was already present
in the network's own computation, not whether a differently-trained network could do
better (that would require retraining, out of scope here).

Do not use this class for anything except this diagnostic. It is not the encoder used by
the RL environment, agents, or regime pipeline.
"""

from __future__ import annotations

import torch

from encoder.model import TFTRegimeEncoder


class TFTRegimeEncoderLastPool(TFTRegimeEncoder):
    """Identical to TFTRegimeEncoder except _encode() pools via the last timestep."""

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x_weighted = self._vsn(x)                        # (B, T, 14)
        lstm_out, _ = self.lstm(x_weighted)               # (B, T, 128)
        h = lstm_out[:, -1, :]                             # (B, 128) — last timestep, not mean
        gate = torch.sigmoid(self.gate(h))                # (B, 128) — unchanged downstream path
        return h * gate                                   # (B, 128)
