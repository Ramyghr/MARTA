from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from encoder.augment import augment_window


class RegimeContrastiveDataset(Dataset):
    """
    Sliding-window contrastive dataset over a single asset's aligned time series.

    Returns (anchor, positive) pairs where both views are independently augmented
    copies of the same raw window (SimCLR design — no raw windows are returned).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        step: int = 1,
        augment_kwargs: dict | None = None,
    ) -> None:
        # Convert once; __getitem__ never touches the DataFrame again
        self.data: np.ndarray = df.values.astype(np.float32)
        self.window_size = window_size
        self.indices: list[int] = list(range(0, len(df) - window_size, step))
        # Forwarded to augment_window() as **kwargs — e.g. {"use_jitter": False}.
        # None/{} (the default) reproduces the original unconditional augment_window(window)
        # call exactly, so any existing caller that doesn't pass this is unaffected.
        self.augment_kwargs: dict = augment_kwargs or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        window = self.data[start : start + self.window_size]  # (60, 14)
        anchor = torch.from_numpy(augment_window(window, **self.augment_kwargs))
        positive = torch.from_numpy(augment_window(window, **self.augment_kwargs))
        return anchor, positive


class PredictiveContrastiveDataset(Dataset):
    """
    CPC-style predictive contrastive dataset (2026-08-12 encoder objective experiment) —
    positive pairs are (context window, NON-overlapping future window) instead of two
    augmented views of the SAME window. Trains the encoder to make its representation of
    "now" informative about what happens next, rather than merely invariant to noise.

    context = data[t-window_size+1 : t+1]   (window_size bars ending at t)
    future  = data[t+1 : t+1+future_size]   (future_size bars immediately following)

    Leakage: the valid-index range requires BOTH window_size bars of context AND
    future_size bars of future to fit inside `df` — the SAME already-split sub-dataframe
    (train_df or val_df) every other dataset in this module receives. A future window can
    therefore never extend past whichever split boundary `df` was already cut to, exactly
    the same guarantee RegimeContrastiveDataset relies on. See encoder/train.py's
    _load_asset_split for where df is sliced and the existing TestSetContaminationError
    check that validates it.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        future_size: int = 24,
        step: int = 1,
        augment_kwargs: dict | None = None,
        asset_id: int = 0,
        return_metadata: bool = False,
    ) -> None:
        self.data: np.ndarray = df.values.astype(np.float32)
        self.window_size = window_size
        self.future_size = future_size
        self.indices: list[int] = list(
            range(window_size - 1, len(df) - future_size, step)
        )
        self.augment_kwargs: dict = augment_kwargs or {}
        # Opt-in (2026-08-12, E1-NegCtrl) — asset_id is a fixed integer for this dataset
        # instance's whole lifetime (one instance = one asset). Deliberately NOT derived
        # from rl.env.CANONICAL_ASSETS, which uses "GOLD" while this pipeline's own asset
        # list uses "XAUUSD" — that exact file-stem/logical-name mismatch has caused real
        # bugs multiple times in this project; the caller passes an id built from its own
        # config's asset list instead (see encoder/train.py's _load_asset_split).
        self.asset_id = asset_id
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        context = self.data[t - self.window_size + 1 : t + 1]      # (window_size, 14)
        future = self.data[t + 1 : t + 1 + self.future_size]        # (future_size, 14)
        context_t = torch.from_numpy(augment_window(context, **self.augment_kwargs))
        future_t = torch.from_numpy(augment_window(future, **self.augment_kwargs))
        if self.return_metadata:
            return context_t, future_t, self.asset_id, t
        return context_t, future_t
