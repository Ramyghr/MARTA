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
    ) -> None:
        # Convert once; __getitem__ never touches the DataFrame again
        self.data: np.ndarray = df.values.astype(np.float32)
        self.window_size = window_size
        self.indices: list[int] = list(range(0, len(df) - window_size, step))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        window = self.data[start : start + self.window_size]  # (60, 14)
        anchor = torch.from_numpy(augment_window(window))
        positive = torch.from_numpy(augment_window(window))
        return anchor, positive
