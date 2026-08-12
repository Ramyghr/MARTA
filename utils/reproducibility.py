"""Shared determinism utility (P0.3) — one seeding call for every RNG source this repo
touches, so train/eval entrypoints don't each reinvent (and potentially diverge from)
their own seeding logic.

Historical context: validation Sharpe was found to swing wildly (~-1.8 to +1.4) across
repeated eval runs on the SAME checkpoint before the 2026-07-31 eval-reliability audit,
traced to unseeded GPU forward passes on the frozen regime encoder. rl/evaluate.py already
seeded before each eval pass; this consolidates that logic and closes the larger gap —
RL training itself (run_rl.py) had no explicit seeding at all beyond stable-baselines3's
own PPO(seed=...) constructor argument, which only covers randomness inside SB3's own
scope and not this repo's own env-level or data-prep randomness.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python's random, NumPy, and PyTorch (CPU + all visible CUDA devices).

    Call once at the top of any train or eval entrypoint, before any other
    randomness-consuming code runs (data loading, env construction, model init).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
