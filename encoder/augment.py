"""
Contrastive augmentation functions for regime windows.

All functions are pure, stateless, and thread-safe: each call creates a local
np.random.Generator seeded from OS entropy — safe for multi-worker DataLoaders.
"""

from __future__ import annotations

import numpy as np


def amplitude_scale(window: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Multiply all values by a scalar drawn from Uniform(0.85, 1.15)."""
    rng = np.random.default_rng()
    if rng.random() >= p:
        return window.copy()
    scale = rng.uniform(0.85, 1.15)
    return (window * scale).astype(np.float32)


def add_jitter(window: np.ndarray, p: float = 0.5, sigma: float = 0.03) -> np.ndarray:
    """Add i.i.d. Gaussian noise with std=sigma to every element."""
    rng = np.random.default_rng()
    if rng.random() >= p:
        return window.copy()
    noise = rng.normal(0.0, sigma, window.shape).astype(np.float32)
    return (window + noise).astype(np.float32)


def channel_mask(window: np.ndarray, p: float = 0.5, n_mask: int = 2) -> np.ndarray:
    """Zero out n_mask randomly chosen feature columns for the entire window."""
    rng = np.random.default_rng()
    if rng.random() >= p:
        return window.copy()
    result = window.copy()
    n_features = window.shape[1]
    mask_cols = rng.choice(n_features, size=n_mask, replace=False)
    result[:, mask_cols] = 0.0
    return result


def time_warp(window: np.ndarray, p: float = 0.3, max_shift: float = 5.0) -> np.ndarray:
    """
    Elastic distortion of the time axis via a smooth sin-wave perturbation.

    Strategy: treat the original data as sampled on a regular grid and resample
    it as if the grid were warped by a sin wave with random phase and amplitude.
    np.interp(src, warp, data) reads from the warped positions and maps them
    back onto the regular output grid, preserving shape (T, F) exactly.
    """
    rng = np.random.default_rng()
    if rng.random() >= p:
        return window.copy()
    T, F = window.shape
    src = np.linspace(0, T - 1, T)
    phase = rng.uniform(0, 2 * np.pi)
    amplitude = rng.uniform(0, max_shift)
    # warp is analytically monotone for amplitude <= max_shift and T=60
    # (min derivative = 1 - amplitude * 2π/T = 1 - 5 * 2π/60 ≈ 0.48 > 0)
    warp = src + amplitude * np.sin(src * 2.0 * np.pi / T + phase)
    warp = np.clip(warp, 0, T - 1)
    result = np.empty_like(window)
    for f in range(F):
        result[:, f] = np.interp(src, warp, window[:, f])
    return result.astype(np.float32)


def augment_window(window: np.ndarray) -> np.ndarray:
    """Apply all four augmentations sequentially and independently."""
    x = amplitude_scale(window)
    x = add_jitter(x)
    x = channel_mask(x)
    x = time_warp(x)
    return x
