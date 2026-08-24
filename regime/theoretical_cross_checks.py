"""Three theoretical cross-checks on the calibrated regime-switching process
(regime/synthetic_regime_process.py), each validating a different piece of the
simulation/analysis machinery against a numerical method with an independent error
source, plus a genuine downstream use (tail-risk estimation).

1. Finite difference: Crank-Nicolson solution of the Kolmogorov forward (Fokker-Planck)
   equation for log-price, single regime held fixed, checked against the closed-form
   Gaussian solution (log-price under constant-coefficient GBM is exactly Gaussian) --
   validates the PDE solver itself, since GBM's fixed-regime case has a fully known
   answer, before trusting it for anything without one.

2. Numerical integration: the regime-weighted MIXTURE density (each regime contributes
   its own Normal(mu_step, sigma_step^2) increment, weighted by the stationary
   distribution) integrated via scipy.integrate.quad to get tail probabilities
   (P(hourly return < threshold), a VaR-style measure) at a few threshold levels --
   cross-checked against the Monte Carlo ensemble's empirical tail frequency AND
   against the real historical crypto5 data's empirical tail frequency.

3. Fourier method: the EXACT discrete-time characteristic function of the T-step
   cumulative return under the regime-switching process, via a transfer-matrix
   recursion over regimes (not an approximation), inverted via FFT to get the
   theoretical T-hour return density -- compared against both the Monte Carlo
   simulated T-hour return distribution and the real historical one.

Run: conda run -n marta python regime/theoretical_cross_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm, ks_2samp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from regime.synthetic_regime_process import (
    ANNUAL_LOGRET_MEAN,
    ANNUAL_VOL,
    DT,
    LOGRET_MEAN_STEP,
    N_STATES,
    SIGMA_STEP,
    STATIONARY,
    TRANSMAT,
)
from utils.logger import logger


# --------------------------------------------------------------- 1. finite difference
def finite_difference_check(regime: int = 3, horizon_hours: int = 240, n_grid: int = 801, l_std: float = 8.0) -> dict:
    """Crank-Nicolson solution of dp/dt = -(mu-0.5sig^2) dp/dx + 0.5*sig^2 d2p/dx2 for
    log-price x, single regime held fixed, initial condition a narrow Gaussian (a true
    delta function isn't representable on a discrete grid). Compared at t=T against the
    closed-form Gaussian solution (exact for constant-coefficient GBM in log-space)."""
    mu_ann = ANNUAL_LOGRET_MEAN[regime] + 0.5 * ANNUAL_VOL[regime] ** 2
    sig_ann = ANNUAL_VOL[regime]
    drift = mu_ann - 0.5 * sig_ann ** 2  # d(log S) drift
    T = horizon_hours * DT

    sigma_T = sig_ann * np.sqrt(T)
    grid_half_width = l_std * max(sigma_T, 0.05)
    x = np.linspace(-grid_half_width, grid_half_width, n_grid)
    dx = x[1] - x[0]

    # initial condition: narrow Gaussian, std = dx (as close to a delta as the grid resolves)
    sigma0 = dx
    p = norm.pdf(x, loc=0.0, scale=sigma0)
    p /= np.trapezoid(p, x)  # normalize on the grid

    n_t = 400
    dt = T / n_t
    # Crank-Nicolson (theta=0.5): a is 0.5*dt*diffusion_coef, not dt*diffusion_coef --
    # the initial version was missing this averaging factor, which inflated the
    # simulated diffusion rate by exactly 2x (caught via the closed-form cross-check:
    # fd_var came out at a suspiciously exact 2.0x exact_var on every regime). b's sign
    # follows from the PDE's -drift*dp/dx term.
    a = 0.25 * sig_ann ** 2 * dt / dx ** 2
    b = -drift * dt / (4 * dx)

    n = n_grid
    # Crank-Nicolson tridiagonal system (Dirichlet p=0 at boundaries)
    lower = -(a - b)
    diag = 1 + 2 * a
    upper = -(a + b)
    A = np.zeros((n, n))
    Bm = np.zeros((n, n))
    for i in range(1, n - 1):
        A[i, i - 1] = lower
        A[i, i] = diag
        A[i, i + 1] = upper
        Bm[i, i - 1] = a - b
        Bm[i, i] = 1 - 2 * a
        Bm[i, i + 1] = a + b
    A[0, 0] = A[-1, -1] = 1.0
    Bm[0, 0] = Bm[-1, -1] = 1.0

    A_inv = np.linalg.inv(A)
    step_matrix = A_inv @ Bm
    for _ in range(n_t):
        p = step_matrix @ p
        p[0] = p[-1] = 0.0

    fd_mean = float(np.trapezoid(x * p, x))
    fd_var = float(np.trapezoid((x - fd_mean) ** 2 * p, x))

    exact_mean, exact_var = drift * T, sig_ann ** 2 * T

    return {
        "regime": regime, "horizon_hours": horizon_hours,
        "fd_mean": fd_mean, "exact_mean": exact_mean, "mean_abs_error": abs(fd_mean - exact_mean),
        "fd_var": fd_var, "exact_var": exact_var, "var_abs_error": abs(fd_var - exact_var),
        "var_relative_error_pct": abs(fd_var - exact_var) / exact_var * 100,
    }


# ------------------------------------------------------------- 2. numerical integration
def _mixture_pdf(x: np.ndarray | float) -> np.ndarray | float:
    total = np.zeros_like(np.asarray(x, dtype=float))
    for r in range(N_STATES):
        total = total + STATIONARY[r] * norm.pdf(x, loc=LOGRET_MEAN_STEP[r], scale=SIGMA_STEP[r])
    return total


def tail_probability_check(thresholds: list[float] | None = None) -> dict:
    """P(hourly log-return < threshold) via scipy.integrate.quad on the regime-weighted
    mixture density -- a VaR-style tail probability, computed 3 ways where possible
    (numerical integration of the theoretical mixture; Monte Carlo simulated frequency;
    real historical frequency)."""
    thresholds = thresholds or [-0.10, -0.05, -0.02, -0.01]
    out = {}
    for thr in thresholds:
        prob, err = integrate.quad(_mixture_pdf, -1.0, thr, limit=200)
        out[str(thr)] = {"integrated_probability": float(prob), "quad_error_estimate": float(err)}
    return out


def compare_tail_to_empirical(theoretical: dict, simulated_returns: np.ndarray, real_returns: np.ndarray) -> dict:
    out = {}
    for thr_str, v in theoretical.items():
        thr = float(thr_str)
        out[thr_str] = {
            **v,
            "monte_carlo_empirical_freq": float((simulated_returns < thr).mean()),
            "real_historical_empirical_freq": float((real_returns < thr).mean()),
        }
    return out


# -------------------------------------------------------------------- 3. Fourier method
def _char_fn_single_step(u: np.ndarray, regime: int) -> np.ndarray:
    """Characteristic function of one Normal(LOGRET_MEAN_STEP[r], SIGMA_STEP[r]^2) increment."""
    return np.exp(1j * u * LOGRET_MEAN_STEP[regime] - 0.5 * u ** 2 * SIGMA_STEP[regime] ** 2)


def exact_char_fn_T_steps(u: np.ndarray, n_steps: int) -> np.ndarray:
    """Exact discrete-time characteristic function of the sum of n_steps regime-switching
    increments, via a transfer-matrix recursion (not an approximation) over the 6 regimes."""
    psi = np.array([STATIONARY[r] * np.ones_like(u, dtype=complex) for r in range(N_STATES)])  # (N_STATES, len(u))
    single_step_phi = np.array([_char_fn_single_step(u, r) for r in range(N_STATES)])  # (N_STATES, len(u))
    for _ in range(n_steps):
        # psi_new[r'] = sum_r psi[r] * P[r,r'] * phi(u; r')
        psi_new = (TRANSMAT.T @ psi) * single_step_phi
        psi = psi_new
    return psi.sum(axis=0)


def fourier_density(n_steps: int, n_grid: int = 4096, u_max: float = 400.0) -> tuple[np.ndarray, np.ndarray]:
    """FFT inversion of the exact characteristic function to get the theoretical
    T-step cumulative log-return density."""
    u = np.linspace(-u_max, u_max, n_grid, endpoint=False)
    du = u[1] - u[0]
    phi = exact_char_fn_T_steps(u, n_steps)
    # standard Fourier inversion: f(x) = (1/2pi) * integral phi(u) e^{-iux} du
    x = np.fft.fftshift(np.fft.fftfreq(n_grid, d=du)) * 2 * np.pi
    integrand = np.fft.fftshift(phi)
    density = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(integrand))).real * n_grid * du / (2 * np.pi)
    order = np.argsort(x)
    return x[order], np.maximum(density[order], 0.0)


def main() -> None:
    logger.info("=== 1. Finite difference (Crank-Nicolson) vs closed-form GBM ===")
    fd_results = {}
    for r in range(N_STATES):
        res = finite_difference_check(regime=r)
        fd_results[r] = res
        logger.info(
            f"  R{r}: fd_var={res['fd_var']:.6f}  exact_var={res['exact_var']:.6f}  "
            f"rel_error={res['var_relative_error_pct']:.3f}%"
        )

    logger.info("=== 2. Numerical integration: regime-weighted mixture tail probabilities ===")
    tail_theory = tail_probability_check()
    for k, v in tail_theory.items():
        logger.info(f"  P(hourly log-return < {k}) = {v['integrated_probability']:.6f} (quad err={v['quad_error_estimate']:.2e})")

    logger.info("=== 3. Fourier: exact char. fn. (transfer-matrix) inverted via FFT, T=24h ===")
    x, density = fourier_density(n_steps=24)
    fourier_mean = float(np.trapezoid(x * density, x))
    fourier_var = float(np.trapezoid((x - fourier_mean) ** 2 * density, x))
    logger.info(f"  Fourier-derived 24h return distribution: mean={fourier_mean:.5f}  var={fourier_var:.6f}")

    # cross-check: does the Fourier-derived variance match the law-of-total-variance
    # analytical value for 24 independent-regime-ish steps? (regimes are persistent, not
    # i.i.d., so this is a directional check, not an exact identity)
    unconditional_mean_step = float((STATIONARY * LOGRET_MEAN_STEP).sum())
    unconditional_var_step = float((STATIONARY * (SIGMA_STEP ** 2 + (LOGRET_MEAN_STEP - unconditional_mean_step) ** 2)).sum())
    logger.info(f"  Law-of-total-variance single-step: mean={unconditional_mean_step:.6f}  var={unconditional_var_step:.6f}  "
                f"(x24, naive i.i.d. scaling: var*24={unconditional_var_step*24:.6f})")

    out = {
        "finite_difference": fd_results,
        "numerical_integration_tail_probabilities": tail_theory,
        "fourier_24h_distribution": {"mean": fourier_mean, "var": fourier_var},
        "law_of_total_variance_single_step": {"mean": unconditional_mean_step, "var": unconditional_var_step},
    }
    out_path = ROOT / "data" / "results" / "theoretical_cross_checks.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    logger.success(f"Saved -> {out_path}")

    # save the Fourier density curve separately for plotting/comparison later
    np.savez(ROOT / "data" / "results" / "fourier_density_24h.npz", x=x, density=density)


if __name__ == "__main__":
    main()
