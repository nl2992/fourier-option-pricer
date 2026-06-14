"""Variance swap and variance option MC pricer under BSM."""

from __future__ import annotations

import numpy as np

from .path_engine import GBMPathSpec, simulate_gbm_paths


def _realized_variance(paths: np.ndarray, T: float) -> np.ndarray:
    """Annualized realized variance from path matrix (n_paths, n_steps+1).

    RV = (1/T) * sum_i (log S_i/S_{i-1})^2; under BSM E[RV] = sigma^2.
    """
    log_returns = np.diff(np.log(paths), axis=1)
    return np.sum(log_returns**2, axis=1) / T


def variance_swap_mc(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_steps: int,
    spec: GBMPathSpec,
) -> float:
    """Variance swap fair rate by MC. Under BSM E[annualized RV] = sigma^2."""
    path_spec = GBMPathSpec(
        n_paths=spec.n_paths,
        n_steps=n_steps,
        seed=spec.seed,
        antithetic=spec.antithetic,
        batch_size=spec.batch_size,
    )
    paths = simulate_gbm_paths(S0, r, q, sigma, T, path_spec)
    rv = _realized_variance(paths, T)
    # Fair value of variance swap = E[RV] (no discounting needed for fair rate)
    return float(np.mean(rv))


def variance_option_mc(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    K_var: float,
    n_steps: int,
    cp: int,
    spec: GBMPathSpec,
) -> float:
    """Variance option price by MC. Payoff = max(cp*(RV - K_var), 0) discounted at r."""
    path_spec = GBMPathSpec(
        n_paths=spec.n_paths,
        n_steps=n_steps,
        seed=spec.seed,
        antithetic=spec.antithetic,
        batch_size=spec.batch_size,
    )
    paths = simulate_gbm_paths(S0, r, q, sigma, T, path_spec)
    rv = _realized_variance(paths, T)
    disc = np.exp(-r * T)
    payoff = disc * np.maximum(cp * (rv - K_var), 0.0)
    return float(np.mean(payoff))
