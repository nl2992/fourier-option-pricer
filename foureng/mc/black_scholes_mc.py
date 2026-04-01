"""Plain Black-Scholes Monte Carlo baseline for European options.

Uses a one-step exact GBM simulation:
    S_T = S0 * exp((r - q - vol^2/2)*T + vol*sqrt(T)*Z)

Intended as a reference MC for convergence tests and a simple sanity check
against closed-form BS prices. For Heston MC see heston_conditional_mc.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCSpec:
    """Configuration for a plain Monte Carlo run.

    Attributes
    ----------
    n_paths : int
        Number of sample paths to simulate.
    seed : int or None
        Seed for NumPy's default_rng. None gives a non-reproducible run.
    """

    n_paths: int
    seed: int | None = None


def european_call_mc(
    S0: float,
    K: np.ndarray,
    T: float,
    r: float,
    q: float,
    vol: float,
    mc: MCSpec,
) -> np.ndarray:
    """Plain BS Monte Carlo baseline for European calls (single time step, exact GBM).

    S_T = S0 * exp((r - q - 0.5*vol^2)*T + vol*sqrt(T)*Z)
    Returns array of prices shape (len(K),).
    """
    K = np.atleast_1d(np.asarray(K, dtype=np.float64))
    if K.size == 0:
        raise ValueError("K/strikes must be non-empty")
    if np.any(~np.isfinite(K)) or np.any(K <= 0.0):
        raise ValueError("All strikes must be finite and strictly positive")
    if not (np.isfinite(S0) and S0 > 0.0):
        raise ValueError(f"S0 must be finite and strictly positive; got {S0}")
    if not (np.isfinite(T) and T > 0.0):
        raise ValueError(f"T must be finite and strictly positive; got {T}")
    if not (np.isfinite(r) and np.isfinite(q)):
        raise ValueError(f"r and q must be finite; got r={r}, q={q}")
    if not (np.isfinite(vol) and vol >= 0.0):
        raise ValueError(f"vol must be finite and non-negative; got {vol}")
    if mc.n_paths <= 0:
        raise ValueError("n_paths must be positive")
    rng = np.random.default_rng(mc.seed)
    Z = rng.standard_normal(mc.n_paths)
    ST = S0 * np.exp((r - q - 0.5 * vol * vol) * T + vol * np.sqrt(T) * Z)
    payoff = np.maximum(ST[:, None] - K[None, :], 0.0)
    return np.exp(-r * T) * payoff.mean(axis=0)
