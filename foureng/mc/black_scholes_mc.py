from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class MCSpec:
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
    K = np.atleast_1d(np.asarray(K, dtype=float))
    if K.size == 0:
        raise ValueError("K/strikes must be non-empty")
    if np.any(K <= 0.0):
        raise ValueError("All strikes must be strictly positive")
    if S0 <= 0.0:
        raise ValueError("S0 must be strictly positive")
    if T <= 0.0:
        raise ValueError("T must be strictly positive")
    if vol < 0.0:
        raise ValueError("vol must be non-negative")
    if mc.n_paths <= 0:
        raise ValueError("n_paths must be positive")
    rng = np.random.default_rng(mc.seed)
    Z = rng.standard_normal(mc.n_paths)
    ST = S0 * np.exp((r - q - 0.5 * vol * vol) * T + vol * np.sqrt(T) * Z)
    payoff = np.maximum(ST[:, None] - K[None, :], 0.0)
    return np.exp(-r * T) * payoff.mean(axis=0)
