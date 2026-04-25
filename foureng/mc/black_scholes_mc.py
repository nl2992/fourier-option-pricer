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
    rng = np.random.default_rng(mc.seed)
    Z = rng.standard_normal(mc.n_paths)
    ST = S0 * np.exp((r - q - 0.5 * vol * vol) * T + vol * np.sqrt(T) * Z)
    payoff = np.maximum(ST[:, None] - K[None, :], 0.0)
    return np.exp(-r * T) * payoff.mean(axis=0)
