"""Floating- and fixed-strike lookback option MC pricer."""

from __future__ import annotations

import numpy as np

from .path_engine import GBMPathSpec, simulate_gbm_paths


def lookback_mc(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_steps: int,
    cp: int,
    spec: GBMPathSpec,
    K: float | None = None,
) -> float:
    """Lookback option price by MC simulation.

    Floating-strike (K is None):
        cp=+1 : payoff = S_T - min_{0..T} S_u
        cp=-1 : payoff = max_{0..T} S_u - S_T

    Fixed-strike (K provided):
        cp=+1 : payoff = max(max_{0..T} S_u - K, 0)
        cp=-1 : payoff = max(K - min_{0..T} S_u, 0)

    S0 is included in the running min/max.

    Parameters
    ----------
    S0, r, q, sigma, T : standard BSM inputs
    n_steps : time steps
    cp      : +1 or -1
    spec    : GBMPathSpec
    K       : fixed strike, or None for floating-strike

    Returns
    -------
    float
        Discounted lookback price.
    """
    path_spec = GBMPathSpec(
        n_paths=spec.n_paths,
        n_steps=n_steps,
        seed=spec.seed,
        antithetic=spec.antithetic,
        batch_size=spec.batch_size,
    )
    paths = simulate_gbm_paths(S0, r, q, sigma, T, path_spec)

    # Include S0 (col 0) in running min/max
    path_min = paths.min(axis=1)
    path_max = paths.max(axis=1)
    S_T = paths[:, -1]

    disc = np.exp(-r * T)

    if K is None:
        # Floating-strike
        if cp == 1:
            payoff = S_T - path_min
        else:
            payoff = path_max - S_T
    else:
        # Fixed-strike
        if cp == 1:
            payoff = np.maximum(path_max - K, 0.0)
        else:
            payoff = np.maximum(K - path_min, 0.0)

    return float(disc * np.mean(payoff))
