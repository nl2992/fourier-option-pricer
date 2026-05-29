"""Path generators for Monte Carlo simulation.

All generators return an ``(n_paths, n_steps + 1)`` float64 array of spot
values, including the initial spot S₀ at index 0.

Conventions
-----------
* Paths are generated under the risk-neutral measure.
* ``rng`` must be a ``numpy.random.Generator`` (e.g. ``default_rng(seed)``).
* ``n_steps=1`` gives a single terminal draw — equivalent to the exact
  one-step GBM formula, so there is no additional Euler error.
"""

from __future__ import annotations

import numpy as np


def gbm_paths(
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
    *,
    antithetic: bool = False,
) -> np.ndarray:
    """Generate GBM paths under the risk-neutral measure.

    Uses the exact log-Euler scheme:  dlog(S) = (r−q−σ²/2)dt + σ dW_t

    Parameters
    ----------
    S0, r, q, T, sigma :
        Spot, risk-free rate, div yield, maturity, lognormal vol.
    n_paths, n_steps :
        Path count and discretisation steps.  ``n_steps=1`` is exact.
    rng : numpy.random.Generator
        Random source.
    antithetic : bool
        If True, ``n_paths`` must be even; the second half uses −Z.
        The returned array still has ``n_paths`` rows.

    Returns
    -------
    np.ndarray shape (n_paths, n_steps + 1)
        S[i, 0] = S0 for all i;  S[i, j] is the spot at time j*dt.
    """
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal((half, n_steps))
        Z = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    log_increments = drift + diff * Z                  # (n_paths, n_steps)
    log_S = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )                                                  # (n_paths, n_steps+1)
    return S0 * np.exp(log_S)


def gbm_terminal(
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    n_paths: int,
    rng: np.random.Generator,
    *,
    antithetic: bool = False,
) -> np.ndarray:
    """Draw only the terminal GBM value S_T (one-step exact).

    Returns
    -------
    np.ndarray shape (n_paths,)
    """
    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal(half)
        Z = np.concatenate([Z_half, -Z_half])
    else:
        Z = rng.standard_normal(n_paths)
    return S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
