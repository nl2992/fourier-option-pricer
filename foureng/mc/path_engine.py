"""Batched GBM path simulator for path-dependent Monte Carlo engines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GBMPathSpec:
    """Configuration for a GBM path simulation run."""

    n_paths: int
    n_steps: int
    seed: int | None = None
    antithetic: bool = True
    batch_size: int = 10_000


def simulate_gbm_paths(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    spec: GBMPathSpec,
) -> np.ndarray:
    """Simulate GBM paths via log-Euler stepping, returning shape (n_paths, n_steps+1).

    Column 0 is S0. When spec.antithetic=True the second half of paths are antithetic
    mirrors of the first half; n_paths should be even.
    """
    n = spec.n_paths
    m = spec.n_steps
    dt = T / m
    drift = (r - q - 0.5 * sigma**2) * dt
    vol_dt = sigma * np.sqrt(dt)

    rng = np.random.default_rng(spec.seed)

    # Pre-allocate output
    paths = np.empty((n, m + 1), dtype=np.float64)
    paths[:, 0] = S0

    # Half-paths needed for antithetic draws
    n_base = n // 2 if spec.antithetic else n
    # Batch over steps to keep peak memory low for large n_steps
    batch = spec.batch_size

    col = 1
    while col <= m:
        end = min(col + batch, m + 1)
        n_cols = end - col

        Z_base = rng.standard_normal((n_base, n_cols))

        if spec.antithetic:
            Z = np.concatenate([Z_base, -Z_base], axis=0)
            # Pad to full n if n is odd
            if Z.shape[0] < n:
                Z = np.vstack([Z, Z_base[:1]])
        else:
            Z = Z_base

        log_increments = drift + vol_dt * Z
        log_prev = np.log(paths[:, col - 1])
        log_steps = np.cumsum(log_increments, axis=1)
        paths[:, col:end] = np.exp(log_prev[:, None] + log_steps)

        col = end

    return paths
