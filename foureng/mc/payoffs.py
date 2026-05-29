"""Payoff functions operating on path arrays.

All functions accept a path matrix of shape ``(n_paths, n_steps+1)``
and return a 1-D payoff vector of shape ``(n_paths,)``.

Terminal-only payoffs (``european_payoff``) accept a 1-D terminal vector.

Barrier monitoring
------------------
``barrier_payoff`` monitors the barrier **at each stored time step**.
For accurate continuous-barrier pricing with a coarse grid, use the optional
Brownian-bridge correction (``use_bb_correction=True``), which probabilistically
fills in unobserved crossings between consecutive steps.
"""

from __future__ import annotations

import numpy as np


# ── vanilla / terminal payoffs ────────────────────────────────────────────

def european_payoff(S_T: np.ndarray, K: float, cp: int = 1) -> np.ndarray:
    """max(cp*(S_T − K), 0) — works on any-shape array."""
    return np.maximum(cp * (S_T - K), 0.0)


# ── path-dependent payoffs ─────────────────────────────────────────────────

def asian_arithmetic_payoff(
    paths: np.ndarray,
    K: float,
    cp: int = 1,
    *,
    skip_initial: bool = True,
) -> np.ndarray:
    """Arithmetic-average Asian payoff.

    Parameters
    ----------
    paths : (n_paths, n_steps+1)
        Full path matrix.
    K : float
        Strike.
    cp : int
        +1 call, −1 put.
    skip_initial : bool
        If True (default), the average excludes S₀ (index 0).

    Returns
    -------
    np.ndarray  (n_paths,)
    """
    avg = paths[:, 1:].mean(axis=1) if skip_initial else paths.mean(axis=1)
    return np.maximum(cp * (avg - K), 0.0)


def asian_geometric_payoff(
    paths: np.ndarray,
    K: float,
    cp: int = 1,
    *,
    skip_initial: bool = True,
) -> np.ndarray:
    """Geometric-average Asian payoff (log-average, then exponentiate).

    Returns
    -------
    np.ndarray  (n_paths,)
    """
    p = paths[:, 1:] if skip_initial else paths
    geo_avg = np.exp(np.log(p).mean(axis=1))
    return np.maximum(cp * (geo_avg - K), 0.0)


def barrier_payoff(
    paths: np.ndarray,
    K: float,
    H: float,
    barrier_type: str,
    cp: int = 1,
    *,
    use_bb_correction: bool = False,
    sigma: float | None = None,
    dt: float | None = None,
) -> np.ndarray:
    """Single-barrier payoff with optional Brownian-bridge correction.

    Parameters
    ----------
    paths : (n_paths, n_steps+1)
        Full path matrix.
    K, H : float
        Strike and barrier level.
    barrier_type : {"down_out", "down_in", "up_out", "up_in"}
    cp : int
    use_bb_correction : bool
        If True, apply the Brownian-bridge crossing probability to account
        for unobserved crossings between discrete time steps.
        Requires ``sigma`` and ``dt``.
    sigma, dt : float or None
        Needed only when ``use_bb_correction=True``.

    Returns
    -------
    np.ndarray  (n_paths,)
    """
    if barrier_type.startswith("down"):
        # Down barrier: breach when S falls to or below H
        if use_bb_correction and sigma is not None and dt is not None:
            _survived = _bb_down_survival(paths, H, sigma, dt)
        else:
            _survived = (paths.min(axis=1) > H)
    elif barrier_type.startswith("up"):
        if use_bb_correction and sigma is not None and dt is not None:
            _survived = _bb_up_survival(paths, H, sigma, dt)
        else:
            _survived = (paths.max(axis=1) < H)
    else:
        raise ValueError(f"Unknown barrier_type: {barrier_type!r}")

    if barrier_type.endswith("_out"):
        active = _survived
    else:  # _in
        active = ~_survived

    S_T = paths[:, -1]
    return np.maximum(cp * (S_T - K), 0.0) * active.astype(float)


# ── Brownian bridge correction ─────────────────────────────────────────────

def _bb_down_survival(
    paths: np.ndarray,
    H: float,
    sigma: float,
    dt: float,
) -> np.ndarray:
    """Survival probability for a down barrier using the BB approximation.

    Between consecutive path points (S_i, S_{i+1}), the exact GBM probability
    of touching a lower barrier H is:

        P(min S_t < H | S_i, S_{i+1}) = exp(-2 * log(S_i/H) * log(S_{i+1}/H) / (σ²dt))

    If log(S_i/H) ≤ 0 or log(S_{i+1}/H) ≤ 0 (already breached or at barrier)
    the interval is treated as definitely crossed.

    We draw U ~ Uniform(0,1) per interval and declare survival if U > crossing_prob.
    """
    rng_bb = np.random.default_rng()  # fresh rng; for reproducibility call with seeded paths
    n_paths, n_steps_p1 = paths.shape
    n_steps = n_steps_p1 - 1

    survived = np.ones(n_paths, dtype=bool)
    for j in range(n_steps):
        S_i  = paths[:, j]
        S_i1 = paths[:, j + 1]
        a = np.log(S_i  / H)
        b = np.log(S_i1 / H)
        # If either is ≤ 0 already breached at discrete grid
        already = (a <= 0) | (b <= 0)
        pos = (~already) & survived
        if not pos.any():
            survived &= ~already
            continue
        p_cross = np.zeros(n_paths)
        p_cross[pos] = np.exp(
            -2.0 * a[pos] * b[pos] / (sigma**2 * dt)
        )
        p_cross = np.clip(p_cross, 0.0, 1.0)
        U = rng_bb.uniform(size=n_paths)
        crossed = already | (U < p_cross)
        survived &= ~crossed
    return survived


def _bb_up_survival(
    paths: np.ndarray,
    H: float,
    sigma: float,
    dt: float,
) -> np.ndarray:
    """Survival probability for an up barrier using the BB approximation."""
    rng_bb = np.random.default_rng()
    n_paths, n_steps_p1 = paths.shape
    n_steps = n_steps_p1 - 1

    survived = np.ones(n_paths, dtype=bool)
    for j in range(n_steps):
        S_i  = paths[:, j]
        S_i1 = paths[:, j + 1]
        a = np.log(H / S_i)
        b = np.log(H / S_i1)
        already = (a <= 0) | (b <= 0)
        pos = (~already) & survived
        if not pos.any():
            survived &= ~already
            continue
        p_cross = np.zeros(n_paths)
        p_cross[pos] = np.exp(
            -2.0 * a[pos] * b[pos] / (sigma**2 * dt)
        )
        p_cross = np.clip(p_cross, 0.0, 1.0)
        U = rng_bb.uniform(size=n_paths)
        crossed = already | (U < p_cross)
        survived &= ~crossed
    return survived
