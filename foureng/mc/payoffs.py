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


def lookback_payoff(
    paths: np.ndarray,
    cp: int = 1,
    *,
    strike_type: str = "floating",
    strike: float = 0.0,
) -> np.ndarray:
    """Lookback payoff from a full path matrix."""
    running_min = paths.min(axis=1)
    running_max = paths.max(axis=1)
    s_t = paths[:, -1]

    if strike_type == "floating":
        if cp == 1:
            return np.maximum(s_t - running_min, 0.0)
        return np.maximum(running_max - s_t, 0.0)

    if strike_type == "fixed":
        if cp == 1:
            return np.maximum(running_max - strike, 0.0)
        return np.maximum(strike - running_min, 0.0)

    raise ValueError(f"Unknown strike_type: {strike_type!r}")


def realized_variance(
    paths: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Annualized realized variance from log returns on observation dates."""
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("times must be a non-empty 1-D array")
    log_returns = np.diff(np.log(paths), axis=1)
    total_time = float(t[-1])
    return np.sum(log_returns**2, axis=1) / total_time


def variance_swap_payoff(
    paths: np.ndarray,
    times: np.ndarray,
    notional: float = 1.0,
) -> np.ndarray:
    """Variance swap payoff under zero fair-strike convention."""
    return notional * realized_variance(paths, times)


def variance_option_payoff(
    paths: np.ndarray,
    times: np.ndarray,
    strike: float,
    cp: int = 1,
    *,
    variance_type: str = "realised",
    sigma: float | None = None,
) -> np.ndarray:
    """Variance option payoff on annualized realized or integrated variance."""
    if variance_type == "integrated":
        if sigma is None:
            raise ValueError("variance_option_payoff requires sigma for integrated variance")
        underlying = np.full(paths.shape[0], sigma**2, dtype=np.float64)
    else:
        underlying = realized_variance(paths, times)
    return np.maximum(cp * (underlying - strike), 0.0)


def cliquet_payoff(
    paths: np.ndarray,
    cp: int = 1,
    *,
    local_floor: float = float("-inf"),
    local_cap: float = float("inf"),
    global_floor: float = float("-inf"),
    global_cap: float = float("inf"),
    payoff_type: str = "additive",
) -> np.ndarray:
    """Cliquet payoff from a path matrix observed on reset dates."""
    gross_returns = paths[:, 1:] / paths[:, :-1] - 1.0
    clipped = np.clip(gross_returns, local_floor, local_cap)

    if payoff_type == "additive":
        total = clipped.sum(axis=1)
        signed = cp * total
        return np.clip(signed, global_floor, global_cap)

    if payoff_type == "multiplicative":
        compounded = np.prod(1.0 + clipped, axis=1) - 1.0
        return np.clip(compounded, global_floor, global_cap)

    raise ValueError(f"Unknown payoff_type: {payoff_type!r}")


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
            _survived = paths.min(axis=1) > H
    elif barrier_type.startswith("up"):
        if use_bb_correction and sigma is not None and dt is not None:
            _survived = _bb_up_survival(paths, H, sigma, dt)
        else:
            _survived = paths.max(axis=1) < H
    else:
        raise ValueError(f"Unknown barrier_type: {barrier_type!r}")

    if barrier_type.endswith("_out"):
        active = _survived
    else:  # _in
        active = ~_survived

    S_T = paths[:, -1]
    return np.maximum(cp * (S_T - K), 0.0) * active.astype(float)


def double_barrier_payoff(
    paths: np.ndarray,
    K: float,
    lower: float,
    upper: float,
    cp: int = 1,
    *,
    knockout: bool = True,
) -> np.ndarray:
    """Double-barrier payoff monitored on the stored path grid."""
    if lower <= 0.0 or upper <= lower:
        raise ValueError("double_barrier_payoff requires 0 < lower < upper")
    survived = (paths.min(axis=1) > lower) & (paths.max(axis=1) < upper)
    active = survived if knockout else ~survived
    terminal = paths[:, -1]
    return np.maximum(cp * (terminal - K), 0.0) * active.astype(float)


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
        S_i = paths[:, j]
        S_i1 = paths[:, j + 1]
        a = np.log(S_i / H)
        b = np.log(S_i1 / H)
        # If either is ≤ 0 already breached at discrete grid
        already = (a <= 0) | (b <= 0)
        pos = (~already) & survived
        if not pos.any():
            survived &= ~already
            continue
        p_cross = np.zeros(n_paths)
        p_cross[pos] = np.exp(-2.0 * a[pos] * b[pos] / (sigma**2 * dt))
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
        S_i = paths[:, j]
        S_i1 = paths[:, j + 1]
        a = np.log(H / S_i)
        b = np.log(H / S_i1)
        already = (a <= 0) | (b <= 0)
        pos = (~already) & survived
        if not pos.any():
            survived &= ~already
            continue
        p_cross = np.zeros(n_paths)
        p_cross[pos] = np.exp(-2.0 * a[pos] * b[pos] / (sigma**2 * dt))
        p_cross = np.clip(p_cross, 0.0, 1.0)
        U = rng_bb.uniform(size=n_paths)
        crossed = already | (U < p_cross)
        survived &= ~crossed
    return survived
