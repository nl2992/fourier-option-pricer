"""Finite-difference BSM vanilla option pricers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams


@dataclass(frozen=True)
class PDEGrid:
    """Finite-difference grid for one-dimensional BSM pricing."""

    spot_steps: int = 400
    time_steps: int = 400
    s_max_mult: float = 4.0
    omega: float = 1.2
    psor_tol: float = 1e-10
    psor_max_iter: int = 10_000


def _boundary(
    s_max: float,
    strike: float,
    tau: float,
    r: float,
    q: float,
    cp: int,
    exercise: str,
) -> tuple[float, float]:
    if cp == 1:
        lower = 0.0
        upper = (
            max(s_max - strike, 0.0)
            if exercise == "american"
            else (s_max * np.exp(-q * tau) - strike * np.exp(-r * tau))
        )
    else:
        lower = strike if exercise == "american" else strike * np.exp(-r * tau)
        upper = 0.0
    return float(lower), float(max(upper, 0.0))


def _solve_lcp_psor(
    lower_diag: np.ndarray,
    diag: np.ndarray,
    upper_diag: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    *,
    omega: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    x = np.maximum(rhs / diag, obstacle)
    n = x.size
    for _ in range(max_iter):
        err = 0.0
        for i in range(n):
            left = lower_diag[i - 1] * x[i - 1] if i > 0 else 0.0
            right = upper_diag[i] * x[i + 1] if i < n - 1 else 0.0
            candidate = (rhs[i] - left - right) / diag[i]
            updated = max(obstacle[i], x[i] + omega * (candidate - x[i]))
            err = max(err, abs(updated - x[i]))
            x[i] = updated
        if err < tol:
            return x
    return x


def bsm_pde_fd_price(
    fwd: ForwardSpec,
    params: BsmParams,
    strike: float,
    *,
    cp: int = 1,
    exercise: str = "european",
    grid: PDEGrid | None = None,
) -> float:
    """Price a BSM vanilla option by an implicit finite-difference scheme."""
    if cp not in (1, -1):
        raise ValueError(f"bsm_pde_fd_price: cp must be +1 or -1, got {cp}")
    if exercise not in {"european", "american"}:
        raise ValueError("exercise must be 'european' or 'american'")
    if strike <= 0.0 or not np.isfinite(strike):
        raise ValueError("strike must be finite and > 0")

    spec = grid or PDEGrid()
    if spec.spot_steps < 20 or spec.time_steps < 1:
        raise ValueError("PDEGrid requires spot_steps >= 20 and time_steps >= 1")
    if spec.s_max_mult <= 1.0:
        raise ValueError("PDEGrid.s_max_mult must be > 1")

    m = int(spec.spot_steps)
    n = int(spec.time_steps)
    s_max = spec.s_max_mult * max(fwd.S0, strike)
    dt = fwd.T / n

    S = np.linspace(0.0, s_max, m + 1)
    values = np.maximum(cp * (S - strike), 0.0)
    interior_S = S[1:m]
    idx = np.arange(1, m, dtype=np.float64)

    a = 0.5 * params.sigma**2 * idx**2 - 0.5 * (fwd.r - fwd.q) * idx
    b = -(params.sigma**2 * idx**2 + fwd.r)
    c = 0.5 * params.sigma**2 * idx**2 + 0.5 * (fwd.r - fwd.q) * idx

    lower_A = a[1:]
    diag_A = b
    upper_A = c[:-1]

    matrix_lower = -dt * lower_A
    matrix_diag = 1.0 - dt * diag_A
    matrix_upper = -dt * upper_A

    ab = np.zeros((3, m - 1), dtype=np.float64)
    ab[0, 1:] = matrix_upper
    ab[1, :] = matrix_diag
    ab[2, :-1] = matrix_lower

    for step in range(n):
        tau_next = (step + 1) * dt
        lower_bc, upper_bc = _boundary(s_max, strike, tau_next, fwd.r, fwd.q, cp, exercise)
        rhs = values[1:m].copy()
        rhs[0] += dt * a[0] * lower_bc
        rhs[-1] += dt * c[-1] * upper_bc

        if exercise == "european":
            interior = solve_banded((1, 1), ab, rhs)
        else:
            obstacle = np.maximum(cp * (interior_S - strike), 0.0)
            interior = _solve_lcp_psor(
                matrix_lower,
                matrix_diag,
                matrix_upper,
                rhs,
                obstacle,
                omega=spec.omega,
                tol=spec.psor_tol,
                max_iter=spec.psor_max_iter,
            )

        values[0] = lower_bc
        values[1:m] = interior
        values[m] = upper_bc

    return float(np.interp(fwd.S0, S, values))


def bsm_pde_fd_price_at_strikes(
    fwd: ForwardSpec,
    params: BsmParams,
    strikes,
    *,
    cp: int = 1,
    exercise: str = "european",
    grid: PDEGrid | None = None,
) -> np.ndarray:
    """Vectorized strike wrapper around :func:`bsm_pde_fd_price`."""
    K = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    return np.asarray(
        [bsm_pde_fd_price(fwd, params, float(k), cp=cp, exercise=exercise, grid=grid) for k in K],
        dtype=np.float64,
    )


__all__ = ["PDEGrid", "bsm_pde_fd_price", "bsm_pde_fd_price_at_strikes"]
