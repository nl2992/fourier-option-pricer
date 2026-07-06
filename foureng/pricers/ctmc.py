"""Continuous-time Markov chain (CTMC) approximation for 1-D diffusions.

Approximates a diffusion for the log-price ``x = log(S/S0)``,

    dx = mu(x) dt + sigma(x) dW,    mu(x) = r - q - sigma(x)^2 / 2,

by a finite-state CTMC on a uniform log-grid with generator ``Q`` built
from the standard finite-volume stencil (Lo & Skindilias 2014;
Mijatovic & Pistorius 2013): for spacing ``h``,

    Q[i, i-1] = sigma_i^2 / (2 h^2) - mu_i / (2h)
    Q[i, i+1] = sigma_i^2 / (2 h^2) + mu_i / (2h)
    Q[i, i]   = -(Q[i, i-1] + Q[i, i+1]),

switching to the upwind stencil at any node where the central weights
would go negative, so ``Q`` is always a valid generator. Prices follow
from dense matrix exponentials of the small (m x m) system:

    European:  V = expm(T (Q - r I)) g,        read off at the S0 node;
    American:  V_M = g,  V_{m-1} = max(g, P V_m),  P = expm(dt (Q - r I)),

i.e. Bermudan time-stepping with ``n_steps`` exercise dates (Richardson-
free; increase ``n_steps`` for tighter early-exercise resolution). The
grid is centered so that ``x = 0`` is a node — no interpolation at spot.

``sigma`` may be a constant (BSM) or a callable ``sigma(S) -> vol`` for
local-volatility / CEV-type diffusions, which is the CTMC's raison
d'etre: it prices under state-dependent coefficients where no CF exists.

References
----------
Mijatovic, A. & Pistorius, M. (2013). Continuously monitored barrier
options under Markov processes. *Mathematical Finance*, 23(1), 1-38.

Lo, C.C. & Skindilias, K. (2014). An improved Markov chain approximation
methodology: derivatives pricing and model calibration. *IJTAF*, 17(7).

Kirkby, J.L., Nguyen, D. & Cui, Z. (2017). A unified approach to Bermudan
and barrier options under stochastic volatility models with jumps.
*Journal of Economic Dynamics and Control*, 80, 75-100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import expm

SigmaLike = float | Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CTMCGrid:
    """CTMC state grid: ``n_states`` log-price nodes spanning ``width``
    standard deviations of the terminal distribution on each side.

    ``n_states`` is forced odd so ``x = 0`` (the spot) is exactly a node.
    """

    n_states: int = 301
    width: float = 6.0

    def __post_init__(self) -> None:
        if self.n_states < 5:
            raise ValueError("CTMCGrid: n_states must be >= 5")
        if self.width <= 0.0:
            raise ValueError("CTMCGrid: width must be > 0")


def _build_generator(x: np.ndarray, r: float, q: float, sig: np.ndarray) -> np.ndarray:
    """Tridiagonal generator matrix from the finite-volume stencil."""
    m = x.size
    h = x[1] - x[0]
    mu = r - q - 0.5 * sig**2
    diff = sig**2 / (2.0 * h * h)

    lower = diff - mu / (2.0 * h)
    upper = diff + mu / (2.0 * h)
    # Upwind fallback where central differencing loses positivity.
    neg = (lower < 0.0) | (upper < 0.0)
    if np.any(neg):
        up = mu > 0
        lower_uw = diff + np.where(up, 0.0, -mu / h)
        upper_uw = diff + np.where(up, mu / h, 0.0)
        lower = np.where(neg, lower_uw, lower)
        upper = np.where(neg, upper_uw, upper)

    Q = np.zeros((m, m))
    idx = np.arange(1, m - 1)
    Q[idx, idx - 1] = lower[idx]
    Q[idx, idx + 1] = upper[idx]
    Q[idx, idx] = -(lower[idx] + upper[idx])
    # Absorbing boundary rows (zero rows): mass that reaches the edge stays.
    return Q


def _grid_and_generator(
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: SigmaLike,
    grid: CTMCGrid,
) -> tuple[np.ndarray, np.ndarray, int]:
    m = grid.n_states if grid.n_states % 2 == 1 else grid.n_states + 1
    sig_ref = float(sigma(np.asarray([S0]))[0]) if callable(sigma) else float(sigma)
    if not (np.isfinite(sig_ref) and sig_ref > 0.0):
        raise ValueError(f"ctmc: reference volatility must be > 0; got {sig_ref}")
    half = grid.width * sig_ref * np.sqrt(max(T, 1e-12)) + abs(r - q) * T
    x = np.linspace(-half, half, m)  # x = 0 is the middle node
    S = S0 * np.exp(x)
    sig = np.asarray(sigma(S), dtype=np.float64) if callable(sigma) else np.full(m, float(sigma))
    if np.any(~np.isfinite(sig)) or np.any(sig <= 0.0):
        raise ValueError("ctmc: sigma(S) must be finite and > 0 on the whole grid")
    Q = _build_generator(x, r, q, sig)
    return S, Q, m // 2


def ctmc_european_price(
    S0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    sigma: SigmaLike,
    *,
    cp: int = 1,
    grid: CTMCGrid | None = None,
) -> float:
    """European vanilla price under a 1-D (local-vol) diffusion via CTMC."""
    if cp not in (1, -1):
        raise ValueError(f"ctmc_european_price: cp must be +1 or -1, got {cp}")
    if K <= 0.0 or S0 <= 0.0 or T <= 0.0:
        raise ValueError("ctmc_european_price: S0, K, T must be > 0")
    grid = grid or CTMCGrid()
    S, Q, i0 = _grid_and_generator(S0, r, q, T, sigma, grid)
    g = np.maximum(cp * (S - K), 0.0)
    V = expm(T * (Q - r * np.eye(S.size))) @ g
    return float(V[i0])


def ctmc_american_price(
    S0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    sigma: SigmaLike,
    *,
    cp: int = 1,
    n_steps: int = 100,
    grid: CTMCGrid | None = None,
) -> float:
    """American vanilla price via CTMC Bermudan time-stepping."""
    if cp not in (1, -1):
        raise ValueError(f"ctmc_american_price: cp must be +1 or -1, got {cp}")
    if K <= 0.0 or S0 <= 0.0 or T <= 0.0:
        raise ValueError("ctmc_american_price: S0, K, T must be > 0")
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("ctmc_american_price: n_steps must be >= 1")
    grid = grid or CTMCGrid()
    S, Q, i0 = _grid_and_generator(S0, r, q, T, sigma, grid)
    g = np.maximum(cp * (S - K), 0.0)
    dt = T / n_steps
    P = expm(dt * (Q - r * np.eye(S.size)))
    V = g.copy()
    for _ in range(n_steps):
        V = np.maximum(g, P @ V)
    return float(V[i0])


def ctmc_european_price_at_strikes(
    S0: float,
    strikes: np.ndarray,
    r: float,
    q: float,
    T: float,
    sigma: SigmaLike,
    *,
    cp: int = 1,
    grid: CTMCGrid | None = None,
) -> np.ndarray:
    """Strip version: one matrix exponential shared across all strikes."""
    if cp not in (1, -1):
        raise ValueError(f"ctmc_european_price_at_strikes: cp must be +1 or -1, got {cp}")
    Ks = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    if np.any(Ks <= 0.0):
        raise ValueError("ctmc_european_price_at_strikes: all strikes must be > 0")
    grid = grid or CTMCGrid()
    S, Q, i0 = _grid_and_generator(S0, r, q, T, sigma, grid)
    row = expm(T * (Q - r * np.eye(S.size)))[i0]  # only the spot row is needed
    payoffs = np.maximum(cp * (S[None, :] - Ks[:, None]), 0.0)
    return payoffs @ row


__all__ = [
    "CTMCGrid",
    "ctmc_american_price",
    "ctmc_european_price",
    "ctmc_european_price_at_strikes",
]
