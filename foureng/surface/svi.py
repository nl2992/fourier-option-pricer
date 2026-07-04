"""SVI (Stochastic Volatility Inspired) smile parameterization.

Gatheral (2004) defines the total implied variance smile w(k) as a function
of log-moneyness k = ln(K/F) at a fixed maturity:

    w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))

where the five raw parameters are:

    a     : vertical translation  (min total-variance at spine)
    b     : slope (≥ 0)
    rho   : correlation (-1 < rho < 1)
    m     : location of the spine (the ATM-like pivot)
    sigma : curvature (> 0)

Implied volatility is recovered as:
    IV(k, T) = sqrt(w(k) / T)

Arbitrage conditions (Lee 2004, Gatheral-Jacquier 2014):
    Static no-butterfly-arbitrage:
        g(k) = (1 - k*w'/(2*w))^2 - (w'/2)^2*(1/4 + 1/w) + w''/2  >=  0 for all k
    Calendar spread no-arbitrage:
        If fitting multiple maturities, w(k, T) must be non-decreasing in T.

References:
    Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility
    parameterization with application to the valuation of volatility derivatives.
    Presentation at Global Derivatives & Risk Management, Madrid.

    Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
    *Quantitative Finance*, 14(1), 59-71.

    Lee, R. (2004). The moment formula for implied volatility at extreme strikes.
    *Mathematical Finance*, 14(3), 469-480.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

# ── parameter container ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters (Gatheral 2004).

    Attributes
    ----------
    a     : vertical translation (total-variance offset)
    b     : wings slope (b >= 0)
    rho   : correlation parameter (-1 < rho < 1)
    m     : smile centre (log-moneyness location)
    sigma : ATM curvature / smoothness (sigma > 0)
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0:
            raise ValueError(f"SVI b must be >= 0, got {self.b}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"SVI rho must be in (-1, 1), got {self.rho}")
        if self.sigma <= 0:
            raise ValueError(f"SVI sigma must be > 0, got {self.sigma}")


# ── core formula ──────────────────────────────────────────────────────────────


def svi_total_variance(k: np.ndarray | float, params: SVIParams) -> np.ndarray:
    """Total implied variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)).

    Parameters
    ----------
    k      : log-moneyness array  (k = ln(K/F))
    params : SVIParams

    Returns
    -------
    w : total variance array, same shape as k; clipped to >= 0
    """
    k = np.asarray(k, dtype=float)
    d = k - params.m
    w = params.a + params.b * (params.rho * d + np.sqrt(d**2 + params.sigma**2))
    return np.maximum(w, 0.0)


def svi_implied_vol(k: np.ndarray | float, T: float, params: SVIParams) -> np.ndarray:
    """BSM implied volatility from SVI total-variance parameterization.

    Parameters
    ----------
    k      : log-moneyness  k = ln(K/F)
    T      : time to expiry (years)
    params : SVIParams

    Returns
    -------
    iv : implied volatility in annual terms
    """
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    w = svi_total_variance(k, params)
    return np.sqrt(w / T)


# ── arbitrage checks ──────────────────────────────────────────────────────────


def svi_butterfly_density(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """Risk-neutral density weight g(k) (must be >= 0 for no butterfly arbitrage).

    Following Gatheral-Jacquier (2014), equation (2.1):

        g(k) = (1 - k*w'(k) / (2*w(k)))^2 - (w'(k)/2)^2 * (1/4 + 1/w(k)) + w''(k)/2

    A positive g(k) everywhere ensures the risk-neutral density is non-negative.

    Parameters
    ----------
    k      : log-moneyness grid
    params : SVIParams

    Returns
    -------
    g : density weight array, same shape as k
    """
    k = np.asarray(k, dtype=float)
    d = k - params.m
    sq = np.sqrt(d**2 + params.sigma**2)
    b, rho = params.b, params.rho

    w = params.a + b * (rho * d + sq)
    w = np.maximum(w, 1e-16)

    # first derivative w'(k)
    w1 = b * (rho + d / sq)

    # second derivative w''(k)
    w2 = b * params.sigma**2 / sq**3

    g = (1 - k * w1 / (2 * w)) ** 2 - (w1 / 2) ** 2 * (1 / 4 + 1 / w) + w2 / 2
    return g


def svi_check_butterfly_arbitrage(
    params: SVIParams,
    k_grid: np.ndarray | None = None,
    tol: float = -1e-6,
) -> bool:
    """Return True if the smile is free of butterfly arbitrage.

    Evaluates g(k) on a dense grid over [-3, 3] by default (or a supplied grid).

    Parameters
    ----------
    params : SVIParams
    k_grid : optional array of log-moneyness nodes to check
    tol    : g(k) is considered non-negative if >= tol (slight slack for numerics)

    Returns
    -------
    bool : True = no arbitrage detected
    """
    if k_grid is None:
        k_grid = np.linspace(-3.0, 3.0, 1001)
    g = svi_butterfly_density(k_grid, params)
    return bool(np.all(g >= tol))


# ── calibration ───────────────────────────────────────────────────────────────


class SVIFitResult(NamedTuple):
    """Result of a single-maturity SVI calibration."""

    params: SVIParams
    rmse: float
    max_err: float
    butterfly_free: bool


def fit_svi_smile(
    k: np.ndarray,
    iv_market: np.ndarray,
    T: float,
    *,
    a0: float | None = None,
    b0: float = 0.1,
    rho0: float = -0.3,
    m0: float = 0.0,
    sigma0: float = 0.3,
    use_global: bool = False,
    maxiter: int = 2000,
) -> SVIFitResult:
    """Fit SVI parameters to a single-maturity implied-volatility smile.

    Minimises the root-mean-squared IV error between the SVI model and market
    implied vols.  The objective is on IV (not total variance) to weight all
    strikes equally in practitioner terms.

    Parameters
    ----------
    k          : log-moneyness array  k = ln(K/F)
    iv_market  : market implied volatility array (same length as k)
    T          : time to expiry in years
    a0         : initial guess for a  (defaults to 0.9 * mean(iv^2 * T))
    b0         : initial guess for b  (default 0.1)
    rho0       : initial guess for rho (default -0.3)
    m0         : initial guess for m  (default 0.0)
    sigma0     : initial guess for sigma (default 0.3)
    use_global : if True use differential evolution first, then refine with L-BFGS-B
    maxiter    : maximum iterations for local optimiser

    Returns
    -------
    SVIFitResult with fitted SVIParams, RMSE, max absolute error, butterfly flag
    """
    k = np.asarray(k, dtype=float)
    iv_market = np.asarray(iv_market, dtype=float)
    if k.shape != iv_market.shape:
        raise ValueError("k and iv_market must have the same shape")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")

    w_market = iv_market**2 * T

    if a0 is None:
        a0 = float(max(0.001, 0.9 * float(np.mean(w_market))))

    # bounds: a ∈ (0, max_w), b ∈ (0, 2), rho ∈ (-0.99, 0.99), m ∈ (-2, 2), sigma ∈ (0.01, 2)
    max_w = float(np.max(w_market)) * 2.0
    bounds = [
        (1e-6, max_w),
        (1e-6, 2.0),
        (-0.99, 0.99),
        (-2.0, 2.0),
        (1e-4, 2.0),
    ]

    def objective(x: np.ndarray) -> float:
        a, b, rho, m, sigma = x
        d = k - m
        w = a + b * (rho * d + np.sqrt(d**2 + sigma**2))
        w = np.maximum(w, 1e-16)
        iv_model = np.sqrt(w / T)
        return float(np.mean((iv_model - iv_market) ** 2))

    x0 = np.array([a0, b0, rho0, m0, sigma0])

    if use_global:
        de_result = differential_evolution(
            objective,
            bounds,
            seed=0,
            maxiter=500,
            tol=1e-8,
            polish=True,
        )
        x0 = de_result.x

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-8},
    )

    a, b, rho, m, sigma = result.x
    fitted = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
    iv_fit = svi_implied_vol(k, T, fitted)
    err = iv_fit - iv_market
    rmse = float(np.sqrt(np.mean(err**2)))
    max_err = float(np.max(np.abs(err)))
    bf_free = svi_check_butterfly_arbitrage(fitted)

    return SVIFitResult(
        params=fitted,
        rmse=rmse,
        max_err=max_err,
        butterfly_free=bf_free,
    )
