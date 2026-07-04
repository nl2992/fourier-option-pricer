"""SABR stochastic volatility model — Hagan, Kumar, Lesniewski & Woodward (2002).

The SABR model:
    dF = sigma_t F^beta dW1
    d sigma_t = nu sigma_t dW2
    <dW1, dW2> = rho dt

where F is the forward price.

This module provides the Hagan (2002) lognormal implied volatility approximation
and convenience call/put pricers via the BSM formula.

The SABR model does not have a closed-form characteristic function and therefore
is NOT registered in MODEL_REGISTRY.  Instead, functions are exported directly.

Public API
----------
SabrParams              -- frozen dataclass holding (alpha, beta, rho, nu)
sabr_hagan_implied_vol  -- Hagan 2002 lognormal IV approximation
sabr_call_price         -- call price via SABR vol + BSM
sabr_put_price          -- put price via SABR vol + BSM

References
----------
Hagan, P.S., Kumar, D., Lesniewski, A.S., Woodward, D.E. (2002).
    "Managing smile risk." Wilmott Magazine, 84-108.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foureng.analytics.bsm_barrier import bsm_call, bsm_put

from .base import ModelSpec


@dataclass(frozen=True)
class SabrParams(ModelSpec):
    """SABR parameters (Hagan et al. 2002).

    dF = sigma_t F^beta dW1
    d sigma_t = nu sigma_t dW2
    <dW1,dW2> = rho dt

    Attributes
    ----------
    alpha : float
        Initial vol level (sigma_0).  Must be > 0.
    beta : float
        CEV exponent in [0, 1].  beta=1 -> lognormal; beta=0 -> normal.
    rho : float
        Correlation in (-1, 1).
    nu : float
        Vol-of-vol >= 0.
    """

    alpha: float
    beta: float
    rho: float
    nu: float

    def __init__(self, alpha: float, beta: float, rho: float, nu: float) -> None:
        if not (np.isfinite(alpha) and alpha > 0.0):
            raise ValueError(f"SabrParams: alpha must be finite and > 0; got {alpha}")
        if not (np.isfinite(beta) and 0.0 <= beta <= 1.0):
            raise ValueError(f"SabrParams: beta must be in [0, 1]; got {beta}")
        if not (np.isfinite(rho) and -1.0 < rho < 1.0):
            raise ValueError(f"SabrParams: rho must be in (-1, 1); got {rho}")
        if not (np.isfinite(nu) and nu >= 0.0):
            raise ValueError(f"SabrParams: nu must be finite and >= 0; got {nu}")
        object.__setattr__(self, "name", "sabr")
        object.__setattr__(self, "alpha", float(alpha))
        object.__setattr__(self, "beta", float(beta))
        object.__setattr__(self, "rho", float(rho))
        object.__setattr__(self, "nu", float(nu))


# ── Hagan (2002) lognormal implied vol ────────────────────────────────────


def sabr_hagan_implied_vol(
    F: float,
    K: float | np.ndarray,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float | np.ndarray:
    """Hagan (2002) lognormal (Black-Scholes) implied volatility for SABR.

    Parameters
    ----------
    F : float
        Forward price.  Must be > 0.
    K : float
        Strike (scalar or array).  Must be > 0.
    T : float
        Time to maturity (years).  Must be > 0.
    alpha : float
        SABR alpha (initial vol).  Must be > 0.
    beta : float
        CEV exponent in [0, 1].
    rho : float
        Correlation in (-1, 1).
    nu : float
        Vol-of-vol >= 0.

    Returns
    -------
    float or np.ndarray
        Lognormal implied volatility sigma_BS >= 0.

    Notes
    -----
    The formula uses the ATM limiting expression when |ln(F/K)| < 1e-6
    to avoid 0/0 issues near the money.  A guard on |z| < 1e-8 prevents
    floating-point cancellation in the z/chi(z) ratio when nu is tiny.

    References
    ----------
    Hagan et al. (2002), equations (2.17a)-(2.17b).
    """
    if F <= 0:
        raise ValueError(f"F must be > 0; got F={F}")
    if T <= 0:
        raise ValueError(f"T must be > 0; got T={T}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0; got alpha={alpha}")

    K_arr = np.asarray(K, dtype=np.float64)
    scalar = K_arr.ndim == 0
    K_arr = np.atleast_1d(K_arr)

    if np.any(K_arr <= 0):
        raise ValueError(f"K must be > 0; got K={K}")

    one_minus_beta = 1.0 - beta
    log_FK = np.log(F / K_arr)
    FK_mid = np.sqrt(F * K_arr)
    FK_beta = FK_mid**one_minus_beta  # (FK)^{(1-beta)/2}

    # Time correction (same for ATM and OTM)
    correction_2 = (
        1.0
        + (
            (one_minus_beta**2 / 24.0) * alpha**2 / (FK_mid ** (2.0 * one_minus_beta))
            + 0.25 * rho * beta * nu * alpha / FK_beta
            + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
        )
        * T
    )

    # Base vol factor
    log_corr = 1.0 + (
        (one_minus_beta**2 / 24.0) * log_FK**2 + (one_minus_beta**4 / 1920.0) * log_FK**4
    )
    sigma_0 = alpha / (FK_beta * log_corr)

    # z / chi(z) skew-curvature correction
    z = (nu / alpha) * FK_beta * log_FK
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z**2)
    chi_z = np.log((sqrt_term + z - rho) / (1.0 - rho))

    # Guard against cancellation when z is tiny (nu->0 or near ATM).
    # Compute z/chi_z only where |z| >= 1e-8 to avoid RuntimeWarning.
    safe_chi_z = np.where(np.abs(z) < 1e-8, 1.0, chi_z)
    correction_1 = np.where(
        np.abs(z) < 1e-8,
        1.0 - 0.5 * rho * z,  # first-order Taylor of z/chi(z) at z=0
        z / safe_chi_z,
    )

    sigma_BS = sigma_0 * correction_1 * correction_2

    # Override with ATM limit where |log(F/K)| < 1e-6
    atm = np.abs(log_FK) < 1e-6
    if np.any(atm):
        f_beta = F**one_minus_beta
        atm_corr = (
            1.0
            + (
                (one_minus_beta**2 / 24.0) * alpha**2 / F ** (2.0 * one_minus_beta)
                + 0.25 * rho * beta * nu * alpha / f_beta
                + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
            )
            * T
        )
        sigma_BS = np.where(atm, alpha / f_beta * atm_corr, sigma_BS)

    sigma_BS = np.maximum(sigma_BS, 0.0)
    return float(sigma_BS[0]) if scalar else sigma_BS


# ── convenience pricers ────────────────────────────────────────────────────


def sabr_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """European call price under SABR via Hagan implied vol + BSM.

    The forward is F = S * exp((r-q)*T).  The SABR lognormal IV is
    computed at (F, K) and fed into the standard BSM call formula.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike.
    T : float
        Time to maturity (years).
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    alpha, beta, rho, nu : float
        SABR parameters.

    Returns
    -------
    float
        European call price.
    """
    F = S * np.exp((r - q) * T)
    sigma_bs = float(sabr_hagan_implied_vol(F, K, T, alpha, beta, rho, nu))
    return bsm_call(S, K, r, q, T, sigma_bs)


def sabr_put_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """European put price under SABR via Hagan implied vol + BSM.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike.
    T : float
        Time to maturity (years).
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    alpha, beta, rho, nu : float
        SABR parameters.

    Returns
    -------
    float
        European put price.
    """
    F = S * np.exp((r - q) * T)
    sigma_bs = float(sabr_hagan_implied_vol(F, K, T, alpha, beta, rho, nu))
    return bsm_put(S, K, r, q, T, sigma_bs)


__all__ = [
    "SabrParams",
    "sabr_hagan_implied_vol",
    "sabr_call_price",
    "sabr_put_price",
]
