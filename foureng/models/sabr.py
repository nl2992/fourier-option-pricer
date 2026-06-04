"""SABR approximation parameters and Hagan implied volatility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ModelSpec


@dataclass(frozen=True)
class SabrParams(ModelSpec):
    """SABR parameters for Hagan's lognormal implied-vol approximation."""

    alpha: float
    beta: float
    rho: float
    nu: float

    def __init__(self, alpha: float, beta: float, rho: float, nu: float):
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


def sabr_hagan_implied_vol(F: float, K, T: float, p: SabrParams):
    """Hagan et al. lognormal SABR implied volatility approximation."""
    K_arr = np.asarray(K, dtype=np.float64)
    if F <= 0.0 or T <= 0.0:
        raise ValueError("SABR Hagan requires F > 0 and T > 0")
    if np.any(K_arr <= 0.0):
        raise ValueError("SABR Hagan requires positive strikes")

    alpha, beta, rho, nu = p.alpha, p.beta, p.rho, p.nu
    one_minus_beta = 1.0 - beta
    log_fk = np.log(F / K_arr)
    fk_beta = (F * K_arr) ** (0.5 * one_minus_beta)

    base = alpha / fk_beta
    z = (nu / alpha) * fk_beta * log_fk if nu > 0.0 else np.zeros_like(K_arr)
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))
    z_over_x = np.empty_like(z)
    small = np.abs(z) < 1e-8
    z_over_x[small] = 1.0 - 0.5 * rho * z[small]
    z_over_x[~small] = z[~small] / x_z[~small]

    log_corr = (
        1.0 + (one_minus_beta**2 / 24.0) * log_fk**2 + (one_minus_beta**4 / 1920.0) * log_fk**4
    )
    time_corr = (
        1.0
        + (
            (one_minus_beta**2 / 24.0) * alpha**2 / ((F * K_arr) ** one_minus_beta)
            + 0.25 * rho * beta * nu * alpha / fk_beta
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        )
        * T
    )

    vol = base * z_over_x * time_corr / log_corr
    atm = np.isclose(log_fk, 0.0, atol=1e-10)
    if np.any(atm):
        f_beta = F**one_minus_beta
        atm_corr = (
            1.0
            + (
                (one_minus_beta**2 / 24.0) * alpha**2 / (F ** (2.0 * one_minus_beta))
                + 0.25 * rho * beta * nu * alpha / f_beta
                + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
            )
            * T
        )
        vol = np.asarray(vol)
        vol[atm] = alpha / f_beta * atm_corr
    return np.asarray(vol, dtype=np.float64)


__all__ = ["SabrParams", "sabr_hagan_implied_vol"]
