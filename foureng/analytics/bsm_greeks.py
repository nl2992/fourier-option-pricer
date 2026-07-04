"""BSM analytical Greeks for European vanilla options.

Standard parameterisation:
  S     : spot price
  K     : strike
  r     : continuously-compounded risk-free rate
  q     : continuous dividend yield (repo / borrow cost)
  T     : time to maturity (years)
  sigma : Black-Scholes lognormal volatility

cp = +1 for a call, -1 for a put.

All Greeks follow the standard finance sign convention:
  - Delta  : ∂V/∂S
  - Gamma  : ∂²V/∂S²   (same for calls and puts)
  - Vega   : ∂V/∂σ     (same for calls and puts; in price/vol-unit)
  - Theta  : ∂V/∂t  = -∂V/∂T  (daily: divide by 365)
  - Rho    : ∂V/∂r     (in price / 1 percentage-point = /100)
  - Vanna  : ∂²V/(∂S ∂σ)
  - Volga  : ∂²V/∂σ²  (also called Vomma)

Reference: Hull, "Options, Futures, and Other Derivatives", 10th ed., Ch. 19.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ── internal helpers ──────────────────────────────────────────────────────────


def _d1d2(S: float, K: float, r: float, q: float, T: float, sigma: float):
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    return d1, d2


# ── first-order Greeks ────────────────────────────────────────────────────────


def bsm_delta(S: float, K: float, r: float, q: float, T: float, sigma: float, cp: int = 1) -> float:
    """∂V/∂S.  Call: e^{-qT}N(d1).  Put: -e^{-qT}N(-d1)."""
    d1, _ = _d1d2(S, K, r, q, T, sigma)
    disc_q = np.exp(-q * T)
    if cp == 1:
        return disc_q * norm.cdf(d1)
    return -disc_q * norm.cdf(-d1)


def bsm_gamma(S: float, K: float, r: float, q: float, T: float, sigma: float, cp: int = 1) -> float:
    """∂²V/∂S².  Identical for calls and puts."""
    d1, _ = _d1d2(S, K, r, q, T, sigma)
    disc_q = np.exp(-q * T)
    return disc_q * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bsm_vega(S: float, K: float, r: float, q: float, T: float, sigma: float, cp: int = 1) -> float:
    """∂V/∂σ.  Identical for calls and puts.  Units: price per 1 vol unit."""
    d1, _ = _d1d2(S, K, r, q, T, sigma)
    disc_q = np.exp(-q * T)
    return S * disc_q * norm.pdf(d1) * np.sqrt(T)


def bsm_theta(S: float, K: float, r: float, q: float, T: float, sigma: float, cp: int = 1) -> float:
    """∂V/∂t = -∂V/∂T.  Returned as price change per calendar year.
    Divide by 365 to get daily theta."""
    d1, d2 = _d1d2(S, K, r, q, T, sigma)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    common = -S * disc_q * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if cp == 1:
        return common - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    return common + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)


def bsm_rho(S: float, K: float, r: float, q: float, T: float, sigma: float, cp: int = 1) -> float:
    """∂V/∂r.  Scaled by 1/100 so units are price per 1 bp (percentage point)."""
    _, d2 = _d1d2(S, K, r, q, T, sigma)
    disc_r = np.exp(-r * T)
    if cp == 1:
        return K * T * disc_r * norm.cdf(d2) / 100
    return -K * T * disc_r * norm.cdf(-d2) / 100


# ── second-order / cross Greeks ───────────────────────────────────────────────


def bsm_vanna(S: float, K: float, r: float, q: float, T: float, sigma: float, cp: int = 1) -> float:
    """∂²V/(∂S ∂σ).  Identical for calls and puts."""
    d1, d2 = _d1d2(S, K, r, q, T, sigma)
    disc_q = np.exp(-q * T)
    return -disc_q * norm.pdf(d1) * d2 / sigma


def bsm_volga(S: float, K: float, r: float, q: float, T: float, sigma: float, cp: int = 1) -> float:
    """∂²V/∂σ²  (Vomma).  Identical for calls and puts."""
    d1, d2 = _d1d2(S, K, r, q, T, sigma)
    disc_q = np.exp(-q * T)
    return S * disc_q * norm.pdf(d1) * np.sqrt(T) * d1 * d2 / sigma


# ── convenience bundle ────────────────────────────────────────────────────────


def bsm_all_greeks(
    S: float, K: float, r: float, q: float, T: float, sigma: float, cp: int = 1
) -> dict[str, float]:
    """Return all Greeks in a single dictionary."""
    return {
        "delta": bsm_delta(S, K, r, q, T, sigma, cp),
        "gamma": bsm_gamma(S, K, r, q, T, sigma, cp),
        "vega": bsm_vega(S, K, r, q, T, sigma, cp),
        "theta": bsm_theta(S, K, r, q, T, sigma, cp),
        "rho": bsm_rho(S, K, r, q, T, sigma, cp),
        "vanna": bsm_vanna(S, K, r, q, T, sigma, cp),
        "volga": bsm_volga(S, K, r, q, T, sigma, cp),
    }
