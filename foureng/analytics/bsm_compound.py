"""Geske (1979) compound option pricing formula.

A compound option is an option on an option with two strikes and maturities:

  T1 < T2 : compound option expires at T1 with strike K1
  T2      : underlying option expires at T2 with strike K2

The four types — call-on-call, call-on-put, put-on-call, put-on-put — are
implemented via the unified Haug (2007) formulation.

The critical stock price S* satisfies:
  underlying_option(S*, K2, T2-T1, ...) = K1

Reference:
  Geske, R. (1979). The Valuation of Compound Options.
  *Journal of Financial Economics* 7(1), 63–81.

  Haug, E.G. (2007). *The Complete Guide to Option Pricing Formulas*,
  2nd ed., Ch. 2.8.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal as _mvn
from scipy.stats import norm

# ── bivariate normal CDF ──────────────────────────────────────────────────────


def _N2(x: float, y: float, rho: float) -> float:
    """P(X ≤ x, Y ≤ y) for standard bivariate normal with correlation rho."""
    if abs(rho) >= 1.0:
        rho = np.clip(rho, -0.9999999, 0.9999999)
    cov = [[1.0, rho], [rho, 1.0]]
    return float(_mvn.cdf([x, y], mean=[0.0, 0.0], cov=cov))


# ── BSM price helper ──────────────────────────────────────────────────────────


def _bsm(S: float, K: float, r: float, q: float, T: float, sigma: float, cp: int) -> float:
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    if cp == 1:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ── critical stock price S* ───────────────────────────────────────────────────


def _critical_stock(
    K1: float, K2: float, r: float, q: float, T1: float, T2: float, sigma: float, cp_inner: int
) -> float:
    """Find S* such that bsm(S*, K2, r, q, T2-T1, sigma, cp_inner) = K1."""
    tau = T2 - T1

    def f(s: float) -> float:
        return _bsm(s, K2, r, q, tau, sigma, cp_inner) - K1

    # For a call, S* > 0 and the function is increasing in S
    # For a put, S* > 0 and the function is decreasing in S
    lo, hi = 1e-8, 1e10
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi > 0:
        # No sign change → trivial extreme: always exercise or never exercise
        if cp_inner == 1 and f_lo > 0:
            return 0.0  # call always worth ≥ K1; holder always exercises
        if cp_inner == -1 and f_lo < 0:
            return float("inf")  # put never worth K1; never exercise
        return 0.0 if f_lo > 0 else float("inf")
    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)


# ── four Haug (2007) formulas ─────────────────────────────────────────────────


def geske_compound_price(
    S: float,
    K1: float,
    K2: float,
    r: float,
    q: float,
    T1: float,
    T2: float,
    sigma: float,
    cp_outer: int = 1,
    cp_inner: int = 1,
) -> float:
    """Price a compound option using the Geske (1979) / Haug (2007) formula.

    Parameters
    ----------
    S        : current spot price
    K1       : strike of the compound option (paid at T1)
    K2       : strike of the underlying option (paid at T2)
    r        : continuously-compounded risk-free rate
    q        : continuous dividend yield
    T1       : maturity of the compound option  (T1 < T2)
    T2       : maturity of the underlying option
    sigma    : lognormal volatility (constant)
    cp_outer : +1 = call on option,  -1 = put on option
    cp_inner : +1 = underlying is a call, -1 = underlying is a put

    Returns
    -------
    float : compound option price ≥ 0
    """
    if T1 >= T2:
        raise ValueError(f"T1={T1} must be strictly less than T2={T2}")
    if T1 <= 0:
        raise ValueError(f"T1 must be > 0, got {T1}")

    S_star = _critical_stock(K1, K2, r, q, T1, T2, sigma, cp_inner)

    # Trivial cases: S* at boundary → option is deep ITM or OTM
    if S_star == 0.0:
        # always exercise: compound = inner option price - K1*disc1
        inner = _bsm(S, K2, r, q, T2, sigma, cp_inner)
        if cp_outer == 1:
            return max(inner - K1 * np.exp(-r * T1), 0.0)
        return max(K1 * np.exp(-r * T1) - inner, 0.0)
    if S_star == float("inf"):
        return 0.0

    sq1 = sigma * np.sqrt(T1)
    sq2 = sigma * np.sqrt(T2)
    rho = np.sqrt(T1 / T2)

    # a1, a2: based on S vs S*  (determines exercise of compound at T1)
    a1 = (np.log(S / S_star) + (r - q + 0.5 * sigma**2) * T1) / sq1
    a2 = a1 - sq1

    # b1, b2: based on S vs K2  (determines value of inner option)
    b1 = (np.log(S / K2) + (r - q + 0.5 * sigma**2) * T2) / sq2
    b2 = b1 - sq2

    disc_q2 = np.exp(-q * T2)
    disc_r2 = np.exp(-r * T2)
    disc_r1 = np.exp(-r * T1)

    if cp_outer == 1 and cp_inner == 1:
        # Call-on-call (Haug 2007, eq. 2.81)
        return (
            S * disc_q2 * _N2(a1, b1, rho)
            - K2 * disc_r2 * _N2(a2, b2, rho)
            - K1 * disc_r1 * norm.cdf(a2)
        )

    if cp_outer == -1 and cp_inner == 1:
        # Put-on-call (Haug 2007, eq. 2.82)
        return (
            -S * disc_q2 * _N2(-a1, b1, -rho)
            + K2 * disc_r2 * _N2(-a2, b2, -rho)
            + K1 * disc_r1 * norm.cdf(-a2)
        )

    if cp_outer == 1 and cp_inner == -1:
        # Call-on-put (Haug 2007, eq. 2.83)
        return (
            -S * disc_q2 * _N2(-a1, -b1, rho)
            + K2 * disc_r2 * _N2(-a2, -b2, rho)
            - K1 * disc_r1 * norm.cdf(-a2)
        )

    # cp_outer == -1 and cp_inner == -1
    # Put-on-put (Haug 2007, eq. 2.84)
    return (
        S * disc_q2 * _N2(a1, -b1, -rho)
        - K2 * disc_r2 * _N2(a2, -b2, -rho)
        + K1 * disc_r1 * norm.cdf(a2)
    )
