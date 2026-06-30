"""Quanto option pricing under the BSM framework.

A **quanto** (quantity-adjusted) option pays in a fixed domestic currency
amount, where the underlying asset is denominated in a foreign currency.
The FX rate is implicitly fixed at 1 unit of domestic per unit of foreign
via the quanto hedge.

The key adjustment is to the forward price of the foreign asset.  Under
domestic risk-neutral measure, the adjusted forward is:

    F_adj = S * exp((r_dom - q_for - rho * sigma_S * sigma_X) * T)

where
    S       -- spot price of the foreign asset (in foreign currency units)
    r_dom   -- domestic risk-free rate
    q_for   -- dividend yield of the foreign asset
    rho     -- correlation between asset log-returns and FX log-returns
    sigma_S -- volatility of the foreign asset
    sigma_X -- volatility of the FX rate (foreign / domestic)
    T       -- time to maturity (years)

Once the adjusted forward is computed the option price in domestic currency
is a standard BSM formula with forward F_adj and volatility sigma_S:

    quanto_call = e^{-r_dom * T} * BSCall(F_adj, K, T, sigma_S)
    quanto_put  = e^{-r_dom * T} * BSPut(F_adj, K, T, sigma_S)

References:
    Reiner, E. (1992). Quanto mechanics. *Risk*, 5(3), 59-63.
    Hull, J. C. (2018). *Options, Futures and Other Derivatives*, 10th ed.
        Pearson.  Chapter 29.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def bsm_quanto_forward(
    S: float,
    r_dom: float,
    r_for: float,
    q_for: float,
    rho: float,
    sigma_S: float,
    sigma_X: float,
    T: float,
) -> float:
    """Compute the quanto-adjusted forward price of a foreign asset.

    Parameters
    ----------
    S : float
        Spot price of the foreign asset in its own currency.
    r_dom : float
        Domestic risk-free rate (continuously compounded).
    r_for : float
        Foreign risk-free rate (continuously compounded).  Not used in the
        adjusted-forward formula but kept for completeness / put-call parity.
    q_for : float
        Continuous dividend yield of the foreign asset.
    rho : float
        Correlation between asset log-returns and FX log-returns.
    sigma_S : float
        Volatility of the foreign asset.
    sigma_X : float
        Volatility of the FX rate (foreign per domestic, or its reciprocal;
        the adjustment is rho * sigma_S * sigma_X so sign is absorbed).
    T : float
        Time to maturity in years.

    Returns
    -------
    float
        Quanto-adjusted forward price F_adj.
    """
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if sigma_S < 0:
        raise ValueError(f"sigma_S must be non-negative, got {sigma_S}")
    if sigma_X < 0:
        raise ValueError(f"sigma_X must be non-negative, got {sigma_X}")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")

    drift = r_dom - q_for - rho * sigma_S * sigma_X
    return S * math.exp(drift * T)


def bsm_quanto_option(
    S: float,
    K: float,
    r_dom: float,
    r_for: float,
    q_for: float,
    rho: float,
    sigma_S: float,
    sigma_X: float,
    T: float,
    cp: int = 1,
) -> float:
    """Price a quanto option (foreign underlying, domestic payout) via BSM.

    The option pays max(cp*(S_T - K), 0) units of domestic currency, where
    S_T is the terminal spot in foreign currency and K is the foreign-currency
    strike.

    Parameters
    ----------
    S : float
        Spot price of the foreign asset in its own currency.
    K : float
        Strike in foreign currency units.
    r_dom : float
        Domestic risk-free rate.
    r_for : float
        Foreign risk-free rate (passed through to ``bsm_quanto_forward``).
    q_for : float
        Dividend yield of the foreign asset.
    rho : float
        Asset / FX return correlation.
    sigma_S : float
        Foreign-asset return volatility.
    sigma_X : float
        FX rate volatility.
    T : float
        Time to maturity (years).  For T == 0 returns intrinsic value.
    cp : int
        +1 for call, -1 for put.

    Returns
    -------
    float
        Quanto option price in domestic currency.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if cp not in (1, -1):
        raise ValueError(f"cp must be +1 or -1, got {cp}")

    F_adj = bsm_quanto_forward(S, r_dom, r_for, q_for, rho, sigma_S, sigma_X, T)
    disc  = math.exp(-r_dom * T)

    if T == 0.0:
        return float(disc * max(cp * (S - K), 0.0))

    if sigma_S == 0.0:
        return float(disc * max(cp * (F_adj - K), 0.0))

    sqrtT  = math.sqrt(T)
    d1     = (math.log(F_adj / K) + 0.5 * sigma_S**2 * T) / (sigma_S * sqrtT)
    d2     = d1 - sigma_S * sqrtT

    if cp == 1:
        price = disc * (F_adj * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = disc * (K * norm.cdf(-d2) - F_adj * norm.cdf(-d1))

    return float(price)
