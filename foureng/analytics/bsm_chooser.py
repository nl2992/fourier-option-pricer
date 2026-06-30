"""Simple chooser option pricing (Rubinstein 1991).

A simple chooser option lets the holder decide at time T_choice whether
they want a call or a put, both with strike K and expiry T_exp > T_choice.

Rubinstein (1991) shows the price equals:

  V = BSM_Call(S, K*, r, q, T_choice, sigma)
    + BSM_Put(S, K,  r, q, T_exp,    sigma)

where  K* = K * exp(-(r-q) * (T_exp - T_choice))

This decomposition holds because:
  max(C(S_{T_c}), P(S_{T_c})) = P(S, K, T_exp)
                               + max(C(S_{T_c}) - P(S_{T_c}), 0)

and by put-call parity the "exchange" part reduces to a call on the
forward K*.

Reference:
  Rubinstein, M. (1991). Double Trouble. Risk Magazine, 4, 73.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bsm_chooser_price(
    S: float,
    K: float,
    r: float,
    q: float,
    T_choice: float,
    T_exp: float,
    sigma: float,
) -> float:
    """Price a simple chooser option (Rubinstein 1991).

    Parameters
    ----------
    S        : current spot price
    K        : common strike for call and put
    r        : continuously-compounded risk-free rate
    q        : continuous dividend yield
    T_choice : time until the choice date (years)
    T_exp    : time until option expiry (years);  T_choice < T_exp
    sigma    : lognormal volatility

    Returns
    -------
    float : chooser option price
    """
    if T_choice >= T_exp:
        raise ValueError(f"T_choice={T_choice} must be < T_exp={T_exp}")
    if T_choice <= 0:
        raise ValueError(f"T_choice must be > 0, got {T_choice}")

    # Discounted strike at T_choice
    K_star = K * np.exp(-(r - q) * (T_exp - T_choice))

    sq_tc = sigma * np.sqrt(T_choice)
    sq_te = sigma * np.sqrt(T_exp)

    # Call component: call on forward K* expiring at T_choice
    d1c = (np.log(S / K_star) + (r - q + 0.5 * sigma**2) * T_choice) / sq_tc
    d2c = d1c - sq_tc
    call_part = (S * np.exp(-q * T_choice) * norm.cdf(d1c)
                 - K_star * np.exp(-r * T_choice) * norm.cdf(d2c))

    # Put component: European put expiring at T_exp
    d1p = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T_exp) / sq_te
    d2p = d1p - sq_te
    put_part = (K * np.exp(-r * T_exp) * norm.cdf(-d2p)
                - S * np.exp(-q * T_exp) * norm.cdf(-d1p))

    return call_part + put_part
