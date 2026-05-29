"""BSM analytic closed-form for geometric-average Asian options.

Kemna & Vorst (1990) showed that a geometric-average Asian option can be
priced as a modified BSM European option.  Under the risk-neutral measure,
the continuously-monitored geometric average

    G_T = exp( (1/T) ∫₀ᵀ log(S_t) dt )

is lognormally distributed with:

    E[log(G_T)] = log(S₀) + (b − σ²/2) T/2          [adjusted forward]
    Var[log(G_T)] = σ²_adj T   where  σ²_adj = σ²/3   [adjusted vol]

Here b = r − q (cost of carry), leading to:

    adjusted_b   = (b − σ²/6) / 2     [drift of log(G_T)]
    adjusted_σ   = σ / √3             [vol of log(G_T)]

The price is then a standard BSM formula with these adjusted parameters.

Discrete arithmetic average options are NOT covered here — they require
Monte Carlo or Fourier methods.

References
----------
Kemna, A.G.Z. & Vorst, A.C.F. (1990).  A pricing method for options based
on average asset values.  *Journal of Banking and Finance*, 14(1), 113-129.

Haug (2007) Ch. 4, "Geometric Average Options".
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bsm_geometric_asian(
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    cp: int = 1,
) -> float:
    """BSM closed-form for a continuously-monitored geometric-average Asian option.

    Parameters
    ----------
    S : float   Spot price.
    K : float   Strike.
    r : float   Risk-free rate (continuous, annualised).
    q : float   Dividend yield (continuous).
    T : float   Maturity (years).
    sigma : float  Lognormal volatility.
    cp : int    +1 for call, −1 for put.

    Returns
    -------
    float
        Option price.

    Notes
    -----
    The adjusted cost-of-carry b̃ and adjusted volatility σ̃ are:

        b  = r − q
        b̃  = (b − σ²/6) / 2
        σ̃  = σ / √3

    These turn the geometric average into a lognormal with the same
    Black-Scholes formula structure.
    """
    b = r - q
    sigma_adj = sigma / np.sqrt(3.0)
    b_adj = 0.5 * (b - sigma**2 / 6.0)

    sq = sigma_adj * np.sqrt(T)
    d1 = (np.log(S / K) + (b_adj + 0.5 * sigma_adj**2) * T) / sq
    d2 = d1 - sq

    disc = np.exp(-r * T)
    fwd_adj = S * np.exp(b_adj * T)  # adjusted forward for geometric average

    if cp == 1:
        return disc * (fwd_adj * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return disc * (K * norm.cdf(-d2) - fwd_adj * norm.cdf(-d1))


def bsm_geometric_asian_parity(
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
) -> float:
    """Put-call parity check value for geometric Asian: call − put = disc*(F_adj − K).

    Returns
    -------
    float
        disc * (F_adj − K), where F_adj = S * exp(b̃ T).
    """
    b = r - q
    b_adj = 0.5 * (b - sigma**2 / 6.0)
    F_adj = S * np.exp(b_adj * T)
    return np.exp(-r * T) * (F_adj - K)
