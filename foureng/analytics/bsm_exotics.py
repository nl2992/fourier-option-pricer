"""BSM analytic closed-forms for exotic European-style products.

Covers:
  * Margrabe (1978) exchange option: max(S₁_T − S₂_T, 0)
  * Goldman-Sosin-Gatto (1979) lookback options (floating strike)
  * BSM forward-start option (Rubinstein 1990)
  * BSM gap option (digital with offset payoff)

All formulas assume continuous dividends and risk-neutral pricing under GBM.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ── Margrabe (1978) exchange option ───────────────────────────────────────


def margrabe_exchange(
    S1: float,
    S2: float,
    q1: float,
    q2: float,
    T: float,
    sigma1: float,
    sigma2: float,
    rho: float,
) -> float:
    """Price of an exchange option: max(S₁_T − S₂_T, 0).

    The payer delivers asset 2 and receives asset 1 at maturity T.
    There is no strike (it is "strike = S₂"); the option is priced as a
    BSM call on asset 1 "financed" by asset 2.

    Parameters
    ----------
    S1, S2 : float   Current prices of assets 1 and 2.
    q1, q2 : float   Continuous dividend yields.
    T : float        Maturity in years.
    sigma1, sigma2 : float   Lognormal vols.
    rho : float      Instantaneous correlation between log-returns.

    Returns
    -------
    float
        Price = S1·e^{-q1T}·N(d1) − S2·e^{-q2T}·N(d2)

    Reference
    ---------
    Margrabe, W. (1978). The value of an option to exchange one asset for
    another. *Journal of Finance*, 33(1), 177–186.
    """
    sigma = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
    if sigma <= 0:
        # Deterministic: pay max(S1*exp(-q1*T) - S2*exp(-q2*T), 0)
        return max(S1 * np.exp(-q1 * T) - S2 * np.exp(-q2 * T), 0.0)
    d1 = (np.log(S1 / S2) + (q2 - q1 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S1 * np.exp(-q1 * T) * norm.cdf(d1) - S2 * np.exp(-q2 * T) * norm.cdf(d2)


# ── BSM forward-start option ──────────────────────────────────────────────


def bsm_forward_start(
    S: float,
    alpha: float,
    t_start: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    cp: int = 1,
) -> float:
    """BSM forward-start option price (Rubinstein 1990).

    The strike is set at time ``t_start`` as K = alpha * S_{t_start}
    (alpha=1 → at-the-money forward-start).

    The price reduces to a standard BSM on a "reset" stock:
      price = S · e^{-q·t_start} · BSM_call(1, alpha, r, q, τ, sigma) / 1

    where τ = T − t_start is the remaining tenor.

    Parameters
    ----------
    S : float        Current spot.
    alpha : float    Strike ratio K/S_{t_start}.  alpha=1 → ATM at start.
    t_start : float  Strike-setting time.
    T : float        Maturity (T > t_start).
    r, q : float     Risk-free rate and dividend yield.
    sigma : float    Lognormal vol.
    cp : int         +1 call, −1 put.

    Returns
    -------
    float
        Forward-start option price.
    """
    tau = T - t_start
    if tau <= 0:
        raise ValueError(f"T ({T}) must be > t_start ({t_start})")
    # At t=0, the value is S * e^{-q*t_start} * (BSM value per unit spot at t_start)
    # The BSM value per unit spot at t_start (with effective spot=1, K=alpha):
    b = r - q
    sq = sigma * np.sqrt(tau)
    d1 = (np.log(1.0 / alpha) + (b + 0.5 * sigma**2) * tau) / sq
    d2 = d1 - sq
    # unit_value = BSM(S=1, K=alpha, r, q, tau) with standard formula:
    #   call = exp(-q*tau)*N(d1) - alpha*exp(-r*tau)*N(d2)
    if cp == 1:
        unit_value = np.exp(-q * tau) * norm.cdf(d1) - alpha * np.exp(-r * tau) * norm.cdf(d2)
    else:
        unit_value = alpha * np.exp(-r * tau) * norm.cdf(-d2) - np.exp(-q * tau) * norm.cdf(-d1)
    return S * np.exp(-q * t_start) * unit_value


# ── Goldman-Sosin-Gatto (1979) floating-strike lookback ──────────────────


def bsm_lookback_floating(
    S: float,
    S_min: float,
    S_max: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    cp: int = 1,
) -> float:
    """Floating-strike lookback option (Goldman, Sosin & Gatto 1979).

    Payoffs:
      cp=+1 (call): S_T  − min(S_t, S_min)   [buy at minimum]
      cp=−1 (put):  max(S_t, S_max) − S_T     [sell at maximum]

    Parameters
    ----------
    S : float        Current spot.
    S_min : float    Running minimum so far (= S at inception).
    S_max : float    Running maximum so far (= S at inception).
    r, q : float     Rates.
    T : float        Remaining time to maturity.
    sigma : float    Vol.
    cp : int         +1 call (buy at min), −1 put (sell at max).

    Returns
    -------
    float

    Reference
    ---------
    Goldman, B., Sosin, H. & Gatto, M.A. (1979).  Path dependent options.
    *Journal of Finance*, 34(5), 1111–1127.
    """
    b = r - q
    sq = sigma * np.sqrt(T)
    sig2 = sigma**2

    if cp == 1:
        m = S_min
        a1 = (np.log(S / m) + (b + 0.5 * sig2) * T) / sq
        a2 = a1 - sq
        a3 = (np.log(S / m) + (-b + 0.5 * sig2) * T) / sq
        if abs(b) < 1e-12:
            # Special case b=0 to avoid division by zero
            return S * np.exp(-q * T) * (
                norm.cdf(a1)
                + sig2
                / (2 * b + 1e-300)
                * (
                    -((S / m) ** (2 * b / sig2 + 1e-300)) * norm.cdf(-a3)
                    + np.exp(b * T) * norm.cdf(a1)
                )
            ) - m * np.exp(-r * T) * norm.cdf(a2)
        ratio = (S / m) ** (-2 * b / sig2)
        return (
            S * np.exp(-q * T) * norm.cdf(a1)
            - m * np.exp(-r * T) * norm.cdf(a2)
            + S
            * np.exp(-q * T)
            * (sig2 / (2 * b))
            * (-(ratio) * norm.cdf(-a3) + np.exp(b * T) * norm.cdf(a1))
        )
    else:
        m = S_max
        a1 = (np.log(S / m) + (b + 0.5 * sig2) * T) / sq
        a2 = a1 - sq
        a3 = (np.log(S / m) + (-b + 0.5 * sig2) * T) / sq
        if abs(b) < 1e-12:
            return (
                m * np.exp(-r * T) * norm.cdf(-a2)
                - S * np.exp(-q * T) * norm.cdf(-a1)
                + S * np.exp(-q * T) * sig2 / (2 * 1e-300)
            )  # degenerate
        ratio = (S / m) ** (-2 * b / sig2)
        return (
            m * np.exp(-r * T) * norm.cdf(-a2)
            - S * np.exp(-q * T) * norm.cdf(-a1)
            + S
            * np.exp(-q * T)
            * (sig2 / (2 * b))
            * (ratio * norm.cdf(a3) - np.exp(b * T) * norm.cdf(-a1))
        )


# ── BSM gap option ─────────────────────────────────────────────────────────


def bsm_gap_call(
    S: float,
    K1: float,
    K2: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
) -> float:
    """BSM gap call: pays S_T − K1 if S_T > K2  (K1 is payment, K2 is trigger).

    If K1 = K2 this reduces to the standard BSM call.

    Returns
    -------
    float
    """
    b = r - q
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K2) + (b + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    return S * np.exp(-q * T) * norm.cdf(d1) - K1 * np.exp(-r * T) * norm.cdf(d2)
