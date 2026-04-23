"""Implied-volatility utilities — robust Brent inversion of the BSM formula.

Given a Fourier-priced option strip we often want the BSM-implied vol
per strike for smile plots and calibration diagnostics. The core
operation is a one-dimensional root-find on the Black-Scholes price
function, which is monotone in sigma on the admissible interval
``(intrinsic, F*disc)`` — a textbook Brent problem.

Implementation note
-------------------
An earlier version of this wrapper delegated to
:meth:`pyfeng.BsmFft.impvol_brentq`. That function has a subtle
normalization bug: it divides the input price by ``df`` before passing
it to an internal pricer that *also* applies discounting, so with
``r != q`` it returns vols that are off by a ``log(F/S)``-sized
amount. We instead run :func:`scipy.optimize.brentq` directly against
a closed-form Black-Scholes call. No dependency on PyFENG here and
the solver is ~2x faster than PyFENG's wrapped version on a strip.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ..models.base import ForwardSpec


def _bs_call(F: float, K: float, T: float, sigma: float, disc: float) -> float:
    """Black-Scholes call in forward/discount form. Robust at sigma=0."""
    if sigma <= 0.0:
        return disc * max(F - K, 0.0)
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))


def _bs_put(F: float, K: float, T: float, sigma: float, disc: float) -> float:
    # put = call - disc*(F - K) = disc*(K*N(-d2) - F*N(-d1))
    if sigma <= 0.0:
        return disc * max(K - F, 0.0)
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol_from_prices(
    prices: np.ndarray,
    strikes: np.ndarray,
    fwd: ForwardSpec,
    *,
    cp: int = 1,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 5.0,
) -> np.ndarray:
    """BSM-implied vol for each ``(K, price)`` via :func:`scipy.optimize.brentq`.

    Parameters
    ----------
    prices :
        1-D array of option prices. Calls if ``cp=1``, puts if ``cp=-1``.
        Interpreted as **discounted** option prices — the same output
        scale produced by every pricer in :mod:`foureng.pipeline`.
    strikes :
        1-D array of strikes, same shape as ``prices``.
    fwd :
        Forward spec (``S0``, ``r``, ``q``, ``T``) consistent with the
        numeraire of ``prices``.
    cp :
        ``+1`` call, ``-1`` put.
    sigma_lo, sigma_hi :
        Brent bracket. The BSM price is monotone in ``sigma`` so these
        just need to straddle the true vol; 1e-6 / 5.0 covers every
        realistic regime.

    Returns
    -------
    np.ndarray
        Implied vols. Entries with prices outside the no-arbitrage
        bracket ``(intrinsic, F*disc)`` for calls or
        ``(intrinsic, K*disc)`` for puts return :data:`np.nan`.
    """
    prices = np.asarray(prices, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)
    if prices.shape != strikes.shape:
        raise ValueError(
            f"prices {prices.shape} and strikes {strikes.shape} must match"
        )
    if cp not in (1, -1):
        raise ValueError(f"cp must be +1 (call) or -1 (put); got {cp}")

    F = fwd.F0
    disc = fwd.disc
    T = fwd.T
    price_fn = _bs_call if cp == 1 else _bs_put

    # No-arbitrage bracket for the option price.
    if cp == 1:
        intrinsic = np.maximum(F - strikes, 0.0) * disc
        upper = np.full_like(strikes, F * disc)
    else:
        intrinsic = np.maximum(strikes - F, 0.0) * disc
        upper = strikes * disc

    iv = np.empty_like(prices)
    tol = 1e-12
    for i, (P, K) in enumerate(zip(prices, strikes)):
        if not np.isfinite(P) or P <= intrinsic[i] + tol or P >= upper[i] - tol:
            iv[i] = np.nan
            continue
        try:
            iv[i] = brentq(
                lambda s, K=K, P=P: price_fn(F, K, T, s, disc) - P,
                sigma_lo, sigma_hi, xtol=1e-12, rtol=1e-12,
            )
        except Exception:
            iv[i] = np.nan
    return iv


__all__ = ["implied_vol_from_prices"]
