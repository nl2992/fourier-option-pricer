"""Implied-volatility utilities  -  vectorised inversion of the BSM formula.

Given a Fourier-priced option strip we often want the BSM-implied vol
per strike for smile plots and calibration diagnostics. The inversion is
delegated to :func:`foureng.iv.lets_be_rational.implied_vol_lets_be_rational`
(Jäckel 2015): machine precision, fully vectorised, no bracketing search.

Implementation note
-------------------
An earlier version of this wrapper delegated to
:meth:`pyfeng.BsmFft.impvol_brentq`. That function has a subtle
normalization bug: it divides the input price by ``df`` before passing
it to an internal pricer that *also* applies discounting, so with
``r != q`` it returns vols that are off by a ``log(F/S)``-sized
amount. This module works in forward/discount form throughout and has no
dependency on PyFENG.
"""

from __future__ import annotations

import numpy as np

from ..iv.lets_be_rational import implied_vol_lets_be_rational
from ..models.base import ForwardSpec


def implied_vol_from_prices(
    prices: np.ndarray,
    strikes: np.ndarray,
    fwd: ForwardSpec,
    *,
    cp: int = 1,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 5.0,
) -> np.ndarray:
    """BSM-implied vol for each ``(K, price)``, vectorised.

    Inverts with :func:`foureng.iv.lets_be_rational.implied_vol_lets_be_rational`
    (machine precision, no bracketing root search).

    Parameters
    ----------
    prices :
        1-D array of option prices. Calls if ``cp=1``, puts if ``cp=-1``.
        Interpreted as **discounted** option prices  -  the same output
        scale produced by every pricer in :mod:`foureng.pipeline`.
    strikes :
        1-D array of strikes, same shape as ``prices``.
    fwd :
        Forward spec (``S0``, ``r``, ``q``, ``T``) consistent with the
        numeraire of ``prices``.
    cp :
        ``+1`` call, ``-1`` put.
    sigma_lo, sigma_hi :
        Admissible vol range; solutions outside it are returned as NaN
        (the bracket of the original Brent implementation).

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
        raise ValueError(f"prices {prices.shape} and strikes {strikes.shape} must match")
    if cp not in (1, -1):
        raise ValueError(f"cp must be +1 (call) or -1 (put); got {cp}")

    F = fwd.F0
    disc = fwd.disc
    T = fwd.T

    # No-arbitrage bracket for the option price.
    if cp == 1:
        intrinsic = np.maximum(F - strikes, 0.0) * disc
        upper = np.full_like(strikes, F * disc)
    else:
        intrinsic = np.maximum(strikes - F, 0.0) * disc
        upper = strikes * disc

    tol = 1e-12
    inside = np.isfinite(prices) & (prices > intrinsic + tol) & (prices < upper - tol)
    iv = np.full_like(prices, np.nan)
    if np.any(inside):
        # Vectorised machine-precision inversion (Jäckel 2015), then honour
        # the documented (sigma_lo, sigma_hi) bracket of the Brent solver.
        sol = implied_vol_lets_be_rational(prices[inside], F, strikes[inside], T, disc=disc, cp=cp)
        sol[(sol < sigma_lo) | (sol > sigma_hi)] = np.nan
        iv[inside] = sol
    return iv


__all__ = ["implied_vol_from_prices"]
