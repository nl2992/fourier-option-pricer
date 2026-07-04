"""Hilbert-transform European pricer (Feng & Linetsky 2008).

Prices European calls and puts from the characteristic function via the
Gil-Pelaez probability representation, discretized on the half-integer sinc
grid that defines the discrete Hilbert transform.

Convention
----------
The model exposes the CF of the forward-normalized log return

    X_T = log(S_T / F_0),    phi(u) = E[e^{i u X_T}],    phi(-i) = 1.

With ``k = log(K / F_0)`` the call decomposes into two tail probabilities:

    C = D * (F_0 * Pi_1 - K * Pi_2)

    Pi_2 = Q(X_T > k)      = 1/2 + (1/pi) * I(phi;     k)
    Pi_1 = Q~(X_T > k)     = 1/2 + (1/pi) * I(phi(.-i); k)

where Q~ is the share (stock-numeraire) measure, phi(u - i) is the CF of X_T
under Q~ (Radon-Nikodym weight e^{X_T}, using phi(-i) = 1), and

    I(f; k) = integral_0^inf Re[ e^{-i u k} f(u) / (i u) ] du.

Discretization
--------------
``I`` is a principal-value / Hilbert-transform integral. Evaluating the
integrand on the half-integer grid ``u_m = (m + 1/2) h`` and summing with
weight ``h`` skips the ``u = 0`` singularity and reproduces the sinc-basis
discrete Hilbert transform of Feng & Linetsky (2008). For characteristic
functions analytic in a horizontal strip (all diffusion and finite-activity
jump models, and most infinite-activity Levy models used here), the
discretization error decays like ``exp(-c/h)``, so modest grids reach near
machine precision.

References
----------
Feng, L. & Linetsky, V. (2008). Pricing discretely monitored barrier options
and defaultable bonds in Levy process models: a fast Hilbert transform
approach. *Mathematical Finance*, 18(3), 337-384.

Stenger, F. (1993). *Numerical Methods Based on Sinc and Analytic Functions*.
Springer.

Gil-Pelaez, J. (1951). Note on the inversion theorem. *Biometrika*, 38,
481-482.
"""

from __future__ import annotations

import numpy as np

from ..models.base import CharFunc, ForwardSpec
from ..utils.grids import HilbertGrid


def _tail_probabilities(
    phi: CharFunc,
    log_moneyness: np.ndarray,
    grid: HilbertGrid,
    *,
    shift: complex = 0.0j,
) -> np.ndarray:
    """Q(X_T > k) for each k via the discrete Hilbert transform.

    ``shift`` moves the CF argument (``phi(u + shift)``); ``shift = -1j``
    produces the share-measure probability Pi_1.
    """
    u = grid.u()
    phi_vals = np.asarray(phi(u + shift), dtype=np.complex128)
    # integrand rows: strikes, cols: frequencies
    osc = np.exp(-1j * np.outer(log_moneyness, u))
    integrand = np.real(osc * (phi_vals / (1j * u))[None, :])
    integral = grid.h * integrand.sum(axis=1)
    return 0.5 + integral / np.pi


def hilbert_itm_probabilities(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes: np.ndarray,
    grid: HilbertGrid | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Share- and cash-measure ITM probabilities (Pi_1, Pi_2) per strike.

    Under BSM these are ``N(d1)`` and ``N(d2)``; they are the building blocks
    of the Hilbert-transform vanilla price and are exposed for digital-style
    diagnostics and tests.
    """
    grid = grid if grid is not None else HilbertGrid()
    strikes = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    if np.any(strikes <= 0.0):
        raise ValueError("hilbert_itm_probabilities: all strikes must be > 0")
    k = np.log(strikes / fwd.F0)
    pi2 = _tail_probabilities(phi, k, grid)
    pi1 = _tail_probabilities(phi, k, grid, shift=-1j)
    return np.clip(pi1, 0.0, 1.0), np.clip(pi2, 0.0, 1.0)


def hilbert_price_at_strikes(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes: np.ndarray,
    *,
    cp: int = 1,
    grid: HilbertGrid | None = None,
) -> np.ndarray:
    """European call/put prices via the Feng-Linetsky Hilbert transform.

    Parameters
    ----------
    phi :
        Forward-normalized log-return CF; must accept complex arguments
        (it is evaluated at ``u - i`` for the share-measure leg).
    fwd :
        Market inputs (spot, rates, maturity).
    strikes :
        1-D array of strictly positive strikes.
    cp :
        ``+1`` calls, ``-1`` puts (via put-call parity).
    grid :
        Optional :class:`~foureng.utils.grids.HilbertGrid`; the default
        (``h = 0.05``, ``N = 8192``) resolves every registry model at
        standard maturities.
    """
    if cp not in (1, -1):
        raise ValueError(f"hilbert_price_at_strikes: cp must be +1 or -1, got {cp}")
    strikes = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    pi1, pi2 = hilbert_itm_probabilities(phi, fwd, strikes, grid)
    calls = fwd.disc * (fwd.F0 * pi1 - strikes * pi2)
    if cp == 1:
        return calls
    return calls - fwd.disc * (fwd.F0 - strikes)


__all__ = ["hilbert_itm_probabilities", "hilbert_price_at_strikes"]
