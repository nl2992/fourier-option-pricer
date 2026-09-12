"""SINC Fourier pricer (Baschetti, Bormetti, Romagnoli & Rossi 2022).

The SINC method expands the sign function on a window ``(-X_c, X_c)`` by its
Shannon/Fourier series, which contains only odd frequencies:

    sgn(y) = (4/pi) sum_{n odd} sin(n pi y / X_c) / n,

so that, for ``X = log(S_T / F)`` and ``k = log(K / F)``,

    Q(X > k) = 1/2 + (2/pi) sum_{n odd} Im[e^{-i n pi k / X_c} phi(n pi / X_c)] / n,

exact up to the mass of ``X`` farther than ``X_c`` from ``k`` and the
truncation of the series. Calls are ``D (F Q~(X > k) - K Q(X > k))`` with the
share-measure CF ``phi(u - i)``.

This is the same sum as the Feng & Linetsky (2008) discrete Hilbert transform
on the half-integer grid ``u_m = (m + 1/2) h`` with ``h = 2 pi / X_c`` (the
``hilbert`` engine); the two derivations meet in one formula. What the SINC
paper adds, and this module provides, is (i) sizing ``X_c`` from the density
support instead of a fixed fine step, which needs far fewer terms, and (ii)
pricing a whole smile at once with one FFT on a matched log-strike grid.

Reference
---------
Baschetti, F., Bormetti, G., Romagnoli, S. & Rossi, P. (2022), The SINC way:
a fast and accurate approach to Fourier pricing, *Quantitative Finance* 22(3),
427-446 (arXiv:2009.00557).
"""

from __future__ import annotations

import numpy as np

from ..models.base import CharFunc, ForwardSpec
from ..utils.grids import HilbertGrid, SincGrid
from .hilbert import hilbert_price_at_strikes

__all__ = ["sinc_price_at_strikes", "sinc_smile"]

_N_MIN, _N_MAX = 32, 1 << 16


def _n_terms(phi: CharFunc, h: float, n: int | None) -> int:
    if n is not None:
        return int(n)
    N = _N_MIN
    while N < _N_MAX:
        u = np.array([(N - 0.5) * h])
        tail = max(abs(complex(phi(u)[0])), abs(complex(phi(u - 1j)[0])))
        if tail <= 1e-15:
            break
        N *= 2
    return N


def _support(cumulants, L: float) -> tuple[float, float]:
    c1, c2, c4 = cumulants
    return float(c1), float(L * np.sqrt(abs(c2) + np.sqrt(abs(c4))))


def sinc_price_at_strikes(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes,
    cumulants,
    *,
    cp: int = 1,
    grid: SincGrid | None = None,
) -> np.ndarray:
    """European prices by the SINC formula at arbitrary strikes.

    ``cumulants = (c1, c2, c4)`` of ``X_T`` size the window: by default
    ``X_c = max_k |k - c1| + 2 L sqrt(c2 + sqrt|c4|)``, so every strike's
    sign-function window covers the density with a full support width to spare
    (exponential jump tails still carry mass one width out).
    """
    grid = grid or SincGrid()
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    if np.any(K <= 0.0):
        raise ValueError("sinc: strikes must be > 0")
    center, width = _support(cumulants, grid.L)
    k = np.log(K / fwd.F0)
    X_c = grid.X_c or float(np.max(np.abs(k - center)) + 2.0 * width)
    h = 2.0 * np.pi / X_c
    hg = HilbertGrid(h=h, N=_n_terms(phi, h, grid.N))
    return hilbert_price_at_strikes(phi, fwd, K, cp=cp, grid=hg)


def sinc_smile(
    phi: CharFunc,
    fwd: ForwardSpec,
    cumulants,
    *,
    cp: int = 1,
    grid: SincGrid | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A whole smile from one FFT: prices on a uniform log-strike grid.

    With ``u_m = (m + 1/2) h`` and log-strike spacing ``Delta = pi / (N h)``,
    ``e^{-i u_m k_j}`` factorises into a length-``2N`` DFT over ``m``. The window
    ``X_c = 2.5 L sqrt(c2 + sqrt|c4|)`` keeps ``|X - k| < X_c`` for every grid
    strike; the grid spans ``c1 +- X_c / 2`` with ``2N`` points.

    Returns
    -------
    strikes, prices : np.ndarray
    """
    grid = grid or SincGrid()
    center, width = _support(cumulants, grid.L)
    X_c = grid.X_c or 2.5 * width
    h = 2.0 * np.pi / X_c
    N = _n_terms(phi, h, grid.N)
    m = np.arange(N)
    u = (m + 0.5) * h
    delta = np.pi / (N * h)
    k0 = center - 0.5 * X_c
    j = np.arange(2 * N)
    k = k0 + j * delta

    def tail(shift: complex) -> np.ndarray:
        vals = np.asarray(phi(u + shift), dtype=complex) * np.exp(-1j * u * k0) / u
        series = np.fft.fft(vals, 2 * N) * np.exp(-0.5j * h * j * delta)
        return 0.5 + (h / np.pi) * np.imag(series)

    pi2, pi1 = tail(0.0), tail(-1j)
    K = fwd.F0 * np.exp(k)
    calls = fwd.disc * (fwd.F0 * np.clip(pi1, 0, 1) - K * np.clip(pi2, 0, 1))
    return K, calls if cp == 1 else calls - fwd.disc * (fwd.F0 - K)
