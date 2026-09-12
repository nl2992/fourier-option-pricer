"""SWIFT: Shannon-wavelet inverse Fourier pricing (Ortiz-Gracia & Oosterlee 2016).

The density of ``X = log(S_T / F)`` is projected on the Shannon scaling
functions of scale ``m``, ``phi_{m,k}(x) = 2^{m/2} sinc(2^m x - k)``:

    f(x) ~ sum_{k=k1}^{k2} c_{m,k} phi_{m,k}(x),
    c_{m,k} = 2^{m/2} / 2 int_{-1}^{1} phi(pi 2^m s) e^{-i pi s k} ds.

SWIFT replaces ``sinc`` by Vieta's cosine product, ``sinc(t) ~ (1/J)
sum_{j=1}^{J} cos(C_j t)`` with ``C_j = (2j - 1) pi / (2J)``, which turns the
frequency integral into a midpoint rule on the odd frequencies
``u_j = 2^m C_j``:

    c_{m,k} ~ (2^{m/2} / J) sum_j Re[phi(2^m C_j) e^{-i C_j k}],

and, with the same cosines for the scaling functions, the payoff projections

    V_{m,k} = int payoff(x) phi_{m,k}(x) dx
            ~ (2^{m/2} / J) sum_j int payoff(x) cos(C_j (2^m x - k)) dx

are closed-form. The price is ``D sum_k c_{m,k} V_{m,k}``. Both approximations
share the discrete orthogonality of the ``J`` cosines, so the result is exact
for the density periodised on ``2J / 2^m`` and band-limited to ``|u| < pi 2^m``:
the scale ``m`` controls the frequency truncation and ``[k1, k2]`` the spatial
one, the wavelet counterpart of COS's ``N`` and ``[a, b]``.

Parameters are chosen as in the paper: ``[k1, k2] = [floor(2^m a), ceil(2^m b)]``
for a cumulant interval ``[a, b]``, ``J`` the smallest power of two with
``2J >= k2 - k1`` (so periodisation does not fold the support), and ``m`` the
smallest scale at which the CF has decayed below 1e-15 at ``pi 2^m``.

Reference
---------
Ortiz-Gracia, L. & Oosterlee, C.W. (2016), A highly efficient Shannon wavelet
inverse Fourier technique for pricing European options, *SIAM J. Sci.
Comput.* 38(1), B118-B143.
"""

from __future__ import annotations

import numpy as np

from ..models.base import CharFunc, ForwardSpec

__all__ = ["swift_price_at_strikes"]

_M_MIN = 1
_MAX_INDEX = 4096  # cap on k2 - k1 (and hence J): keeps the (k, j) matrices small


def _scale(phi: CharFunc, m: int | None, width: float) -> int:
    """Smallest scale at which the CF has decayed at ``pi 2^m``, capped so that the
    wavelet index range ``2^m width`` stays below ``_MAX_INDEX`` (slowly decaying
    CFs, e.g. short-dated VG, then converge only algebraically)."""
    if m is not None:
        return int(m)
    m_max = max(_M_MIN, int(np.floor(np.log2(_MAX_INDEX / width))))
    for mm in range(_M_MIN, m_max + 1):
        u = np.array([np.pi * 2.0**mm])
        if max(abs(complex(phi(u)[0])), abs(complex(phi(u - 1j)[0]))) <= 1e-15:
            return mm
    return m_max


def swift_price_at_strikes(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes,
    cumulants,
    *,
    cp: int = 1,
    m: int | None = None,
    L: float = 12.0,
) -> np.ndarray:
    """European prices by the SWIFT method.

    Parameters
    ----------
    phi :
        CF of ``X_T = log(S_T / F_0)``.
    fwd :
        Forward spec.
    strikes :
        Strikes.
    cumulants :
        ``(c1, c2, c4)`` of ``X_T``; they place the wavelet index range.
    cp :
        ``+1`` calls, ``-1`` puts (parity).
    m :
        Wavelet scale; ``None`` picks it from the CF decay.
    L :
        Cumulant truncation multiplier for ``[a, b]``.
    """
    if cp not in (1, -1):
        raise ValueError(f"cp must be +1 or -1; got {cp}")
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    if np.any(K <= 0.0):
        raise ValueError("swift: strikes must be > 0")
    c1, c2, c4 = cumulants
    half = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    a, b = c1 - half, c1 + half

    mm = _scale(phi, m, b - a)
    two_m = 2.0**mm
    k1, k2 = int(np.floor(two_m * a)), int(np.ceil(two_m * b))
    J = 1 << int(np.ceil(np.log2(max(k2 - k1, 2))))
    C = (2.0 * np.arange(1, J + 1) - 1.0) * np.pi / (2.0 * J)
    D = two_m * C  # frequencies u_j
    idx = np.arange(k1, k2 + 1, dtype=float)
    norm = 2.0 ** (mm / 2.0) / J

    ck = np.outer(idx, C)  # C_j k, shape (K_idx, J)
    coeffs = norm * (np.cos(ck) @ np.real(phi(D)) + np.sin(ck) @ np.imag(phi(D)))  # c_{m,k}
    # cos(C_j (2^m x - k)) = cos(D_j x) cos(C_j k) + sin(D_j x) sin(C_j k): fold the
    # k-sum into the coefficients once, leaving O(J) work per strike.
    alpha = coeffs @ np.cos(ck)
    beta = coeffs @ np.sin(ck)

    F = fwd.F0
    x_hi = k2 / two_m
    x_lo = np.maximum(np.log(K / F), k1 / two_m)[:, None]
    kk = K[:, None]

    def parts(x):
        cx, sx = np.cos(D * x), np.sin(D * x)
        ex = F * np.exp(x) / (1.0 + D * D)
        A = ex * (cx + D * sx) - kk * sx / D
        B = ex * (sx - D * cx) + kk * cx / D
        return A, B

    a_hi, b_hi = parts(np.full_like(x_lo, x_hi))
    a_lo, b_lo = parts(x_lo)
    prices = fwd.disc * norm * ((a_hi - a_lo) @ alpha + (b_hi - b_lo) @ beta)
    prices = np.where(x_lo[:, 0] < x_hi, prices, 0.0)
    calls = np.maximum(prices, 0.0)
    return calls if cp == 1 else calls - fwd.disc * (F - K)
