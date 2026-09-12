"""European options on an optimally shifted contour with double-exponential quadrature.

A high-precision reference engine. With ``X = log(S_T / F)``, ``k = log(K / F)``
and the call payoff transform ``g^(z) = -e^{(1 + iz) k} / (z (z - i))``,

    C / (D F) = R(c) + (1/pi) int_0^inf Re[ g^(w + ic) phi(-w - ic) ] dw,

for any contour height ``c`` with ``E[e^{cX}] < inf`` and ``c`` not 0 or 1.
Moving the contour across the poles of ``g^`` at ``z = i`` and ``z = 0`` adds
the residues ``R(c) = 0`` for ``c > 1``, ``1`` for ``0 < c < 1`` and
``1 - e^k`` for ``c < 0``; puts subtract ``1 - e^k`` more (Lee 2004).

Two choices make the integral cheap and accurate:

* ``c`` is the Lord & Kahl (2007) optimum: it minimises the integrand at
  ``w = 0``, ``(1 - c) k + log M(c) - log|c (c - 1)|`` with ``M`` the moment
  generating function, which is the saddle point of the integrand. Out-of-the-
  money options land in the residue-free region, so their values come out as
  a pure integral with full *relative* precision, where COS-type expansions
  stop at an absolute error floor.
* The semi-infinite integral uses the exp-sinh double-exponential rule
  ``w = s exp(pi/2 sinh t)`` (Takahasi & Mori 1974), halving the step until two
  levels agree; ``s = 1 / sqrt(Var X)`` sets the frequency scale. This is the
  quadrature Andersen & Lake (2018) advocate for high-precision Fourier pricing
  (their deformed contours are not used: the horizontal contour is valid for
  every model with a finite ``M(c)``).

The admissible range of ``c`` is found by scanning ``M(c) = phi(-ic)``
outward from ``[0, 1]`` and stopping at the first value that is not finite,
real and positive, or where ``log M`` stops being convex (beyond a moment
explosion some CFs return finite but meaningless values).

References
----------
* Lord, R. & Kahl, C. (2007), Optimal Fourier inversion in semi-analytical
  option pricing, *Journal of Computational Finance* 10(4).
* Lee, R. (2004), Option pricing by transform methods: extensions,
  unification and error control, *Journal of Computational Finance* 7(3).
* Andersen, L. & Lake, M. (2018), Robust high-precision option pricing by
  Fourier transforms: contour deformations and double-exponential quadrature.
* Takahasi, H. & Mori, M. (1974), Double exponential formulas for numerical
  integration, *Publ. RIMS Kyoto Univ.* 9.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..models.base import ForwardSpec
from ..utils.grids import ContourGrid

__all__ = ["contour_price_at_strikes"]

_SCAN_STEP = 0.02
_POLE_GAP = 0.02
_T_MAX = 3.9  # exp-sinh truncation: w in [s e^-38, s e^38]


def _log_mgf(phi: Callable, c: np.ndarray) -> np.ndarray:
    """``log M(c)`` where valid, ``nan`` where ``phi(-ic)`` is not a finite positive real."""
    with np.errstate(all="ignore"):
        vals = np.asarray(phi(-1j * c), dtype=complex)
        re, im = vals.real, vals.imag
        ok = np.isfinite(re) & np.isfinite(im) & (re > 0.0) & (np.abs(im) <= 1e-8 * np.abs(re))
        return np.where(ok, np.log(np.where(ok, re, 1.0)), np.nan)


def _scan_offsets(c_bound: float) -> np.ndarray:
    """Distances from the [0, 1] band: fine steps near it, geometric further out."""
    fine = _SCAN_STEP * np.arange(1, int(8.0 / _SCAN_STEP) + 1)
    n_geo = int(np.ceil(np.log(max(c_bound, 8.0) / 8.0) / np.log(1.02)))
    coarse = 8.0 * 1.02 ** np.arange(1, n_geo + 1)
    return np.concatenate([fine, coarse[coarse <= c_bound]])


def _admissible_side(c: np.ndarray, log_m: np.ndarray) -> int:
    """Number of leading scan points (moving away from [0, 1]) that are usable."""
    bad = ~np.isfinite(log_m) | (log_m > 600.0)
    n_ok = int(np.argmax(bad)) if np.any(bad) else len(c)
    if n_ok >= 3:  # log M is convex; a downward kink means we left the strip
        slope = np.diff(log_m[:n_ok]) / np.diff(c[:n_ok])
        d2 = np.diff(slope) * np.sign(c[1] - c[0])
        kink = d2 < -1e-7 * (1.0 + np.abs(slope[1:]))
        if np.any(kink):
            n_ok = int(np.argmax(kink)) + 1
    return max(n_ok - 2, 0)  # keep a margin from the boundary


def _contour_heights(phi, k: np.ndarray, grid: ContourGrid) -> tuple[np.ndarray, float]:
    """Lord-Kahl optimal ``c`` per strike, and the frequency scale ``s``."""
    log_m_half = float(_log_mgf(phi, np.array([0.5]))[0])
    var = -8.0 * log_m_half if np.isfinite(log_m_half) and log_m_half < 0.0 else 1.0
    scale = 1.0 / np.sqrt(max(var, 1e-12))
    if grid.c is not None:
        if grid.c in (0.0, 1.0):
            raise ValueError("ContourGrid.c must not be 0 or 1 (poles of the payoff transform)")
        return np.full(k.shape, float(grid.c)), scale

    offsets = _scan_offsets(grid.c_bound)
    up, down = 1.0 + offsets, -offsets
    mid = np.linspace(_POLE_GAP, 1.0 - _POLE_GAP, 49)
    lm_up, lm_down, lm_mid = _log_mgf(phi, up), _log_mgf(phi, down), _log_mgf(phi, mid)
    n_up, n_down = _admissible_side(up, lm_up), _admissible_side(down, lm_down)
    cand = np.concatenate([mid, up[:n_up], down[:n_down]])
    log_m = np.concatenate([lm_mid, lm_up[:n_up], lm_down[:n_down]])
    keep = np.isfinite(log_m) & (np.abs(cand) >= _POLE_GAP) & (np.abs(cand - 1.0) >= _POLE_GAP)
    cand, log_m = cand[keep], log_m[keep]
    if cand.size == 0:
        raise ValueError("contour: no admissible contour height; the CF looks invalid")
    obj = (1.0 - cand)[None, :] * k[:, None] + log_m[None, :] - np.log(np.abs(cand * (cand - 1.0)))
    return cand[np.argmin(obj, axis=1)], scale


def _integrals(phi, k: np.ndarray, c: np.ndarray, scale: float, grid: ContourGrid) -> np.ndarray:
    """``(1/pi) int_0^inf Re[g^(w + ic) phi(-w - ic)] dw`` per strike (exp-sinh rule)."""

    def f(t: np.ndarray) -> np.ndarray:
        w = scale * np.exp(0.5 * np.pi * np.sinh(t))
        dw = 0.5 * np.pi * np.cosh(t) * w
        z = w[None, :] + 1j * c[:, None]
        with np.errstate(all="ignore"):
            vals = np.asarray(phi(-z.ravel()), dtype=complex).reshape(z.shape)
            # g^(z) without its e^{(1-c)k} factor, which is applied at the end.
            integrand = -np.exp(1j * w[None, :] * k[:, None]) * vals / (z * (z - 1j))
            out = np.real(integrand) * dw[None, :]
        return np.where(np.isfinite(out), out, 0.0)

    h = 0.5
    t = np.arange(-_T_MAX, _T_MAX + 1e-12, h)
    total = f(t).sum(axis=1)
    estimate = h * total
    converged = np.zeros(k.shape, dtype=bool)
    for _ in range(grid.max_levels):
        t_new = t[:-1] + 0.5 * h  # midpoints: the nested half-step nodes
        total = total + f(t_new).sum(axis=1)
        t = np.sort(np.concatenate([t, t_new]))
        h *= 0.5
        new = h * total
        converged = np.abs(new - estimate) <= grid.rel_tol * np.abs(new) + 1e-300
        estimate = new
        if np.all(converged):
            break
    return np.exp((1.0 - c) * k) * estimate / np.pi


def contour_price_at_strikes(
    phi: Callable[[np.ndarray], np.ndarray],
    fwd: ForwardSpec,
    strikes,
    *,
    cp: int = 1,
    grid: ContourGrid | None = None,
) -> np.ndarray:
    """European prices by optimal-contour Fourier inversion (see module docstring).

    Parameters
    ----------
    phi :
        CF of ``X_T = log(S_T / F_0)``; must accept complex arguments in the
        strip where the moment generating function is finite.
    fwd :
        Forward spec (``F0``, ``disc``).
    strikes :
        Strikes (any shape broadcastable to 1-D).
    cp :
        ``+1`` calls, ``-1`` puts.
    grid :
        :class:`~foureng.utils.grids.ContourGrid` tolerances and overrides.

    Returns
    -------
    np.ndarray
        Discounted prices. Out-of-the-money values keep ~``rel_tol`` relative
        accuracy however small they are.
    """
    if cp not in (1, -1):
        raise ValueError(f"cp must be +1 (call) or -1 (put); got {cp}")
    grid = grid or ContourGrid()
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    if np.any(K <= 0.0):
        raise ValueError("contour: strikes must be > 0")
    k = np.log(K / fwd.F0)
    c, scale = _contour_heights(phi, k, grid)
    integral = _integrals(phi, k, c, scale, grid)
    residue_call = np.where(c > 1.0, 0.0, np.where(c > 0.0, 1.0, -np.expm1(k)))
    residue = residue_call if cp == 1 else residue_call + np.expm1(k)
    return fwd.disc * fwd.F0 * (residue + integral)
