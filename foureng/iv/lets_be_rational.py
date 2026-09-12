"""Vectorised machine-precision Black implied volatility ("Let's Be Rational").

Implements the algorithm of Jäckel (2015), *Let's Be Rational*, Wilmott
2015(75), 40-53, in NumPy, vectorised over arbitrary broadcast shapes:

1. Reduce every quote to an out-of-the-money call in normalised form,
   ``beta = price / (disc * sqrt(F K))`` with ``x = ln(F/K) <= 0``, so the
   normalised price is ``b(x, s) = e^{x/2} N(x/s + s/2) - e^{-x/2} N(x/s - s/2)``
   with total volatility ``s = sigma sqrt(T)`` and ``0 <= b < e^{x/2}``.
2. Split ``b(s)`` at its inflection point ``s_c = sqrt(2|x|)`` and the two
   tangents through it into four branches and build an initial guess on each by
   (transformed) rational cubic interpolation, which is already accurate to a
   few digits.
3. Refine with order-3 Householder steps on branch-specific objectives
   (``1/ln b`` in the lower wing, ``b`` in the middle, ``ln(b_max - b)`` near
   the upper bound), inside a bisection bracket for robustness. Two iterations
   typically reach machine precision; a small fixed cap guards the rest.

The normalised Black function is evaluated without catastrophic cancellation:
a Taylor series in ``t = s/2`` for small ``t``, a scaled-complementary-error
form in the lower region, and an exact complement ``b_max - b`` above the
inflection point. The implementation is written from the paper's mathematics;
the branch structure, maps, and objective functions follow Jäckel (2015).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erfcinv, erfcx, erfinv, ndtr, ndtri

__all__ = ["black_price", "implied_vol_lets_be_rational"]

_EPS = float(np.finfo(float).eps)
_SQRT_2 = float(np.sqrt(2.0))
_SQRT_3 = float(np.sqrt(3.0))
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
_SQRT_PI_OVER_2 = float(np.sqrt(0.5 * np.pi))
_TWO_PI_OVER_SQRT_27 = float(2.0 * np.pi / np.sqrt(27.0))
_SMALL_T = 0.21  # below this the Taylor series in t is used (Jäckel's threshold)
_SMALL_T_TERMS = 12
_MILLS_SWITCH = 4.0  # -h above this: continued fraction for a(h)
_MILLS_CF_TERMS = 60
_MAX_ITER = 8
_R_MIN = -(1.0 - float(np.sqrt(_EPS)))
_R_MAX = 2.0 / (_EPS * _EPS)


# ---------------------------------------------------------------------------
# Normalised Black function for out-of-the-money calls (x <= 0)
# ---------------------------------------------------------------------------


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / _SQRT_2PI


def _mills_a(h: np.ndarray) -> np.ndarray:
    """``a(h) = 1 + h N(h) / phi(h)`` for ``h <= 0`` without cancellation.

    ``a`` behaves like ``1/h^2`` as ``h -> -inf``; the direct formula loses
    ``log10(h^2)`` digits there, so large ``|h|`` uses the Laplace continued
    fraction of the Mills ratio rearranged as ``a = 1 / (1 + z C(z))`` with
    ``C(z) = z + 2/(z + 3/(z + 4/(z + ...)))`` and ``z = -h``.
    """
    z = -h
    out = np.empty_like(h)
    near = z < _MILLS_SWITCH
    if np.any(near):
        hn = h[near]
        out[near] = 1.0 + hn * _SQRT_PI_OVER_2 * erfcx(-hn / _SQRT_2)
    far = ~near
    if np.any(far):
        zf = z[far]
        c = zf.copy()
        for n in range(_MILLS_CF_TERMS, 1, -1):
            c = zf + n / c
        out[far] = 1.0 / (1.0 + zf * c)
    return out


def _nb_small_t(h: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Series ``b = 2 t phi(h) sum_k (-t^2/2)^k / k! j_k(h)`` for small ``t``.

    Obtained by integrating the vega ``db/ds`` from 0 to ``s``; ``j_0 = a(h)``
    and ``j_k = (1 - h^2 j_{k-1}) / (2k + 1)``. Every error in the recurrence is
    damped by powers of ``t^2``, so the relative error stays ``O(eps x^2)``.
    """
    j = _mills_a(h)
    acc = j.copy()
    coef = np.ones_like(t)
    m = -0.5 * t * t
    h2 = h * h
    for k in range(1, _SMALL_T_TERMS + 1):
        j = (1.0 - h2 * j) / (2 * k + 1)
        coef = coef * m / k
        acc = acc + coef * j
    return 2.0 * t * _phi(h) * acc


def _nb_otm(x: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised OTM call ``b(x, s)`` and its complement ``e^{x/2} - b``.

    ``x <= 0`` and ``s > 0`` elementwise. The complement is exact (no
    subtraction) wherever ``s`` exceeds the inflection point, which is where the
    upper-branch objective needs it.
    """
    h = x / s
    t = 0.5 * s
    b = np.empty_like(s)
    c = np.empty_like(s)
    bmax = np.exp(0.5 * x)

    small = t < _SMALL_T
    upper = ~small & (h + t > 0.0)
    lower = ~small & ~upper

    if np.any(small):
        b[small] = _nb_small_t(h[small], t[small])
        c[small] = bmax[small] - b[small]
    if np.any(lower):
        hl, tl = h[lower], t[lower]
        pref = 0.5 * np.exp(-0.5 * (hl * hl + tl * tl))
        b[lower] = pref * (erfcx(-(hl + tl) / _SQRT_2) - erfcx((tl - hl) / _SQRT_2))
        c[lower] = bmax[lower] - b[lower]
    if np.any(upper):
        hu, tu = h[upper], t[upper]
        pref = 0.5 * np.exp(-0.5 * (hu * hu + tu * tu))
        c[upper] = pref * (erfcx((hu + tu) / _SQRT_2) + erfcx((tu - hu) / _SQRT_2))
        b[upper] = bmax[upper] - c[upper]
    return b, c


def _vega(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``db/ds = exp(-(x^2/s^2 + s^2/4)/2) / sqrt(2 pi)``."""
    h = x / s
    t = 0.5 * s
    return np.exp(-0.5 * (h * h + t * t)) / _SQRT_2PI


# ---------------------------------------------------------------------------
# Rational cubic interpolation (Delbourgo & Gregory 1985)
# ---------------------------------------------------------------------------


def _rc_interp(x, x_l, x_r, y_l, y_r, d_l, d_r, r):
    h = x_r - x_l
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(h > 0.0, (x - x_l) / np.where(h > 0.0, h, 1.0), 0.5)
    omt = 1.0 - t
    t2, omt2 = t * t, omt * omt
    num = y_r * t2 * t + (r * y_r - h * d_r) * t2 * omt + (r * y_l + h * d_l) * t * omt2
    num = num + y_l * omt2 * omt
    rational = num / (1.0 + (r - 3.0) * t * omt)
    linear = y_r * t + y_l * omt
    return np.where(np.isfinite(r) & (r < _R_MAX), rational, linear)


def _r_min(d_l, d_r, slope, prefer_shape: bool):
    """Smallest control parameter that keeps the interpolant shape-preserving."""
    monotonic = (d_l * slope >= 0.0) & (d_r * slope >= 0.0)
    convex = (d_l <= slope) & (slope <= d_r)
    concave = (d_l >= slope) & (slope >= d_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(
            monotonic & (np.abs(slope) > 0.0),
            (d_r + d_l) / slope,
            np.where(monotonic & prefer_shape, _R_MAX, -np.inf),
        )
        ok = (np.abs(slope - d_l) > 0.0) & (np.abs(d_r - slope) > 0.0)
        r2_cc = np.where(
            ok,
            np.maximum(np.abs((d_r - d_l) / (d_r - slope)), np.abs((d_r - d_l) / (slope - d_l))),
            _R_MAX if prefer_shape else -np.inf,
        )
    r2 = np.where(convex | concave, r2_cc, np.where(monotonic & prefer_shape, _R_MAX, -np.inf))
    r = np.maximum(_R_MIN, np.maximum(r1, r2))
    return np.where(monotonic | convex | concave, r, _R_MIN)


def _r_fit_left(x_l, x_r, y_l, y_r, d_l, d_r, d2_l, prefer_shape: bool):
    h = x_r - x_l
    slope = (y_r - y_l) / h
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (0.5 * h * d2_l + (d_r - d_l)) / (slope - d_l)
    r = np.where(np.isfinite(r), r, _R_MAX)
    return np.maximum(r, _r_min(d_l, d_r, slope, prefer_shape))


def _r_fit_right(x_l, x_r, y_l, y_r, d_l, d_r, d2_r, prefer_shape: bool):
    h = x_r - x_l
    slope = (y_r - y_l) / h
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (0.5 * h * d2_r + (d_r - d_l)) / (d_r - slope)
    r = np.where(np.isfinite(r), r, _R_MAX)
    return np.maximum(r, _r_min(d_l, d_r, slope, prefer_shape))


# ---------------------------------------------------------------------------
# Lower / upper maps (Jäckel 2015, sections 3-4)
# ---------------------------------------------------------------------------


def _f_lower_map(x, s):
    """``f = 2 pi |x| / sqrt(27) * N(-z)^3`` with ``z = |x| / (sqrt(3) s)``,
    and its first two derivatives with respect to ``beta``."""
    ax = np.abs(x)
    z = ax / (_SQRT_3 * s)
    y = z * z
    s2 = s * s
    mills = _SQRT_PI_OVER_2 * erfcx(z / _SQRT_2)  # N(-z) / phi(z)
    big_phi = ndtr(-z)
    f = _TWO_PI_OVER_SQRT_27 * ax * big_phi**3
    fp = y * mills * mills * np.exp(0.125 * s2)
    with np.errstate(over="ignore", invalid="ignore"):
        fpp = (
            (np.pi / 6.0)
            * y
            / (s2 * s)
            * big_phi
            * (8.0 * _SQRT_3 * s * ax + (3.0 * s2 * (s2 - 8.0) - 8.0 * x * x) * mills)
            * np.exp(2.0 * y + 0.25 * s2)
        )
    return f, fp, fpp


def _inverse_f_lower_map(x, f):
    ax = np.abs(x)
    u = _SQRT_3 * np.cbrt(f / (2.0 * np.pi * ax))
    return ax / (_SQRT_3 * -ndtri(u))


def _initial_guess(beta, x, bmax, s_c, b_c, v_c, s_l, b_l, s_u, b_u):
    s = np.empty_like(beta)

    lo = beta < b_l
    if np.any(lo):
        xs, bs, sl, bl = x[lo], beta[lo], s_l[lo], b_l[lo]
        f_l, fp_l, fpp_l = _f_lower_map(xs, sl)
        zero = np.zeros_like(bs)
        one = np.ones_like(bs)
        r = _r_fit_right(zero, bl, zero, f_l, one, fp_l, fpp_l, True)
        f = _rc_interp(bs, zero, bl, zero, f_l, one, fp_l, r)
        tt = bs / bl
        f = np.where(f > 0.0, f, (f_l * tt + bl * (1.0 - tt)) * tt)
        s[lo] = _inverse_f_lower_map(xs, f)

    m1 = (beta >= b_l) & (beta <= b_c)
    if np.any(m1):
        xs, bs = x[m1], beta[m1]
        v_l = _vega(xs, s_l[m1])
        d_l, d_r = 1.0 / v_l, 1.0 / v_c[m1]
        zero = np.zeros_like(bs)
        r = _r_fit_right(b_l[m1], b_c[m1], s_l[m1], s_c[m1], d_l, d_r, zero, False)
        s[m1] = _rc_interp(bs, b_l[m1], b_c[m1], s_l[m1], s_c[m1], d_l, d_r, r)

    m2 = (beta > b_c) & (beta <= b_u)
    if np.any(m2):
        xs, bs = x[m2], beta[m2]
        v_u = _vega(xs, s_u[m2])
        d_l, d_r = 1.0 / v_c[m2], 1.0 / v_u
        zero = np.zeros_like(bs)
        r = _r_fit_left(b_c[m2], b_u[m2], s_c[m2], s_u[m2], d_l, d_r, zero, False)
        s[m2] = _rc_interp(bs, b_c[m2], b_u[m2], s_c[m2], s_u[m2], d_l, d_r, r)

    up = beta > b_u
    if np.any(up):
        xs, bs, su, bu, bm = x[up], beta[up], s_u[up], b_u[up], bmax[up]
        f_u = ndtr(-0.5 * su)
        w = (xs / su) ** 2
        fp_u = -0.5 * np.exp(0.5 * w)
        fpp_u = _SQRT_PI_OVER_2 * np.exp(w + 0.125 * su * su) * w / su
        half = np.full_like(bs, -0.5)
        zero = np.zeros_like(bs)
        r = _r_fit_left(bu, bm, f_u, zero, fp_u, half, fpp_u, True)
        f = _rc_interp(bs, bu, bm, f_u, zero, fp_u, half, r)
        hh = bm - bu
        tt = (bs - bu) / hh
        f = np.where(f > 0.0, f, (f_u * (1.0 - tt) + 0.5 * hh * tt) * (1.0 - tt))
        s[up] = -2.0 * ndtri(f)
    return s


# ---------------------------------------------------------------------------
# Core solver on normalised OTM inputs
# ---------------------------------------------------------------------------


def _implied_total_vol_otm(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Total volatility ``s`` with ``b(x, s) = beta`` for ``x <= 0``, ``0 < beta < e^{x/2}``."""
    s_out = np.empty_like(beta)
    bmax = np.exp(0.5 * x)

    atm = x == 0.0
    if np.any(atm):
        # b(0, s) = erf(s / (2 sqrt 2)) exactly.
        ba = beta[atm]
        s_out[atm] = 2.0 * _SQRT_2 * np.where(ba < 0.5, erfinv(ba), erfcinv(1.0 - ba))
    gen = ~atm
    if not np.any(gen):
        return s_out

    beta, x, bmax = beta[gen], x[gen], bmax[gen]
    s_c = np.sqrt(-2.0 * x)
    b_c, _ = _nb_otm(x, s_c)
    v_c = _vega(x, s_c)
    s_l = s_c - b_c / v_c
    s_u = s_c + (bmax - b_c) / v_c
    pos = s_l > 0.0
    b_l = np.zeros_like(beta)
    if np.any(pos):
        b_l[pos], _ = _nb_otm(x[pos], s_l[pos])
    b_u, _ = _nb_otm(x, s_u)

    s = _initial_guess(beta, x, bmax, s_c, b_c, v_c, s_l, b_l, s_u, b_u)

    branch_lo = beta < b_l
    branch_up = beta > b_u
    lo = np.where(branch_lo, 0.0, np.where(branch_up, s_u, s_l))
    hi = np.where(branch_lo, s_l, np.where(branch_up, np.inf, s_u))
    lo = np.maximum(lo, 0.0)
    bad = ~np.isfinite(s) | (s <= lo) | (s >= hi)
    s = np.where(bad, np.where(np.isfinite(hi), 0.5 * (lo + hi), 2.0 * lo + 1.0), s)
    s = np.maximum(s, np.finfo(float).tiny)

    log_beta = np.log(beta)
    active = np.ones_like(beta, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(_MAX_ITER):
            if not np.any(active):
                break
            idx = np.flatnonzero(active)
            sa, xa, ba = s[idx], x[idx], beta[idx]
            b, c = _nb_otm(xa, sa)
            bp = _vega(xa, sa)
            h = xa / sa
            b2 = h * h / sa - 0.25 * sa  # b''/b'
            b3 = b2 * b2 - 3.0 * (h / sa) ** 2 - 0.25  # b'''/b'

            lo_a, hi_a = lo[idx], hi[idx]
            hi_a = np.where(b > ba, np.minimum(hi_a, sa), hi_a)
            lo_a = np.where(b < ba, np.maximum(lo_a, sa), lo_a)

            newton = (ba - b) / bp
            halley = b2.copy()
            hh3 = b3.copy()

            in_lo = branch_lo[idx]
            if np.any(in_lo):
                bl_ = b[in_lo]
                ln_b = np.log(bl_)
                p = bp[in_lo] / bl_
                lb = log_beta[idx][in_lo]
                newton[in_lo] = np.log(ba[in_lo] / bl_) * ln_b / lb / p
                halley[in_lo] = b2[in_lo] - p * (1.0 + 2.0 / ln_b)
                hh3[in_lo] = (
                    b3[in_lo]
                    + 2.0 * p * p * (1.0 + 3.0 / ln_b * (1.0 + 1.0 / ln_b))
                    - 3.0 * b2[in_lo] * p * (1.0 + 2.0 / ln_b)
                )
            in_up = branch_up[idx]
            if np.any(in_up):
                cu = c[in_up]
                g = bp[in_up] / cu
                newton[in_up] = np.log(cu / (bmax[idx][in_up] - ba[in_up])) / g
                halley[in_up] = b2[in_up] + g
                hh3[in_up] = b3[in_up] + g * (2.0 * g + 3.0 * b2[in_up])

            factor = (1.0 + 0.5 * halley * newton) / (1.0 + newton * (halley + hh3 * newton / 6.0))
            ds = newton * factor
            ds = np.where(in_lo, np.maximum(ds, -0.5 * sa), ds)
            s_new = sa + ds
            # A step at rounding level means convergence; at that point b(s) may
            # sit a hair either side of beta and has already shrunk the bracket
            # onto s, so it must not trigger the bisection safeguard.
            tiny = np.abs(ds) <= 16.0 * _EPS * sa
            fallback = ~tiny & (~np.isfinite(s_new) | (s_new < lo_a) | (s_new > hi_a) | ~(b > 0.0))
            bisect = np.where(np.isfinite(hi_a), 0.5 * (lo_a + hi_a), 2.0 * sa)
            s_new = np.where(fallback, bisect, s_new)

            done = tiny | (hi_a - lo_a <= 4.0 * _EPS * s_new)
            s[idx], lo[idx], hi[idx] = s_new, lo_a, hi_a
            active[idx[done]] = False

    s_out[gen] = s
    return s_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def black_price(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    *,
    disc: ArrayLike = 1.0,
    cp: ArrayLike = 1,
) -> np.ndarray:
    """Black (1976) forward-measure option price, vectorised and cancellation-free.

    Parameters
    ----------
    F, K, T, sigma : array_like
        Forward, strike, maturity in years, and lognormal volatility. Broadcast.
    disc : array_like, default 1.0
        Discount factor to the payment date, e.g. ``exp(-r T)``.
    cp : array_like, default 1
        ``+1`` for calls, ``-1`` for puts.

    Returns
    -------
    np.ndarray
        ``disc * E[(cp (F_T - K))^+]``. Far out-of-the-money prices keep full
        relative precision, which a naive ``F N(d1) - K N(d2)`` does not.
    """
    arrays = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (F, K, T, sigma, disc, cp)))
    shape = arrays[0].shape
    F, K, T, sigma, disc, cp = (a.ravel() for a in arrays)
    if np.any((cp != 1.0) & (cp != -1.0)):
        raise ValueError("cp must be +1 (call) or -1 (put)")
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(F / K)
        s = sigma * np.sqrt(np.maximum(T, 0.0))
    out = np.maximum(cp * (F - K), 0.0)
    pos = s > 0.0
    if np.any(pos):
        b, _ = _nb_otm(-np.abs(x[pos]), s[pos])
        out[pos] += np.sqrt(F[pos] * K[pos]) * b
    return (disc * out).reshape(shape)


def implied_vol_lets_be_rational(
    price: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    *,
    disc: ArrayLike = 1.0,
    cp: ArrayLike = 1,
) -> np.ndarray:
    """Black implied volatility to machine precision, vectorised (Jäckel 2015).

    Inverts :func:`black_price` without a bracketing root search: a rational
    initial guess on one of four branches plus at most a couple of order-3
    Householder steps. All inputs broadcast against each other.

    Parameters
    ----------
    price : array_like
        Discounted option prices.
    F, K, T : array_like
        Forward, strike, and maturity in years.
    disc : array_like, default 1.0
        Discount factor the prices carry, e.g. ``exp(-r T)``.
    cp : array_like, default 1
        ``+1`` for calls, ``-1`` for puts.

    Returns
    -------
    np.ndarray
        Implied volatilities. ``0.0`` where the price equals intrinsic value;
        ``nan`` where the price is below intrinsic, at or above the upper
        no-arbitrage bound (``disc * F`` for calls, ``disc * K`` for puts), or
        where an input is non-finite or ``T <= 0``.

    Examples
    --------
    >>> import numpy as np, foureng as fe
    >>> p = fe.black_price(100.0, np.array([80.0, 100.0, 125.0]), 1.0, 0.25)
    >>> fe.implied_vol_lets_be_rational(p, 100.0, np.array([80.0, 100.0, 125.0]), 1.0)
    array([0.25, 0.25, 0.25])
    """
    arrays = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (price, F, K, T, disc, cp)))
    shape = arrays[0].shape
    price, F, K, T, disc, cp = (a.ravel() for a in arrays)
    if np.any((cp != 1.0) & (cp != -1.0)):
        raise ValueError("cp must be +1 (call) or -1 (put)")
    out = np.full(price.shape, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (
            np.isfinite(price)
            & np.isfinite(F)
            & np.isfinite(K)
            & np.isfinite(disc)
            & (F > 0.0)
            & (K > 0.0)
            & (T > 0.0)
            & (disc > 0.0)
        )
        undisc = price / disc
        itm_value = np.maximum(cp * (F - K), 0.0)
        otm_price = undisc - itm_value  # ITM quotes become OTM of the other type
        root_fk = np.sqrt(F * K)
        beta = otm_price / root_fk
        x = -np.abs(np.log(F / K))
        bmax = np.exp(0.5 * x)

    out[valid & (beta == 0.0)] = 0.0
    solve = valid & (beta > 0.0) & (beta < bmax)
    if np.any(solve):
        s = _implied_total_vol_otm(beta[solve], x[solve])
        out[solve] = s / np.sqrt(T[solve])
    return out.reshape(shape)
