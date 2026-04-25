from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class BSInputs:
    F0: float
    K: float
    T: float
    r: float
    q: float
    is_call: bool = True


def bs_price_from_fwd(vol: float, inp: BSInputs) -> float:
    """Black-76 call/put price from forward and discount (vol >= 0).

    Handles degenerate cases: vol -> 0 returns intrinsic; T -> 0 returns intrinsic.
    """
    F, K, T, r = inp.F0, inp.K, inp.T, inp.r
    disc = np.exp(-r * T)
    if vol <= 0.0 or T <= 0.0:
        payoff = max(F - K, 0.0) if inp.is_call else max(K - F, 0.0)
        return disc * payoff
    sqT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol * vol * T) / (vol * sqT)
    d2 = d1 - vol * sqT
    if inp.is_call:
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def _bs_vega_from_fwd(vol: float, inp: BSInputs) -> float:
    F, K, T, r = inp.F0, inp.K, inp.T, inp.r
    if vol <= 0.0 or T <= 0.0:
        return 0.0
    sqT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol * vol * T) / (vol * sqT)
    return np.exp(-r * T) * F * norm.pdf(d1) * sqT


def implied_vol_brent(price: float, inp: BSInputs, lo: float = 1e-6, hi: float = 5.0) -> float:
    """Brent IV baseline; returns NaN if bracket fails."""
    def f(sig: float) -> float:
        return bs_price_from_fwd(sig, inp) - price
    try:
        return float(brentq(f, lo, hi, maxiter=200))
    except Exception:
        return float("nan")


def implied_vol_newton_safeguarded(
    price: float,
    inp: BSInputs,
    vol0: float = 0.2,
    iters: int = 20,
    tol: float = 1e-10,
    lo: float = 1e-6,
    hi: float = 5.0,
) -> float:
    """Safeguarded Newton: Newton step if it stays inside [lo,hi] and reduces |f|,
    otherwise fall back to bisection on the current bracket.
    """
    def f(sig: float) -> float:
        return bs_price_from_fwd(sig, inp) - price

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    # ensure f(lo) < 0 < f(hi) (call price increasing in vol)
    if f_lo > 0:
        lo, hi = hi, lo
        f_lo, f_hi = f_hi, f_lo

    x = float(np.clip(vol0, min(lo, hi) + 1e-8, max(lo, hi) - 1e-8))
    fx = f(x)
    for _ in range(iters):
        if abs(fx) < tol:
            return x
        vega = _bs_vega_from_fwd(x, inp)
        x_newton = x - fx / vega if vega > 1e-14 else None
        # keep bracket updated
        if fx < 0:
            lo, f_lo = x, fx
        else:
            hi, f_hi = x, fx
        lo_s, hi_s = min(lo, hi), max(lo, hi)
        # accept Newton iff it stays inside and moves at least a little
        if x_newton is not None and lo_s < x_newton < hi_s:
            x_new = x_newton
        else:
            x_new = 0.5 * (lo + hi)
        if abs(x_new - x) < tol:
            return x_new
        x, fx = x_new, f(x_new)
    return x
