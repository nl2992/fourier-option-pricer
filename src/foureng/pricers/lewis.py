"""
Lewis (2001) Fourier pricer for European options.

Implements the single-integral Parseval-style formula using a
forward-normalized log-return characteristic function.

Convention
----------
We assume the model exposes the characteristic function of X_T where

    S_T = F_0(T) * exp(X_T),

so that

    phi_X(u) = E[e^{i u X_T}]
    phi_X(-i) = 1.

Let
    F = forward price at maturity T
    D = exp(-r T)
    k = log(K / F)

Then the Lewis call formula is

    C = D * [ F - sqrt(F K) / pi * I(k) ]

with

    I(k) = integral_0^inf Re(
              exp(-i u k) * phi_X(u - i/2) / (u^2 + 1/4)
           ) du

Put prices are obtained by put-call parity.

Notes
-----
- This module is compatible with today's free-function pipeline.
- A LewisPricer class is also provided so Pass 2/3 can adopt it
  without rewriting the numerical core.
- If your model currently returns the CF of log-price rather than
  forward-normalized log-return, use `cf_from_logprice_cf(...)`
  below to adapt it first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import math
import numpy as np

from .base import BasePricer

# NumPy 2.0 renamed ``np.trapz`` -> ``np.trapezoid``. Keep both paths working
# so the module runs under NumPy 1.x (CI had 1.26 historically) and 2.x.
_trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]

ArrayLike = np.ndarray | list[float] | tuple[float, ...]
ComplexOrArray = complex | np.ndarray
CF = Callable[[ComplexOrArray], ComplexOrArray]


def _as_1d_float_array(x: float | ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _forward_and_discount(
    spot: float,
    texp: float,
    intr: float,
    divr: float,
    is_fwd: bool,
) -> tuple[float, float]:
    """
    Return (forward, discount_factor).

    If is_fwd=True, `spot` is interpreted as the forward F_0(T).
    """
    df = math.exp(-intr * texp)
    if is_fwd:
        fwd = float(spot)
    else:
        fwd = float(spot) * math.exp((intr - divr) * texp)
    return fwd, df


def cf_from_logprice_cf(logprice_cf: CF, x0: float) -> CF:
    """
    Convert a log-price CF into a forward-normalized log-return CF.

    Suppose logprice_cf(u) = E[e^{i u log(S_T)}].
    If x0 = log(F_0(T)), then X_T = log(S_T) - x0 and

        phi_X(u) = exp(-i u x0) * logprice_cf(u)

    This helper builds that adapted CF.
    """
    def _wrapped(u: ComplexOrArray) -> ComplexOrArray:
        return np.exp(-1j * u * x0) * logprice_cf(u)

    return _wrapped


def _lewis_integrand(
    u: np.ndarray,
    k_log_moneyness: float,
    cf_shifted_vals: np.ndarray,
) -> np.ndarray:
    """
    Lewis integrand evaluated on a real u-grid.

    Parameters
    ----------
    u
        Real integration grid, shape (n,)
    k_log_moneyness
        log(K / F)
    cf_shifted_vals
        phi_X(u - i/2), shape (n,)
    """
    denom = u * u + 0.25
    osc = np.exp(-1j * u * k_log_moneyness)
    return np.real(osc * cf_shifted_vals / denom)


def _lewis_integral_trapz(
    cf: CF,
    k_log_moneyness: np.ndarray,
    *,
    u_max: float,
    n_u: int,
) -> np.ndarray:
    """
    Vectorized Lewis integral using a shared trapezoidal grid.

    Good for pricing a moderate strip of strikes with one CF evaluation grid.
    """
    u = np.linspace(0.0, u_max, int(n_u) + 1, dtype=float)
    cf_shifted = np.asarray(cf(u - 0.5j), dtype=np.complex128)

    # shape: (n_strikes, n_u+1)
    integrand = _lewis_integrand(
        u[None, :],
        k_log_moneyness[:, None],
        cf_shifted[None, :],
    )
    return _trapz(integrand, u, axis=1)


def _lewis_integral_quad(
    cf: CF,
    k_log_moneyness: np.ndarray,
    *,
    u_max: float,
    epsabs: float,
    epsrel: float,
) -> np.ndarray:
    """
    Lewis integral using scipy.integrate.quad, one strike at a time.

    More robust for single-strike / sparse-strike use.
    """
    try:
        from scipy.integrate import quad
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "scipy is required for method='quad'. "
            "Either install scipy or use method='trapz'."
        ) from exc

    out = np.empty_like(k_log_moneyness, dtype=float)

    for i, k in enumerate(k_log_moneyness):
        def _f(u: float) -> float:
            val = cf(u - 0.5j)
            return float(np.real(np.exp(-1j * u * k) * val / (u * u + 0.25)))

        integral, _err = quad(
            _f,
            0.0,
            float(u_max),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=500,
        )
        out[i] = integral

    return out


def lewis_call_prices(
    cf: CF,
    strikes: float | ArrayLike,
    spot: float,
    texp: float,
    *,
    intr: float = 0.0,
    divr: float = 0.0,
    is_fwd: bool = False,
    method: Literal["trapz", "quad"] = "trapz",
    u_max: float = 200.0,
    n_u: int = 4096,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
) -> np.ndarray:
    """
    Price European calls under the Lewis formula.

    Parameters
    ----------
    cf
        Forward-normalized log-return CF phi_X(u).
    strikes
        Scalar or array of strikes.
    spot
        Spot price, unless is_fwd=True in which case this is the forward.
    texp
        Time to expiry.
    intr, divr
        Interest and dividend/foreign rates.
    is_fwd
        If True, `spot` is interpreted as the forward F_0(T).
    method
        "trapz" for a shared deterministic grid,
        "quad" for adaptive quadrature per strike.
    u_max
        Upper cutoff of the Lewis integral.
    n_u
        Number of u-steps for trapz.
    epsabs, epsrel
        Quadrature tolerances for method="quad".
    """
    strikes_arr = _as_1d_float_array(strikes)
    if np.any(strikes_arr <= 0.0):
        raise ValueError("All strikes must be strictly positive.")
    if spot <= 0.0:
        raise ValueError("spot/forward must be strictly positive.")
    if texp <= 0.0:
        raise ValueError("texp must be strictly positive.")

    fwd, df = _forward_and_discount(
        spot=spot,
        texp=texp,
        intr=intr,
        divr=divr,
        is_fwd=is_fwd,
    )

    k = np.log(strikes_arr / fwd)

    if method == "trapz":
        integral = _lewis_integral_trapz(
            cf,
            k,
            u_max=u_max,
            n_u=n_u,
        )
    elif method == "quad":
        integral = _lewis_integral_quad(
            cf,
            k,
            u_max=u_max,
            epsabs=epsabs,
            epsrel=epsrel,
        )
    else:
        raise ValueError(f"Unknown Lewis integration method: {method}")

    calls = df * (fwd - np.sqrt(fwd * strikes_arr) / math.pi * integral)
    return calls


def lewis_prices(
    cf: CF,
    strikes: float | ArrayLike,
    spot: float,
    texp: float,
    *,
    cp: int = 1,
    intr: float = 0.0,
    divr: float = 0.0,
    is_fwd: bool = False,
    method: Literal["trapz", "quad"] = "trapz",
    u_max: float = 200.0,
    n_u: int = 4096,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
) -> np.ndarray:
    """
    Price European calls or puts via Lewis + put-call parity.

    cp = +1 for calls, -1 for puts.
    """
    strikes_arr = _as_1d_float_array(strikes)
    calls = lewis_call_prices(
        cf=cf,
        strikes=strikes_arr,
        spot=spot,
        texp=texp,
        intr=intr,
        divr=divr,
        is_fwd=is_fwd,
        method=method,
        u_max=u_max,
        n_u=n_u,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    if cp == 1:
        return calls
    if cp == -1:
        fwd, df = _forward_and_discount(
            spot=spot,
            texp=texp,
            intr=intr,
            divr=divr,
            is_fwd=is_fwd,
        )
        puts = calls - df * (fwd - strikes_arr)
        return puts

    raise ValueError("cp must be +1 (call) or -1 (put).")


@dataclass(slots=True)
class LewisPricer(BasePricer):
    """
    Future-friendly pricer wrapper around the Lewis free-function core.

    This already fits the intended Pass-2 pricer-class shape, even though
    today's pipeline can continue calling `lewis_prices(...)` directly.
    """

    method_name: str = "lewis"
    integration_method: Literal["trapz", "quad"] = "trapz"
    u_max: float = 200.0
    n_u: int = 4096
    epsabs: float = 1e-10
    epsrel: float = 1e-10

    def price(self, model, strike, spot, texp, cp=1, **kwargs):
        """
        Accept either:
        - a model object exposing `charfunc_logreturn(u, texp)`
        - a callable CF directly
        """
        if callable(model) and not hasattr(model, "charfunc_logreturn"):
            cf = model
            intr = kwargs.pop("intr", 0.0)
            divr = kwargs.pop("divr", 0.0)
            is_fwd = kwargs.pop("is_fwd", False)
        else:
            if not hasattr(model, "charfunc_logreturn"):
                raise TypeError(
                    "LewisPricer.price expects either a CF callable "
                    "or a model exposing charfunc_logreturn(u, texp)."
                )
            cf = lambda u: model.charfunc_logreturn(u, texp)
            intr = kwargs.pop("intr", getattr(model, "intr", 0.0))
            divr = kwargs.pop("divr", getattr(model, "divr", 0.0))
            is_fwd = kwargs.pop("is_fwd", getattr(model, "is_fwd", False))

        return lewis_prices(
            cf=cf,
            strikes=strike,
            spot=spot,
            texp=texp,
            cp=cp,
            intr=intr,
            divr=divr,
            is_fwd=is_fwd,
            method=kwargs.pop("method", self.integration_method),
            u_max=kwargs.pop("u_max", self.u_max),
            n_u=kwargs.pop("n_u", self.n_u),
            epsabs=kwargs.pop("epsabs", self.epsabs),
            epsrel=kwargs.pop("epsrel", self.epsrel),
        )


__all__ = [
    "LewisPricer",
    "cf_from_logprice_cf",
    "lewis_call_prices",
    "lewis_prices",
]