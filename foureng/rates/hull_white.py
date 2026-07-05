"""Hull-White (1990) one-factor extended-Vasicek model.

Model
-----
    dr_t = ( theta(t) - a * r_t ) dt + sigma dW_t,          r_0 given

Constant mean-reversion speed ``a`` and volatility ``sigma``.  The
time-dependent drift ``theta(t)`` is chosen so the model reproduces the
initial market discount curve ``P^M(0, T)``.  The classical fit is

    theta(t) = ∂ f^M(0, t) / ∂t  +  a * f^M(0, t)  +  ( sigma^2 / (2 a) )
               * ( 1 - exp(-2 a t) ),

where ``f^M(0, t) = -∂ log P^M(0, t) / ∂t`` is the market instantaneous
forward rate (Brigo-Mercurio 2006, eq (3.34)).  Note that ``theta(t)``
enters neither the bond price nor the variance of ``I_T`` because both
are determined entirely by ``(P^M, a, sigma)``:

    P(0, T)      = P^M(0, T)      (perfect calibration to the initial curve)

    Var[I_T]     = (sigma^2 / a^2)
                   * [ T
                       - 2 * (1 - exp(-a T)) / a
                       + (1 - exp(-2 a T)) / (2 a) ]

    E[I_T]       = -log P^M(0, T)  +  Var[I_T] / 2
                    (convexity correction; enforces the exact identity
                     E[exp(-I_T)] = P^M(0, T))

    phi_{I_T}(u) = exp( i * u * E[I_T]  -  0.5 * u^2 * Var[I_T] )

This class exposes a caller-supplied initial curve ``initial_discount(T)``
(default: flat forward at ``r0``, giving ``P^M(0, T) = exp(-r0 * T)``).
No numerical fitting is required at construction time.

LevFin bridge
-------------
For make-whole and soft-call valuation, ``E[I_T]`` is exactly the negative
log-discount of the caller's live yield curve while ``Var[I_T]`` captures
the interest-rate optionality embedded in the call decision.  Callers can
plug in a stripped SOFR curve directly and get a research-grade
stochastic-discount pricer with no calibration overhead.

References
----------
* Hull, J. & White, A. (1990), "Pricing interest-rate derivative
  securities", *Review of Financial Studies*, 3, 573-592.
* Brigo, D. & Mercurio, F. (2006), *Interest Rate Models  -  Theory and
  Practice*, 2nd ed., Springer, §3.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def _flat_curve_factory(r0: float) -> Callable[[float], float]:
    """Default initial discount curve: flat forward at ``r0``."""

    def initial(T: float) -> float:
        return float(np.exp(-r0 * T))

    return initial


@dataclass(frozen=True)
class HullWhiteParams:
    """Hull-White one-factor parameters plus an initial discount curve.

    Parameters
    ----------
    a : float
        Mean-reversion speed, must be ``> 0``.
    sigma : float
        Volatility, must be ``> 0``.
    r0 : float
        Initial short rate (any finite real).  Only used by the default
        flat initial curve; ignored if ``initial_discount`` is supplied
        as a callable that already encodes the market curve.
    initial_discount : Callable[[float], float] or None
        Market discount curve ``P^M(0, T)``, mapping maturity in years to
        a discount factor in ``(0, 1]``.  Defaults to a flat forward at
        ``r0``: ``P^M(0, T) = exp(-r0 * T)``.  Not part of ``__eq__``;
        set to ``None`` when hashing.
    """

    a: float
    sigma: float
    r0: float
    initial_discount: Callable[[float], float] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not (np.isfinite(self.a) and self.a > 0):
            raise ValueError(f"HullWhiteParams: a must be > 0; got {self.a}")
        if not (np.isfinite(self.sigma) and self.sigma > 0):
            raise ValueError(f"HullWhiteParams: sigma must be > 0; got {self.sigma}")
        if not np.isfinite(self.r0):
            raise ValueError(f"HullWhiteParams: r0 must be finite; got {self.r0}")

    def curve(self) -> Callable[[float], float]:
        """Return the effective initial discount curve (falls back to flat)."""
        if self.initial_discount is not None:
            return self.initial_discount
        return _flat_curve_factory(self.r0)


def _hw_integrated_variance(a: float, sigma: float, T: float) -> float:
    """Var[I_T] under Hull-White (identical to Vasicek variance).

    Numerically stable for small ``a*T`` via a Taylor fallback.
    """
    if T == 0.0:
        return 0.0
    aT = a * T
    if aT < 1e-6:
        # Taylor expand to O(T^5): Var = (sigma^2 / 3) * T^3 * (1 - aT/4 + ...)
        return (sigma * sigma) * (T**3) / 3.0 * (1.0 - 0.25 * aT)

    e1 = np.exp(-aT)
    e2 = np.exp(-2.0 * aT)
    bracket = T - 2.0 * (1.0 - e1) / a + (1.0 - e2) / (2.0 * a)
    return (sigma * sigma / (a * a)) * bracket


def hull_white_discount_bond(p: HullWhiteParams, T: float) -> float:
    """Zero-coupon bond price P(0, T) under Hull-White.

    Because Hull-White is fitted to the initial discount curve by
    construction, ``P(0, T) = P^M(0, T)`` exactly.

    Parameters
    ----------
    p : HullWhiteParams
        Model parameters (including the initial curve).
    T : float
        Maturity in years, must be ``>= 0``.
    """
    if not (np.isfinite(T) and T >= 0):
        raise ValueError(f"hull_white_discount_bond: T must be >= 0; got {T}")
    if T == 0.0:
        return 1.0
    return float(p.curve()(T))


def hull_white_integrated_rate_cumulants(
    p: HullWhiteParams,
    T: float,
) -> tuple[float, float]:
    """Mean and variance of ``I_T = ∫_0^T r_s ds`` under Hull-White.

    Mean is fixed by the calibration constraint
    ``E^Q[exp(-I_T)] = P^M(0, T)``.  Since ``I_T`` is Gaussian under HW,
    ``E[exp(-I_T)] = exp(-E[I_T] + Var[I_T]/2)``, hence

        E[I_T] = -log P^M(0, T)  +  Var[I_T] / 2.

    The convexity term ``Var[I_T]/2`` is small under realistic
    calibrations but essential so that ``phi_{I_T}(i) = P^M(0, T)``
    holds *exactly* (rather than to first order in the variance).

    Variance depends only on ``(a, sigma, T)``:

        Var[I_T] = (sigma^2 / a^2)
                   * [ T - 2*(1-e^(-aT))/a + (1-e^(-2aT))/(2a) ].

    Parameters
    ----------
    p : HullWhiteParams
        Model parameters.
    T : float
        Maturity in years, ``T >= 0``.
    """
    if not (np.isfinite(T) and T >= 0):
        raise ValueError(f"hull_white_integrated_rate_cumulants: T must be >= 0; got {T}")
    if T == 0.0:
        return 0.0, 0.0

    P = p.curve()(T)
    if not (np.isfinite(P) and P > 0):
        raise ValueError(
            f"hull_white_integrated_rate_cumulants: initial_discount(T={T}) = "
            f"{P} is not in (0, inf); check the supplied curve."
        )
    variance = _hw_integrated_variance(p.a, p.sigma, T)
    mean = -float(np.log(P)) + 0.5 * variance
    return mean, variance


def hull_white_integrated_rate_cf(
    u: np.ndarray | complex | float,
    p: HullWhiteParams,
    T: float,
) -> np.ndarray:
    """Characteristic function of ``I_T`` under Hull-White.

    ``I_T`` is Gaussian, so

        phi_{I_T}(u) = exp( i*u * E[I_T]  -  0.5 * u^2 * Var[I_T] ).

    Parameters
    ----------
    u : array-like or scalar
        Fourier argument (real or complex).  Broadcasts elementwise.
    p : HullWhiteParams
        Model parameters.
    T : float
        Maturity in years, ``T >= 0``.
    """
    mean, var = hull_white_integrated_rate_cumulants(p, T)
    u_arr = np.asarray(u, dtype=np.complex128)
    return np.exp(1j * u_arr * mean - 0.5 * u_arr * u_arr * var)
