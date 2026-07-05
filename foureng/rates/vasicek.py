"""Vasicek (1977) short-rate model  -  closed-form bond prices and CFs.

Model
-----
    dr_t = kappa * (theta - r_t) dt + sigma dW_t,        r_0 = r0

The Vasicek short rate is a Gaussian Ornstein-Uhlenbeck process:
mean-reverting to ``theta`` at speed ``kappa`` with constant volatility
``sigma``.  Because both ``r_t`` and ``∫_0^T r_s ds`` are Gaussian under
the risk-neutral measure Q, every relevant object is available in closed
form.

Zero-coupon bond
----------------
    P(0, T) = A(T) * exp(-B(T) * r0)

    B(T) = (1 - exp(-kappa * T)) / kappa
    A(T) = exp( (theta - sigma^2 / (2*kappa^2)) * (B(T) - T)
                - sigma^2 * B(T)^2 / (4*kappa) )

Integrated rate ``I_T = ∫_0^T r_s ds``
--------------------------------------
Since ``r`` is Gaussian, ``I_T`` is also Gaussian, with

    E^Q[I_T]    = r0 * B(T) + theta * (T - B(T))
    Var^Q[I_T]  = (sigma^2 / kappa^2) * ( T - B(T)
                                          - kappa * B(T)^2 / 2 )

and characteristic function

    phi_{I_T}(u) = exp( i*u * E[I_T]  -  0.5 * u^2 * Var[I_T] ).

Notes
-----
The bond price ``P(0, T)`` can be recovered from ``integrated_rate_cf`` at
``u = i`` via ``P(0, T) = phi_{I_T}(i)`` (Laplace-transform view of the
discount factor).  This equivalence is used as a self-consistency check
in the tests.

References
----------
* Vasicek, O. (1977), "An equilibrium characterization of the term
  structure", *Journal of Financial Economics*, 5, 177-188.
* Brigo, D. & Mercurio, F. (2006), *Interest Rate Models  -  Theory and
  Practice*, 2nd ed., Springer, §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VasicekParams:
    """Vasicek short-rate parameters.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed, must be ``> 0``.
    theta : float
        Long-run mean of the short rate (any finite real).
    sigma : float
        Instantaneous volatility, must be ``> 0``.
    r0 : float
        Initial short rate at ``t = 0`` (any finite real).
    """

    kappa: float
    theta: float
    sigma: float
    r0: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.kappa) and self.kappa > 0):
            raise ValueError(f"VasicekParams: kappa must be > 0; got {self.kappa}")
        if not np.isfinite(self.theta):
            raise ValueError(f"VasicekParams: theta must be finite; got {self.theta}")
        if not (np.isfinite(self.sigma) and self.sigma > 0):
            raise ValueError(f"VasicekParams: sigma must be > 0; got {self.sigma}")
        if not np.isfinite(self.r0):
            raise ValueError(f"VasicekParams: r0 must be finite; got {self.r0}")


def _B(kappa: float, T: float) -> float:
    """Helper coefficient B(T) = (1 - exp(-kappa*T)) / kappa.

    Numerically stable for small ``kappa*T`` via first-order fallback.
    """
    if kappa * T < 1e-8:
        # Taylor: (1 - (1 - kT + (kT)^2/2 - ...)) / k = T - kT^2/2 + ...
        return T - 0.5 * kappa * T * T
    return (1.0 - np.exp(-kappa * T)) / kappa


def vasicek_discount_bond(p: VasicekParams, T: float) -> float:
    """Zero-coupon bond price P(0, T) under Vasicek.

    Parameters
    ----------
    p : VasicekParams
        Model parameters.
    T : float
        Maturity in years, must be ``>= 0``.

    Returns
    -------
    float
        Discount factor ``P(0, T)`` in (0, 1] (Vasicek can produce
        ``P > 1`` for pathological parameters with negative mean rate;
        this is a known model limitation, not a bug).
    """
    if not (np.isfinite(T) and T >= 0):
        raise ValueError(f"vasicek_discount_bond: T must be >= 0; got {T}")
    if T == 0.0:
        return 1.0

    B = _B(p.kappa, T)
    long_run = p.theta - 0.5 * p.sigma * p.sigma / (p.kappa * p.kappa)
    log_A = long_run * (B - T) - 0.25 * p.sigma * p.sigma * B * B / p.kappa
    return float(np.exp(log_A - B * p.r0))


def vasicek_integrated_rate_cumulants(p: VasicekParams, T: float) -> tuple[float, float]:
    """Mean and variance of ``I_T = ∫_0^T r_s ds`` under Vasicek.

    Parameters
    ----------
    p : VasicekParams
        Model parameters.
    T : float
        Maturity in years, ``T >= 0``.

    Returns
    -------
    (mean, variance) : tuple of floats
        First two cumulants of ``I_T``; higher cumulants are zero
        because ``I_T`` is Gaussian.
    """
    if not (np.isfinite(T) and T >= 0):
        raise ValueError(f"vasicek_integrated_rate_cumulants: T must be >= 0; got {T}")
    if T == 0.0:
        return 0.0, 0.0

    B = _B(p.kappa, T)
    mean = p.r0 * B + p.theta * (T - B)
    var = (p.sigma * p.sigma / (p.kappa * p.kappa)) * (T - B - 0.5 * p.kappa * B * B)
    return float(mean), float(var)


def vasicek_integrated_rate_cf(
    u: np.ndarray | complex | float,
    p: VasicekParams,
    T: float,
) -> np.ndarray:
    """Characteristic function of ``I_T = ∫_0^T r_s ds`` under Vasicek.

    ``phi_{I_T}(u) = E^Q[ exp(i u I_T) ]
                    = exp( i u * mean  -  0.5 * u^2 * var )``.

    Parameters
    ----------
    u : array-like or scalar
        Fourier argument (real or complex).  Broadcasts elementwise.
    p : VasicekParams
        Model parameters.
    T : float
        Maturity in years, ``T >= 0``.

    Returns
    -------
    np.ndarray
        Complex-valued CF evaluated at each ``u``.
    """
    mean, var = vasicek_integrated_rate_cumulants(p, T)
    u_arr = np.asarray(u, dtype=np.complex128)
    return np.exp(1j * u_arr * mean - 0.5 * u_arr * u_arr * var)
