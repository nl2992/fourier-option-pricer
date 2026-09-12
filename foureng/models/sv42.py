"""4/2 stochastic volatility model (Grasselli 2017)  -  native closed-form CF.

The instantaneous volatility is a combination of a Heston ("1/2") and a 3/2
term driven by one CIR factor:

    dS/S = (r - q) dt + (a sqrt(v) + b / sqrt(v)) dW
    dv   = kappa (theta - v) dt + nu sqrt(v) dZ,        <dW, dZ> = rho dt.

``b = 0`` is Heston with variance ``a^2 v``; ``a = 0`` is the 3/2 model for the
instantaneous variance ``V = b^2 / v``. With both terms the volatility is
bounded below by ``2 sqrt(ab)``, and the Heston and 3/2 smile behaviours mix.

Characteristic function
-----------------------
Writing ``W = rho Z + sqrt(1 - rho^2) W_perp``, conditioning on the variance
path, and expressing both stochastic integrals through the CIR dynamics
(``int sqrt(v) dZ`` is linear in ``v_T`` and ``int v``; ``int dZ / sqrt(v)``
is linear in ``ln v_T`` and ``int 1/v``) gives

    phi(u) = exp(c(u)) E[ v_T^gamma exp(-lam v_T - mu int v - eta int 1/v) ]

with ``gamma, lam`` linear and ``mu, eta`` quadratic in ``u``. The expectation
is evaluated in closed form by two measure changes: a Girsanov change of the
mean-reversion speed to ``kappa~ = sqrt(kappa^2 + 2 nu^2 mu)`` absorbs
``int v``, and a change of the CIR dimension to ``nu^2/2 + D`` with
``D = sqrt((kappa theta - nu^2/2)^2 + 2 eta nu^2)`` absorbs ``int 1/v`` at the
cost of a power of ``v_T``. The remaining moment of the noncentral chi-square
law is a Kummer function, ``E[Y^P e^{-sY}] ~ 1F1(d/2 + P; d/2; zeta/(2(1+2s)))``.

The Kummer series is summed in log space with the ``exp(-zeta/2)`` factor
folded in, so its terms behave like Poisson weights; no complex-parameter
special-function library is needed. Both square roots have positive real part
for every real ``u``, so principal branches are continuous. Validated against
50-digit mpmath, against Heston (``b = 0``) and the 3/2 model (``a = 0``) to
~1e-15, and against Monte Carlo for general ``(a, b)``.

References
----------
* Grasselli, M. (2017), "The 4/2 stochastic volatility model: a unified
  approach for the Heston and the 3/2 model", *Mathematical Finance* 27(4),
  1013-1034.
* Heston, S. (1993); Heston, S. (1997) / Platen, E. (1997) for the 3/2 model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import loggamma

from .base import ForwardSpec, ModelSpec

_CUMULANT_CACHE: dict = {}


@dataclass(frozen=True)
class Sv42Params(ModelSpec):
    """4/2 stochastic volatility parameters.

    Parameters
    ----------
    v0 :
        Initial value of the CIR factor ``v``. Must be ``> 0``.
    kappa, theta :
        Mean-reversion speed and long-run level of ``v``. Both ``> 0``.
    nu :
        Volatility of ``v`` (coefficient of ``sqrt(v) dZ``). Must be ``> 0``.
    rho :
        Correlation of ``dW`` and ``dZ``, in ``(-1, 1)``.
    a, b :
        Loadings of the Heston (``a sqrt(v)``) and 3/2 (``b / sqrt(v)``)
        volatility components. ``a, b >= 0``, not both zero. ``b > 0``
        requires the Feller condition ``2 kappa theta >= nu^2`` so that ``v``
        stays positive and ``int 1/v`` is finite.
    """

    v0: float
    kappa: float
    theta: float
    nu: float
    rho: float
    a: float
    b: float

    def __init__(
        self,
        v0: float,
        kappa: float,
        theta: float,
        nu: float,
        rho: float,
        a: float,
        b: float,
    ):
        for name, val in (("v0", v0), ("kappa", kappa), ("theta", theta), ("nu", nu)):
            if not (np.isfinite(val) and val > 0):
                raise ValueError(f"Sv42Params: {name} must be > 0; got {val}")
        if not (np.isfinite(rho) and -1.0 < rho < 1.0):
            raise ValueError(f"Sv42Params: rho must be in (-1, 1); got {rho}")
        for name, val in (("a", a), ("b", b)):
            if not (np.isfinite(val) and val >= 0):
                raise ValueError(f"Sv42Params: {name} must be >= 0; got {val}")
        if a == 0 and b == 0:
            raise ValueError("Sv42Params: a and b cannot both be zero")
        if b > 0 and 2.0 * kappa * theta < nu * nu:
            raise ValueError(
                "Sv42Params: b > 0 requires the Feller condition 2*kappa*theta >= nu^2; "
                f"got 2*kappa*theta={2.0 * kappa * theta:.6g} < nu^2={nu * nu:.6g}"
            )
        object.__setattr__(self, "name", "sv42")
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)


def sv42_cf(u: np.ndarray, fwd: ForwardSpec, p: Sv42Params) -> np.ndarray:
    """CF of ``X_T = log(S_T / F_0)`` under the 4/2 model (see module docstring)."""
    u = np.asarray(u, dtype=complex)
    T = fwd.T
    v0, kappa, theta, sig, rho, a, b = p.v0, p.kappa, p.theta, p.nu, p.rho, p.a, p.b
    iu = 1j * u
    kth = kappa * theta - 0.5 * sig * sig

    psi = 0.5 * (iu + u * u * (1.0 - rho * rho))
    mu = psi * a * a - iu * rho * a * kappa / sig  # coefficient of int v
    eta = psi * b * b + iu * rho * b * kth / sig  # coefficient of int 1/v
    lam = -iu * rho * a / sig  # coefficient of v_T
    gam = iu * rho * b / sig  # power of v_T
    const = (
        -2.0 * a * b * psi * T
        - iu * rho * (a / sig) * (v0 + kappa * theta * T)
        + iu * rho * (b / sig) * (kappa * T - np.log(v0))
    )

    kt = np.sqrt(kappa * kappa + 2.0 * sig * sig * mu)
    k1 = (kt - kappa) / (sig * sig)
    if b == 0.0:
        D = np.full_like(u, kth)  # no int 1/v term: keep the plain CIR dimension
    else:
        D = np.sqrt(kth * kth + 2.0 * eta * sig * sig)
    b1 = (D - kth) / (sig * sig)

    log_c = np.log(0.25 * sig * sig) + np.log(-np.expm1(-kt * T)) - np.log(kt)
    half_d = 1.0 + 2.0 * D / (sig * sig)
    zeta = v0 * np.exp(-kt * T - log_c)
    one_2s = 1.0 + 2.0 * (lam - k1) * np.exp(log_c)
    P = gam - b1
    A = half_d + P
    z = zeta / (2.0 * one_2s)

    # exp(-zeta/2) * 1F1(A; half_d; z), summed as log-space Poisson-like terms.
    zmax = float(np.max(np.abs(z))) if z.size else 0.0
    n_terms = int(np.ceil(zmax + 12.0 * np.sqrt(zmax) + 40.0))
    # Streaming log-sum-exp: O(len(u)) memory however many terms are needed.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_z = np.log(z)
        L = -0.5 * zeta
        scale = L.real.copy()
        series = np.exp(L - scale)
        for j in range(n_terms):
            L = L + np.log(A + j) - np.log(half_d + j) + log_z - np.log(j + 1.0)
            new_scale = np.maximum(scale, L.real)
            series = series * np.exp(scale - new_scale) + np.exp(L - new_scale)
            scale = new_scale

    log_moment = (
        P * (log_c + np.log(2.0))
        - A * np.log(one_2s)
        + loggamma(A)
        - loggamma(half_d)
        + scale
        + np.log(series)
    )
    return np.exp(const - k1 * (v0 + kappa * theta * T) - b1 * (kt * T - np.log(v0)) + log_moment)


def sv42_cumulants(fwd: ForwardSpec, p: Sv42Params) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of ``X_T`` by Cauchy integration of the CF."""
    cached = _CUMULANT_CACHE.get((p, fwd))
    if cached is not None:
        return cached
    from ..utils.cumulants import cumulants_from_cf

    c = cumulants_from_cf(lambda u: sv42_cf(u, fwd, p), order=4, radius=0.25, M=64)
    out = (float(c[0]), float(c[1]), float(c[3]))
    _CUMULANT_CACHE[(p, fwd)] = out
    return out
