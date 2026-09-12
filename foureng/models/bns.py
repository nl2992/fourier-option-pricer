"""Barndorff-Nielsen & Shephard (2001) stochastic volatility with Gamma-OU variance.

    d log S = (r - q - lam k(rho) - v/2) dt + sqrt(v) dW + rho dz_{lam t},
    dv      = -lam v dt + dz_{lam t},

where the background driving Levy process ``z`` is compound Poisson with
intensity ``a`` and Exp(``b``) jumps, so ``v`` has stationary Gamma(a, b)
marginals and ``k(u) = log E[e^{u z_1}] = a u / (b - u)``. Variance jumps up
while, with ``rho < 0``, the price jumps down at the same instants (leverage).

The CF of ``X_T = log(S_T / F_0)`` is closed form (Nicolato & Venardos 2003;
Schoutens 2003, section 7.2.2):

    log phi(u) = -iu a lam rho T / (b - rho)
                 - (u^2 + iu) (1 - e^{-lam T}) v0 / (2 lam)
                 + a / (b - f2) * (b log((b - f1) / (b - iu rho)) + f2 lam T),
    f1 = iu rho - (u^2 + iu) (1 - e^{-lam T}) / (2 lam),
    f2 = iu rho - (u^2 + iu) / (2 lam).

For real ``u`` both ``b - f1`` and ``b - iu rho`` lie in the right half-plane,
so their logs are taken separately on principal branches; ``b - f2`` never
vanishes. With ``a = 0`` the variance decays deterministically and the model
is Black-Scholes with total variance ``v0 (1 - e^{-lam T}) / lam``.

References
----------
* Barndorff-Nielsen, O. E. & Shephard, N. (2001), Non-Gaussian Ornstein-
  Uhlenbeck-based models and some of their uses in financial economics,
  *J. Royal Statistical Society B* 63(2), 167-241.
* Nicolato, E. & Venardos, E. (2003), Option pricing in stochastic volatility
  models of the Ornstein-Uhlenbeck type, *Mathematical Finance* 13(4), 445-466.
* Schoutens, W. (2003), *Levy Processes in Finance*, Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec

_CUMULANT_CACHE: dict = {}


@dataclass(frozen=True)
class BNSParams(ModelSpec):
    """BNS Gamma-OU parameters.

    Parameters
    ----------
    v0 : float
        Initial variance ``v(0) > 0``.
    lam : float
        Mean-reversion rate of the variance, ``> 0``.
    a, b : float
        Jump intensity and Exp rate of the driving compound Poisson process
        (stationary variance Gamma(a, b), mean ``a / b``); ``a >= 0``, ``b > 0``.
    rho : float
        Price response to a variance jump (leverage), usually ``<= 0``; must be
        ``< b`` so that ``E[e^{rho z}]`` is finite.
    """

    v0: float
    lam: float
    a: float
    b: float
    rho: float

    def __init__(self, v0: float, lam: float, a: float, b: float, rho: float):
        for name, val in (("v0", v0), ("lam", lam), ("b", b)):
            if not (np.isfinite(val) and val > 0):
                raise ValueError(f"BNSParams: {name} must be > 0; got {val}")
        if not (np.isfinite(a) and a >= 0):
            raise ValueError(f"BNSParams: a must be >= 0; got {a}")
        if not (np.isfinite(rho) and rho < b):
            raise ValueError(f"BNSParams: rho must be finite and < b; got rho={rho}, b={b}")
        object.__setattr__(self, "name", "bns")
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "lam", lam)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "rho", rho)


def bns_cf(u: np.ndarray, fwd: ForwardSpec, p: BNSParams) -> np.ndarray:
    """CF of ``X_T = log(S_T / F_0)`` under BNS Gamma-OU (see module docstring)."""
    u = np.asarray(u, dtype=complex)
    T, lam, a, b, rho = fwd.T, p.lam, p.a, p.b, p.rho
    iu = 1j * u
    q = u * u + iu
    decay = -np.expm1(-lam * T)  # 1 - e^{-lam T}
    f1 = iu * rho - q * decay / (2.0 * lam)
    f2 = iu * rho - q / (2.0 * lam)
    log_phi = -iu * a * lam * rho * T / (b - rho) - q * decay * p.v0 / (2.0 * lam)
    if a > 0.0:
        log_ratio = np.log(b - f1) - np.log(b - iu * rho)
        log_phi = log_phi + a / (b - f2) * (b * log_ratio + f2 * lam * T)
    return np.exp(log_phi)


def bns_cumulants(fwd: ForwardSpec, p: BNSParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` by Cauchy integration of the CF."""
    cached = _CUMULANT_CACHE.get((p, fwd))
    if cached is not None:
        return cached
    from ..utils.cumulants import cumulants_from_cf

    c = cumulants_from_cf(lambda u: bns_cf(u, fwd, p), order=4, radius=0.25, M=64)
    out = (float(c[0]), float(c[1]), float(c[3]))
    _CUMULANT_CACHE[(p, fwd)] = out
    return out
