"""Normal Inverse Gaussian (Barndorff-Nielsen 1997) CF — PyFENG-backed.

NIG is, like Variance Gamma, a **pure-jump Lévy model built by subordinating
a drifted Brownian motion to a random time change** — but the time change
is an inverse-Gaussian (IG) subordinator rather than a Gamma one:

    X_t = theta * tau_t + sigma * W_{tau_t},   tau_t ~ IG(t, nu).

The resulting distribution has heavier tails than VG (semi-heavy, like
the NIG-density namesake) and is a staple in equity / FX modelling for
short-maturity smile fitting. Infinite-activity, same 3-parameter shape
as VG (``sigma, nu, theta``), same API here.

PyFENG exposes the model as :class:`pyfeng.ExpNigFft` with the same MGF
convention as :class:`pyfeng.VarGammaFft`:

    MGF(u) = exp( T/nu * [mu*u + 1 - sqrt(1 - 2*theta*nu*u - sigma^2*nu*u^2)] )
    mu      = -1 + sqrt(1 - 2*theta*nu - sigma^2*nu)

with the martingale correction ``mu`` baked in so that ``MGF(1) = 1``.

Adapter
-------
PyFENG's :class:`ExpNigFft` inherits from :class:`SvABC`, which means its
``__init__`` takes the SV-style ``(sigma, vov, rho, mr, theta, ...)``
signature even though only ``sigma``, ``vov``, ``theta`` matter for the
pure-Lévy NIG — the extra ``rho`` / ``mr`` are unused dead weight. We
default them in the factory and expose only the three meaningful
parameters in our dataclass.

References
----------
* Barndorff-Nielsen, O. E. (1997), "Normal inverse Gaussian distributions
  and stochastic volatility modelling", Scand. J. Statist.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec
from ._pyfeng_backend import build_cached, import_pyfeng


@dataclass(frozen=True)
class NigParams(ModelSpec):
    """NIG Lévy parameters (same 3-parameter shape as :class:`VGParams`).

    Attributes
    ----------
    sigma :
        Diffusion volatility of the subordinated Brownian motion. Must be
        ``> 0``.
    nu :
        IG time-subordinator rate (akin to VG's ``nu``). Controls the
        excess kurtosis — larger ``nu`` = heavier tails.
    theta :
        Drift of the subordinated BM. Controls skew — ``theta < 0``
        produces the familiar negative equity skew.

    Existence condition: ``1 - 2*theta*nu - sigma^2*nu > 0``, otherwise
    the martingale correction ``mu`` is complex. PyFENG surfaces this as
    a NaN from ``sqrt``.
    """

    sigma: float
    nu: float
    theta: float

    def __init__(self, sigma: float, nu: float, theta: float):
        object.__setattr__(self, "name", "nig")
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "theta", theta)


# ---------------------------------------------------------------------------
# PyFENG-backed CF
# ---------------------------------------------------------------------------

_NIG_MODEL_CACHE: dict[tuple, object] = {}


def _pyfeng_nig_model(fwd: ForwardSpec, p: NigParams):
    """Build-and-cache a :class:`pyfeng.ExpNigFft` for ``(fwd, p)``.

    Kwarg mapping:

        sigma  <-  p.sigma
        vov    <-  p.nu
        theta  <-  p.theta
        rho    <-  0.0   (unused by the NIG CF — SvABC baggage)
        mr     <-  0.01  (unused, but SvABC insists on it; PyFENG default)
    """
    def _factory():
        pf = import_pyfeng()
        return pf.ExpNigFft(
            sigma=p.sigma,
            vov=p.nu,
            theta=p.theta,
            intr=fwd.r,
            divr=fwd.q,
        )
    return build_cached(_NIG_MODEL_CACHE, (p, fwd), _factory)


def nig_cf(u: np.ndarray, fwd: ForwardSpec, p: NigParams) -> np.ndarray:
    """CF of ``X_T = log(S_T / F_0)`` under NIG — via PyFENG's ``ExpNigFft``."""
    m = _pyfeng_nig_model(fwd, p)
    u_arr = np.asarray(u)
    return np.asarray(m.charfunc_logprice(u_arr, texp=fwd.T),
                      dtype=np.complex128)


# ---------------------------------------------------------------------------
# Cumulants — numerical Cauchy integral on the PyFENG CF
# ---------------------------------------------------------------------------

def nig_cumulants(fwd: ForwardSpec, p: NigParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of ``X_T`` under NIG.

    A closed form exists (differentiate the Lévy exponent analytically)
    but algebra around the ``sqrt(1 - 2*theta*nu*u - sigma^2*nu*u^2)``
    term is noisy and easy to typo. The NIG CF is analytic on a disk
    around ``u=0`` under the existence condition, so the same FFT-on-
    circle Cauchy-integral routine we use for OUSV / CGMY converges to
    machine precision here too.
    """
    from ..utils.cumulants import cumulants_from_cf

    phi = lambda u: nig_cf(u, fwd, p)
    c = cumulants_from_cf(phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
