"""Ornstein-Uhlenbeck Stochastic Volatility (Schöbel-Zhu 1999) — PyFENG-backed.

OUSV is the "vol-OU" companion to Heston's "variance-CIR": **the
instantaneous volatility itself** follows an OU process (not the
variance). Concretely:

    d log S = (r - q - 0.5 * sigma_t^2) dt + sigma_t dW_1,
    d sigma = kappa (theta - sigma) dt + nu dW_2,
    <dW_1, dW_2> = rho dt.

Compared with Heston: same parameter *count*, different variance
dynamics. The OU process admits negative values of ``sigma``, but the
model's option prices depend only on ``sigma_t^2`` entering the
log-price diffusion — so sign-flipping has no effect on European
payoffs. That's the reason OUSV doesn't need a Feller-type condition.

PyFENG ships :class:`pyfeng.OusvFft` with the same public surface as
:class:`pyfeng.HestonFft`: ``charfunc_logprice``, ``price``, and
``impvol_brentq``. Adapter conventions mirror :mod:`.heston` exactly —
our param dataclass uses the academic names ``(sigma0, kappa, theta,
nu, rho)`` and we translate to PyFENG's ``(sigma, mr, theta, vov, rho)``
inside the model factory.

References
----------
* Schöbel, R. and Zhu, J. (1999), "Stochastic Volatility With an
  Ornstein–Uhlenbeck Process: An Extension", European Finance Review.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec
from ._pyfeng_backend import build_cached, import_pyfeng


@dataclass(frozen=True)
class OusvParams(ModelSpec):
    """Schöbel-Zhu OUSV parameters.

    Attributes
    ----------
    sigma0 :
        Initial instantaneous volatility ``sigma_0``. (PyFENG kwarg:
        ``sigma``.)
    kappa :
        Mean-reversion speed of the OU process. (PyFENG kwarg: ``mr``.)
    theta :
        Long-run mean of the OU process. (PyFENG kwarg: ``theta``.)
    nu :
        Vol-of-vol / OU diffusion coefficient. (PyFENG kwarg: ``vov``.)
    rho :
        Correlation between the price and vol Brownian motions.
    """

    sigma0: float
    kappa: float
    theta: float
    nu: float
    rho: float

    def __init__(self, sigma0: float, kappa: float, theta: float,
                 nu: float, rho: float):
        object.__setattr__(self, "name", "ousv")
        object.__setattr__(self, "sigma0", sigma0)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)


# ---------------------------------------------------------------------------
# PyFENG-backed CF
# ---------------------------------------------------------------------------

_OUSV_MODEL_CACHE: dict[tuple, object] = {}


def _pyfeng_ousv_model(fwd: ForwardSpec, p: OusvParams):
    """Build-and-cache a :class:`pyfeng.OusvFft` for ``(fwd, p)``.

    PyFENG kwarg mapping — mirrors :class:`pyfeng.HestonFft`:

        sigma  <-  p.sigma0   (initial vol, not variance)
        mr     <-  p.kappa
        theta  <-  p.theta
        vov    <-  p.nu
        rho    <-  p.rho
    """
    def _factory():
        pf = import_pyfeng()
        return pf.OusvFft(
            sigma=p.sigma0,
            mr=p.kappa,
            theta=p.theta,
            vov=p.nu,
            rho=p.rho,
            intr=fwd.r,
            divr=fwd.q,
        )
    return build_cached(_OUSV_MODEL_CACHE, (p, fwd), _factory)


def ousv_cf(u: np.ndarray, fwd: ForwardSpec, p: OusvParams) -> np.ndarray:
    """CF of ``X_T = log(S_T / F_0)`` under OUSV — via PyFENG's ``OusvFft``."""
    m = _pyfeng_ousv_model(fwd, p)
    u_arr = np.asarray(u)
    return np.asarray(m.charfunc_logprice(u_arr, texp=fwd.T),
                      dtype=np.complex128)


# ---------------------------------------------------------------------------
# Cumulants — numerical via Cauchy integral on the CF
# ---------------------------------------------------------------------------

def ousv_cumulants(fwd: ForwardSpec, p: OusvParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of ``X_T`` under OUSV.

    Closed-form OUSV cumulants are long and parameter-regime-restricted;
    we use the same FFT-on-circle Cauchy-integral routine used for
    Heston and CGMY. Works for any CF analytic on a neighbourhood of
    ``u = 0`` — which OUSV is under the standard parameter regime.
    """
    from ..utils.cumulants import cumulants_from_cf

    phi = lambda u: ousv_cf(u, fwd, p)
    c = cumulants_from_cf(phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
