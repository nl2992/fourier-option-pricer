"""Variance-Gamma characteristic function.

Implementation note — **PyFENG is the CF implementation for this model**.
``vg_cf`` is a thin wrapper around :meth:`pyfeng.VarGammaFft.charfunc_logprice`;
parameters are in the canonical CM1999 ``(sigma, nu, theta)`` convention
the project uses throughout, and PyFENG's CF agrees with the analytic
VG formula to ~1e-18 under that convention (verified).

The analytic cumulants are still computed in-house from FO2008 Table 10
— PyFENG doesn't expose cumulants and we need them for the COS
auto-grid truncation.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec
from ._pyfeng_backend import build_cached, import_pyfeng


@dataclass(frozen=True)
class VGParams(ModelSpec):
    """Variance Gamma parameters (Madan, Carr, Chang 1998)."""

    sigma: float
    nu: float
    theta: float

    def __init__(self, sigma: float, nu: float, theta: float):
        object.__setattr__(self, "name", "variance_gamma")
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "theta", theta)


# ---------------------------------------------------------------------------
# PyFENG-backed CF — uses the shared (cache, lazy-import) backend helpers so
# every PyFENG-backed wrapper (BSM / Heston / OUSV / VG / CGMY / NIG) has the
# same 10-line factory shape.
# ---------------------------------------------------------------------------

_VG_MODEL_CACHE: dict[tuple, object] = {}


def _pyfeng_vg_model(fwd: ForwardSpec, p: VGParams):
    """Build-and-cache a :class:`pyfeng.VarGammaFft` for (fwd, p).

    PyFENG kwarg mapping (CM1999 ``(sigma, nu, theta)`` -> VarGammaFft):

        sigma  <-  p.sigma   (diffusion vol)
        vov    <-  p.nu      (variance-of-time subordinator rate)
        theta  <-  p.theta   (drift asymmetry)
    """
    def _factory():
        pf = import_pyfeng()
        return pf.VarGammaFft(
            sigma=p.sigma,
            vov=p.nu,
            theta=p.theta,
            intr=fwd.r,
            divr=fwd.q,
        )
    return build_cached(_VG_MODEL_CACHE, (p, fwd), _factory)


def vg_cf(u: np.ndarray, fwd: ForwardSpec, p: VGParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under VG, via PyFENG's :class:`VarGammaFft`.

    Martingale correction
    ---------------------
    The canonical VG CF is

        phi(u) = exp(i*u*omega*T) * (1 - i*theta*nu*u + 0.5*sigma^2*nu*u^2)^(-T/nu)

    with ``omega = (1/nu) * log(1 - theta*nu - 0.5*sigma^2*nu)`` ensuring
    ``E[exp(X_T)] = 1``. Requires ``1 - theta*nu - 0.5*sigma^2*nu > 0`` for
    the log to be real — we do not re-check that condition here; PyFENG
    handles it.
    """
    m = _pyfeng_vg_model(fwd, p)
    u_arr = np.asarray(u)
    return np.asarray(m.charfunc_logprice(u_arr, texp=fwd.T), dtype=np.complex128)


def vg_cumulants(fwd: ForwardSpec, p: VGParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of log(S_T/F0) under VG (FO 2008 Table 10).

    Closed form — kept in-house because PyFENG does not expose cumulants
    and we need them for the COS truncation-interval rule.
    """
    sigma, nu, theta = p.sigma, p.nu, p.theta
    T = fwd.T
    cond = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
    omega = np.log(cond) / nu
    c1 = (theta + omega) * T
    c2 = (sigma * sigma + nu * theta * theta) * T
    c4 = 3.0 * (
        sigma ** 4 * nu
        + 2.0 * theta ** 4 * nu ** 3
        + 4.0 * sigma * sigma * theta * theta * nu * nu
    ) * T
    return float(c1), float(c2), float(c4)
