"""Heston (1993) characteristic function.

Implementation note — **PyFENG is the CF implementation for this model**.
We use :meth:`pyfeng.HestonFft.charfunc_logprice` directly so the project
does not duplicate the professor's code. The previous in-house analytic
"Formulation 2" CF has been retired; :data:`heston_cf_form2` is kept as a
back-compat alias pointing at :func:`heston_cf`.

Convention
----------
This project uses ``X_T = log(S_T / F_0)`` throughout (log-forward).
PyFENG's method name says "logprice" but it numerically agrees with our
log-forward CF to ~1e-18 on the Lewis parameters (verified across
``u in [-10, 10]``). If a future PyFENG release changes this, the single
convention shift ``phi * exp(-1j*u*log(F0))`` belongs here — marked
inline below.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class HestonParams(ModelSpec):
    """Heston (1993) parameters.

    dS/S = (r - q) dt + sqrt(v) dW1
    dv   = kappa*(theta - v) dt + nu*sqrt(v) dW2,   <dW1,dW2> = rho*dt

    Feller condition for strictly positive variance: 2*kappa*theta >= nu^2.
    """

    kappa: float
    theta: float
    nu: float
    rho: float
    v0: float

    def __init__(self, kappa: float, theta: float, nu: float, rho: float, v0: float):
        object.__setattr__(self, "name", "heston")
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "v0", v0)


# ---------------------------------------------------------------------------
# PyFENG-backed CF
# ---------------------------------------------------------------------------

_HESTON_MODEL_CACHE: dict[tuple, Any] = {}


def _pyfeng_heston_model(fwd: ForwardSpec, p: HestonParams):
    """Build-and-cache a :class:`pyfeng.HestonFft` for (fwd, p).

    PyFENG's constructor is relatively expensive; :meth:`charfunc_logprice`
    is cheap. We cache per ``(p, fwd)`` so repeated CF evaluations — e.g.
    the 64-point contour integral in :func:`heston_cumulants` and a
    subsequent COS/FRFT/Carr-Madan pricing pass — all hit the same model.
    """
    key = (p, fwd)
    m = _HESTON_MODEL_CACHE.get(key)
    if m is not None:
        return m
    try:
        import pyfeng as pf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "foureng.char_func.heston requires pyfeng; install with "
            "`pip install pyfeng`."
        ) from exc
    m = pf.HestonFft(
        sigma=p.v0,      # PyFENG: sigma = v0 (variance, not sqrt)
        vov=p.nu,
        rho=p.rho,
        mr=p.kappa,
        theta=p.theta,
        intr=fwd.r,
        divr=fwd.q,
    )
    _HESTON_MODEL_CACHE[key] = m
    return m


def heston_cf(u: np.ndarray, fwd: ForwardSpec, p: HestonParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) via PyFENG's :class:`HestonFft`.

    Parameters
    ----------
    u : array_like
        Frequency grid (real or complex).
    fwd : ForwardSpec
    p : HestonParams

    Returns
    -------
    np.ndarray
        Complex-valued CF, same shape as ``u``.
    """
    m = _pyfeng_heston_model(fwd, p)
    u_arr = np.asarray(u)
    phi = np.asarray(m.charfunc_logprice(u_arr, texp=fwd.T), dtype=np.complex128)
    # Convention-shift hook (currently a no-op — PyFENG's charfunc_logprice
    # is already in log-forward convention for this class, verified). If
    # this ever changes upstream: phi *= np.exp(-1j * u_arr * np.log(fwd.F0))
    return phi


# Back-compat alias: the project previously exported an analytic "Formulation 2"
# CF under this name and 16 files still import it. Same symbol, PyFENG body.
heston_cf_form2 = heston_cf


# ---------------------------------------------------------------------------
# Cumulants (used by COS auto-grid). Derived numerically from the CF so the
# formula is independent of whether the CF is analytic or PyFENG-backed.
# ---------------------------------------------------------------------------

def heston_cumulants(fwd: ForwardSpec, p: HestonParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of X_T = log(S_T/F_0) via Cauchy integration
    on the CF. Matches the project's convention documented in
    ``utils/cumulants.py`` and used by :func:`cos_auto_grid`.
    """
    from ..utils.cumulants import cumulants_from_cf

    phi = lambda u: heston_cf(u, fwd, p)
    c = cumulants_from_cf(phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
