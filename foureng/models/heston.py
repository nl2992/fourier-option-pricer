"""Heston (1993) characteristic function.

Implementation note  -  **PyFENG is the CF implementation for this model**.
We use :meth:`pyfeng.HestonFft.logp_cf` directly so the project
does not duplicate the professor's code. The previous in-house analytic
"Formulation 2" CF has been retired; :data:`heston_cf_form2` is kept as a
back-compat alias pointing at :func:`heston_cf`.

Convention
----------
This project uses ``X_T = log(S_T / F_0)`` throughout (log-forward).
PyFENG's method name says "logprice" but it numerically agrees with our
log-forward CF to ~1e-18 on the Lewis parameters (verified across
``u in [-10, 10]``). If a future PyFENG release changes this, the single
convention shift ``phi * exp(-1j*u*log(F0))`` belongs here  -  marked
inline below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any  # retained for _HESTON_MODEL_CACHE type annotation

import numpy as np

from ._pyfeng_backend import build_cached, import_pyfeng
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
        if not (np.isfinite(kappa) and kappa > 0):
            raise ValueError(f"HestonParams: kappa must be > 0; got {kappa}")
        if not (np.isfinite(theta) and theta > 0):
            raise ValueError(f"HestonParams: theta must be > 0; got {theta}")
        if not (np.isfinite(nu) and nu >= 0):
            raise ValueError(f"HestonParams: nu must be >= 0; got {nu}")
        if not (np.isfinite(rho) and -1.0 < rho < 1.0):
            raise ValueError(f"HestonParams: rho must be in (-1, 1); got {rho}")
        if not (np.isfinite(v0) and v0 > 0):
            raise ValueError(f"HestonParams: v0 must be > 0; got {v0}")
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
_HESTON_CUMULANT_CACHE: dict[tuple[HestonParams, ForwardSpec], tuple[float, float, float]] = {}


def _pyfeng_heston_model(fwd: ForwardSpec, p: HestonParams):
    """Build-and-cache a :class:`pyfeng.HestonFft` for (fwd, p).

    PyFENG's constructor is relatively expensive; :meth:`logp_cf`
    is cheap. We cache per ``(p, fwd)`` so repeated CF evaluations  -  e.g.
    the 64-point contour integral in :func:`heston_cumulants` and a
    subsequent COS/FRFT/Carr-Madan pricing pass  -  all hit the same model.
    """
    return build_cached(
        _HESTON_MODEL_CACHE,
        key=(p, fwd),
        factory=lambda: import_pyfeng().HestonFft(
            sigma=p.v0,  # PyFENG: sigma = v0 (variance, not sqrt)
            vov=p.nu,
            rho=p.rho,
            mr=p.kappa,
            theta=p.theta,
            intr=fwd.r,
            divr=fwd.q,
        ),
    )


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
    phi = np.asarray(m.logp_cf(u_arr, texp=fwd.T), dtype=np.complex128)
    # Convention-shift hook (currently a no-op  -  PyFENG's logp_cf
    # is already in log-forward convention for this class, verified). If
    # this ever changes upstream: phi *= np.exp(-1j * u_arr * np.log(fwd.F0))
    return phi


# Back-compat alias: the project previously exported an analytic "Formulation 2"
# CF under this name and 16 files still import it. Same symbol, PyFENG body.
heston_cf_form2 = heston_cf


# ---------------------------------------------------------------------------
# Riccati decomposition  -  needed by Double Heston and any model that
# composes two independent Heston variance factors.
# ---------------------------------------------------------------------------


def heston_riccati_cd(
    u: np.ndarray, T: float | np.ndarray, p: "HestonParams"
) -> tuple[np.ndarray, np.ndarray]:
    """Riccati (C, D) such that phi_Heston(u; T) = exp(C + D * v0).

    Uses Form 2 ("Little Heston Trap", Albrecher et al. 2007) which avoids
    the complex-log branch discontinuity present in the original Form 1 when
    (rho*nu*iu - kappa - d) crosses the negative real axis mid-maturity.

    Parameters
    ----------
    u : array_like (real or complex)
        Frequency grid.
    T : float
        Time to expiry.
    p : HestonParams
        Model parameters. Requires ``nu > 0``.

    Returns
    -------
    C, D : np.ndarray (complex128)
        ``phi_Heston(u; T) = exp(C + D * p.v0)`` element-wise.

    Notes
    -----
    The Albrecher Form 2 formula is::

        xi    = kappa - rho * nu * iu
        d     = sqrt(xi^2 + nu^2 * (iu + u^2))
        g2    = (xi - d) / (xi + d)
        B     = (xi - d) / nu^2 * (1 - exp(-d*T)) / (1 - g2*exp(-d*T))
        A     = kappa*theta / nu^2 * ((xi-d)*T - 2*log((1-g2*exp(-d*T))/(1-g2)))
        phi   = exp(A + B*v0)

    so C = A and D = B.

    References
    ----------
    * Albrecher, H., Mayer, P., Schachermayer, W. & Teugels, J. (2007),
      "The Little Heston Trap", *Wilmott Magazine*, Jan/Feb, 83-92.
    """
    if p.nu == 0.0:
        raise ValueError("heston_riccati_cd requires nu > 0; use BSM for nu = 0.")
    u_c = np.asarray(u, dtype=np.complex128)
    kappa, theta, nu, rho = p.kappa, p.theta, p.nu, p.rho

    # Form 2 has removable singularities at u=0 and u=-i (where iu + u² = 0),
    # both of which give phi=1 (C=D=0). Detect them and substitute a safe
    # dummy value for the intermediate algebra, then override the output.
    inner = 1j * u_c + u_c**2  # = u*(u + i); zero iff u=0 or u=-i
    degen = np.abs(inner) < 1e-14
    u_eval = np.where(degen, 1.0 + 1j, u_c)  # safe non-degenerate substitute

    xi = kappa - rho * nu * 1j * u_eval
    d = np.sqrt(xi**2 + nu**2 * (1j * u_eval + u_eval**2))

    g2 = (xi - d) / (xi + d)
    exp_neg_dT = np.exp(-d * T)

    # Albrecher Form 2: B from the Riccati solution
    B = (xi - d) / nu**2 * (1.0 - exp_neg_dT) / (1.0 - g2 * exp_neg_dT)

    # A: integrated kappa*theta term
    log_term = np.log((1.0 - g2 * exp_neg_dT) / (1.0 - g2))
    A = kappa * theta / nu**2 * ((xi - d) * T - 2.0 * log_term)

    # phi = exp(A + B*v0);  at degenerate points phi=1 → C=D=0
    C = np.where(degen, 0j, A)
    D = np.where(degen, 0j, B)
    return C, D


# ---------------------------------------------------------------------------
# Cumulants (used by COS auto-grid). Derived numerically from the CF so the
# formula is independent of whether the CF is analytic or PyFENG-backed.
# ---------------------------------------------------------------------------


def heston_cumulants(fwd: ForwardSpec, p: HestonParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of X_T = log(S_T/F_0) via Cauchy integration
    on the CF. Matches the project's convention documented in
    ``utils/cumulants.py`` and used by :func:`cos_auto_grid`.
    """
    cached = _HESTON_CUMULANT_CACHE.get((p, fwd))
    if cached is not None:
        return cached

    from ..utils.cumulants import cumulants_from_cf

    def _phi(u):
        return heston_cf(u, fwd, p)

    c = cumulants_from_cf(_phi, order=4, radius=0.25, M=64)
    out = (float(c[0]), float(c[1]), float(c[3]))
    _HESTON_CUMULANT_CACHE[(p, fwd)] = out
    return out
