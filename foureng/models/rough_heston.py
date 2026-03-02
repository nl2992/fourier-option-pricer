"""Rough Heston model (El Euch & Rosenbaum 2019) — PyFENG-backed.

The Rough Heston model replaces the Markovian variance process of the
classical Heston model with a fractional Brownian motion driver, producing
a variance process with long memory and path roughness controlled by the
Hurst parameter H (or equivalently the fractional exponent alpha):

    dS/S = sqrt(v_t) dW1
    v_t  = v0 + (1/Γ(alpha)) ∫_0^t (t-s)^{alpha-1} [kappa*(theta - v_s) ds
                                                  + nu*sqrt(v_s) dW2_s]

where alpha ∈ (0, 1) and <dW1, dW2> = rho dt.  When alpha = 0.5 (i.e.,
H = 0.5 in some parameterisations), the model reduces to the standard
Heston SDE.  Rough paths (H < 0.5 in the half-integer convention) produce
a steeper short-maturity ATM skew consistent with empirical observations.

CF derivation
-------------
El Euch & Rosenbaum (2019) show the CF of X_T = log(S_T/F_0) is expressed
via a solution to a fractional Riccati ODE.  PyFENG implements two
numerical solvers: Adam's method (default, Method 1) and the fast hybrid
approximation of Callegaro et al. (2021).

PyFENG import
-------------
``pyfeng.ex`` re-exports ``RoughHestonFft`` but that import breaks under
newer SciPy (``from scipy.misc import derivative`` was removed).  We import
directly from ``pyfeng.sv_fft``:

    from pyfeng.sv_fft import RoughHestonFft

Numerical note
--------------
The fractional Riccati ODE is solved numerically, so repeated CF
evaluations on the same grid are expensive. We cache the PyFENG model
object per ``(params, fwd)`` using :func:`._pyfeng_backend.build_cached`.

References
----------
* El Euch, O. & Rosenbaum, M. (2019), "The characteristic function of rough
  Heston models", *Mathematical Finance*, 29, 3–38.
* Callegaro, G., Grasselli, M. & Pagès, G. (2021), "Fast hybrid schemes for
  fractional Riccati equations (rough is not so tough)", *Mathematics of
  Operations Research*, 46, 221–254.

PyFENG benchmark (first docstring example, verified numerically)::

    sigma=0.0392, vov=0.1, mr=0.3156, rho=-0.681, theta=0.3156, alpha=0.62
    strikes=[60, 70, 100, 140], spot=100, texp=1.0, r=q=0
    → prices ≈ [40.407, 31.356, 11.473, 1.888]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._pyfeng_backend import build_cached, import_pyfeng
from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class RoughHestonParams(ModelSpec):
    """Rough Heston parameters.

    Parameters
    ----------
    sigma : float
        Initial instantaneous variance (``v0``). Must be ``> 0``.
        Note: *variance*, not volatility.
    vov : float
        Vol-of-vol (``nu``) — coefficient on the sqrt(v) fractional
        diffusion term.
    mr : float
        Mean-reversion speed (``kappa``). Must be ``> 0``.
    rho : float
        Spot-variance correlation. Must be in ``(-1, 1)``.
    theta : float
        Long-term variance mean. Must be ``> 0``.
    alpha : float
        Fractional exponent, in ``(0, 1)``. Governs path roughness;
        ``alpha = 0.5`` recovers the standard Heston SDE.
    """

    sigma: float
    vov: float
    mr: float
    rho: float
    theta: float
    alpha: float

    def __init__(
        self,
        sigma: float,
        vov: float,
        mr: float,
        rho: float,
        theta: float,
        alpha: float = 0.62,
    ):
        if not (np.isfinite(sigma) and sigma > 0):
            raise ValueError(
                f"RoughHestonParams: sigma (initial variance) must be > 0; got {sigma}"
            )
        if not np.isfinite(vov):
            raise ValueError(f"RoughHestonParams: vov must be finite; got {vov}")
        if not (np.isfinite(mr) and mr > 0):
            raise ValueError(f"RoughHestonParams: mr must be > 0; got {mr}")
        if not (np.isfinite(rho) and -1.0 < rho < 1.0):
            raise ValueError(f"RoughHestonParams: rho must be in (-1, 1); got {rho}")
        if not (np.isfinite(theta) and theta > 0):
            raise ValueError(f"RoughHestonParams: theta must be > 0; got {theta}")
        if not (np.isfinite(alpha) and 0.0 < alpha < 1.0):
            raise ValueError(f"RoughHestonParams: alpha must be in (0, 1); got {alpha}")
        object.__setattr__(self, "name", "rough_heston")
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "vov", vov)
        object.__setattr__(self, "mr", mr)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "alpha", alpha)


# ---------------------------------------------------------------------------
# PyFENG-backed CF
# ---------------------------------------------------------------------------

_ROUGH_HESTON_MODEL_CACHE: dict[tuple, object] = {}


def _pyfeng_rough_heston_model(fwd: ForwardSpec, p: RoughHestonParams):
    """Build-and-cache a :class:`pyfeng.sv_fft.RoughHestonFft` for ``(fwd, p)``.

    We import directly from ``pyfeng.sv_fft`` to avoid the broken
    ``pyfeng.ex`` path (which imports ``scipy.misc.derivative``).

    PyFENG kwarg mapping::

        sigma  <->  p.sigma    (initial variance)
        vov    <->  p.vov      (vol-of-vol)
        mr     <->  p.mr       (mean-reversion)
        rho    <->  p.rho      (spot-var correlation)
        theta  <->  p.theta    (long-term variance)
        alpha  <->  p.alpha    (fractional exponent)
    """

    def _factory():
        import_pyfeng()
        # Import directly from sv_fft — pyfeng.ex is broken under newer SciPy
        from pyfeng.sv_fft import RoughHestonFft  # type: ignore

        return RoughHestonFft(
            sigma=p.sigma,
            vov=p.vov,
            mr=p.mr,
            rho=p.rho,
            theta=p.theta,
            alpha=p.alpha,
            intr=fwd.r,
            divr=fwd.q,
        )

    return build_cached(_ROUGH_HESTON_MODEL_CACHE, (p, fwd), _factory)


def rough_heston_cf(u: np.ndarray, fwd: ForwardSpec, p: RoughHestonParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under the rough Heston model.

    Evaluates via PyFENG's :class:`RoughHestonFft.logp_cf`, which
    solves the fractional Riccati equation numerically (Adam's method by
    default) and is already in the log-forward convention.

    Parameters
    ----------
    u : array_like (real or complex)
        Frequency grid.
    fwd : ForwardSpec
    p : RoughHestonParams

    Returns
    -------
    np.ndarray (complex128)
    """
    m = _pyfeng_rough_heston_model(fwd, p)
    u_arr = np.asarray(u)
    return np.asarray(m.logp_cf(u_arr, texp=fwd.T), dtype=np.complex128)


# ---------------------------------------------------------------------------
# Cumulants — numerical Cauchy integral
# ---------------------------------------------------------------------------


def rough_heston_cumulants(fwd: ForwardSpec, p: RoughHestonParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of X_T under the rough Heston model.

    The fractional Riccati CF has no closed-form cumulants at u=0, so we
    use the standard numerical Cauchy-circle FFT. The CF is analytic near
    u=0 for typical parameter choices.
    """
    from ..utils.cumulants import cumulants_from_cf

    def _phi(u):
        return rough_heston_cf(u, fwd, p)

    c = cumulants_from_cf(_phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
