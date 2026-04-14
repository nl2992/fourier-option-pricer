"""3/2 Stochastic Volatility model (Lewis 2000)  -  PyFENG-backed.

The 3/2 model is a stochastic-volatility model where the *instantaneous
variance* v_t satisfies a mean-reverting SDE with diffusion term proportional
to v^(3/2) rather than sqrt(v) (Heston). The faster diffusion rate produces
a much larger smile curvature for short maturities, fitting equity skew better
than Heston at the short end.

    dS/S = sqrt(v_t) dW1
    dv   = kappa * v * (theta - v) dt + nu * v^(3/2) dW2,
           <dW1, dW2> = rho dt

Unlike Heston's SDE, the coefficient on dW2 is v^(3/2), giving the "3/2" name.

PyFENG exposes the Fourier pricer as :class:`pyfeng.Sv32Fft`. The CF follows
from Lewis (2000) via a Girsanov-style transform: the Laplace transform of the
integrated variance process has a closed-form Bessel representation, which
PyFENG evaluates numerically via its standard FFT grid.

References
----------
* Lewis, A. L. (2000), *Option Valuation Under Stochastic Volatility*,
  Finance Press. (Original CF derivation via Laplace transform of ∫v_t dt.)
* Carr, P. and Sun, J. (2007), "A new approach for option pricing under
  stochastic volatility", *Review of Derivatives Research*.

PyFENG benchmark (from docstring)::

    sigma=0.06, mr=20.48, theta=0.218, vov=3.20, rho=-0.99
    strikes=[95, 100, 105], spot=100, texp=0.5
    → prices ≈ [11.7235, 8.9978, 6.7091]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._pyfeng_backend import build_cached, import_pyfeng
from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class Sv32Params(ModelSpec):
    """3/2 SV model parameters.

    Parameters
    ----------
    v0 :
        Initial instantaneous variance (``= sigma`` in PyFENG notation).
        Must be ``> 0``.  Note: this is *variance*, not volatility; the
        initial vol is ``sqrt(v0)``.
    kappa :
        Mean-reversion speed of the variance process (``mr`` in PyFENG).
        Must be ``> 0``.
    theta :
        Long-term mean of the variance process (``theta`` in PyFENG).
        Must be ``> 0``.
    nu :
        Vol-of-vol  -  coefficient on the v^(3/2) diffusion term (``vov``
        in PyFENG).  Must be ``> 0``.
    rho :
        Correlation between the spot and variance Brownian motions.
        Must be in ``(-1, 1)``.
    """

    v0: float
    kappa: float
    theta: float
    nu: float
    rho: float

    def __init__(
        self,
        v0: float,
        kappa: float,
        theta: float,
        nu: float,
        rho: float,
    ):
        if not (np.isfinite(v0) and v0 > 0):
            raise ValueError(f"Sv32Params: v0 must be > 0; got {v0}")
        if not (np.isfinite(kappa) and kappa > 0):
            raise ValueError(f"Sv32Params: kappa must be > 0; got {kappa}")
        if not (np.isfinite(theta) and theta > 0):
            raise ValueError(f"Sv32Params: theta must be > 0; got {theta}")
        if not (np.isfinite(nu) and nu > 0):
            raise ValueError(f"Sv32Params: nu must be > 0; got {nu}")
        if not (np.isfinite(rho) and -1.0 < rho < 1.0):
            raise ValueError(f"Sv32Params: rho must be in (-1, 1); got {rho}")
        object.__setattr__(self, "name", "sv32")
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)


# ---------------------------------------------------------------------------
# PyFENG-backed CF
# ---------------------------------------------------------------------------

_SV32_MODEL_CACHE: dict[tuple, object] = {}


def _pyfeng_sv32_model(fwd: ForwardSpec, p: Sv32Params):
    """Build-and-cache a :class:`pyfeng.Sv32Fft` for ``(fwd, p)``.

    PyFENG kwarg mapping::

        sigma  <->  p.v0       (initial variance)
        mr     <->  p.kappa    (mean-reversion speed)
        theta  <->  p.theta    (long-term variance)
        vov    <->  p.nu       (vol-of-vol)
        rho    <->  p.rho      (spot-var correlation)
    """

    def _factory():
        pf = import_pyfeng()
        return pf.Sv32Fft(
            sigma=p.v0,
            vov=p.nu,
            mr=p.kappa,
            rho=p.rho,
            theta=p.theta,
            intr=fwd.r,
            divr=fwd.q,
        )

    return build_cached(_SV32_MODEL_CACHE, (p, fwd), _factory)


def sv32_cf(u: np.ndarray, fwd: ForwardSpec, p: Sv32Params) -> np.ndarray:
    """CF of ``X_T = log(S_T / F_0)`` under the 3/2 SV model.

    Evaluates via PyFENG's :class:`Sv32Fft.logp_cf`, which is
    already in the log-forward convention consistent with this project's
    martingale condition ``φ(-i) = 1``.

    Parameters
    ----------
    u : array_like
        Frequency grid (real or complex).
    fwd : ForwardSpec
    p : Sv32Params

    Returns
    -------
    np.ndarray
        Complex CF values, same shape as ``u``.
    """
    m = _pyfeng_sv32_model(fwd, p)
    u_arr = np.asarray(u)
    return np.asarray(m.logp_cf(u_arr, texp=fwd.T), dtype=np.complex128)


# ---------------------------------------------------------------------------
# Cumulants  -  numerical Cauchy integral
# ---------------------------------------------------------------------------


def sv32_cumulants(fwd: ForwardSpec, p: Sv32Params) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of ``X_T`` under the 3/2 SV model.

    The 3/2 CF has no simple closed-form cumulant formula (the Bessel
    function representation does not differentiate cleanly at u=0), so we
    use the standard numerical Cauchy-circle FFT that all PyFENG-backed
    models fall back to.
    """
    from ..utils.cumulants import cumulants_from_cf

    def _phi(u):
        return sv32_cf(u, fwd, p)

    c = cumulants_from_cf(_phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
