"""Heston-VG SVJ model  -  Heston SV + Variance-Gamma Lévy jumps.

Variance Gamma (VG) is a pure-jump, infinite-activity Lévy process
obtained by subordinating a drifted Brownian motion with a Gamma clock.
Combined with Heston's stochastic-volatility diffusion the CF factorises
under independence:

    phi(u) = phi_H(u) * phi_VG_jump(u)

where ``phi_H`` is the Heston CF (PyFENG-backed) and ``phi_VG_jump`` is
the martingale-corrected VG pure-jump component.

VG Lévy exponent
----------------
For the VG process parameterised by ``(sigma, nu, theta)`` (Madan, Carr &
Chang 1998) the per-unit-time log-CF (Lévy exponent) is:

    psi_VG(u) = -(1/nu) * log(1 - i*theta*nu*u + sigma^2*nu*u^2/2)

At ``u = -i``:

    psi_VG(-i) = -(1/nu) * log(1 - theta*nu - sigma^2*nu/2)

which is real.  Existence condition: ``1 - theta*nu - sigma^2*nu/2 > 0``.

Martingale correction
---------------------
    phi_VG_jump(u) = exp(T * (psi_VG(u) - iu * psi_VG(-i)))

so ``phi_VG_jump(-i) = exp(T * 0) = 1``, preserving the log-forward
martingale invariant.

Reduction limits
----------------
* ``vg_sigma → 0, vg_theta → 0``: VG exponent → 0, recovers pure Heston.
* Heston ``nu → 0`` (deterministic variance at v0): Heston collapses to
  BSM with diffusion ``sqrt(v0)``, giving BSM + VG.

References
----------
* Madan, D. B., Carr, P. & Chang, E. C. (1998), "The Variance Gamma
  Process and Option Pricing", *European Finance Review*, 2, 79-105.
* Heston, S. (1993), "A Closed-Form Solution for Options with Stochastic
  Volatility", *Review of Financial Studies*, 6(2), 327-343.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press. (Section 15.4 for SV + Lévy construction.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec
from .heston import HestonParams, heston_cf


@dataclass(frozen=True)
class HestonVGParams(ModelSpec):
    """Heston-VG SVJ parameters.

    Heston block
    ------------
    kappa : float  Mean-reversion speed of the variance process. ``> 0``.
    theta : float  Long-run mean of the variance process. ``> 0``.
    nu : float     Vol-of-vol. Must be ``> 0`` (PyFENG requirement).
    rho : float    Correlation between price and variance Brownians. ``[-1, 1]``.
    v0 : float     Initial instantaneous variance. ``> 0``.

    VG jump block
    -------------
    vg_sigma : float  Diffusion coefficient of the BM in the VG subordination.
                      Must be ``> 0``.
    vg_nu : float     Variance of the Gamma clock per unit time. Must be ``> 0``.
    vg_theta : float  Drift of the BM in the VG subordination (skew parameter).

    Existence condition
    -------------------
    ``1 - vg_theta * vg_nu - vg_sigma^2 * vg_nu / 2 > 0``
    (required so that ``E[exp(J_T)]`` is finite and the martingale
    correction ``psi_VG(-i)`` is real).
    """

    kappa: float
    theta: float
    nu: float
    rho: float
    v0: float
    vg_sigma: float
    vg_nu: float
    vg_theta: float

    def __init__(
        self,
        kappa: float,
        theta: float,
        nu: float,
        rho: float,
        v0: float,
        vg_sigma: float,
        vg_nu: float,
        vg_theta: float,
    ):
        if not (np.isfinite(kappa) and kappa > 0):
            raise ValueError(f"HestonVGParams: kappa must be > 0; got {kappa}")
        if not (np.isfinite(theta) and theta > 0):
            raise ValueError(f"HestonVGParams: theta must be > 0; got {theta}")
        if not (np.isfinite(nu) and nu > 0):
            raise ValueError(f"HestonVGParams: nu must be > 0 (PyFENG); got {nu}")
        if not (np.isfinite(rho) and -1.0 <= rho <= 1.0):
            raise ValueError(f"HestonVGParams: rho must be in [-1,1]; got {rho}")
        if not (np.isfinite(v0) and v0 > 0):
            raise ValueError(f"HestonVGParams: v0 must be > 0; got {v0}")
        if not (np.isfinite(vg_sigma) and vg_sigma > 0):
            raise ValueError(f"HestonVGParams: vg_sigma must be > 0; got {vg_sigma}")
        if not (np.isfinite(vg_nu) and vg_nu > 0):
            raise ValueError(f"HestonVGParams: vg_nu must be > 0; got {vg_nu}")
        if not np.isfinite(vg_theta):
            raise ValueError(f"HestonVGParams: vg_theta must be finite; got {vg_theta}")
        cond = 1.0 - vg_theta * vg_nu - 0.5 * vg_sigma**2 * vg_nu
        if cond <= 0.0:
            raise ValueError(
                f"HestonVGParams: VG existence condition violated: "
                f"1 - vg_theta*vg_nu - vg_sigma^2*vg_nu/2 = {cond:.6g} <= 0"
            )
        object.__setattr__(self, "name", "heston_vg")
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "vg_sigma", vg_sigma)
        object.__setattr__(self, "vg_nu", vg_nu)
        object.__setattr__(self, "vg_theta", vg_theta)

    @property
    def heston_params(self) -> HestonParams:
        return HestonParams(
            kappa=self.kappa,
            theta=self.theta,
            nu=self.nu,
            rho=self.rho,
            v0=self.v0,
        )


# ---------------------------------------------------------------------------
# VG Lévy exponent (per-unit-time log-CF)  -  (sigma, nu, theta) convention
# ---------------------------------------------------------------------------


def _vg_levy_exponent_cm98(u: np.ndarray, sigma: float, nu: float, theta: float) -> np.ndarray:
    """VG Lévy exponent in the Madan-Carr-Chang (1998) (sigma, nu, theta) convention.

    psi_VG(u) = -(1/nu) * log(1 - i*theta*nu*u + sigma^2*nu*u^2/2)

    Valid when the argument of the log has strictly positive real part, which
    holds on the real axis whenever the existence condition is satisfied.
    """
    u_c = np.asarray(u, dtype=np.complex128)
    arg = 1.0 - 1j * theta * nu * u_c + 0.5 * sigma**2 * nu * u_c**2
    return -(1.0 / nu) * np.log(arg)


# ---------------------------------------------------------------------------
# Heston-VG characteristic function
# ---------------------------------------------------------------------------


def heston_vg_cf(u: np.ndarray, fwd: ForwardSpec, p: HestonVGParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under Heston-VG.

    Factorised as phi_H(u) * phi_VG_jump(u) under SV-Lévy independence.

    The VG jump component is martingale-corrected:
        phi_VG_jump(u) = exp(T * (psi_VG(u) - iu * psi_VG(-i)))
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T

    # Heston component (PyFENG-backed, log-forward convention)
    phi_H = heston_cf(u_c, fwd, p.heston_params)

    # VG Lévy exponents
    psi_u = _vg_levy_exponent_cm98(u_c, p.vg_sigma, p.vg_nu, p.vg_theta)
    psi_mi = _vg_levy_exponent_cm98(np.array([-1j]), p.vg_sigma, p.vg_nu, p.vg_theta)[0]

    # Martingale-corrected VG jump CF
    phi_VG_jump = np.exp(T * (psi_u - 1j * u_c * psi_mi))

    return phi_H * phi_VG_jump


# ---------------------------------------------------------------------------
# Cumulants  -  numerical Cauchy integral
# ---------------------------------------------------------------------------


def heston_vg_cumulants(fwd: ForwardSpec, p: HestonVGParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of X_T under Heston-VG.

    Computed numerically via a 64-point Cauchy FFT on a small circle
    around u=0 (same technique as Heston-CGMY and VGSA).
    """
    from ..utils.cumulants import cumulants_from_cf

    c = cumulants_from_cf(lambda u: heston_vg_cf(u, fwd, p), order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
