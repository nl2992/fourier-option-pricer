"""Heston-NIG SVJ model  -  Heston SV + Normal Inverse Gaussian Lévy jumps.

Normal Inverse Gaussian (NIG) is a pure-jump, infinite-activity Lévy model
built via an inverse-Gaussian time subordination of drifted Brownian motion.
Combined with Heston's stochastic-volatility diffusion the CF factorises under
independence:

    phi(u) = phi_H(u) * phi_NIG_jump(u)

where ``phi_H`` is the Heston CF (PyFENG-backed via :func:`heston_cf`) and
``phi_NIG_jump`` is the martingale-corrected NIG pure-jump component.

NIG Lévy exponent
-----------------
For the NIG process parameterised by ``(sigma, nu, theta)`` the Lévy exponent
per unit time is:

    psi_NIG(u) = (1/nu) * (1 - sqrt(1 - 2i*theta*nu*u + sigma^2*nu*u^2))

At ``u = -i`` this evaluates to the real scalar:

    psi_NIG(-i) = (1/nu) * (1 - sqrt(1 - 2*theta*nu - sigma^2*nu))

which is exactly the martingale correction ``-mu/nu`` of the standalone NIG model
(same existence condition: ``1 - 2*theta*nu - sigma^2*nu > 0``).

Martingale correction
---------------------
The NIG jump component is compensated identically to Heston-CGMY:

    phi_NIG_jump(u) = exp( T * (psi_NIG(u) - iu * psi_NIG(-i)) )

so phi_NIG_jump(-i) = exp(T * 0) = 1, preserving the log-forward martingale.

Reduction to Heston
-------------------
Setting ``sigma → 0`` and ``theta → 0`` simultaneously drives the NIG jump
contribution to zero (psi_NIG(u) → 0), recovering pure Heston. In practice
verify by checking the ratio phi_HN / phi_H → 1 pointwise.

Reduction to NIG
----------------
Setting Heston ``nu → 0`` makes the variance deterministic at v0, collapsing
Heston to BSM (phi_H → phi_BSM(sigma=sqrt(v0))). The result is a BSM + NIG
model, NOT standalone NIG; standalone NIG is recovered only if v0 → 0 as well
(which is not admitted since v0 > 0 is required by HestonParams).

References
----------
* Heston, S. (1993), "A Closed-Form Solution for Options with Stochastic
  Volatility", *Review of Financial Studies*, 6(2), 327–343.
* Barndorff-Nielsen, O. E. (1997), "Normal inverse Gaussian distributions
  and stochastic volatility modelling", *Scand. J. Statist.*, 24, 1–13.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press. (Section 4.4 and Chapter 15 for the SV+Lévy construction.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec
from .heston import HestonParams, heston_cf


@dataclass(frozen=True)
class HestonNIGParams(ModelSpec):
    """Heston-NIG SVJ parameters.

    Heston block (stochastic variance, same meaning as :class:`HestonParams`)
    --------------------------------------------------------------------------
    kappa : float
        Mean-reversion speed of the variance process. Must be ``> 0``.
    theta : float
        Long-run mean of the variance process. Must be ``> 0``.
    nu : float
        Vol-of-vol (coefficient on the ``sqrt(v) dW_v`` term). Must be ``>= 0``.
    rho : float
        Correlation between spot and variance Brownian motions. Must be in
        ``(-1, 1)``.
    v0 : float
        Initial instantaneous variance. Must be ``> 0``.

    NIG jump block (pure-jump infinite-activity Lévy component)
    -----------------------------------------------------------
    nig_sigma : float
        Diffusion scale of the NIG subordinated BM. Must be ``> 0``.
    nig_nu : float
        Inverse-Gaussian time-subordinator rate. Controls excess kurtosis
        (heavier tails for larger ``nig_nu``). Must be ``> 0``.
    nig_theta : float
        Drift of the NIG subordinated BM. Controls skew; ``nig_theta < 0``
        produces the standard negative equity skew.

    Existence condition
    -------------------
    ``1 - 2*nig_theta*nig_nu - nig_sigma^2*nig_nu > 0`` must hold so that the
    NIG martingale correction ``psi_NIG(-i)`` is real-valued.
    """

    # Heston part
    kappa: float
    theta: float
    nu: float
    rho: float
    v0: float
    # NIG jump part
    nig_sigma: float
    nig_nu: float
    nig_theta: float

    def __init__(
        self,
        kappa: float,
        theta: float,
        nu: float,
        rho: float,
        v0: float,
        nig_sigma: float,
        nig_nu: float,
        nig_theta: float,
    ):
        if not (np.isfinite(kappa) and kappa > 0):
            raise ValueError(f"HestonNIGParams: kappa must be > 0; got {kappa}")
        if not (np.isfinite(theta) and theta > 0):
            raise ValueError(f"HestonNIGParams: theta must be > 0; got {theta}")
        if not (np.isfinite(nu) and nu > 0):
            raise ValueError(f"HestonNIGParams: nu must be > 0 (PyFENG HestonFft requires vov > 0); got {nu}")
        if not (np.isfinite(rho) and -1.0 < rho < 1.0):
            raise ValueError(f"HestonNIGParams: rho must be in (-1, 1); got {rho}")
        if not (np.isfinite(v0) and v0 > 0):
            raise ValueError(f"HestonNIGParams: v0 must be > 0; got {v0}")
        if not (np.isfinite(nig_sigma) and nig_sigma > 0):
            raise ValueError(f"HestonNIGParams: nig_sigma must be > 0; got {nig_sigma}")
        if not (np.isfinite(nig_nu) and nig_nu > 0):
            raise ValueError(f"HestonNIGParams: nig_nu must be > 0; got {nig_nu}")
        if not np.isfinite(nig_theta):
            raise ValueError(f"HestonNIGParams: nig_theta must be finite; got {nig_theta}")
        existence = 1.0 - 2.0 * nig_theta * nig_nu - nig_sigma**2 * nig_nu
        if existence <= 0:
            raise ValueError(
                f"HestonNIGParams: NIG existence condition "
                f"1 - 2*nig_theta*nig_nu - nig_sigma^2*nig_nu > 0 violated "
                f"(got {existence:.6g}); martingale correction would be complex."
            )
        object.__setattr__(self, "name", "heston_nig")
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "nig_sigma", nig_sigma)
        object.__setattr__(self, "nig_nu", nig_nu)
        object.__setattr__(self, "nig_theta", nig_theta)

    @property
    def heston_params(self) -> HestonParams:
        """View of the Heston block for reuse in CF and cumulant computations."""
        return HestonParams(
            kappa=self.kappa,
            theta=self.theta,
            nu=self.nu,
            rho=self.rho,
            v0=self.v0,
        )


# ---------------------------------------------------------------------------
# NIG Lévy exponent helper
# ---------------------------------------------------------------------------


def _nig_levy_exponent(u: np.ndarray, nig_sigma: float, nig_nu: float, nig_theta: float) -> np.ndarray:
    """NIG Lévy exponent per unit time (no martingale correction).

        psi_NIG(u) = (1/nu) * (1 - sqrt(1 - 2i*theta*nu*u + sigma^2*nu*u^2))

    Derived from the NIG MGF M(s) = exp(T/nu * [mu*s + 1 - sqrt(1 - 2*theta*nu*s - sigma^2*nu*s^2)])
    by evaluating at s = iu and extracting the "raw" component without drift mu.

    At u = -i:  psi_NIG(-i) = (1/nu) * (1 - sqrt(1 - 2*theta*nu - sigma^2*nu))
    which is real and negative for theta <= 0, and is the negation of the NIG
    drift correction mu/nu (where mu = -1 + sqrt(1 - 2*theta*nu - sigma^2*nu)).
    """
    u_c = np.asarray(u, dtype=np.complex128)
    disc = 1.0 - 2j * nig_theta * nig_nu * u_c + nig_sigma**2 * nig_nu * u_c**2
    return (1.0 / nig_nu) * (1.0 - np.sqrt(disc))


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------


def heston_nig_cf(u: np.ndarray, fwd: ForwardSpec, p: HestonNIGParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under Heston-NIG SVJ.

    Independent factorisation:

        phi(u) = phi_H(u) * exp( T * ( psi_NIG(u) - iu * psi_NIG(-i) ) )

    where ``phi_H`` is the PyFENG-backed Heston CF and ``psi_NIG`` is the NIG
    Lévy exponent.  The ``-iu * psi_NIG(-i)`` term is the compensator that
    ensures ``phi(-i) = 1`` (log-forward martingale condition).

    Parameters
    ----------
    u : array_like (real or complex)
        Frequency grid.
    fwd : ForwardSpec
    p : HestonNIGParams

    Returns
    -------
    np.ndarray (complex128)
    """
    T = fwd.T
    u_c = np.asarray(u, dtype=np.complex128)

    # Heston (stochastic-vol) component via PyFENG
    phi_H = heston_cf(u_c, fwd, p.heston_params)

    # NIG pure-jump component with martingale correction
    psi_u = _nig_levy_exponent(u_c, p.nig_sigma, p.nig_nu, p.nig_theta)
    psi_minus_i = _nig_levy_exponent(np.array([-1j]), p.nig_sigma, p.nig_nu, p.nig_theta)[0]
    phi_NIG_jump = np.exp(T * (psi_u - 1j * u_c * psi_minus_i))

    return phi_H * phi_NIG_jump


# ---------------------------------------------------------------------------
# Cumulants for COS auto-grid (numerical Cauchy integral)
# ---------------------------------------------------------------------------


def heston_nig_cumulants(fwd: ForwardSpec, p: HestonNIGParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of X_T under Heston-NIG.

    Independent components add: cumulants of the Heston diffusion plus
    cumulants of the NIG jump process.  We delegate to :func:`heston_cumulants`
    for the Heston block and compute the NIG cumulants analytically.

    NIG cumulant generating function per unit time (without drift correction):
        L_NIG(s) = (1/nu) * (1 - sqrt(1 - 2*theta*nu*s - sigma^2*nu*s^2))

    Differentiating at s = 0:
        c1_NIG = T * (L_NIG'(0) - psi_NIG(-i))   [drift-corrected mean]
        c2_NIG = T * L_NIG''(0)
        c4_NIG = T * L_NIG''''(0)

    where:
        L_NIG'(0)  = theta + sigma^2 / (2*sqrt(...)) ... evaluated analytically.

    Rather than carrying the full analytic cumulant formulae (which become
    noisy near the boundary of the existence region), we use the same
    FFT-on-circle Cauchy-integral method used for Heston-CGMY.
    """
    from ..utils.cumulants import cumulants_from_cf

    def _phi(u):
        return heston_nig_cf(u, fwd, p)

    c = cumulants_from_cf(_phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
