"""Bates (1996) SV + lognormal-jump model — SVJ.

Bates = Heston diffusion + Merton (1976) compound-Poisson jumps with
log-normal jump sizes, all under the project's log-forward convention
``X_T = log(S_T/F_0)``. The CF factorises:

    phi_B(u) = phi_H(u) * phi_jump(u)

where ``phi_H`` is Heston's CF (PyFENG-backed via :mod:`.heston`) and
``phi_jump`` is the usual martingale-corrected compound-Poisson factor
— the same pattern already used for Kou in :mod:`.kou`.

References
----------
* Bates, D. (1996), "Jumps and Stochastic Volatility: Exchange Rate
  Processes Implicit in Deutsche Mark Options".
* Merton, R. (1976), "Option pricing when underlying stock returns are
  discontinuous".
* Fang & Oosterlee (2008) for the COS truncation rule this cumulants
  function feeds.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec
from .heston import HestonParams, heston_cf, heston_cumulants


@dataclass(frozen=True)
class BatesParams(ModelSpec):
    """Bates (SVJ) parameters — Heston block + log-normal jump block.

    Heston block
    ------------
    ``kappa, theta, nu, rho, v0`` — same meaning as :class:`HestonParams`.
    Feller condition: ``2*kappa*theta >= nu^2``.

    Jump block (Merton, 1976)
    -------------------------
    ``lam_j`` : Poisson intensity (jumps per unit time).
    ``mu_j``  : mean of the log-jump size.
    ``sigma_j`` : std. dev. of the log-jump size.

    The jump-size variable is ``Y ~ Normal(mu_j, sigma_j^2)`` and the
    martingale correction ``omega_j = -lam_j * zeta`` with
    ``zeta = exp(mu_j + 0.5*sigma_j^2) - 1`` is applied inside
    :func:`bates_cf` / :func:`bates_cumulants`.
    """

    # Heston part
    kappa: float
    theta: float
    nu: float
    rho: float
    v0: float
    # Jump part
    lam_j: float
    mu_j: float
    sigma_j: float

    def __init__(
        self,
        kappa: float,
        theta: float,
        nu: float,
        rho: float,
        v0: float,
        lam_j: float,
        mu_j: float,
        sigma_j: float,
    ):
        object.__setattr__(self, "name", "bates")
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "lam_j", lam_j)
        object.__setattr__(self, "mu_j", mu_j)
        object.__setattr__(self, "sigma_j", sigma_j)

    @property
    def heston_params(self) -> HestonParams:
        """View of the Heston block for reuse by :func:`heston_cf` etc."""
        return HestonParams(
            kappa=self.kappa,
            theta=self.theta,
            nu=self.nu,
            rho=self.rho,
            v0=self.v0,
        )


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------

def bates_cf(u: np.ndarray, fwd: ForwardSpec, p: BatesParams) -> np.ndarray:
    """Bates CF of X_T = log(S_T/F_0).

        phi_B(u) = phi_H(u) * phi_jump(u)

    with
        phi_H(u)    = :func:`heston_cf`  (PyFENG-backed Heston),
        zeta        = exp(mu_j + 0.5*sigma_j^2) - 1,
        omega_j     = -lam_j * zeta,
        phi_Y(u)    = exp(i*u*mu_j - 0.5*sigma_j^2*u^2),
        phi_jump(u) = exp( T * ( i*u*omega_j + lam_j*(phi_Y(u) - 1) ) ).

    The compensator makes ``E[exp(X_T)] = 1`` so ``F_0`` remains the
    martingale of the discounted price, consistent with every other CF
    in this project.
    """
    T = fwd.T
    u_c = np.asarray(u, dtype=np.complex128)

    # Continuous part — delegates to PyFENG via heston_cf.
    phi_H = heston_cf(u_c, fwd, p.heston_params)

    # Jump part — Merton-style compound Poisson, log-forward convention.
    lam_j, mu_j, sig_j = p.lam_j, p.mu_j, p.sigma_j
    zeta = np.exp(mu_j + 0.5 * sig_j * sig_j) - 1.0
    omega_j = -lam_j * zeta
    phi_Y = np.exp(1j * u_c * mu_j - 0.5 * sig_j * sig_j * u_c * u_c)
    phi_jump = np.exp(T * (1j * u_c * omega_j + lam_j * (phi_Y - 1.0)))

    return phi_H * phi_jump


# ---------------------------------------------------------------------------
# Cumulants (independent components → cumulants add)
# ---------------------------------------------------------------------------

def bates_cumulants(fwd: ForwardSpec, p: BatesParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of X_T under Bates.

    Heston and jump contributions are independent, so cumulants add:

        c_n^B = c_n^H + c_n^jump.

    Compound-Poisson contribution with Y ~ Normal(mu_j, sigma_j^2):

        E[Y]    = mu_j
        E[Y^2]  = mu_j^2 + sigma_j^2
        E[Y^4]  = mu_j^4 + 6*mu_j^2*sigma_j^2 + 3*sigma_j^4

        c1_j = lam_j * T * (mu_j - zeta)       [includes omega_j drift]
        c2_j = lam_j * T * E[Y^2]
        c4_j = lam_j * T * E[Y^4].

    The Heston block is handed off to :func:`heston_cumulants` (which
    reads ``c2`` numerically from the CF and returns ``c4 = 0`` by the
    conservative convention used by :func:`cos_auto_grid`).
    """
    T = fwd.T
    lam_j, mu_j, sig_j = p.lam_j, p.mu_j, p.sigma_j

    # Diffusion cumulants from the Heston CF (shared convention).
    c1_H, c2_H, c4_H = heston_cumulants(fwd, p.heston_params)

    # Jump raw moments and compensator.
    zeta = float(np.exp(mu_j + 0.5 * sig_j * sig_j) - 1.0)
    EY2 = mu_j * mu_j + sig_j * sig_j
    EY4 = mu_j ** 4 + 6.0 * mu_j * mu_j * sig_j * sig_j + 3.0 * sig_j ** 4

    c1_j = lam_j * T * (mu_j - zeta)
    c2_j = lam_j * T * EY2
    c4_j = lam_j * T * EY4

    return float(c1_H + c1_j), float(c2_H + c2_j), float(c4_H + c4_j)
