"""Heston-Kou SVJ model — Heston SV + Kou double-exponential jumps.

Heston's stochastic-volatility diffusion combined with a compound-Poisson
jump whose size follows Kou's (2002) double-exponential law. The CF
factorises under independence:

    phi(u) = phi_H(u) * phi_jump(u)

where ``phi_H`` is Heston (PyFENG-backed via :func:`heston_cf`) and
``phi_jump`` is the log-forward, martingale-corrected compound-Poisson
factor — the same construction already used in :mod:`.kou` and
:mod:`.bates`.

References
----------
* Heston, S. (1993).
* Kou, S. (2002), "A Jump-Diffusion Model for Option Pricing".
* Fang & Oosterlee (2008) — COS truncation rule consumes the cumulants
  returned here.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec
from .heston import HestonParams, heston_cf, heston_cumulants


@dataclass(frozen=True)
class HestonKouParams(ModelSpec):
    """Heston-Kou SVJ parameters.

    Heston block
    ------------
    ``kappa, theta, nu, rho, v0`` — same meaning as :class:`HestonParams`.
    Feller condition: ``2*kappa*theta >= nu^2``.

    Kou jump block (double-exponential)
    -----------------------------------
    ``lam_j`` : Poisson intensity (jumps per unit time).
    ``p_j``   : probability of an up-jump, in (0, 1).
    ``eta1``  : up-jump rate; requires ``eta1 > 1`` for a finite jump mean
                and a valid martingale correction.
    ``eta2``  : down-jump rate; requires ``eta2 > 0``.
    """

    # Heston part
    kappa: float
    theta: float
    nu: float
    rho: float
    v0: float
    # Kou jump part
    lam_j: float
    p_j: float
    eta1: float
    eta2: float

    def __init__(
        self,
        kappa: float,
        theta: float,
        nu: float,
        rho: float,
        v0: float,
        lam_j: float,
        p_j: float,
        eta1: float,
        eta2: float,
    ):
        object.__setattr__(self, "name", "heston_kou")
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "lam_j", lam_j)
        object.__setattr__(self, "p_j", p_j)
        object.__setattr__(self, "eta1", eta1)
        object.__setattr__(self, "eta2", eta2)

    @property
    def heston_params(self) -> HestonParams:
        """View of the Heston block for reuse."""
        return HestonParams(
            kappa=self.kappa,
            theta=self.theta,
            nu=self.nu,
            rho=self.rho,
            v0=self.v0,
        )


def heston_kou_cf(u: np.ndarray, fwd: ForwardSpec, p: HestonKouParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under Heston-Kou SVJ.

        phi(u)       = phi_H(u) * phi_jump(u)
        zeta         = p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1
        omega_j      = -lam_j * zeta                              (drift)
        phi_Y(u)     = p*eta1/(eta1 - i u) + (1-p)*eta2/(eta2 + i u)
        phi_jump(u)  = exp( T * ( i u omega_j + lam_j (phi_Y(u) - 1) ) )

    ``phi_H`` comes from the PyFENG-backed :func:`heston_cf`. The
    compensator choice enforces ``E[exp(X_T)] = 1`` so ``F_0`` remains
    the martingale of the discounted price — the same convention used
    by every CF in this project.
    """
    if p.eta1 <= 1.0:
        raise ValueError(f"Heston-Kou requires eta1 > 1; got {p.eta1}")
    if p.eta2 <= 0.0:
        raise ValueError(f"Heston-Kou requires eta2 > 0; got {p.eta2}")

    T = fwd.T
    u_c = np.asarray(u, dtype=np.complex128)

    # Continuous part (Heston via PyFENG)
    phi_H = heston_cf(u_c, fwd, p.heston_params)

    # Jump part
    pp, eta1, eta2, lam = p.p_j, p.eta1, p.eta2, p.lam_j
    zeta = pp * eta1 / (eta1 - 1.0) + (1.0 - pp) * eta2 / (eta2 + 1.0) - 1.0
    omega_j = -lam * zeta
    phi_Y = pp * eta1 / (eta1 - 1j * u_c) + (1.0 - pp) * eta2 / (eta2 + 1j * u_c)
    phi_jump = np.exp(T * (1j * u_c * omega_j + lam * (phi_Y - 1.0)))

    return phi_H * phi_jump


def heston_kou_cumulants(
    fwd: ForwardSpec, p: HestonKouParams
) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of X_T under Heston-Kou.

    Independent components → cumulants add. Double-exponential raw
    moments (Y has density ``p*eta1*exp(-eta1*y) 1_{y>0} +
    (1-p)*eta2*exp(eta2*y) 1_{y<0}``):

        E[Y]   = p/eta1 - (1-p)/eta2
        E[Y^2] = 2p/eta1^2 + 2(1-p)/eta2^2
        E[Y^4] = 24p/eta1^4 + 24(1-p)/eta2^4

    Compound-Poisson contributions:

        c1_j = lam_j * T * (E[Y] - zeta)   [includes the omega_j drift]
        c2_j = lam_j * T * E[Y^2]
        c4_j = lam_j * T * E[Y^4]

    Heston block delegated to :func:`heston_cumulants`.
    """
    T = fwd.T
    pp, eta1, eta2, lam = p.p_j, p.eta1, p.eta2, p.lam_j

    c1_H, c2_H, c4_H = heston_cumulants(fwd, p.heston_params)

    zeta = pp * eta1 / (eta1 - 1.0) + (1.0 - pp) * eta2 / (eta2 + 1.0) - 1.0
    EY = pp / eta1 - (1.0 - pp) / eta2
    EY2 = 2.0 * pp / eta1 ** 2 + 2.0 * (1.0 - pp) / eta2 ** 2
    EY4 = 24.0 * pp / eta1 ** 4 + 24.0 * (1.0 - pp) / eta2 ** 4

    c1_j = lam * T * (EY - zeta)
    c2_j = lam * T * EY2
    c4_j = lam * T * EY4

    return float(c1_H + c1_j), float(c2_H + c2_j), float(c4_H + c4_j)
