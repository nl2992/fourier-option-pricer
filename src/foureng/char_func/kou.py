from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class KouParams(ModelSpec):
    """Kou (2002) double-exponential jump-diffusion parameters.

    sigma : diffusion vol
    lam   : Poisson intensity
    p     : prob. of up-jump (in (0,1))
    eta1  : up-jump rate   (eta1 > 1 for finite mean)
    eta2  : down-jump rate (eta2 > 0)
    """
    sigma: float
    lam: float
    p: float
    eta1: float
    eta2: float

    def __init__(self, sigma: float, lam: float, p: float, eta1: float, eta2: float):
        object.__setattr__(self, "name", "kou")
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "lam", lam)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "eta1", eta1)
        object.__setattr__(self, "eta2", eta2)


def kou_cf(u: np.ndarray, fwd: ForwardSpec, p: KouParams) -> np.ndarray:
    """CF of X_T = log(S_T/F0) under Kou.

    Martingale correction:
        zeta = p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1
        omega = -0.5*sigma^2 - lam*zeta
    so that E[exp(X_T)] = 1.

        phi(u) = exp( T * [ i*u*omega - 0.5*sigma^2*u^2
                            + lam*( p*eta1/(eta1 - i*u) + (1-p)*eta2/(eta2 + i*u) - 1 ) ] )
    """
    if p.eta1 <= 1.0:
        raise ValueError(f"Kou requires eta1 > 1 for finite jump mean; got {p.eta1}")
    if p.eta2 <= 0.0:
        raise ValueError(f"Kou requires eta2 > 0; got {p.eta2}")

    T = fwd.T
    sigma, lam, pp, eta1, eta2 = p.sigma, p.lam, p.p, p.eta1, p.eta2

    zeta = pp * eta1 / (eta1 - 1.0) + (1.0 - pp) * eta2 / (eta2 + 1.0) - 1.0
    omega = -0.5 * sigma * sigma - lam * zeta

    u = np.asarray(u, dtype=np.complex128)
    jump = pp * eta1 / (eta1 - 1j * u) + (1.0 - pp) * eta2 / (eta2 + 1j * u) - 1.0
    expo = 1j * u * omega - 0.5 * sigma * sigma * (u * u) + lam * jump
    return np.exp(T * expo)


def kou_cumulants(fwd: ForwardSpec, p: KouParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of log(S_T/F0) under Kou.

    Compound Poisson part has kappa_n = lam*T*E[Y^n] where Y ~ double-exp:
        E[Y]   = p/eta1 - (1-p)/eta2
        E[Y^2] = 2p/eta1^2 + 2(1-p)/eta2^2
        E[Y^4] = 24p/eta1^4 + 24(1-p)/eta2^4
    Added to Brownian part (contributes only c2 = sigma^2*T) and martingale drift.
    """
    T = fwd.T
    sigma, lam, pp, eta1, eta2 = p.sigma, p.lam, p.p, p.eta1, p.eta2

    zeta = pp * eta1 / (eta1 - 1.0) + (1.0 - pp) * eta2 / (eta2 + 1.0) - 1.0
    omega = -0.5 * sigma * sigma - lam * zeta

    EY  = pp / eta1 - (1.0 - pp) / eta2
    EY2 = 2.0 * pp / eta1 ** 2 + 2.0 * (1.0 - pp) / eta2 ** 2
    EY4 = 24.0 * pp / eta1 ** 4 + 24.0 * (1.0 - pp) / eta2 ** 4

    c1 = T * (omega + lam * EY)
    c2 = T * (sigma * sigma + lam * EY2)
    c4 = T * lam * EY4
    return float(c1), float(c2), float(c4)
