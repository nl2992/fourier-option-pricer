from __future__ import annotations
import numpy as np
from dataclasses import dataclass
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


def heston_cf_form2(u: np.ndarray, fwd: ForwardSpec, p: HestonParams) -> np.ndarray:
    """Stable Heston CF for X_T = log(S_T/F0), in Albrecher "Formulation 2".

    phi(u) = exp( C(u,T) + D(u,T)*v0 ),  with
      b = kappa - i*rho*nu*u
      d = sqrt(b^2 + nu^2*(u^2 + i*u))
      g = (b - d) / (b + d)
      D = (b - d)/nu^2 * (1 - exp(-d*T))/(1 - g*exp(-d*T))
      C = (kappa*theta/nu^2) * ((b - d)*T - 2*log((1 - g*exp(-d*T))/(1 - g)))

    The log-return CF does not contain the forward drift factor exp(i*u*log F0)
    — that is applied by the pricer when it needs the log-price CF.
    """
    u = np.asarray(u, dtype=np.complex128)
    T = fwd.T
    kappa, theta, nu, rho, v0 = p.kappa, p.theta, p.nu, p.rho, p.v0

    b = kappa - 1j * rho * nu * u
    d = np.sqrt(b * b + nu * nu * (u * u + 1j * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)

    D = (b - d) / (nu * nu) * (1.0 - e) / (1.0 - g * e)
    C = (kappa * theta / (nu * nu)) * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))

    return np.exp(C + D * v0)


def heston_cumulants(fwd: ForwardSpec, p: HestonParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of X_T = log(S_T/F0) for COS truncation.

    Computed numerically from the CF via the Cauchy integral (see
    ``foureng.utils.cumulants.cumulants_from_cf``). This is more reliable
    than the FO2008 Table 11 analytic c2 formula (which we verified deviates
    from the CF-derived value by ~0.5% on Lewis and ~2% on FO2008 Table 1
    parameters — a paper typo or transcription error). Two completely
    independent numerical schemes (5-point FD on the imaginary axis and the
    FFT-on-circle implemented here) agree to 6+ digits, so we take the
    CF-derived values as authoritative.

    c4 is returned as-is (signed); the COS truncation rule takes sqrt(|c4|).
    """
    from ..utils.cumulants import cumulants_from_cf

    phi = lambda u: heston_cf_form2(u, fwd, p)
    c = cumulants_from_cf(phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
