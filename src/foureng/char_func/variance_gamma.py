from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class VGParams(ModelSpec):
    """Variance Gamma parameters (Madan, Carr, Chang 1998)."""
    sigma: float
    nu: float
    theta: float

    def __init__(self, sigma: float, nu: float, theta: float):
        object.__setattr__(self, "name", "variance_gamma")
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "theta", theta)


def vg_cf(u: np.ndarray, fwd: ForwardSpec, p: VGParams) -> np.ndarray:
    """CF of X_T = log(S_T/F0) under VG.

    The martingale correction omega makes E[exp(X_T)] = 1:
        omega = (1/nu) * log(1 - theta*nu - 0.5*sigma^2*nu)
    Requires  1 - theta*nu - 0.5*sigma^2*nu > 0 for the log to be real.

        phi(u) = exp(i*u*omega*T) * (1 - i*theta*nu*u + 0.5*sigma^2*nu*u^2)^(-T/nu)
    """
    sigma, nu, theta = p.sigma, p.nu, p.theta
    T = fwd.T

    cond = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
    if cond <= 0.0:
        raise ValueError(
            f"VG martingale condition violated: 1 - theta*nu - 0.5*sigma^2*nu = {cond:.4g}"
        )
    omega = np.log(cond) / nu

    u = np.asarray(u, dtype=np.complex128)
    drift = np.exp(1j * u * omega * T)
    base = 1.0 - 1j * theta * nu * u + 0.5 * sigma * sigma * nu * (u * u)
    return drift * base ** (-T / nu)


def vg_cumulants(fwd: ForwardSpec, p: VGParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of log(S_T/F0) under VG (FO 2008 Table 10)."""
    sigma, nu, theta = p.sigma, p.nu, p.theta
    T = fwd.T
    cond = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
    omega = np.log(cond) / nu
    c1 = (theta + omega) * T
    c2 = (sigma * sigma + nu * theta * theta) * T
    c4 = 3.0 * (sigma ** 4 * nu
                + 2.0 * theta ** 4 * nu ** 3
                + 4.0 * sigma * sigma * theta * theta * nu * nu) * T
    return float(c1), float(c2), float(c4)
