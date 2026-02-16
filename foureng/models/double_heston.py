"""Double Heston (two-factor stochastic volatility) characteristic function.

The Double Heston model runs two *independent* Heston variance factors in
parallel. The log-forward CF factorises as a product of two single-Heston
CFs, which in turn factorises in the exponent:

    dS/S   = sqrt(v1_t + v2_t) dW0
    dv1    = kappa1*(theta1 - v1) dt + nu1*sqrt(v1) dW1,  <dW0,dW1> = rho1 dt
    dv2    = kappa2*(theta2 - v2) dt + nu2*sqrt(v2) dW2,  <dW0,dW2> = rho2 dt
             <dW1, dW2> = 0  (the two variance factors are independent)

Because the two variance Brownian motions are independent of each other
(though each is correlated with the spot), the CF factorises:

    phi_DH(u; T) = exp(C1 + D1*v01 + C2 + D2*v02)
                 = phi_H1(u; T) * phi_H2(u; T)

where (C_i, D_i) = heston_riccati_cd(u, T, factor_i_params).

This decomposition is exact and requires no approximation beyond the
standard Heston CF assumptions.

Parameterisation
----------------
We follow the project convention of using ``nu`` for vol-of-vol (some
sources use ``sigma``, ``gamma``, or ``vov`` — all equivalent). The
``v0i`` parameters are initial *variances* (not volatilities).

Paper benchmark
---------------
Kelly (2025) reports Double Heston prices for:

    S0=100, r=0.1, T=1, strikes=[60, 80, 100, 110, 130]
    kappa1=0.9, theta1=0.1, nu1=0.1, v01=0.36, rho1=-0.5
    kappa2=1.2, theta2=0.15, nu2=0.2, v02=0.49, rho2=-0.5

    calls = [52.7623, 42.2321, 33.9625, 30.5234, 24.7718]
    puts  = [7.0526,  14.6191, 24.4463, 30.0556, 42.4007]

Tolerance against this benchmark: 5e-4 (source rounds to 4 decimals).

References
----------
* Christoffersen, P., Heston, S. & Jacobs, K. (2009), "The Shape and
  Term Structure of the Index Option Smirk: Why Multifactor Stochastic
  Volatility Models Work so Well", *Management Science*, 55(12), 1914–1932.
* Albrecher, H. et al. (2007), "The Little Heston Trap",
  *Wilmott Magazine*, Jan/Feb, 83–92.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec
from .heston import HestonParams, heston_riccati_cd


@dataclass(frozen=True)
class DoubleHestonParams(ModelSpec):
    """Double Heston two-factor SV parameters.

    Factor 1
    --------
    kappa1 : float
        Mean-reversion speed of variance factor 1. Must be ``> 0``.
    theta1 : float
        Long-run mean of variance factor 1. Must be ``>= 0``; ``0``
        reduces the factor to a pure deterministic decay.
    nu1 : float
        Vol-of-vol for factor 1 (coefficient on the ``sqrt(v1) dW1``
        diffusion). Must be ``> 0``.
    rho1 : float
        Correlation between spot and variance factor 1. Must be in
        ``(-1, 1)``.
    v01 : float
        Initial variance factor 1. Must be ``>= 0``; ``0`` kills the
        factor entirely.

    Factor 2
    --------
    kappa2, theta2, nu2, rho2, v02 : same conventions as factor 1.

    Notes
    -----
    The Feller condition for each factor is ``2*kappa_i*theta_i >= nu_i^2``.
    Violations do not prevent pricing but may cause the variance process to
    hit zero.

    ``nu_i > 0`` is required; the formula calls :func:`heston_riccati_cd`
    which is undefined for ``nu = 0``.
    """

    kappa1: float
    theta1: float
    nu1: float
    rho1: float
    v01: float
    kappa2: float
    theta2: float
    nu2: float
    rho2: float
    v02: float

    def __init__(
        self,
        kappa1: float,
        theta1: float,
        nu1: float,
        rho1: float,
        v01: float,
        kappa2: float,
        theta2: float,
        nu2: float,
        rho2: float,
        v02: float,
    ):
        for tag, kappa, theta, nu, rho, v0 in [
            ("1", kappa1, theta1, nu1, rho1, v01),
            ("2", kappa2, theta2, nu2, rho2, v02),
        ]:
            if not (np.isfinite(kappa) and kappa > 0):
                raise ValueError(f"DoubleHestonParams: kappa{tag} must be > 0; got {kappa}")
            if not (np.isfinite(theta) and theta >= 0):
                raise ValueError(f"DoubleHestonParams: theta{tag} must be >= 0; got {theta}")
            if not (np.isfinite(nu) and nu > 0):
                raise ValueError(f"DoubleHestonParams: nu{tag} must be > 0; got {nu}")
            if not (np.isfinite(rho) and -1.0 < rho < 1.0):
                raise ValueError(f"DoubleHestonParams: rho{tag} must be in (-1, 1); got {rho}")
            if not (np.isfinite(v0) and v0 >= 0):
                raise ValueError(f"DoubleHestonParams: v0{tag} must be >= 0; got {v0}")
        object.__setattr__(self, "name", "double_heston")
        object.__setattr__(self, "kappa1", kappa1)
        object.__setattr__(self, "theta1", theta1)
        object.__setattr__(self, "nu1", nu1)
        object.__setattr__(self, "rho1", rho1)
        object.__setattr__(self, "v01", v01)
        object.__setattr__(self, "kappa2", kappa2)
        object.__setattr__(self, "theta2", theta2)
        object.__setattr__(self, "nu2", nu2)
        object.__setattr__(self, "rho2", rho2)
        object.__setattr__(self, "v02", v02)

    @property
    def factor1(self) -> HestonParams:
        return HestonParams(
            kappa=self.kappa1,
            theta=self.theta1,
            nu=self.nu1,
            rho=self.rho1,
            v0=max(self.v01, 1e-12),  # HestonParams requires v0 > 0
        )

    @property
    def factor2(self) -> HestonParams:
        return HestonParams(
            kappa=self.kappa2,
            theta=self.theta2 if self.theta2 > 0 else 1e-12,
            nu=self.nu2,
            rho=self.rho2,
            v0=max(self.v02, 1e-12),
        )


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------


def double_heston_cf(u: np.ndarray, fwd: ForwardSpec, p: DoubleHestonParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under Double Heston.

        phi_DH(u; T) = exp(C1 + D1*v01 + C2 + D2*v02)

    where (C_i, D_i) = heston_riccati_cd(u, T, factor_i) from the stable
    Form 2 Albrecher decomposition. The two factors are independent, so
    their log-CFs add.

    Parameters
    ----------
    u : array_like (real or complex)
        Frequency grid.
    fwd : ForwardSpec
    p : DoubleHestonParams

    Returns
    -------
    np.ndarray (complex128)
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T

    h1 = HestonParams(
        kappa=p.kappa1,
        theta=p.theta1 if p.theta1 > 0 else 1e-12,
        nu=p.nu1,
        rho=p.rho1,
        v0=max(p.v01, 1e-12),
    )
    h2 = HestonParams(
        kappa=p.kappa2,
        theta=p.theta2 if p.theta2 > 0 else 1e-12,
        nu=p.nu2,
        rho=p.rho2,
        v0=max(p.v02, 1e-12),
    )

    C1, D1 = heston_riccati_cd(u_c, T, h1)
    C2, D2 = heston_riccati_cd(u_c, T, h2)

    return np.exp(C1 + D1 * p.v01 + C2 + D2 * p.v02)


# ---------------------------------------------------------------------------
# Cumulants — numerical Cauchy integral
# ---------------------------------------------------------------------------


def double_heston_cumulants(fwd: ForwardSpec, p: DoubleHestonParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of X_T under Double Heston.

    Computed numerically via the standard Cauchy-circle FFT on the CF.
    Since the two factors are independent their cumulants add; the
    numerical approach is simpler and avoids re-deriving Heston's
    cumulant algebra for two sets of parameters.
    """
    from ..utils.cumulants import cumulants_from_cf

    def _phi(u):
        return double_heston_cf(u, fwd, p)

    c = cumulants_from_cf(_phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
