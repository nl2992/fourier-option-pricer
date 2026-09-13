"""Two-asset models given by their joint characteristic function.

Each model defines the joint CF of ``X = (log(S1_T / F1), log(S2_T / F2))``,

    Phi(u1, u2) = E[exp(i u1 X1 + i u2 X2)],

for complex ``u`` inside the strip where the moments exist. Both marginals are
martingales (``Phi(-i, 0) = Phi(0, -i) = 1``). These are the three examples of
Hurd & Zhou (2010):

``bsm2d``
    Correlated geometric Brownian motions,
    ``log Phi = -T/2 (u' C u + i sum_j u_j sigma_j^2)`` with ``C = [[s1^2, r s1 s2], [r s1 s2, s2^2]]``.
``vg2d``
    Brownian motions with drifts ``theta_j`` run on one gamma clock ``G``
    (mean ``T``, variance ``nu T``), so the jumps of the two assets arrive
    together:
    ``Phi = exp(i u . omega T) (1 - i nu u . theta + nu/2 u' C u)^(-T/nu)``,
    ``omega_j = log(1 - nu theta_j - nu sigma_j^2 / 2) / nu``. Each marginal is
    the registry ``variance_gamma`` model.
``heston2d``
    Hurd and Zhou's three-factor stochastic volatility model: one CIR variance
    ``dv = kappa (theta - v) dt + nu sqrt(v) dW_v`` drives both assets,
    ``dX_j = -sigma_j^2 v / 2 dt + sigma_j sqrt(v) dW_j``, with
    ``corr(W_1, W_2) = rho`` and ``corr(W_j, W_v) = rho_j``. For fixed ``u`` the
    exponent ``u . X`` is a one-factor Heston process, so the CF is the Heston
    Riccati solution with ``a = -(u' C u + i sum_j u_j sigma_j^2) / 2`` and
    ``b = kappa - i nu sum_j rho_j sigma_j u_j``. With ``sigma_1 = 1`` the first
    marginal is the registry ``heston`` model.

Reference
---------
Hurd, T. R. & Zhou, Z. (2010), A Fourier transform method for spread option
pricing, *SIAM Journal on Financial Mathematics* 1, 142-157.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .base import ModelSpec


def _check_positive(cls: str, **vals: float) -> None:
    for name, val in vals.items():
        if not (np.isfinite(val) and val > 0):
            raise ValueError(f"{cls}: {name} must be > 0; got {val}")


def _check_corr(cls: str, **vals: float) -> None:
    for name, val in vals.items():
        if not (np.isfinite(val) and -1.0 <= val <= 1.0):
            raise ValueError(f"{cls}: {name} must lie in [-1, 1]; got {val}")


@dataclass(frozen=True)
class Bsm2dParams(ModelSpec):
    """Two correlated geometric Brownian motions."""

    sigma1: float
    sigma2: float
    rho: float

    def __init__(self, sigma1: float, sigma2: float, rho: float):
        _check_positive("Bsm2dParams", sigma1=sigma1, sigma2=sigma2)
        _check_corr("Bsm2dParams", rho=rho)
        object.__setattr__(self, "name", "bsm2d")
        object.__setattr__(self, "sigma1", sigma1)
        object.__setattr__(self, "sigma2", sigma2)
        object.__setattr__(self, "rho", rho)


@dataclass(frozen=True)
class Vg2dParams(ModelSpec):
    """Two variance gamma assets on a common gamma clock (Hurd & Zhou 2010)."""

    sigma1: float
    sigma2: float
    theta1: float
    theta2: float
    nu: float
    rho: float

    def __init__(
        self, sigma1: float, sigma2: float, theta1: float, theta2: float, nu: float, rho: float
    ):
        _check_positive("Vg2dParams", sigma1=sigma1, sigma2=sigma2, nu=nu)
        _check_corr("Vg2dParams", rho=rho)
        for name, th, sg in (("1", theta1, sigma1), ("2", theta2, sigma2)):
            if not np.isfinite(th):
                raise ValueError(f"Vg2dParams: theta{name} must be finite; got {th}")
            if 1.0 - nu * th - 0.5 * nu * sg * sg <= 0.0:
                raise ValueError(
                    f"Vg2dParams: asset {name} has no finite mean "
                    "(need 1 - nu theta - nu sigma^2 / 2 > 0)"
                )
        object.__setattr__(self, "name", "vg2d")
        object.__setattr__(self, "sigma1", sigma1)
        object.__setattr__(self, "sigma2", sigma2)
        object.__setattr__(self, "theta1", theta1)
        object.__setattr__(self, "theta2", theta2)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)


@dataclass(frozen=True)
class Heston2dParams(ModelSpec):
    """Two assets driven by one CIR variance (Hurd & Zhou's three-factor SV model)."""

    v0: float
    kappa: float
    theta: float
    nu: float
    sigma1: float
    sigma2: float
    rho: float
    rho1: float
    rho2: float

    def __init__(
        self,
        v0: float,
        kappa: float,
        theta: float,
        nu: float,
        sigma1: float,
        sigma2: float,
        rho: float,
        rho1: float,
        rho2: float,
    ):
        _check_positive(
            "Heston2dParams", v0=v0, kappa=kappa, theta=theta, nu=nu, sigma1=sigma1, sigma2=sigma2
        )
        _check_corr("Heston2dParams", rho=rho, rho1=rho1, rho2=rho2)
        corr = np.array([[1.0, rho, rho1], [rho, 1.0, rho2], [rho1, rho2, 1.0]])
        if np.linalg.eigvalsh(corr)[0] < -1e-12:
            raise ValueError("Heston2dParams: (rho, rho1, rho2) is not a valid correlation matrix")
        object.__setattr__(self, "name", "heston2d")
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "sigma1", sigma1)
        object.__setattr__(self, "sigma2", sigma2)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "rho1", rho1)
        object.__setattr__(self, "rho2", rho2)


def _quad_forms(u1, u2, s1: float, s2: float, rho: float):
    """``u' C u`` and ``sum_j u_j sigma_j^2`` for the diffusion covariance ``C``."""
    quad = s1 * s1 * u1 * u1 + 2.0 * rho * s1 * s2 * u1 * u2 + s2 * s2 * u2 * u2
    lin = s1 * s1 * u1 + s2 * s2 * u2
    return quad, lin


def _bsm2d_log_cf(u1, u2, T: float, p: Bsm2dParams):
    quad, lin = _quad_forms(u1, u2, p.sigma1, p.sigma2, p.rho)
    return -0.5 * T * (quad + 1j * lin)


def _vg2d_log_cf(u1, u2, T: float, p: Vg2dParams):
    nu = p.nu
    quad = (
        p.sigma1**2 * u1 * u1 + 2.0 * p.rho * p.sigma1 * p.sigma2 * u1 * u2 + p.sigma2**2 * u2 * u2
    )
    lin = p.theta1 * u1 + p.theta2 * u2
    om1 = np.log(1.0 - nu * p.theta1 - 0.5 * nu * p.sigma1**2) / nu
    om2 = np.log(1.0 - nu * p.theta2 - 0.5 * nu * p.sigma2**2) / nu
    base = 1.0 - 1j * nu * lin + 0.5 * nu * quad
    return 1j * (u1 * om1 + u2 * om2) * T - (T / nu) * np.log(base)


def _heston2d_log_cf(u1, u2, T: float, p: Heston2dParams):
    quad, lin = _quad_forms(u1, u2, p.sigma1, p.sigma2, p.rho)
    a = -0.5 * (quad + 1j * lin)
    b = p.kappa - 1j * p.nu * (p.rho1 * p.sigma1 * u1 + p.rho2 * p.sigma2 * u2)
    nu2 = p.nu * p.nu
    d = np.sqrt(b * b - 2.0 * nu2 * a)
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    B = (b - d) / nu2 * (1.0 - e) / (1.0 - g * e)
    A = p.kappa * p.theta / nu2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    return A + B * p.v0


_LOG_CF: dict[str, tuple[Callable[..., np.ndarray], type]] = {
    "bsm2d": (_bsm2d_log_cf, Bsm2dParams),
    "vg2d": (_vg2d_log_cf, Vg2dParams),
    "heston2d": (_heston2d_log_cf, Heston2dParams),
}

JOINT_MODELS = tuple(_LOG_CF)


def joint_log_cf(model: str, u1, u2, T: float, params) -> np.ndarray:
    """``log Phi(u1, u2)`` for the two-asset model ``model`` at maturity ``T``."""
    if model not in _LOG_CF:
        raise ValueError(f"joint_cf: unknown two-asset model {model!r}; choose from {JOINT_MODELS}")
    fn, cls = _LOG_CF[model]
    if not isinstance(params, cls):
        raise TypeError(f"joint_cf: model {model!r} needs {cls.__name__}, got {type(params)}")
    u1 = np.asarray(u1, dtype=complex)
    u2 = np.asarray(u2, dtype=complex)
    return fn(u1, u2, float(T), params)


def joint_cf(model: str, u1, u2, T: float, params) -> np.ndarray:
    """Joint CF ``E[exp(i u1 X1 + i u2 X2)]`` of the two log-forward returns."""
    return np.exp(joint_log_cf(model, u1, u2, T, params))


__all__ = [
    "JOINT_MODELS",
    "Bsm2dParams",
    "Heston2dParams",
    "Vg2dParams",
    "joint_cf",
    "joint_log_cf",
]
