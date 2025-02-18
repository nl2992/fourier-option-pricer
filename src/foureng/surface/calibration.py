"""Model calibration to a market implied-vol surface.

Fits Heston / VG / Kou parameters by minimising the sum of squared residuals
between model-implied and market-implied Black-76 vols on a (maturities, strikes)
grid.

Default method is Nelder-Mead (gradient-free, bounded). The natural choice
would be L-BFGS-B, but the IV-residual objective inherits ~1e-8 numerical
noise per grid cell from COS pricing + safeguarded-Newton IV inversion. That
noise trips L-BFGS-B's Wolfe line search on wider smile surfaces ("ABNORMAL
TERMINATION IN LNSRCH" — the classic noisy-objective failure). Nelder-Mead
is robust to this and converges reliably once parameters are normalised to a
unit box.

Parameters are internally rescaled to [0, 1]^d so the simplex moves evenly
across each dimension regardless of raw units (kappa in [0,20] vs theta in
[0,2] vs rho in [-1,1]). Callers still pass real-world parameter values;
the normalisation is invisible.

Calibrate on IVs rather than prices because IV residuals are roughly on the
same scale across strikes (prices vary by orders of magnitude ITM/OTM). For
pathological inputs (stale quotes, arbitrage violations) switch to a
price-space loss with vega weights.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable
from scipy.optimize import minimize

from ..char_func.base import ForwardSpec
from ..char_func.heston import HestonParams, heston_cf_form2, heston_cumulants
from ..char_func.variance_gamma import VGParams, vg_cf, vg_cumulants
from ..char_func.kou import KouParams, kou_cf, kou_cumulants
from .vol_surface import SurfaceSpec, model_iv_surface


@dataclass
class CalibrationResult:
    params: dict            # best-fit parameters as a dict
    loss: float             # final objective (weighted SSE on IVs)
    success: bool
    nfev: int
    residuals: np.ndarray   # (nT, nK) IV residuals model - market


# --- Param-vector <-> model converters ---------------------------------------

def _heston_from_vec(x: np.ndarray) -> HestonParams:
    kappa, theta, nu, rho, v0 = x
    return HestonParams(kappa=float(kappa), theta=float(theta), nu=float(nu),
                        rho=float(rho), v0=float(v0))


def _vg_from_vec(x: np.ndarray) -> VGParams:
    sigma, nu, theta = x
    return VGParams(sigma=float(sigma), nu=float(nu), theta=float(theta))


def _kou_from_vec(x: np.ndarray) -> KouParams:
    sigma, lam, p, eta1, eta2 = x
    return KouParams(sigma=float(sigma), lam=float(lam), p=float(p),
                     eta1=float(eta1), eta2=float(eta2))


# --- Core calibration loop ---------------------------------------------------

def _calibrate(
    spec: SurfaceSpec,
    market_ivs: np.ndarray,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    weights: np.ndarray | None,
    unpack: Callable,
    cf_factory_from_params: Callable,
    cumulant_factory_from_params: Callable,
    N: int,
    L: float,
    method: str,
    fd_step: float,
    maxiter: int,
    ftol: float,
) -> CalibrationResult:
    """Generic box-constrained calibration on IV residuals.

    Parameters are rescaled to [0, 1]^d internally; the optimiser sees a
    well-conditioned problem regardless of raw units. ``method`` controls
    which scipy optimiser is used; Nelder-Mead is the default because the
    IV-residual objective is too noisy for gradient line searches.
    """
    if market_ivs.shape != (len(spec.maturities), len(spec.strikes)):
        raise ValueError(
            f"market_ivs shape {market_ivs.shape} != "
            f"({len(spec.maturities)}, {len(spec.strikes)})"
        )
    if weights is None:
        weights = np.ones_like(market_ivs)

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    rng = hi - lo

    def unscale(z: np.ndarray) -> np.ndarray:
        return lo + np.clip(z, 0.0, 1.0) * rng

    nfev = [0]
    last_residuals = [np.zeros_like(market_ivs)]

    def objective(z: np.ndarray) -> float:
        nfev[0] += 1
        try:
            params = unpack(unscale(z))
            cf = cf_factory_from_params(params)
            cum = cumulant_factory_from_params(params)
            ivs = model_iv_surface(spec, cf, cum, N=N, L=L)
        except Exception:
            return 1e12  # infeasible region — push optimiser away
        if not np.all(np.isfinite(ivs)):
            return 1e12
        r = ivs - market_ivs
        last_residuals[0] = r
        return float(np.sum(weights * r * r))

    z0 = np.clip((np.asarray(x0, dtype=float) - lo) / rng, 1e-6, 1.0 - 1e-6)
    z_bounds = [(1e-6, 1.0 - 1e-6)] * len(bounds)

    if method.lower() in ("nelder-mead", "neldermead", "nm"):
        opts = dict(maxiter=maxiter, xatol=1e-6, fatol=ftol, adaptive=True)
        res = minimize(objective, x0=z0, bounds=z_bounds, method="Nelder-Mead", options=opts)
    elif method.upper() == "L-BFGS-B":
        opts = dict(maxiter=maxiter, ftol=ftol, eps=fd_step)
        res = minimize(objective, x0=z0, bounds=z_bounds, method="L-BFGS-B", options=opts)
    else:
        res = minimize(objective, x0=z0, bounds=z_bounds, method=method,
                       options=dict(maxiter=maxiter))

    best_x = unscale(res.x)
    best_params = unpack(best_x)
    # Refresh stored residuals at the reported optimum.
    objective(res.x)
    return CalibrationResult(
        params=best_params.__dict__ if hasattr(best_params, "__dict__") else dict(vars(best_params)),
        loss=float(res.fun),
        success=bool(res.success),
        nfev=int(nfev[0]),
        residuals=last_residuals[0],
    )


# --- Model-specific entry points ---------------------------------------------

HESTON_DEFAULT_BOUNDS = [
    (1e-3, 20.0),     # kappa
    (1e-4, 2.0),      # theta (long-run variance)
    (1e-3, 5.0),      # nu    (vol of vol)
    (-0.999, 0.999),  # rho
    (1e-4, 2.0),      # v0
]

VG_DEFAULT_BOUNDS = [
    (1e-3, 2.0),      # sigma
    (1e-4, 5.0),      # nu
    (-2.0, 2.0),      # theta
]

KOU_DEFAULT_BOUNDS = [
    (1e-3, 2.0),      # sigma
    (1e-4, 20.0),     # lam
    (1e-3, 1.0 - 1e-3),   # p
    (1.0 + 1e-3, 50.0),   # eta1 (> 1 for finite jump mean)
    (1e-3, 50.0),     # eta2
]


def calibrate_heston(
    spec: SurfaceSpec,
    market_ivs: np.ndarray,
    initial: HestonParams,
    bounds: list[tuple[float, float]] | None = None,
    weights: np.ndarray | None = None,
    N: int = 192,
    L: float = 10.0,
    method: str = "Nelder-Mead",
    fd_step: float = 1e-5,
    maxiter: int = 1000,
    ftol: float = 1e-10,
) -> CalibrationResult:
    bounds = bounds or HESTON_DEFAULT_BOUNDS
    x0 = np.array([initial.kappa, initial.theta, initial.nu, initial.rho, initial.v0])
    return _calibrate(
        spec=spec,
        market_ivs=market_ivs,
        x0=x0,
        bounds=bounds,
        weights=weights,
        unpack=_heston_from_vec,
        cf_factory_from_params=lambda p: (lambda fwd: (lambda u: heston_cf_form2(u, fwd, p))),
        cumulant_factory_from_params=lambda p: (lambda fwd: heston_cumulants(fwd, p)),
        N=N, L=L, method=method, fd_step=fd_step, maxiter=maxiter, ftol=ftol,
    )


def calibrate_vg(
    spec: SurfaceSpec,
    market_ivs: np.ndarray,
    initial: VGParams,
    bounds: list[tuple[float, float]] | None = None,
    weights: np.ndarray | None = None,
    N: int = 512,  # VG needs higher N due to heavy tails
    L: float = 10.0,
    method: str = "Nelder-Mead",
    fd_step: float = 1e-5,
    maxiter: int = 1000,
    ftol: float = 1e-10,
) -> CalibrationResult:
    bounds = bounds or VG_DEFAULT_BOUNDS
    x0 = np.array([initial.sigma, initial.nu, initial.theta])
    return _calibrate(
        spec=spec,
        market_ivs=market_ivs,
        x0=x0,
        bounds=bounds,
        weights=weights,
        unpack=_vg_from_vec,
        cf_factory_from_params=lambda p: (lambda fwd: (lambda u: vg_cf(u, fwd, p))),
        cumulant_factory_from_params=lambda p: (lambda fwd: vg_cumulants(fwd, p)),
        N=N, L=L, method=method, fd_step=fd_step, maxiter=maxiter, ftol=ftol,
    )


def calibrate_kou(
    spec: SurfaceSpec,
    market_ivs: np.ndarray,
    initial: KouParams,
    bounds: list[tuple[float, float]] | None = None,
    weights: np.ndarray | None = None,
    N: int = 192,
    L: float = 10.0,
    method: str = "Nelder-Mead",
    fd_step: float = 1e-5,
    maxiter: int = 1000,
    ftol: float = 1e-10,
) -> CalibrationResult:
    bounds = bounds or KOU_DEFAULT_BOUNDS
    x0 = np.array([initial.sigma, initial.lam, initial.p, initial.eta1, initial.eta2])
    return _calibrate(
        spec=spec,
        market_ivs=market_ivs,
        x0=x0,
        bounds=bounds,
        weights=weights,
        unpack=_kou_from_vec,
        cf_factory_from_params=lambda p: (lambda fwd: (lambda u: kou_cf(u, fwd, p))),
        cumulant_factory_from_params=lambda p: (lambda fwd: kou_cumulants(fwd, p)),
        N=N, L=L, method=method, fd_step=fd_step, maxiter=maxiter, ftol=ftol,
    )
