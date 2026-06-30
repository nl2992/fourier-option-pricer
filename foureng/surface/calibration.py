"""Model calibration to a market implied-vol surface.

Fits Heston / VG / Kou parameters by minimising the sum of squared residuals
between model-implied and market-implied Black-76 vols on a (maturities, strikes)
grid.

Default method is Nelder-Mead (gradient-free, bounded). The natural choice
would be L-BFGS-B, but the IV-residual objective inherits ~1e-8 numerical
noise per grid cell from COS pricing + safeguarded-Newton IV inversion. That
noise trips L-BFGS-B's Wolfe line search on wider smile surfaces ("ABNORMAL
TERMINATION IN LNSRCH"  -  the classic noisy-objective failure). Nelder-Mead
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

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from ..models.cgmy import CgmyParams, cgmy_cf, cgmy_cumulants
from ..models.heston import HestonParams, heston_cf_form2, heston_cumulants
from ..models.kou import KouParams, kou_cf, kou_cumulants
from ..models.nig import NigParams, nig_cf, nig_cumulants
from ..models.sabr import SabrParams, sabr_hagan_implied_vol
from ..models.variance_gamma import VGParams, vg_cf, vg_cumulants
from .vol_surface import SurfaceSpec, model_iv_surface


@dataclass
class CalibrationResult:
    """Output of a model calibration run.

    Attributes
    ----------
    params : dict
        Best-fit model parameters as a plain dict (parameter names to values).
    loss : float
        Final objective value (sum of squared IV residuals, model minus market).
    success : bool
        True if the optimiser reported convergence.
    nfev : int
        Number of objective function evaluations.
    residuals : np.ndarray
        Shape (nT, nK). Per-cell IV residuals (model IV minus market IV).
    """

    params: dict
    loss: float
    success: bool
    nfev: int
    residuals: np.ndarray


# --- Param-vector <-> model converters ---------------------------------------


def _heston_from_vec(x: np.ndarray) -> HestonParams:
    kappa, theta, nu, rho, v0 = x
    return HestonParams(
        kappa=float(kappa), theta=float(theta), nu=float(nu), rho=float(rho), v0=float(v0)
    )


def _vg_from_vec(x: np.ndarray) -> VGParams:
    sigma, nu, theta = x
    return VGParams(sigma=float(sigma), nu=float(nu), theta=float(theta))


def _kou_from_vec(x: np.ndarray) -> KouParams:
    sigma, lam, p, eta1, eta2 = x
    return KouParams(
        sigma=float(sigma), lam=float(lam), p=float(p), eta1=float(eta1), eta2=float(eta2)
    )


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
            f"market_ivs shape {market_ivs.shape} != ({len(spec.maturities)}, {len(spec.strikes)})"
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
            return 1e12  # infeasible region  -  push optimiser away
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
        res = minimize(
            objective, x0=z0, bounds=z_bounds, method=method, options=dict(maxiter=maxiter)
        )

    best_x = unscale(res.x)
    best_params = unpack(best_x)
    # Refresh stored residuals at the reported optimum.
    objective(res.x)
    return CalibrationResult(
        params=best_params.__dict__
        if hasattr(best_params, "__dict__")
        else dict(vars(best_params)),
        loss=float(res.fun),
        success=bool(res.success),
        nfev=int(nfev[0]),
        residuals=last_residuals[0],
    )


# --- Model-specific entry points ---------------------------------------------

HESTON_DEFAULT_BOUNDS = [
    (1e-3, 20.0),  # kappa
    (1e-4, 2.0),  # theta (long-run variance)
    (1e-3, 5.0),  # nu    (vol of vol)
    (-0.999, 0.999),  # rho
    (1e-4, 2.0),  # v0
]

VG_DEFAULT_BOUNDS = [
    (1e-3, 2.0),  # sigma
    (1e-4, 5.0),  # nu
    (-2.0, 2.0),  # theta
]

KOU_DEFAULT_BOUNDS = [
    (1e-3, 2.0),  # sigma
    (1e-4, 20.0),  # lam
    (1e-3, 1.0 - 1e-3),  # p
    (1.0 + 1e-3, 50.0),  # eta1 (> 1 for finite jump mean)
    (1e-3, 50.0),  # eta2
]

CGMY_DEFAULT_BOUNDS = [
    (1e-4, 20.0),  # C  (intensity)
    (1e-3, 50.0),  # G  (left-tail damping)
    (1.0 + 1e-3, 50.0),  # M  (right-tail; must be > 1 for finite mean)
    (1e-4, 1.99),  # Y  (activity index; Y < 2 for finite variance)
]

NIG_DEFAULT_BOUNDS = [
    (1e-3, 2.0),  # sigma
    (1e-3, 5.0),  # nu
    (-2.0, 2.0),  # theta
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
        cf_factory_from_params=lambda p: lambda fwd: lambda u: heston_cf_form2(u, fwd, p),
        cumulant_factory_from_params=lambda p: lambda fwd: heston_cumulants(fwd, p),
        N=N,
        L=L,
        method=method,
        fd_step=fd_step,
        maxiter=maxiter,
        ftol=ftol,
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
        cf_factory_from_params=lambda p: lambda fwd: lambda u: vg_cf(u, fwd, p),
        cumulant_factory_from_params=lambda p: lambda fwd: vg_cumulants(fwd, p),
        N=N,
        L=L,
        method=method,
        fd_step=fd_step,
        maxiter=maxiter,
        ftol=ftol,
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
        cf_factory_from_params=lambda p: lambda fwd: lambda u: kou_cf(u, fwd, p),
        cumulant_factory_from_params=lambda p: lambda fwd: kou_cumulants(fwd, p),
        N=N,
        L=L,
        method=method,
        fd_step=fd_step,
        maxiter=maxiter,
        ftol=ftol,
    )


def _cgmy_from_vec(x: np.ndarray) -> CgmyParams:
    C, G, M, Y = x
    return CgmyParams(C=float(C), G=float(G), M=float(M), Y=float(Y))


def _nig_from_vec(x: np.ndarray) -> NigParams:
    sigma, nu, theta = x
    return NigParams(sigma=float(sigma), nu=float(nu), theta=float(theta))


def calibrate_cgmy(
    spec: SurfaceSpec,
    market_ivs: np.ndarray,
    initial: CgmyParams,
    bounds: list[tuple[float, float]] | None = None,
    weights: np.ndarray | None = None,
    N: int = 512,
    L: float = 12.0,
    method: str = "Nelder-Mead",
    fd_step: float = 1e-5,
    maxiter: int = 1000,
    ftol: float = 1e-10,
) -> CalibrationResult:
    """Calibrate CGMY model parameters to a market IV surface."""
    bounds = bounds or CGMY_DEFAULT_BOUNDS
    x0 = np.array([initial.C, initial.G, initial.M, initial.Y])
    return _calibrate(
        spec=spec,
        market_ivs=market_ivs,
        x0=x0,
        bounds=bounds,
        weights=weights,
        unpack=_cgmy_from_vec,
        cf_factory_from_params=lambda p: lambda fwd: lambda u: cgmy_cf(u, fwd, p),
        cumulant_factory_from_params=lambda p: lambda fwd: cgmy_cumulants(fwd, p),
        N=N,
        L=L,
        method=method,
        fd_step=fd_step,
        maxiter=maxiter,
        ftol=ftol,
    )


def calibrate_nig(
    spec: SurfaceSpec,
    market_ivs: np.ndarray,
    initial: NigParams,
    bounds: list[tuple[float, float]] | None = None,
    weights: np.ndarray | None = None,
    N: int = 512,
    L: float = 12.0,
    method: str = "Nelder-Mead",
    fd_step: float = 1e-5,
    maxiter: int = 1000,
    ftol: float = 1e-10,
) -> CalibrationResult:
    """Calibrate NIG model parameters to a market IV surface."""
    bounds = bounds or NIG_DEFAULT_BOUNDS
    x0 = np.array([initial.sigma, initial.nu, initial.theta])
    return _calibrate(
        spec=spec,
        market_ivs=market_ivs,
        x0=x0,
        bounds=bounds,
        weights=weights,
        unpack=_nig_from_vec,
        cf_factory_from_params=lambda p: lambda fwd: lambda u: nig_cf(u, fwd, p),
        cumulant_factory_from_params=lambda p: lambda fwd: nig_cumulants(fwd, p),
        N=N,
        L=L,
        method=method,
        fd_step=fd_step,
        maxiter=maxiter,
        ftol=ftol,
    )


# ── SABR smile calibration ────────────────────────────────────────────────


SABR_ALPHA_BOUNDS = (1e-4, 5.0)
SABR_RHO_BOUNDS   = (-0.999, 0.999)
SABR_NU_BOUNDS    = (1e-4, 5.0)


@dataclass
class SabrSmileCalibResult:
    """Output of a SABR smile calibration run.

    Attributes
    ----------
    params : SabrParams
        Best-fit SABR parameters.
    loss : float
        Final objective (sum of squared IV residuals).
    success : bool
        True if the optimiser reported convergence.
    nfev : int
        Number of objective evaluations.
    residuals : np.ndarray
        Per-strike IV residuals (model IV − market IV).
    """

    params: SabrParams
    loss: float
    success: bool
    nfev: int
    residuals: np.ndarray


def calibrate_sabr_smile(
    F: float,
    T: float,
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    initial: SabrParams,
    *,
    fit_beta: bool = False,
    weights: np.ndarray | None = None,
    alpha_bounds: tuple[float, float] = SABR_ALPHA_BOUNDS,
    rho_bounds: tuple[float, float] = SABR_RHO_BOUNDS,
    nu_bounds: tuple[float, float] = SABR_NU_BOUNDS,
    beta_bounds: tuple[float, float] = (0.0, 1.0),
    method: str = "Nelder-Mead",
    maxiter: int = 2000,
    ftol: float = 1e-12,
) -> SabrSmileCalibResult:
    """Calibrate SABR parameters to a single-maturity implied vol smile.

    By default beta is kept fixed at ``initial.beta`` and only (alpha, rho, nu)
    are optimised.  Pass ``fit_beta=True`` to include beta in the fit.

    The SABR Hagan (2002) formula gives IVs directly — no COS pricing round-trip
    is needed, so calibration is typically 10–100× faster than CF-based models.

    Parameters
    ----------
    F : float
        Forward price at maturity T.
    T : float
        Time to maturity (years).
    strikes : np.ndarray
        Market strike grid.  All > 0.
    market_ivs : np.ndarray
        Market implied vols at ``strikes``.  Same shape.
    initial : SabrParams
        Starting parameters.  ``initial.beta`` is used as the fixed backbone
        unless ``fit_beta=True``.
    fit_beta : bool
        If True, also calibrate beta.  Default False.
    weights : np.ndarray or None
        Per-strike weights for the IV residual loss.  Default uniform.
    alpha_bounds, rho_bounds, nu_bounds, beta_bounds :
        Box constraints for each parameter.
    method : str
        Scipy optimiser name.  Default ``"Nelder-Mead"``.
    maxiter : int
        Maximum optimiser iterations.
    ftol : float
        Function-value tolerance.

    Returns
    -------
    SabrSmileCalibResult
    """
    K = np.asarray(strikes, dtype=float)
    mkt = np.asarray(market_ivs, dtype=float)
    if K.shape != mkt.shape:
        raise ValueError(f"strikes and market_ivs must have the same shape; got {K.shape} vs {mkt.shape}")
    if weights is None:
        weights = np.ones_like(mkt)
    w = np.asarray(weights, dtype=float)

    beta_fixed = initial.beta

    if fit_beta:
        bounds = [alpha_bounds, rho_bounds, nu_bounds, beta_bounds]
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        x0_raw = np.array([initial.alpha, initial.rho, initial.nu, initial.beta])
    else:
        bounds = [alpha_bounds, rho_bounds, nu_bounds]
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        x0_raw = np.array([initial.alpha, initial.rho, initial.nu])

    rng = hi - lo
    z0 = np.clip((x0_raw - lo) / rng, 1e-6, 1.0 - 1e-6)
    z_bounds = [(1e-6, 1.0 - 1e-6)] * len(bounds)

    nfev = [0]
    last_residuals = [np.zeros_like(mkt)]

    def unscale(z):
        return lo + np.clip(z, 0.0, 1.0) * rng

    def objective(z):
        nfev[0] += 1
        x = unscale(z)
        try:
            if fit_beta:
                alpha, rho, nu, beta = float(x[0]), float(x[1]), float(x[2]), float(x[3])
            else:
                alpha, rho, nu = float(x[0]), float(x[1]), float(x[2])
                beta = beta_fixed
            model_ivs = sabr_hagan_implied_vol(F, K, T, alpha, beta, rho, nu)
        except Exception:
            return 1e12
        if not np.all(np.isfinite(model_ivs)):
            return 1e12
        r = model_ivs - mkt
        last_residuals[0] = r
        return float(np.sum(w * r * r))

    if method.lower() in ("nelder-mead", "neldermead", "nm"):
        opts = dict(maxiter=maxiter, xatol=1e-8, fatol=ftol, adaptive=True)
        res = minimize(objective, x0=z0, bounds=z_bounds, method="Nelder-Mead", options=opts)
    else:
        res = minimize(objective, x0=z0, bounds=z_bounds, method=method,
                       options=dict(maxiter=maxiter, ftol=ftol))

    best = unscale(res.x)
    objective(res.x)  # refresh residuals at optimum
    if fit_beta:
        best_params = SabrParams(alpha=float(best[0]), beta=float(best[3]),
                                 rho=float(best[1]), nu=float(best[2]))
    else:
        best_params = SabrParams(alpha=float(best[0]), beta=beta_fixed,
                                 rho=float(best[1]), nu=float(best[2]))

    return SabrSmileCalibResult(
        params=best_params,
        loss=float(res.fun),
        success=bool(res.success),
        nfev=int(nfev[0]),
        residuals=last_residuals[0],
    )
