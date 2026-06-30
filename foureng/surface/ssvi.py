"""SSVI (Surface SVI) joint surface parameterization.

Gatheral & Jacquier (2014) proposed SSVI, which parameterizes the entire
implied-volatility surface jointly and guarantees absence of static
arbitrage (calendar spread and butterfly) under mild conditions.

The SSVI total variance is

    w(k, theta_t) = (theta_t / 2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + (1 - rho^2)))

where theta_t = ATM total variance at maturity T and phi = phi(theta_t)
is a smooth positive function of theta_t that controls the smile shape.

Two phi parametrizations are supported:

1. **Power-law** (Gatheral & Jacquier 2014, recommended):

       phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))

   with eta > 0 and gamma in (0, 1).

   Butterfly-free iff: eta*(1 + |rho|) <= 4  (sufficient condition)
   Calendar-free iff: phi is non-increasing in theta (automatically satisfied
   when gamma in (0, 1)).

2. **Heston-like** (closed-form in Heston limit):

       phi(theta) = (1 / theta) * (1 - (1 - e^{-theta}) / theta)

   This is a one-parameter family fixed by the Heston structure; no extra
   parameters needed.

References:
    Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
    *Quantitative Finance*, 14(1), 59-71.

    Hendriks, S., & Martini, C. (2019). The extended SSVI volatility surface.
    *Journal of Computational Finance*, 22(5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, NamedTuple, Sequence

import numpy as np
from scipy.optimize import minimize


# ── parameter dataclasses ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SSVIParams:
    """Parameters for the SSVI surface (power-law phi).

    Attributes
    ----------
    rho : float
        Correlation parameter, in (-1, 1).  Controls the overall skew of
        the surface.
    eta : float
        Scale parameter for phi, > 0.
    gamma : float
        Exponent for theta in phi, in (0, 1).
    """

    rho: float
    eta: float
    gamma: float

    def __post_init__(self) -> None:
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")


# ── phi functions ─────────────────────────────────────────────────────────────


def ssvi_phi_power_law(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """Power-law phi: eta / (theta^gamma * (1 + theta)^(1 - gamma)).

    Parameters
    ----------
    theta : ndarray
        ATM total variance values, > 0.
    eta, gamma : float
        SSVI parameters.
    """
    theta = np.asarray(theta, dtype=float)
    return eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))


def ssvi_phi_heston(theta: np.ndarray) -> np.ndarray:
    """Heston-like phi: (theta + e^{-theta} - 1) / theta^2.

    Equivalent to (1/theta)*(1 - (1 - e^{-theta})/theta), numerically
    stable via expm1 for small theta.  Converges to 1/2 as theta -> 0.

    Parameters
    ----------
    theta : ndarray
        ATM total variance values, > 0.
    """
    theta = np.asarray(theta, dtype=float)
    ex = np.expm1(-theta)            # e^{-theta} - 1  (accurate for small theta)
    return (theta + ex) / theta**2   # = (theta - 1 + e^{-theta}) / theta^2


# ── SSVI total variance ───────────────────────────────────────────────────────


def ssvi_total_variance(
    k: np.ndarray,
    theta: float,
    params: SSVIParams,
) -> np.ndarray:
    """Compute SSVI total variance w(k, theta) for a single maturity.

    Parameters
    ----------
    k : ndarray
        Log-moneyness grid k = ln(K/F).
    theta : float
        ATM total variance at this maturity, theta = sigma_ATM^2 * T > 0.
    params : SSVIParams
        SSVI parameters (rho, eta, gamma).

    Returns
    -------
    ndarray
        Total variance w(k; theta, params).
    """
    k     = np.asarray(k, dtype=float)
    phi   = float(ssvi_phi_power_law(np.array([theta]), params.eta, params.gamma)[0])
    rho   = params.rho

    pk    = phi * k + rho
    w     = (theta / 2.0) * (1.0 + rho * phi * k + np.sqrt(pk**2 + (1.0 - rho**2)))
    return np.maximum(w, 0.0)


def ssvi_implied_vol(
    k: np.ndarray,
    T: float,
    theta: float,
    params: SSVIParams,
) -> np.ndarray:
    """Compute SSVI implied volatility.

    Parameters
    ----------
    k : ndarray
        Log-moneyness.
    T : float
        Maturity in years.
    theta : float
        ATM total variance = sigma_ATM^2 * T.
    params : SSVIParams
        SSVI parameters.

    Returns
    -------
    ndarray
        Implied vol sigma(k, T).
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    w = ssvi_total_variance(k, theta, params)
    return np.sqrt(w / T)


# ── arbitrage checks ──────────────────────────────────────────────────────────


def ssvi_check_butterfly_free(params: SSVIParams) -> bool:
    """Check the sufficient condition for butterfly-free SSVI (power-law phi).

    The condition eta*(1 + |rho|) <= 4 (Gatheral & Jacquier 2014, Prop. 4.1)
    is sufficient but not necessary.

    Returns
    -------
    bool
        True if the sufficient condition is satisfied.
    """
    return params.eta * (1.0 + abs(params.rho)) <= 4.0


def ssvi_check_calendar_free(
    theta_t: np.ndarray,
    params: SSVIParams,
) -> bool:
    """Check that phi(theta) is non-increasing (calendar-spread free condition).

    For the power-law phi, phi is non-increasing iff d(phi)/d(theta) <= 0,
    which holds when gamma in (0, 1).  This function numerically verifies it
    on the provided theta grid.

    Parameters
    ----------
    theta_t : ndarray
        ATM total variance values (in increasing order).
    params : SSVIParams

    Returns
    -------
    bool
        True if phi is non-increasing on the theta grid.
    """
    theta_t = np.asarray(theta_t, dtype=float)
    if len(theta_t) < 2:
        return True
    phi_vals = ssvi_phi_power_law(theta_t, params.eta, params.gamma)
    return bool(np.all(np.diff(phi_vals) <= 1e-12))


# ── fit result ────────────────────────────────────────────────────────────────


class SSVIFitResult(NamedTuple):
    """Result of fitting SSVI surface parameters.

    Attributes
    ----------
    params : SSVIParams
        Fitted SSVI parameters (rho, eta, gamma).
    theta_t : ndarray, shape (nT,)
        ATM total variances at each fitted maturity.
    rmse : float
        Root-mean-square error in implied vol (aggregated across all slices).
    max_err : float
        Maximum absolute error in implied vol.
    butterfly_free : bool
        Whether the sufficient butterfly-free condition holds.
    calendar_free : bool
        Whether phi is non-increasing on the maturity grid.
    """

    params: SSVIParams
    theta_t: np.ndarray
    rmse: float
    max_err: float
    butterfly_free: bool
    calendar_free: bool


# ── joint surface calibration ────────────────────────────────────────────────


def fit_ssvi_surface(
    k_list: Sequence[np.ndarray],
    iv_list: Sequence[np.ndarray],
    maturities: Sequence[float],
    *,
    initial: SSVIParams | None = None,
    max_iter: int = 2000,
) -> SSVIFitResult:
    """Fit SSVI surface (power-law phi) to implied-vol data at multiple maturities.

    The fit is performed in two stages:
    1. Per-maturity: estimate ATM total variance theta_t = sigma_ATM^2 * T for
       each maturity slice from the market data (using the ATM point or
       nearest-ATM interpolation).
    2. Joint: optimize (rho, eta, gamma) to minimise sum of squared IV residuals
       across all slices simultaneously, with theta_t fixed from stage 1.

    Parameters
    ----------
    k_list : sequence of ndarray
        Log-moneyness grids, one per maturity.
    iv_list : sequence of ndarray
        Market implied volatilities, same shape as k_list.
    maturities : sequence of float
        Maturity values in years, same length as k_list.
    initial : SSVIParams or None
        Initial guess for (rho, eta, gamma).  If None, uses (rho=-0.3, eta=1.0, gamma=0.5).
    max_iter : int
        Maximum L-BFGS-B iterations.

    Returns
    -------
    SSVIFitResult
    """
    T_arr = np.asarray(maturities, dtype=float)
    nT    = len(T_arr)

    if not (len(k_list) == len(iv_list) == nT):
        raise ValueError("k_list, iv_list, and maturities must all have the same length.")
    if nT < 2:
        raise ValueError("SSVI requires at least 2 maturities for a surface fit.")

    # ── Stage 1: per-maturity theta_t ────────────────────────────────────────
    theta_t = np.empty(nT)
    for i, (k_i, iv_i, T_i) in enumerate(zip(k_list, iv_list, T_arr)):
        k_i  = np.asarray(k_i, dtype=float)
        iv_i = np.asarray(iv_i, dtype=float)
        # Find nearest-ATM IV (smallest |k|)
        idx_atm = int(np.argmin(np.abs(k_i)))
        sigma_atm = float(iv_i[idx_atm])
        theta_t[i] = sigma_atm**2 * float(T_i)

    theta_t = np.maximum(theta_t, 1e-8)

    # ── Stage 2: joint (rho, eta, gamma) optimization ─────────────────────────
    if initial is None:
        initial = SSVIParams(rho=-0.3, eta=1.0, gamma=0.5)

    def _objective(x):
        rho, eta, gamma = x[0], x[1], x[2]
        if not (-1.0 < rho < 1.0) or eta <= 0 or not (0 < gamma < 1):
            return 1e9
        try:
            p = SSVIParams(rho=rho, eta=eta, gamma=gamma)
        except ValueError:
            return 1e9

        total_sq = 0.0
        for i, (k_i, iv_i, T_i, th_i) in enumerate(zip(k_list, iv_list, T_arr, theta_t)):
            k_i  = np.asarray(k_i, dtype=float)
            iv_i = np.asarray(iv_i, dtype=float)
            try:
                iv_fit = ssvi_implied_vol(k_i, float(T_i), float(th_i), p)
            except Exception:
                return 1e9
            total_sq += float(np.sum((iv_fit - iv_i) ** 2))
        return total_sq

    x0     = np.array([initial.rho, initial.eta, initial.gamma])
    bounds = [(-0.9999, 0.9999), (1e-4, 20.0), (1e-4, 0.9999)]
    res    = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})

    rho_f, eta_f, gamma_f = float(res.x[0]), float(res.x[1]), float(res.x[2])
    best = SSVIParams(rho=rho_f, eta=eta_f, gamma=gamma_f)

    # Compute RMSE and max error
    all_errs: list[float] = []
    for k_i, iv_i, T_i, th_i in zip(k_list, iv_list, T_arr, theta_t):
        k_i  = np.asarray(k_i, dtype=float)
        iv_i = np.asarray(iv_i, dtype=float)
        iv_f = ssvi_implied_vol(k_i, float(T_i), float(th_i), best)
        all_errs.extend((iv_f - iv_i).tolist())

    errs = np.array(all_errs)
    rmse    = float(np.sqrt(np.mean(errs**2)))
    max_err = float(np.max(np.abs(errs)))

    return SSVIFitResult(
        params=best,
        theta_t=theta_t,
        rmse=rmse,
        max_err=max_err,
        butterfly_free=ssvi_check_butterfly_free(best),
        calendar_free=ssvi_check_calendar_free(theta_t, best),
    )
