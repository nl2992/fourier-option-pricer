"""Local volatility (Dupire 1994) surface extraction.

Dupire (1994) showed that the entire risk-neutral dynamics of an asset can
be captured by a *local volatility* function sigma_loc(K, T), uniquely
determined by the market implied-vol surface via:

    sigma_loc^2(K, T) = dw/dT / g(k, w)

where w(k, T) = sigma_BS^2(k, T) * T is the total implied variance,
k = ln(K/F(T)) is log-moneyness, and g is the Gatheral-Jacquier density
weight:

    g(k, w) = (1 - k*w'_k/(2w))^2 - (w'_k)^2*(1/4 + 1/w)/4 + w''_kk/2

Two routes are supported:

1. **Analytical from SVI smiles** (``dupire_local_vol_from_svi``):
   Given a list of SVI parameter sets at multiple maturities, computes
   analytical k-derivatives and finite-difference T-derivatives.  Fastest
   and most stable when the SVI fits are already available.

2. **Numerical from an IV grid** (``dupire_local_vol_grid``):
   Given a (nT x nK) matrix of market implied vols, uses central finite
   differences in both T and k.  More general but requires a dense grid and
   some smoothing to avoid noise amplification.

References:
    Dupire, B. (1994). Pricing with a smile.
    *Risk*, 7(1), 18-20.

    Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*.
    Wiley Finance.

    Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
    *Quantitative Finance*, 14(1), 59-71.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .svi import SVIParams, svi_total_variance


# ── helpers ───────────────────────────────────────────────────────────────────


def _svi_w_derivatives(k: np.ndarray, p: SVIParams):
    """Return w, dw/dk, d2w/dk2 analytically from SVI params."""
    d  = k - p.m
    sq = np.sqrt(d**2 + p.sigma**2)
    b, rho = p.b, p.rho

    w   = p.a + b * (rho * d + sq)
    w1  = b * (rho + d / sq)                     # dw/dk
    w2  = b * p.sigma**2 / sq**3                 # d2w/dk2

    return np.maximum(w, 1e-16), w1, w2


# ── Route 1: analytical from SVI ─────────────────────────────────────────────


@dataclass(frozen=True)
class LocalVolSurface:
    """Result of a Dupire local-vol extraction.

    Attributes
    ----------
    log_moneyness : ndarray, shape (nK,)
        Log-moneyness grid k = ln(K/F).
    maturities : ndarray, shape (nT,)
        Maturity grid (years); for the SVI route these are the T midpoints.
    local_var : ndarray, shape (nT, nK)
        Dupire local variance sigma_loc^2(k, T).  Negative entries are
        clipped to zero.
    local_vol : ndarray, shape (nT, nK)
        Local volatility sigma_loc = sqrt(max(local_var, 0)).
    """

    log_moneyness: np.ndarray
    maturities: np.ndarray
    local_var: np.ndarray
    local_vol: np.ndarray


def dupire_local_vol_from_svi(
    log_moneyness: np.ndarray,
    maturities: np.ndarray,
    svi_params: Sequence[SVIParams],
    *,
    r: float = 0.0,
    q: float = 0.0,
    clip_negative: bool = True,
) -> LocalVolSurface:
    """Compute Dupire local-vol surface from SVI smiles at multiple maturities.

    The k-derivatives of w are obtained analytically from the SVI formula.
    The T-derivative is approximated by central (interior) or forward/backward
    (boundary) finite differences between adjacent maturities.

    Parameters
    ----------
    log_moneyness : ndarray, shape (nK,)
        Log-moneyness grid k = ln(K/F).  Should cover the range of interest.
    maturities : ndarray, shape (nT,)
        Maturities in years, strictly increasing.  Must have nT >= 2.
    svi_params : sequence of SVIParams, length nT
        One calibrated SVI parameter set per maturity, same order as maturities.
    r, q : float
        Risk-free rate and dividend yield (for the cost-of-carry in the
        log-moneyness shift, if r != q).  Not needed for the formula itself
        but documented for completeness.
    clip_negative : bool
        If True (default), clip local_var to >= 0 before taking square root.

    Returns
    -------
    LocalVolSurface
        Local vol evaluated at midpoint maturities (len(maturities) - 1 rows)
        and on the full log_moneyness grid.
    """
    k    = np.asarray(log_moneyness, dtype=float)
    T    = np.asarray(maturities, dtype=float)
    nT   = len(T)
    nK   = len(k)

    if nT < 2:
        raise ValueError("Need at least 2 maturities to compute dw/dT.")
    if len(svi_params) != nT:
        raise ValueError(
            f"len(svi_params)={len(svi_params)} must match len(maturities)={nT}"
        )
    if not np.all(np.diff(T) > 0):
        raise ValueError("maturities must be strictly increasing.")

    # Build w(k, T) matrix — shape (nT, nK)
    W = np.array([svi_total_variance(k, p) for p in svi_params])  # (nT, nK)

    # Midpoint maturities
    T_mid  = 0.5 * (T[:-1] + T[1:])
    nT_mid = len(T_mid)

    local_var = np.empty((nT_mid, nK))

    for i in range(nT_mid):
        T1, T2 = T[i], T[i + 1]
        p1, p2 = svi_params[i], svi_params[i + 1]
        dT = T2 - T1

        # Evaluate at the midpoint SVI params (linear blend in parameter space)
        # We use simple finite diff: dw/dT ≈ (w(k,T2) - w(k,T1)) / (T2 - T1)
        dw_dT = (W[i + 1] - W[i]) / dT

        # Use average k-derivatives at the midpoint
        w1a, w1_k_a, w2_k_a = _svi_w_derivatives(k, p1)
        w1b, w1_k_b, w2_k_b = _svi_w_derivatives(k, p2)

        w_mid  = 0.5 * (w1a  + w1b)
        w1_k   = 0.5 * (w1_k_a + w1_k_b)
        w2_k   = 0.5 * (w2_k_a + w2_k_b)

        w_mid = np.maximum(w_mid, 1e-16)

        # Gatheral-Jacquier denominator g(k, w)
        g = (1.0 - k * w1_k / (2.0 * w_mid))**2 \
            - (w1_k**2 / 4.0) * (1.0 / 4.0 + 1.0 / w_mid) \
            + w2_k / 2.0

        # Avoid division by zero or negative g
        g = np.where(g < 1e-12, np.nan, g)

        lv2 = dw_dT / g

        if clip_negative:
            lv2 = np.where(np.isnan(lv2), 0.0, np.maximum(lv2, 0.0))

        local_var[i] = lv2

    local_vol = np.sqrt(local_var)

    return LocalVolSurface(
        log_moneyness=k,
        maturities=T_mid,
        local_var=local_var,
        local_vol=local_vol,
    )


# ── Route 2: numerical from IV grid ──────────────────────────────────────────


def dupire_local_vol_grid(
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    iv_surface: np.ndarray,
    *,
    smooth_sigma: float = 0.0,
    clip_negative: bool = True,
) -> LocalVolSurface:
    """Compute Dupire local-vol surface from a market implied-vol grid.

    Uses central finite differences for interior points and one-sided
    differences at boundaries.  No smoothing is applied by default; if the
    IV surface has market noise, pass ``smooth_sigma > 0`` to apply a
    Gaussian blur before differentiating.

    Parameters
    ----------
    maturities : ndarray, shape (nT,)
        Maturity grid, strictly increasing.  nT >= 3 recommended.
    log_moneyness : ndarray, shape (nK,)
        Log-moneyness grid k = ln(K/F), strictly increasing.
    iv_surface : ndarray, shape (nT, nK)
        Market implied volatilities sigma_BS(k, T).
    smooth_sigma : float
        Standard deviation (in grid-index units) for Gaussian smoothing before
        differentiating.  0 = no smoothing.
    clip_negative : bool
        If True, clip local_var to >= 0 before square-root.

    Returns
    -------
    LocalVolSurface
        Local vol on the interior T-midpoint grid (nT-1 rows).
    """
    T = np.asarray(maturities, dtype=float)
    k = np.asarray(log_moneyness, dtype=float)
    iv = np.asarray(iv_surface, dtype=float)

    nT = len(T)
    nK = len(k)
    if nT < 2:
        raise ValueError("Need at least 2 maturities.")
    if iv.shape != (nT, nK):
        raise ValueError(
            f"iv_surface shape {iv.shape} doesn't match ({nT}, {nK})"
        )
    if not np.all(np.diff(T) > 0):
        raise ValueError("maturities must be strictly increasing.")
    if not np.all(np.diff(k) > 0):
        raise ValueError("log_moneyness must be strictly increasing.")

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        iv = gaussian_filter(iv, sigma=smooth_sigma)

    # Total variance w(T, k) = iv^2 * T,  shape (nT, nK)
    W = iv**2 * T[:, np.newaxis]

    # --- dw/dT: finite differences along T axis ---
    dW_dT = np.empty_like(W)
    for i in range(nT):
        if i == 0:
            dW_dT[i] = (W[1] - W[0]) / (T[1] - T[0])
        elif i == nT - 1:
            dW_dT[i] = (W[-1] - W[-2]) / (T[-1] - T[-2])
        else:
            dW_dT[i] = (W[i + 1] - W[i - 1]) / (T[i + 1] - T[i - 1])

    # --- dw/dk, d2w/dk2: finite differences along k axis ---
    dk = np.gradient(k)  # non-uniform grid step

    dW_dk  = np.gradient(W, k, axis=1)
    d2W_dk = np.gradient(dW_dk, k, axis=1)

    # Evaluate at midpoint T grid
    T_mid = 0.5 * (T[:-1] + T[1:])
    nT_mid = len(T_mid)
    local_var = np.empty((nT_mid, nK))

    for i in range(nT_mid):
        # Interpolate at midpoint
        w_mid   = 0.5 * (W[i]      + W[i + 1])
        dw_dT_m = 0.5 * (dW_dT[i] + dW_dT[i + 1])
        w1_k    = 0.5 * (dW_dk[i]  + dW_dk[i + 1])
        w2_k    = 0.5 * (d2W_dk[i] + d2W_dk[i + 1])

        w_mid = np.maximum(w_mid, 1e-16)
        k_mid = k  # same k grid

        # Gatheral-Jacquier denominator g(k, w)
        g = (1.0 - k_mid * w1_k / (2.0 * w_mid))**2 \
            - (w1_k**2 / 4.0) * (1.0 / 4.0 + 1.0 / w_mid) \
            + w2_k / 2.0

        g = np.where(g < 1e-12, np.nan, g)
        lv2 = dw_dT_m / g

        if clip_negative:
            lv2 = np.where(np.isnan(lv2), 0.0, np.maximum(lv2, 0.0))

        local_var[i] = lv2

    local_vol = np.sqrt(local_var)

    return LocalVolSurface(
        log_moneyness=k,
        maturities=T_mid,
        local_var=local_var,
        local_vol=local_vol,
    )
