"""Tests for the Dupire (1994) local volatility surface extraction.

Verification strategy:
  1. LocalVolSurface construction and shape checks
  2. Flat implied vol -> flat local vol (sigma_loc = sigma_BS everywhere)
  3. Positivity and finiteness of the local vol surface
  4. dupire_local_vol_from_svi: round-trip with SVI smiles
  5. dupire_local_vol_grid: numerical route matches SVI analytical route
  6. Monotone input checks
  7. Error handling (too few maturities, mismatched shapes)
  8. Public API: all local vol symbols importable from foureng
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.surface.local_vol import (
    LocalVolSurface,
    dupire_local_vol_from_svi,
    dupire_local_vol_grid,
)
from foureng.surface.svi import SVIParams, svi_implied_vol

# ── shared fixtures ───────────────────────────────────────────────────────────

K_GRID = np.linspace(-1.0, 1.0, 31)  # log-moneyness
MATURITIES = np.array([0.25, 0.5, 1.0, 2.0])

# Flat smile: a = sigma^2 * T, b = 0, so w(k) = sigma^2 * T for all k
SIGMA_FLAT = 0.25


def _flat_svi(T: float, sigma: float = SIGMA_FLAT) -> SVIParams:
    """Return SVI params giving flat smile at vol=sigma for maturity T."""
    a = sigma**2 * T
    return SVIParams(a=a, b=1e-8, rho=0.0, m=0.0, sigma=0.3)


def _skew_svi(T: float) -> SVIParams:
    """Return SVI params with a mild negative skew."""
    a = 0.04 * T
    return SVIParams(a=a, b=0.3 * np.sqrt(T), rho=-0.4, m=0.0, sigma=0.25)


# ── 1. Return type and shape ─────────────────────────────────────────────────


def test_local_vol_surface_is_dataclass():
    params = [_flat_svi(T) for T in MATURITIES]
    lv = dupire_local_vol_from_svi(K_GRID, MATURITIES, params)
    assert isinstance(lv, LocalVolSurface)


def test_shape_from_svi():
    params = [_flat_svi(T) for T in MATURITIES]
    lv = dupire_local_vol_from_svi(K_GRID, MATURITIES, params)
    nT_mid = len(MATURITIES) - 1
    assert lv.local_var.shape == (nT_mid, len(K_GRID))
    assert lv.local_vol.shape == (nT_mid, len(K_GRID))
    assert len(lv.maturities) == nT_mid
    assert len(lv.log_moneyness) == len(K_GRID)


def test_midpoint_maturities_from_svi():
    params = [_flat_svi(T) for T in MATURITIES]
    lv = dupire_local_vol_from_svi(K_GRID, MATURITIES, params)
    expected = 0.5 * (MATURITIES[:-1] + MATURITIES[1:])
    np.testing.assert_allclose(lv.maturities, expected)


# ── 2. Flat implied vol -> flat local vol ─────────────────────────────────────


def test_flat_smile_gives_constant_local_vol():
    """w(k, T) = sigma^2 * T => dw/dT = sigma^2, g(k, w) = 1 => sigma_loc = sigma."""
    params = [_flat_svi(T) for T in MATURITIES]
    lv = dupire_local_vol_from_svi(K_GRID, MATURITIES, params)
    # Interior k points only (edge effects from w'_k = 0 near b=0)
    k_interior = np.abs(K_GRID) < 0.8
    sigma_loc_interior = lv.local_vol[:, k_interior]
    expected = SIGMA_FLAT * np.ones_like(sigma_loc_interior)
    np.testing.assert_allclose(sigma_loc_interior, expected, rtol=1e-2, atol=1e-3)


# ── 3. Non-negativity and finiteness ──────────────────────────────────────────


def test_local_var_non_negative_from_svi():
    params = [_skew_svi(T) for T in MATURITIES]
    lv = dupire_local_vol_from_svi(K_GRID, MATURITIES, params)
    assert np.all(lv.local_var >= 0.0)


def test_local_vol_finite_from_svi():
    params = [_skew_svi(T) for T in MATURITIES]
    lv = dupire_local_vol_from_svi(K_GRID, MATURITIES, params)
    assert np.all(np.isfinite(lv.local_vol))


def test_local_vol_positive_interior():
    """LV should be strictly positive away from extreme strikes."""
    params = [_skew_svi(T) for T in MATURITIES]
    lv = dupire_local_vol_from_svi(K_GRID, MATURITIES, params)
    k_interior = np.abs(K_GRID) < 0.6
    assert np.all(lv.local_vol[:, k_interior] > 0.0)


# ── 4. Two-maturity minimum ───────────────────────────────────────────────────


def test_two_maturities_minimum():
    T2 = np.array([0.5, 1.0])
    params = [_skew_svi(T) for T in T2]
    lv = dupire_local_vol_from_svi(K_GRID, T2, params)
    assert lv.local_var.shape == (1, len(K_GRID))


# ── 5. Error handling (SVI route) ────────────────────────────────────────────


def test_raises_too_few_maturities():
    with pytest.raises(ValueError, match="2 maturities"):
        dupire_local_vol_from_svi(K_GRID, np.array([1.0]), [_flat_svi(1.0)])


def test_raises_mismatched_params():
    with pytest.raises(ValueError, match="len.svi_params"):
        dupire_local_vol_from_svi(K_GRID, MATURITIES, [_flat_svi(1.0)])


def test_raises_non_increasing_maturities():
    T_bad = np.array([1.0, 0.5, 2.0])
    params = [_flat_svi(T) for T in T_bad]
    with pytest.raises(ValueError, match="strictly increasing"):
        dupire_local_vol_from_svi(K_GRID, T_bad, params)


# ── 6. dupire_local_vol_grid (numerical) ─────────────────────────────────────


def _build_iv_grid(T_arr, k_arr, svi_params_list):
    """Build an IV grid from SVI params at multiple maturities."""
    nT, nK = len(T_arr), len(k_arr)
    iv = np.zeros((nT, nK))
    for i, (T, p) in enumerate(zip(T_arr, svi_params_list)):
        iv[i] = svi_implied_vol(k_arr, T, p)
    return iv


def test_grid_route_shape():
    params = [_skew_svi(T) for T in MATURITIES]
    iv = _build_iv_grid(MATURITIES, K_GRID, params)
    lv = dupire_local_vol_grid(MATURITIES, K_GRID, iv)
    assert lv.local_var.shape == (len(MATURITIES) - 1, len(K_GRID))


def test_grid_route_non_negative():
    params = [_skew_svi(T) for T in MATURITIES]
    iv = _build_iv_grid(MATURITIES, K_GRID, params)
    lv = dupire_local_vol_grid(MATURITIES, K_GRID, iv)
    assert np.all(lv.local_var >= 0.0)


def test_grid_route_finite():
    params = [_skew_svi(T) for T in MATURITIES]
    iv = _build_iv_grid(MATURITIES, K_GRID, params)
    lv = dupire_local_vol_grid(MATURITIES, K_GRID, iv)
    assert np.all(np.isfinite(lv.local_vol))


def test_grid_route_flat_smile():
    """Flat IV grid should give approx flat LV."""
    # Dense grid for finite-diff accuracy
    T_dense = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    k_dense = np.linspace(-0.8, 0.8, 51)
    params_dense = [_flat_svi(T) for T in T_dense]
    iv_dense = _build_iv_grid(T_dense, k_dense, params_dense)
    lv = dupire_local_vol_grid(T_dense, k_dense, iv_dense)
    # Interior k (exclude edges where FD accuracy drops)
    k_mid = np.abs(k_dense) < 0.5
    lv_interior = lv.local_vol[:, k_mid]
    np.testing.assert_allclose(lv_interior, SIGMA_FLAT, rtol=0.03, atol=0.01)


def test_grid_route_raises_too_few_maturities():
    with pytest.raises(ValueError, match="2 maturities"):
        dupire_local_vol_grid(np.array([1.0]), K_GRID, np.ones((1, len(K_GRID))))


def test_grid_route_raises_shape_mismatch():
    with pytest.raises((ValueError, Exception)):
        dupire_local_vol_grid(
            MATURITIES,
            K_GRID,
            np.ones((len(MATURITIES), len(K_GRID) + 1)),  # wrong K size
        )


def test_grid_route_raises_non_increasing_k():
    T2 = np.array([0.5, 1.0])
    k_bad = np.array([0.5, -0.5, 0.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        dupire_local_vol_grid(T2, k_bad, np.ones((2, 4)))


# ── 7. SVI route vs grid route consistency ────────────────────────────────────


def test_svi_route_consistent_with_grid_route():
    """Both routes should give broadly consistent LV on the same surface."""
    T_arr = np.array([0.25, 0.5, 1.0, 2.0])
    k_arr = np.linspace(-0.7, 0.7, 41)
    params = [_skew_svi(T) for T in T_arr]
    iv = _build_iv_grid(T_arr, k_arr, params)

    lv_svi = dupire_local_vol_from_svi(k_arr, T_arr, params)
    lv_grid = dupire_local_vol_grid(T_arr, k_arr, iv)

    k_interior = np.abs(k_arr) < 0.5
    svi_int = lv_svi.local_vol[:, k_interior]
    grid_int = lv_grid.local_vol[:, k_interior]

    # Grid route uses FD so allow 10% relative tolerance
    np.testing.assert_allclose(svi_int, grid_int, rtol=0.10, atol=0.005)


# ── 8. Public API ─────────────────────────────────────────────────────────────


def test_importable_from_foureng():
    import foureng as fe

    assert hasattr(fe, "LocalVolSurface")
    assert hasattr(fe, "dupire_local_vol_from_svi")
    assert hasattr(fe, "dupire_local_vol_grid")


def test_callable_from_foureng():
    import foureng as fe

    params = [_flat_svi(T) for T in MATURITIES]
    lv = fe.dupire_local_vol_from_svi(K_GRID, MATURITIES, params)
    assert isinstance(lv, fe.LocalVolSurface)
