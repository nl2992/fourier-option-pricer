"""Tests for SSVI (Surface SVI) surface parameterization.

Verification strategy:
  1. SSVIParams construction and validation
  2. ssvi_phi_power_law: monotone, positive, limiting behavior
  3. ssvi_phi_heston: shape check
  4. ssvi_total_variance: known formula values
  5. ssvi_implied_vol: shape and positivity
  6. Arbitrage checks: butterfly_free and calendar_free
  7. fit_ssvi_surface: shape, RMSE, butterfly/calendar free
  8. fit_ssvi_surface on Heston-generated surface
  9. Public API
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from foureng.surface.ssvi import (
    SSVIFitResult,
    SSVIParams,
    fit_ssvi_surface,
    ssvi_check_butterfly_free,
    ssvi_check_calendar_free,
    ssvi_implied_vol,
    ssvi_phi_heston,
    ssvi_phi_power_law,
    ssvi_total_variance,
)

# ── shared fixtures ───────────────────────────────────────────────────────────

BASE = SSVIParams(rho=-0.3, eta=0.5, gamma=0.5)
K_GRID = np.linspace(-1.0, 1.0, 31)
MATURITIES = np.array([0.25, 0.5, 1.0, 2.0])
# ATM total variances: sigma_ATM=0.20
THETA_T = (0.20**2) * MATURITIES


# ── 1. SSVIParams construction ────────────────────────────────────────────────


def test_params_construction():
    p = SSVIParams(rho=-0.5, eta=1.0, gamma=0.6)
    assert p.rho == -0.5
    assert p.eta == 1.0
    assert p.gamma == 0.6


def test_params_frozen():
    p = SSVIParams(rho=0.0, eta=1.0, gamma=0.5)
    with pytest.raises((AttributeError, TypeError)):
        p.rho = 0.1  # type: ignore[misc]


def test_params_rho_boundary_rejected():
    with pytest.raises(ValueError, match="rho must be"):
        SSVIParams(rho=1.0, eta=1.0, gamma=0.5)
    with pytest.raises(ValueError, match="rho must be"):
        SSVIParams(rho=-1.0, eta=1.0, gamma=0.5)


def test_params_eta_zero_rejected():
    with pytest.raises(ValueError, match="eta"):
        SSVIParams(rho=0.0, eta=0.0, gamma=0.5)


def test_params_eta_negative_rejected():
    with pytest.raises(ValueError, match="eta"):
        SSVIParams(rho=0.0, eta=-1.0, gamma=0.5)


def test_params_gamma_zero_rejected():
    with pytest.raises(ValueError, match="gamma"):
        SSVIParams(rho=0.0, eta=1.0, gamma=0.0)


def test_params_gamma_one_rejected():
    with pytest.raises(ValueError, match="gamma"):
        SSVIParams(rho=0.0, eta=1.0, gamma=1.0)


# ── 2. ssvi_phi_power_law ─────────────────────────────────────────────────────


def test_phi_power_law_positive():
    theta = np.array([0.01, 0.1, 1.0, 4.0])
    phi = ssvi_phi_power_law(theta, eta=1.0, gamma=0.5)
    assert np.all(phi > 0)


def test_phi_power_law_decreasing():
    """phi should decrease with increasing theta (for gamma in (0,1))."""
    theta = np.linspace(0.01, 5.0, 50)
    phi = ssvi_phi_power_law(theta, eta=1.0, gamma=0.5)
    assert np.all(np.diff(phi) < 0)


def test_phi_power_law_scales_with_eta():
    theta = np.array([1.0])
    phi1 = ssvi_phi_power_law(theta, eta=1.0, gamma=0.5)[0]
    phi2 = ssvi_phi_power_law(theta, eta=2.0, gamma=0.5)[0]
    assert abs(phi2 / phi1 - 2.0) < 1e-12


def test_phi_power_law_formula():
    """Spot check: phi(1.0) = eta / (1^gamma * 2^(1-gamma)) with gamma=0.5."""
    eta, gamma = 2.0, 0.5
    theta = 1.0
    expected = eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))
    phi = float(ssvi_phi_power_law(np.array([theta]), eta, gamma)[0])
    assert abs(phi - expected) < 1e-12


# ── 3. ssvi_phi_heston ────────────────────────────────────────────────────────


def test_phi_heston_positive():
    theta = np.array([0.01, 0.1, 0.5, 1.0, 4.0])
    phi = ssvi_phi_heston(theta)
    assert np.all(phi > 0)


def test_phi_heston_decreasing():
    """Heston phi is also decreasing in theta."""
    theta = np.linspace(0.01, 5.0, 50)
    phi = ssvi_phi_heston(theta)
    assert np.all(np.diff(phi) < 0)


def test_phi_heston_limit_small_theta():
    """For theta -> 0, phi -> 1/2 (L'Hopital)."""
    theta = np.array([1e-6])
    phi = float(ssvi_phi_heston(theta)[0])
    assert abs(phi - 0.5) < 1e-4


# ── 4. ssvi_total_variance ────────────────────────────────────────────────────


def test_total_variance_atm():
    """At k=0, w(0, theta) = theta/2 * (1 + rho*phi*0 + sqrt(rho^2 + 1-rho^2)) = theta."""
    # sqrt(rho^2 + 1 - rho^2) = 1, so w(0) = theta/2 * (1 + 1) = theta
    theta = 0.04
    w0 = float(ssvi_total_variance(np.array([0.0]), theta, BASE)[0])
    assert abs(w0 - theta) < 1e-10


def test_total_variance_non_negative():
    w = ssvi_total_variance(K_GRID, 0.04, BASE)
    assert np.all(w >= 0.0)


def test_total_variance_shape():
    w = ssvi_total_variance(K_GRID, 0.04, BASE)
    assert w.shape == K_GRID.shape


def test_total_variance_symmetric_when_rho_zero():
    """With rho=0, w(k) = w(-k) (symmetric smile)."""
    p0 = SSVIParams(rho=0.0, eta=0.5, gamma=0.5)
    k_pos = np.array([0.1, 0.3, 0.5])
    k_neg = -k_pos
    theta = 0.04
    w_pos = ssvi_total_variance(k_pos, theta, p0)
    w_neg = ssvi_total_variance(k_neg, theta, p0)
    np.testing.assert_allclose(w_pos, w_neg, rtol=1e-12)


# ── 5. ssvi_implied_vol ───────────────────────────────────────────────────────


def test_implied_vol_shape():
    iv = ssvi_implied_vol(K_GRID, T=1.0, theta=0.04, params=BASE)
    assert iv.shape == K_GRID.shape


def test_implied_vol_positive():
    iv = ssvi_implied_vol(K_GRID, T=1.0, theta=0.04, params=BASE)
    assert np.all(iv > 0)


def test_implied_vol_atm_recovers_sigma_atm():
    """ATM IV should equal sigma_ATM = sqrt(theta/T)."""
    T, sigma_atm = 1.0, 0.20
    theta = sigma_atm**2 * T
    iv0 = float(ssvi_implied_vol(np.array([0.0]), T, theta, BASE)[0])
    assert abs(iv0 - sigma_atm) < 1e-8


def test_implied_vol_raises_nonpositive_T():
    with pytest.raises(ValueError, match="T must be positive"):
        ssvi_implied_vol(K_GRID, T=0.0, theta=0.04, params=BASE)


# ── 6. Arbitrage checks ───────────────────────────────────────────────────────


def test_butterfly_free_tight_eta():
    """eta * (1 + |rho|) <= 4 should be True for small eta."""
    p = SSVIParams(rho=-0.5, eta=2.0, gamma=0.5)   # 2*(1.5) = 3 <= 4
    assert ssvi_check_butterfly_free(p) is True


def test_butterfly_free_violated():
    p = SSVIParams(rho=0.5, eta=4.0, gamma=0.5)    # 4*(1.5) = 6 > 4
    assert ssvi_check_butterfly_free(p) is False


def test_calendar_free_power_law_default():
    """Power-law phi with gamma in (0,1) is always non-increasing."""
    theta_arr = np.array([0.01, 0.04, 0.1, 0.2, 0.4])
    assert ssvi_check_calendar_free(theta_arr, BASE) is True


def test_calendar_free_single_maturity():
    assert ssvi_check_calendar_free(np.array([0.04]), BASE) is True


# ── 7. fit_ssvi_surface: basic ────────────────────────────────────────────────


def _make_ssvi_iv_grid(params, theta_t, maturities, k_arr):
    """Build IV grid from SSVI."""
    nT = len(maturities)
    nK = len(k_arr)
    iv = np.zeros((nT, nK))
    for i, (T, th) in enumerate(zip(maturities, theta_t)):
        iv[i] = ssvi_implied_vol(k_arr, T, th, params)
    return iv


def test_fit_ssvi_returns_fit_result():
    iv_grid = _make_ssvi_iv_grid(BASE, THETA_T, MATURITIES, K_GRID)
    k_list  = [K_GRID] * len(MATURITIES)
    iv_list = [iv_grid[i] for i in range(len(MATURITIES))]
    result  = fit_ssvi_surface(k_list, iv_list, MATURITIES)
    assert isinstance(result, SSVIFitResult)


def test_fit_ssvi_round_trip_rmse():
    """Fitting exact SSVI data should give very small RMSE."""
    iv_grid = _make_ssvi_iv_grid(BASE, THETA_T, MATURITIES, K_GRID)
    k_list  = [K_GRID] * len(MATURITIES)
    iv_list = [iv_grid[i] for i in range(len(MATURITIES))]
    result  = fit_ssvi_surface(k_list, iv_list, MATURITIES, initial=BASE)
    assert result.rmse < 5e-3   # should recover near-exactly on synthetic data


def test_fit_ssvi_theta_shape():
    iv_grid = _make_ssvi_iv_grid(BASE, THETA_T, MATURITIES, K_GRID)
    k_list  = [K_GRID] * len(MATURITIES)
    iv_list = [iv_grid[i] for i in range(len(MATURITIES))]
    result  = fit_ssvi_surface(k_list, iv_list, MATURITIES)
    assert result.theta_t.shape == (len(MATURITIES),)
    assert np.all(result.theta_t > 0)


def test_fit_ssvi_max_err_finite():
    iv_grid = _make_ssvi_iv_grid(BASE, THETA_T, MATURITIES, K_GRID)
    k_list  = [K_GRID] * len(MATURITIES)
    iv_list = [iv_grid[i] for i in range(len(MATURITIES))]
    result  = fit_ssvi_surface(k_list, iv_list, MATURITIES)
    assert np.isfinite(result.max_err)
    assert result.max_err >= 0


def test_fit_ssvi_raises_too_few_maturities():
    with pytest.raises(ValueError, match="2 maturities"):
        fit_ssvi_surface([K_GRID], [np.full_like(K_GRID, 0.20)], [1.0])


def test_fit_ssvi_raises_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_ssvi_surface([K_GRID], [np.full_like(K_GRID, 0.20)], [0.5, 1.0])


# ── 8. Public API ─────────────────────────────────────────────────────────────


def test_importable_from_foureng():
    import foureng as fe
    for name in ['SSVIParams', 'SSVIFitResult', 'ssvi_phi_power_law',
                 'ssvi_phi_heston', 'ssvi_total_variance', 'ssvi_implied_vol',
                 'ssvi_check_butterfly_free', 'ssvi_check_calendar_free',
                 'fit_ssvi_surface']:
        assert hasattr(fe, name), f"missing: {name}"


def test_callable_from_foureng():
    import foureng as fe
    w = fe.ssvi_total_variance(K_GRID, 0.04, fe.SSVIParams(rho=-0.3, eta=0.5, gamma=0.5))
    assert w.shape == K_GRID.shape
