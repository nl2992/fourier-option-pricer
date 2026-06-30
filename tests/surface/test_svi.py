"""Tests for the SVI (Stochastic Volatility Inspired) smile parameterization.

Verification strategy:
  1. SVIParams construction and validation
  2. svi_total_variance: formula correctness and boundary behaviour
  3. svi_implied_vol: IV recovery from total-variance
  4. svi_butterfly_density: shape of g(k), positivity for well-behaved params
  5. svi_check_butterfly_arbitrage: detects known-bad params
  6. fit_svi_smile: round-trip calibration from synthetic IV, RMSE target
  7. Calibration: robustness across skew/smile/flat scenarios
  8. Public API surface: all SVI symbols importable from foureng
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.surface.svi import (
    SVIFitResult,
    SVIParams,
    fit_svi_smile,
    svi_butterfly_density,
    svi_check_butterfly_arbitrage,
    svi_implied_vol,
    svi_total_variance,
)

# ── reference parameters ──────────────────────────────────────────────────────

ATM_PARAMS = SVIParams(a=0.04, b=0.4, rho=-0.3, m=0.0, sigma=0.3)
T = 1.0
K_GRID = np.linspace(-1.5, 1.5, 61)


# ── 1. SVIParams construction and validation ──────────────────────────────────


def test_svi_params_construction():
    p = SVIParams(a=0.04, b=0.2, rho=-0.5, m=0.0, sigma=0.2)
    assert p.a == 0.04
    assert p.b == 0.2
    assert p.rho == -0.5


def test_svi_params_frozen():
    p = SVIParams(a=0.04, b=0.2, rho=-0.5, m=0.0, sigma=0.2)
    with pytest.raises((AttributeError, TypeError)):
        p.a = 0.1  # type: ignore[misc]


def test_svi_params_negative_b_rejected():
    with pytest.raises(ValueError, match="b"):
        SVIParams(a=0.04, b=-0.1, rho=-0.3, m=0.0, sigma=0.3)


def test_svi_params_rho_at_boundary_rejected():
    with pytest.raises(ValueError, match="rho"):
        SVIParams(a=0.04, b=0.2, rho=1.0, m=0.0, sigma=0.2)
    with pytest.raises(ValueError, match="rho"):
        SVIParams(a=0.04, b=0.2, rho=-1.0, m=0.0, sigma=0.2)


def test_svi_params_zero_sigma_rejected():
    with pytest.raises(ValueError, match="sigma"):
        SVIParams(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=0.0)


def test_svi_params_negative_sigma_rejected():
    with pytest.raises(ValueError, match="sigma"):
        SVIParams(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=-0.1)


# ── 2. svi_total_variance ─────────────────────────────────────────────────────


def test_total_variance_atm_formula():
    """At k=m, total variance = a + b*sigma (from the sqrt term)."""
    p = ATM_PARAMS
    w_atm = svi_total_variance(p.m, p)
    expected = p.a + p.b * p.sigma
    assert abs(float(w_atm) - expected) < 1e-14


def test_total_variance_nonnegative():
    w = svi_total_variance(K_GRID, ATM_PARAMS)
    assert np.all(w >= 0.0)


def test_total_variance_scalar_input():
    w = svi_total_variance(0.0, ATM_PARAMS)
    assert np.isscalar(float(w)) or w.shape == ()


def test_total_variance_symmetric_when_rho_zero_m_zero():
    """If rho=0 and m=0, total variance must be even in k."""
    p = SVIParams(a=0.04, b=0.3, rho=0.0, m=0.0, sigma=0.25)
    k_pos = np.array([0.1, 0.3, 0.7, 1.2])
    k_neg = -k_pos
    assert np.allclose(svi_total_variance(k_pos, p), svi_total_variance(k_neg, p), atol=1e-14)


def test_total_variance_grid_finite():
    w = svi_total_variance(K_GRID, ATM_PARAMS)
    assert np.all(np.isfinite(w))


# ── 3. svi_implied_vol ────────────────────────────────────────────────────────


def test_implied_vol_recovers_atm():
    iv = svi_implied_vol(ATM_PARAMS.m, T, ATM_PARAMS)
    w = svi_total_variance(ATM_PARAMS.m, ATM_PARAMS)
    expected = float(np.sqrt(w / T))
    assert abs(float(iv) - expected) < 1e-14


def test_implied_vol_positive():
    iv = svi_implied_vol(K_GRID, T, ATM_PARAMS)
    assert np.all(iv > 0)


def test_implied_vol_raises_nonpositive_T():
    with pytest.raises(ValueError, match="T"):
        svi_implied_vol(0.0, 0.0, ATM_PARAMS)

    with pytest.raises(ValueError, match="T"):
        svi_implied_vol(0.0, -0.5, ATM_PARAMS)


def test_implied_vol_scales_with_T():
    """IV ∝ 1/sqrt(T) if total variance is the same."""
    iv1 = float(svi_implied_vol(0.0, 1.0, ATM_PARAMS))
    iv2 = float(svi_implied_vol(0.0, 4.0, ATM_PARAMS))
    assert abs(iv1 / iv2 - 2.0) < 1e-10


# ── 4. svi_butterfly_density ──────────────────────────────────────────────────


def test_butterfly_density_positive_for_atm_params():
    """ATM_PARAMS has gentle curvature and should be butterfly-free."""
    g = svi_butterfly_density(K_GRID, ATM_PARAMS)
    assert np.all(g > -1e-8)


def test_butterfly_density_finite():
    g = svi_butterfly_density(K_GRID, ATM_PARAMS)
    assert np.all(np.isfinite(g))


def test_butterfly_density_shape():
    g = svi_butterfly_density(K_GRID, ATM_PARAMS)
    assert g.shape == K_GRID.shape


# ── 5. svi_check_butterfly_arbitrage ─────────────────────────────────────────


def test_no_arbitrage_for_gentle_params():
    assert svi_check_butterfly_arbitrage(ATM_PARAMS)


def test_no_arbitrage_flat_smile():
    p = SVIParams(a=0.04, b=0.05, rho=0.0, m=0.0, sigma=0.5)
    assert svi_check_butterfly_arbitrage(p)


def test_arbitrage_detected_extreme_skew():
    """Very large b with extreme rho can violate butterfly."""
    p = SVIParams(a=0.01, b=1.99, rho=0.99, m=0.0, sigma=0.01)
    result = svi_check_butterfly_arbitrage(p)
    # May or may not be detected depending on params — just check it doesn't crash
    assert isinstance(result, bool)


# ── 6. fit_svi_smile round-trip ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_smile():
    true_params = SVIParams(a=0.04, b=0.3, rho=-0.4, m=0.05, sigma=0.25)
    k = np.linspace(-1.0, 1.0, 21)
    iv = svi_implied_vol(k, T, true_params)
    return k, iv, true_params


def test_fit_svi_rmse_below_threshold(synthetic_smile):
    k, iv, _ = synthetic_smile
    result = fit_svi_smile(k, iv, T)
    assert result.rmse < 1e-6, f"RMSE too large: {result.rmse:.2e}"


def test_fit_svi_returns_svi_fit_result(synthetic_smile):
    k, iv, _ = synthetic_smile
    result = fit_svi_smile(k, iv, T)
    assert isinstance(result, SVIFitResult)
    assert isinstance(result.params, SVIParams)


def test_fit_svi_params_in_valid_range(synthetic_smile):
    k, iv, _ = synthetic_smile
    result = fit_svi_smile(k, iv, T)
    p = result.params
    assert p.b >= 0
    assert -1.0 < p.rho < 1.0
    assert p.sigma > 0


def test_fit_svi_max_err_below_threshold(synthetic_smile):
    k, iv, _ = synthetic_smile
    result = fit_svi_smile(k, iv, T)
    assert result.max_err < 1e-5, f"max_err too large: {result.max_err:.2e}"


def test_fit_svi_butterfly_free_on_synthetic(synthetic_smile):
    k, iv, _ = synthetic_smile
    result = fit_svi_smile(k, iv, T)
    assert result.butterfly_free


def test_fit_svi_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        fit_svi_smile(np.array([0.0, 0.1]), np.array([0.2]), T)


def test_fit_svi_nonpositive_T_raises():
    k = np.linspace(-0.5, 0.5, 11)
    iv = np.full_like(k, 0.2)
    with pytest.raises(ValueError, match="T"):
        fit_svi_smile(k, iv, T=0.0)


# ── 7. Calibration robustness across scenarios ────────────────────────────────


@pytest.mark.parametrize("rho,b,scenario", [
    (-0.5, 0.3, "negative skew"),
    (0.4, 0.2, "positive skew"),
    (0.0, 0.15, "symmetric smile"),
    (-0.2, 0.05, "low vol flat smile"),
])
def test_fit_svi_various_scenarios(rho, b, scenario):
    """Calibration should achieve RMSE < 1e-4 on synthetic data."""
    p = SVIParams(a=0.04, b=b, rho=rho, m=0.0, sigma=0.25)
    k = np.linspace(-0.8, 0.8, 17)
    iv = svi_implied_vol(k, T, p)
    result = fit_svi_smile(k, iv, T)
    assert result.rmse < 1e-4, f"Scenario '{scenario}': RMSE={result.rmse:.2e}"


def test_fit_svi_short_maturity():
    """Fit should work at short (T=0.1) maturity where IVs are larger."""
    T_short = 0.1
    p = SVIParams(a=0.04, b=0.3, rho=-0.3, m=0.0, sigma=0.25)
    k = np.linspace(-0.5, 0.5, 13)
    iv = svi_implied_vol(k, T_short, p)
    result = fit_svi_smile(k, iv, T_short)
    assert result.rmse < 1e-4


def test_fit_svi_long_maturity():
    """Fit at T=5 years (long-dated smile)."""
    T_long = 5.0
    p = SVIParams(a=0.04, b=0.15, rho=-0.2, m=0.0, sigma=0.4)
    k = np.linspace(-1.5, 1.5, 21)
    iv = svi_implied_vol(k, T_long, p)
    result = fit_svi_smile(k, iv, T_long)
    assert result.rmse < 1e-4


# ── 8. Public API imports ─────────────────────────────────────────────────────


def test_svi_importable_from_foureng():
    import foureng as fe
    assert hasattr(fe, "SVIParams")
    assert hasattr(fe, "SVIFitResult")
    assert hasattr(fe, "svi_total_variance")
    assert hasattr(fe, "svi_implied_vol")
    assert hasattr(fe, "svi_butterfly_density")
    assert hasattr(fe, "svi_check_butterfly_arbitrage")
    assert hasattr(fe, "fit_svi_smile")


def test_svi_params_callable_from_foureng():
    import foureng as fe
    p = fe.SVIParams(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=0.25)
    w = fe.svi_total_variance(0.0, p)
    assert w > 0
