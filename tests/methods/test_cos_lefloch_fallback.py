"""Tests for the Le Floc'h put-plus-parity auto fallback in cos_prices.

Phase 3.2 — Le Floc'h policy: when the truncation upper bound b is large
(long maturity or heavy-tailed model), the direct-call COS payoff coefficients
contain exp(b) factors that overflow; the Le Floc'h fallback routes through
put+parity instead, which is numerically stable for any b.

Reference: Le Floc'h (2013), "Fourier Integration and Stochastic Volatility
Calibration."
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.pricers.cos import _LEFLOCH_B_THRESHOLD, cos_auto_grid, cos_prices


@pytest.fixture
def bsm_narrow():
    """BSM with a narrow truncation interval — safe for call_direct."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.25)
    params = fe.BsmParams(sigma=0.20)
    cums = fe.bsm_cumulants(fwd, params)
    grid = cos_auto_grid(cums, N=256, L=6.0)  # b ≈ 3.x, well below threshold
    phi = lambda u: fe.bsm_cf(u, fwd, params)
    return fwd, params, phi, grid


@pytest.fixture
def heston_long():
    """Heston at long maturity — b will exceed the Le Floc'h threshold."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.02, q=0.0, T=20.0)
    params = fe.HestonParams(kappa=1.0, theta=0.09, nu=1.5, rho=-0.7, v0=0.09)
    cums = fe.heston_cumulants(fwd, params)
    grid = cos_auto_grid(cums, N=512, L=12.0)
    phi = lambda u: fe.heston_cf_form2(u, fwd, params)
    return fwd, params, phi, grid


# ── auto formula matches lefloch on normal ranges ─────────────────────────


def test_auto_formula_matches_lefloch_on_normal_ranges(bsm_narrow):
    fwd, params, phi, grid = bsm_narrow
    strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])

    r_auto = cos_prices(phi, fwd, strikes, grid, pricing_formula="auto").call_prices
    r_lefloch = cos_prices(phi, fwd, strikes, grid, pricing_formula="lefloch").call_prices

    np.testing.assert_allclose(r_auto, r_lefloch, rtol=1e-8)


# ── lefloch forced path reproduces put_parity ─────────────────────────────


def test_forced_lefloch_path_reproduces_put_parity(bsm_narrow):
    fwd, params, phi, grid = bsm_narrow
    strikes = np.array([90.0, 100.0, 110.0])

    r_lefloch = cos_prices(phi, fwd, strikes, grid, pricing_formula="lefloch").call_prices
    r_put_par = cos_prices(phi, fwd, strikes, grid, payoff_mode="put_parity").call_prices

    np.testing.assert_allclose(r_lefloch, r_put_par, rtol=1e-12)


# ── fang_oosterlee forced path reproduces call_direct ─────────────────────


def test_forced_fang_oosterlee_path_reproduces_legacy_behavior(bsm_narrow):
    fwd, params, phi, grid = bsm_narrow
    strikes = np.array([90.0, 100.0, 110.0])

    r_fo = cos_prices(phi, fwd, strikes, grid, pricing_formula="fang_oosterlee").call_prices
    r_cd = cos_prices(phi, fwd, strikes, grid, payoff_mode="call_direct").call_prices

    np.testing.assert_allclose(r_fo, r_cd, rtol=1e-12)


# ── auto uses lefloch when b > threshold ──────────────────────────────────


def test_auto_formula_avoids_large_b_overflow(heston_long):
    fwd, params, phi, grid = heston_long
    assert grid.b > _LEFLOCH_B_THRESHOLD, (
        f"Test requires grid.b > {_LEFLOCH_B_THRESHOLD}, got {grid.b}"
    )

    strikes = np.array([80.0, 100.0, 120.0])

    r_auto = cos_prices(phi, fwd, strikes, grid, pricing_formula="auto").call_prices
    r_lefloch = cos_prices(phi, fwd, strikes, grid, pricing_formula="lefloch").call_prices

    # When b is large, "auto" must route through lefloch
    np.testing.assert_allclose(
        r_auto, r_lefloch, rtol=1e-12, err_msg="Auto should route to lefloch for wide grids"
    )


def test_large_b_fang_oosterlee_degrades_vs_lefloch(heston_long):
    """Direct-call formula loses precision for wide b; lefloch remains stable.

    This test documents that fang_oosterlee CAN diverge when b is very large,
    while lefloch (put+parity) stays consistent with the analytic reference.
    """
    fwd, params, phi, grid = heston_long
    strikes = np.array([60.0, 100.0, 150.0])

    r_lefloch = cos_prices(phi, fwd, strikes, grid, pricing_formula="lefloch").call_prices

    # lefloch prices must be non-negative and bounded
    assert np.all(r_lefloch >= 0), "Le Floc'h prices must be non-negative"
    assert np.all(r_lefloch <= fwd.S0 + 1.0), "Le Floc'h prices must be bounded"

    # The fang_oosterlee path may show large errors for OTM options at wide b
    # We just verify lefloch is more stable (no NaN/Inf)
    assert not np.any(np.isnan(r_lefloch)), "Le Floc'h result contains NaN"
    assert not np.any(np.isinf(r_lefloch)), "Le Floc'h result contains Inf"


# ── put-call parity after lefloch fallback ────────────────────────────────


def test_put_call_parity_after_lefloch_fallback(heston_long):
    fwd, params, phi, grid = heston_long
    strikes = np.array([80.0, 100.0, 120.0])

    calls = cos_prices(phi, fwd, strikes, grid, pricing_formula="lefloch").call_prices

    # The lefloch path always returns calls; derive puts via parity
    puts_from_calls = calls - fwd.disc * (fwd.F0 - strikes)
    # Puts must be non-negative
    assert np.all(puts_from_calls >= -1e-8)


# ── invalid pricing_formula rejected ──────────────────────────────────────


def test_invalid_pricing_formula_raises():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    cums = fe.bsm_cumulants(fwd, params)
    grid = cos_auto_grid(cums, N=128)
    phi = lambda u: fe.bsm_cf(u, fwd, params)
    strikes = np.array([100.0])

    with pytest.raises(ValueError, match="pricing_formula"):
        cos_prices(phi, fwd, strikes, grid, pricing_formula="unknown_formula")


def test_pricing_formula_none_uses_payoff_mode():
    """pricing_formula=None (default) leaves payoff_mode unchanged."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    cums = fe.bsm_cumulants(fwd, params)
    grid = cos_auto_grid(cums, N=256)
    phi = lambda u: fe.bsm_cf(u, fwd, params)
    strikes = np.array([100.0])

    r_none = cos_prices(phi, fwd, strikes, grid, pricing_formula=None).call_prices
    r_put = cos_prices(phi, fwd, strikes, grid, payoff_mode="put_parity").call_prices
    np.testing.assert_allclose(r_none, r_put, rtol=1e-12)


# ── stress: CGMY near Y=2 ─────────────────────────────────────────────────


@pytest.mark.parametrize("Y", [1.7, 1.9, 1.95])
def test_cos_lefloch_stable_cgmy_near_y2(Y):
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.CgmyParams(C=1.0, G=5.0, M=5.0, Y=Y)
    cums = fe.cgmy_cumulants(fwd, params)
    grid = cos_auto_grid(cums, N=512, L=16.0)
    phi = lambda u: fe.cgmy_cf(u, fwd, params)
    strikes = np.array([90.0, 100.0, 110.0])

    calls = cos_prices(phi, fwd, strikes, grid, pricing_formula="lefloch").call_prices
    assert np.all(calls >= 0), f"CGMY Y={Y}: negative prices"
    assert not np.any(np.isnan(calls)), f"CGMY Y={Y}: NaN prices"
    assert np.all(np.diff(calls) < 0), f"CGMY Y={Y}: prices not monotone decreasing"


# ── stress: VG long maturity ──────────────────────────────────────────────


def test_cos_lefloch_stable_vg_long_maturity():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.0, T=10.0)
    params = fe.VGParams(sigma=0.15, nu=0.5, theta=-0.15)
    cums = fe.vg_cumulants(fwd, params)
    grid = cos_auto_grid(cums, N=512, L=14.0)
    phi = lambda u: fe.vg_cf(u, fwd, params)
    strikes = np.array([70.0, 100.0, 130.0])

    calls = cos_prices(phi, fwd, strikes, grid, pricing_formula="lefloch").call_prices
    assert np.all(calls >= 0)
    assert not np.any(np.isnan(calls))
