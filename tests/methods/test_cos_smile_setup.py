"""Tests for COSSmileSetup — strike-independent COS smile caching."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import foureng as fe
from foureng.pricers.cos_setup import COSSmileSetup
from foureng.pricers.cos import cos_auto_grid, cos_prices


@pytest.fixture
def heston_setup():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)
    return fwd, params


@pytest.fixture
def bsm_setup():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    return fwd, params


@pytest.fixture
def vg_setup():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.VGParams(sigma=0.12, nu=0.3, theta=-0.14)
    return fwd, params


# ── Correctness: matches vectorised cos_prices ─────────────────────────────

def test_smile_setup_matches_vectorized_cos_heston(heston_setup):
    fwd, params = heston_setup
    strikes = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])

    setup = COSSmileSetup.build("heston", fwd, params)
    calls_setup = setup.price_call(strikes)

    calls_strip = fe.price_strip("heston", "cos_improved", strikes, fwd, params)
    np.testing.assert_allclose(calls_setup, calls_strip, rtol=1e-4,
                               err_msg="COSSmileSetup calls should match price_strip")


def test_smile_setup_matches_vectorized_cos_bsm(bsm_setup):
    fwd, params = bsm_setup
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])

    setup = COSSmileSetup.build("bsm", fwd, params)
    calls_setup = setup.price_call(strikes)
    calls_strip = fe.price_strip("bsm", "cos_improved", strikes, fwd, params)

    np.testing.assert_allclose(calls_setup, calls_strip, rtol=1e-5)


def test_smile_setup_matches_scalar_loop(bsm_setup):
    fwd, params = bsm_setup
    strikes = np.array([90.0, 100.0, 110.0])

    setup = COSSmileSetup.build("bsm", fwd, params)
    calls_setup = setup.price_call(strikes)

    # price each strike individually via price_strip
    for i, K in enumerate(strikes):
        scalar = float(fe.price_strip("bsm", "cos_improved", [K], fwd, params)[0])
        assert calls_setup[i] == approx(scalar, rel=1e-5), f"Mismatch at K={K}"


def test_smile_setup_put_matches_call_via_parity(bsm_setup):
    fwd, params = bsm_setup
    strikes = np.array([90.0, 100.0, 110.0])

    setup = COSSmileSetup.build("bsm", fwd, params)
    calls = setup.price_call(strikes)
    puts = setup.price_put(strikes)

    # put-call parity: C - P = disc * (F0 - K)
    lhs = calls - puts
    rhs = fwd.disc * (fwd.F0 - strikes)
    np.testing.assert_allclose(lhs, rhs, atol=1e-8)


# ── Determinism ────────────────────────────────────────────────────────────

def test_repeated_smile_setup_is_deterministic(heston_setup):
    fwd, params = heston_setup
    strikes = np.array([95.0, 100.0, 105.0])

    s1 = COSSmileSetup.build("heston", fwd, params)
    s2 = COSSmileSetup.build("heston", fwd, params)

    np.testing.assert_array_equal(s1.density_coeffs, s2.density_coeffs)
    np.testing.assert_array_equal(s1.price_call(strikes), s2.price_call(strikes))


# ── Maturity validation ─────────────────────────────────────────────────────

def test_smile_setup_rejects_changed_maturity(bsm_setup):
    fwd, params = bsm_setup
    setup = COSSmileSetup.build("bsm", fwd, params)
    with pytest.raises(ValueError, match="maturity"):
        setup.validate_maturity(2.0)


def test_smile_setup_validates_same_maturity(bsm_setup):
    fwd, params = bsm_setup
    setup = COSSmileSetup.build("bsm", fwd, params)
    setup.validate_maturity(1.0)  # should not raise


# ── Error handling ──────────────────────────────────────────────────────────

def test_smile_setup_unknown_model_raises():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    with pytest.raises(ValueError, match="unknown model"):
        COSSmileSetup.build("unicorn_model", fwd, fe.BsmParams(sigma=0.2))


# ── VG model ───────────────────────────────────────────────────────────────

def test_smile_setup_vg_matches_price_strip(vg_setup):
    fwd, params = vg_setup
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])

    setup = COSSmileSetup.build("vg", fwd, params)
    calls_setup = setup.price_call(strikes)
    calls_strip = fe.price_strip("vg", "cos_improved", strikes, fwd, params)

    np.testing.assert_allclose(calls_setup, calls_strip, rtol=1e-4)


# ── Explicit grid override ─────────────────────────────────────────────────

def test_smile_setup_explicit_grid(bsm_setup):
    fwd, params = bsm_setup
    cums = fe.bsm_cumulants(fwd, params)
    grid = cos_auto_grid(cums, N=256, L=10.0)

    setup = COSSmileSetup.build("bsm", fwd, params, grid=grid)
    assert setup.grid is grid

    strikes = np.array([95.0, 100.0, 105.0])
    calls = setup.price_call(strikes)
    assert np.all(calls > 0)
    assert np.all(np.diff(calls) < 0)  # calls decrease with strike
