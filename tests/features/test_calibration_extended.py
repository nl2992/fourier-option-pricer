"""Tests for CGMY and NIG calibration (Sprint 7).

Verifies that calibrate_cgmy and calibrate_nig can recover true parameters
(or at least match market IVs closely) when calibrated to synthetic surfaces.

We use a single-maturity, 5-strike surface to keep tests fast.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe


@pytest.fixture
def surface_spec():
    """5-strike 1-maturity surface."""
    return fe.SurfaceSpec(
        S0=100.0,
        r=0.05,
        q=0.0,
        maturities=np.array([1.0]),
        strikes=np.array([85.0, 92.5, 100.0, 107.5, 115.0]),
    )


# ── CGMY calibration ──────────────────────────────────────────────────────

def test_calibrate_cgmy_recovers_market_ivs(surface_spec):
    """CGMY calibration from nearby start → residuals < 0.5% IV."""
    true_params = fe.CgmyParams(C=1.0, G=5.0, M=5.0, Y=1.5)
    market_ivs = fe.model_iv_surface(
        surface_spec,
        lambda fwd: lambda u: fe.cgmy_cf(u, fwd, true_params),
        lambda fwd: fe.cgmy_cumulants(fwd, true_params),
    )

    init = fe.CgmyParams(C=1.1, G=4.8, M=4.8, Y=1.4)
    result = fe.calibrate_cgmy(surface_spec, market_ivs, init, maxiter=500, N=512, L=12.0)

    # The calibrated model should reproduce market IVs to < 0.5 vol point
    assert result.loss < 1e-4, (
        f"CGMY calibration residuals too large: loss={result.loss:.4e}"
    )
    assert np.all(np.abs(result.residuals) < 0.005), (
        f"Per-cell residuals exceed 0.5%: {result.residuals}"
    )


def test_calibrate_cgmy_returns_valid_params(surface_spec):
    """Calibrated CGMY parameters are within their natural bounds."""
    true_params = fe.CgmyParams(C=0.5, G=3.0, M=4.0, Y=1.2)
    market_ivs = fe.model_iv_surface(
        surface_spec,
        lambda fwd: lambda u: fe.cgmy_cf(u, fwd, true_params),
        lambda fwd: fe.cgmy_cumulants(fwd, true_params),
    )
    init = fe.CgmyParams(C=0.7, G=3.5, M=4.5, Y=1.0)
    result = fe.calibrate_cgmy(surface_spec, market_ivs, init, maxiter=300)

    assert result.params["C"] > 0
    assert result.params["G"] > 0
    assert result.params["M"] > 1  # finite mean condition
    assert 0 < result.params["Y"] < 2  # finite variance condition


# ── NIG calibration ───────────────────────────────────────────────────────

def test_calibrate_nig_recovers_market_ivs(surface_spec):
    """NIG calibration from nearby start → residuals < 0.1 vol point."""
    true_params = fe.NigParams(sigma=0.15, nu=0.3, theta=-0.14)
    market_ivs = fe.model_iv_surface(
        surface_spec,
        lambda fwd: lambda u: fe.nig_cf(u, fwd, true_params),
        lambda fwd: fe.nig_cumulants(fwd, true_params),
    )

    init = fe.NigParams(sigma=0.18, nu=0.35, theta=-0.10)
    result = fe.calibrate_nig(surface_spec, market_ivs, init, maxiter=500)

    assert result.loss < 1e-8, (
        f"NIG calibration loss too large: {result.loss:.4e}"
    )


def test_calibrate_nig_exact_start_gives_zero_loss(surface_spec):
    """Calibrating from the true params gives essentially zero loss."""
    true_params = fe.NigParams(sigma=0.15, nu=0.3, theta=-0.14)
    market_ivs = fe.model_iv_surface(
        surface_spec,
        lambda fwd: lambda u: fe.nig_cf(u, fwd, true_params),
        lambda fwd: fe.nig_cumulants(fwd, true_params),
    )
    # Calibrate from true params (should stay near zero)
    result = fe.calibrate_nig(surface_spec, market_ivs, true_params, maxiter=100)
    assert result.loss < 1e-10, f"Loss from true start: {result.loss:.4e}"


# ── CalibrationResult structure ────────────────────────────────────────────

def test_calibration_result_has_correct_residual_shape(surface_spec):
    true_params = fe.NigParams(sigma=0.15, nu=0.3, theta=-0.14)
    market_ivs = fe.model_iv_surface(
        surface_spec,
        lambda fwd: lambda u: fe.nig_cf(u, fwd, true_params),
        lambda fwd: fe.nig_cumulants(fwd, true_params),
    )
    result = fe.calibrate_nig(surface_spec, market_ivs, true_params, maxiter=5)
    assert result.residuals.shape == (len(surface_spec.maturities), len(surface_spec.strikes))
    assert result.nfev > 0
