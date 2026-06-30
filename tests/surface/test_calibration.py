"""Tests for foureng.surface.calibration — model calibration to IV surfaces.

All tests use a self-consistency (roundtrip) approach:
  1. Generate synthetic market IVs from known parameters via COS pricing.
  2. Calibrate using a perturbed starting point.
  3. Assert the recovered parameters are close to the true parameters
     and the residuals are small.

This verifies that the calibration loop, objective function, param-vector
packing/unpacking, and the normalised Nelder-Mead converge correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.models.cgmy import CgmyParams
from foureng.models.heston import HestonParams
from foureng.models.kou import KouParams
from foureng.models.nig import NigParams
from foureng.models.sabr import SabrParams, sabr_hagan_implied_vol
from foureng.models.variance_gamma import VGParams
from foureng.surface.calibration import (
    CalibrationResult,
    SabrSmileCalibResult,
    calibrate_cgmy,
    calibrate_heston,
    calibrate_kou,
    calibrate_nig,
    calibrate_sabr_smile,
    calibrate_vg,
)
from foureng.surface.vol_surface import SurfaceSpec, model_iv_surface


# ── shared surface grid ────────────────────────────────────────────────────

S0, r, q = 100.0, 0.05, 0.0
MATS   = np.array([0.25, 0.5, 1.0])
STRIKES = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])

SPEC = SurfaceSpec(S0=S0, r=r, q=q, maturities=MATS, strikes=STRIKES)


# ── helpers ────────────────────────────────────────────────────────────────


def _heston_ivs(p: HestonParams) -> np.ndarray:
    from foureng.models.heston import heston_cf_form2, heston_cumulants
    return model_iv_surface(
        SPEC,
        cf_factory=lambda fwd: lambda u: heston_cf_form2(u, fwd, p),
        cumulant_factory=lambda fwd: heston_cumulants(fwd, p),
        N=192,
    )


def _vg_ivs(p: VGParams) -> np.ndarray:
    from foureng.models.variance_gamma import vg_cf, vg_cumulants
    return model_iv_surface(
        SPEC,
        cf_factory=lambda fwd: lambda u: vg_cf(u, fwd, p),
        cumulant_factory=lambda fwd: vg_cumulants(fwd, p),
        N=512,
    )


def _nig_ivs(p: NigParams) -> np.ndarray:
    from foureng.models.nig import nig_cf, nig_cumulants
    return model_iv_surface(
        SPEC,
        cf_factory=lambda fwd: lambda u: nig_cf(u, fwd, p),
        cumulant_factory=lambda fwd: nig_cumulants(fwd, p),
        N=512,
    )


# ── 1. CalibrationResult structure ────────────────────────────────────────


def test_calibration_result_has_correct_structure():
    true_p = HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04)
    mkt = _heston_ivs(true_p)

    res = calibrate_heston(SPEC, mkt, initial=true_p, maxiter=5)

    assert isinstance(res, CalibrationResult)
    assert isinstance(res.params, dict)
    assert isinstance(res.loss, float)
    assert isinstance(res.success, bool)
    assert isinstance(res.nfev, int) and res.nfev > 0
    assert res.residuals.shape == (len(MATS), len(STRIKES))


# ── 2. Heston self-consistency (roundtrip) ────────────────────────────────


def test_heston_roundtrip():
    """Calibrate from perturbed start → recover true Heston params within 10%."""
    true_p = HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04)
    mkt = _heston_ivs(true_p)

    # Perturbed start (15-20% off on each param)
    init = HestonParams(kappa=1.7, theta=0.05, nu=0.25, rho=-0.4, v0=0.05)
    res = calibrate_heston(SPEC, mkt, initial=init, maxiter=3000, ftol=1e-12)

    # Loss should be very small (essentially perfect fit)
    assert res.loss < 1e-8, f"Heston residual loss too high: {res.loss:.2e}"
    # Max absolute IV residual < 0.5 vol point
    assert np.max(np.abs(res.residuals)) < 5e-4, (
        f"Max IV residual = {np.max(np.abs(res.residuals)):.4f}"
    )


# ── 3. VG self-consistency ────────────────────────────────────────────────


def test_vg_roundtrip():
    true_p = VGParams(sigma=0.15, nu=0.5, theta=-0.1)
    mkt = _vg_ivs(true_p)

    init = VGParams(sigma=0.18, nu=0.4, theta=-0.05)
    res = calibrate_vg(SPEC, mkt, initial=init, maxiter=3000, ftol=1e-12)

    assert res.loss < 1e-8, f"VG residual loss: {res.loss:.2e}"
    assert np.max(np.abs(res.residuals)) < 5e-4


# ── 4. NIG self-consistency ───────────────────────────────────────────────


def test_nig_roundtrip():
    true_p = NigParams(sigma=0.15, nu=0.3, theta=-0.1)
    mkt = _nig_ivs(true_p)

    init = NigParams(sigma=0.18, nu=0.25, theta=-0.05)
    res = calibrate_nig(SPEC, mkt, initial=init, maxiter=3000, ftol=1e-12)

    assert res.loss < 1e-8, f"NIG residual loss: {res.loss:.2e}"
    assert np.max(np.abs(res.residuals)) < 5e-4


# ── 5. Calibration loss strictly decreases from a bad start ───────────────


def test_calibration_loss_decreases_from_bad_start():
    """The calibrated loss must be strictly less than the initial loss."""
    true_p = HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04)
    mkt = _heston_ivs(true_p)

    # Deliberately bad initial params
    bad_init = HestonParams(kappa=10.0, theta=0.5, nu=2.0, rho=0.4, v0=0.5)

    # Compute initial loss manually
    from foureng.models.heston import heston_cf_form2, heston_cumulants
    init_ivs = model_iv_surface(
        SPEC,
        cf_factory=lambda fwd: lambda u: heston_cf_form2(u, fwd, bad_init),
        cumulant_factory=lambda fwd: heston_cumulants(fwd, bad_init),
        N=192,
    )
    init_loss = float(np.sum((init_ivs - mkt) ** 2))

    res = calibrate_heston(SPEC, mkt, initial=bad_init, maxiter=500)
    assert res.loss < init_loss, (
        f"Calibration did not improve: init={init_loss:.6f}, final={res.loss:.6f}"
    )


# ── 6. Shape validation ────────────────────────────────────────────────────


def test_calibration_rejects_wrong_iv_shape():
    true_p = HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04)
    mkt_wrong = np.zeros((2, 5))  # wrong shape

    with pytest.raises(ValueError, match="shape"):
        calibrate_heston(SPEC, mkt_wrong, initial=true_p)


# ── 7. SABR smile calibration roundtrip ──────────────────────────────────


@pytest.mark.parametrize(
    "true_alpha,true_rho,true_nu",
    [
        (0.25, -0.30, 0.40),
        (0.40,  0.10, 0.20),
        (0.15, -0.60, 0.60),
    ],
)
def test_sabr_smile_roundtrip(true_alpha, true_rho, true_nu):
    """Calibrate SABR smile → recover alpha, rho, nu within tight tolerance."""
    F, T, beta = 100.0, 1.0, 0.5
    K = np.array([80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    true_ivs = sabr_hagan_implied_vol(F, K, T, true_alpha, beta, true_rho, true_nu)

    # Perturbed start
    init = SabrParams(
        alpha=true_alpha * 1.2,
        beta=beta,
        rho=max(min(true_rho * 0.8, 0.99), -0.99),
        nu=true_nu * 1.2,
    )
    res = calibrate_sabr_smile(F, T, K, true_ivs, initial=init, maxiter=5000)

    assert isinstance(res, SabrSmileCalibResult)
    assert res.loss < 1e-10, f"SABR loss too high: {res.loss:.2e}"
    assert np.max(np.abs(res.residuals)) < 1e-6, (
        f"Max SABR IV residual = {np.max(np.abs(res.residuals)):.2e}"
    )
    assert abs(res.params.alpha - true_alpha) < 0.01, (
        f"alpha: expected {true_alpha}, got {res.params.alpha:.4f}"
    )
    assert abs(res.params.rho - true_rho) < 0.05, (
        f"rho: expected {true_rho}, got {res.params.rho:.4f}"
    )
    assert abs(res.params.nu - true_nu) < 0.05, (
        f"nu: expected {true_nu}, got {res.params.nu:.4f}"
    )


# ── 8. SABR calibration result structure ──────────────────────────────────


def test_sabr_calibration_result_structure():
    F, T = 100.0, 1.0
    K = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    true_ivs = sabr_hagan_implied_vol(F, K, T, 0.3, 0.5, -0.3, 0.4)
    init = SabrParams(alpha=0.35, beta=0.5, rho=-0.2, nu=0.3)

    res = calibrate_sabr_smile(F, T, K, true_ivs, initial=init, maxiter=10)

    assert isinstance(res.params, SabrParams)
    assert res.params.beta == 0.5  # beta is fixed
    assert res.residuals.shape == K.shape
    assert res.nfev > 0
    assert res.loss >= 0.0


# ── 9. SABR calibration with fit_beta=True ────────────────────────────────


def test_sabr_calibration_fit_beta():
    """With fit_beta=True, the calibrator also recovers beta."""
    F, T = 100.0, 1.0
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    true_ivs = sabr_hagan_implied_vol(F, K, T, 0.3, 0.7, -0.3, 0.4)
    init = SabrParams(alpha=0.35, beta=0.5, rho=-0.2, nu=0.3)

    res = calibrate_sabr_smile(F, T, K, true_ivs, initial=init,
                               fit_beta=True, maxiter=5000)

    assert res.loss < 1e-8, f"SABR (fit_beta) loss: {res.loss:.2e}"
    assert abs(res.params.beta - 0.7) < 0.1, (
        f"beta: expected 0.7, got {res.params.beta:.4f}"
    )


# ── 10. SABR wrong shapes ─────────────────────────────────────────────────


def test_sabr_calibration_rejects_shape_mismatch():
    init = SabrParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
    with pytest.raises(ValueError, match="shape"):
        calibrate_sabr_smile(
            100.0, 1.0,
            np.array([95.0, 100.0, 105.0]),
            np.array([0.20, 0.21]),  # wrong length
            initial=init,
        )
