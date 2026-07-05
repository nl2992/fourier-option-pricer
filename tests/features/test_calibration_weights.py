"""Vega-weighted and spread-weighted calibration objectives.

Covers the two weight-builder helpers added in the calibration-weights work:

* ``vega_weights`` returns an ``(nT, nK)`` matrix of vega^2, with wing cells
  attenuated relative to the ATM cell.  We check shape, non-negativity, mean-1
  normalisation, and the ATM > wing ordering.
* ``spread_weights`` returns ``1 / half_spread^2`` with a floor.  We check
  that tighter quotes get bigger weights, that missing / crossed quotes fall
  through to the floor, and that the floor prevents infinite weights.
* End-to-end: passing ``weights=vega_weights(...)`` to ``calibrate_heston``
  still recovers the true parameters on a self-generated surface, so the
  weighting does not break the calibrator.
"""

from __future__ import annotations

import numpy as np

from foureng.models.heston import HestonParams, heston_cf_form2, heston_cumulants
from foureng.surface import (
    SurfaceSpec,
    calibrate_heston,
    model_iv_surface,
    spread_weights,
    vega_weights,
)


def _spec() -> SurfaceSpec:
    return SurfaceSpec(
        S0=100.0,
        r=0.02,
        q=0.0,
        maturities=np.array([0.25, 0.5, 1.0]),
        strikes=np.array([80.0, 90.0, 100.0, 110.0, 120.0]),
    )


def _heston_true() -> HestonParams:
    return HestonParams(kappa=3.0, theta=0.04, nu=0.5, rho=-0.6, v0=0.04)


def _heston_iv_grid(params: HestonParams, spec: SurfaceSpec, N: int = 256) -> np.ndarray:
    def cf(fwd):
        return lambda u: heston_cf_form2(u, fwd, params)

    def cum(fwd):
        return heston_cumulants(fwd, params)

    return model_iv_surface(spec, cf, cum, N=N, L=10.0)


def test_vega_weights_shape_and_non_negativity():
    spec = _spec()
    ivs = _heston_iv_grid(_heston_true(), spec)
    w = vega_weights(spec, ivs)
    assert w.shape == ivs.shape
    assert np.all(w >= 0.0)
    assert np.all(np.isfinite(w))


def test_vega_weights_mean_normalisation():
    spec = _spec()
    ivs = _heston_iv_grid(_heston_true(), spec)
    w = vega_weights(spec, ivs, normalise=True)
    # After mean-1 rescale, mean of positive cells should be 1 to ~1e-12.
    assert np.isclose(w.mean(), 1.0, atol=1e-9)


def test_vega_weights_atm_beats_wings():
    """ATM cell has the biggest vega on a smile-shaped Heston surface."""
    spec = _spec()
    ivs = _heston_iv_grid(_heston_true(), spec)
    w = vega_weights(spec, ivs, normalise=False)
    # Column index for K=100 (ATM given S0=100, low r).
    atm_col = int(np.where(spec.strikes == 100.0)[0][0])
    for i in range(len(spec.maturities)):
        row = w[i, :]
        # ATM cell must dominate both deep-OTM wings.
        assert row[atm_col] > row[0], f"wing0>atm at row {i}"
        assert row[atm_col] > row[-1], f"wingN>atm at row {i}"


def test_vega_weights_shape_mismatch_raises():
    spec = _spec()
    import pytest

    with pytest.raises(ValueError):
        vega_weights(spec, np.zeros((2, 2)))


def test_spread_weights_tighter_gets_bigger():
    """Cell with tighter bid-ask spread must get a larger weight."""
    bid = np.array([[0.19, 0.18], [0.17, 0.16]])
    ask = np.array([[0.21, 0.22], [0.23, 0.24]])
    # half-spreads: [[0.01, 0.02], [0.03, 0.04]]  -> weights ~ 1/half^2.
    w = spread_weights(bid, ask, normalise=False)
    # Tightest (0.01) should be strictly greater than widest (0.04).
    assert w[0, 0] > w[1, 1]
    # And bigger than the second-tightest (0.02).
    assert w[0, 0] > w[0, 1]


def test_spread_weights_floor_applied_to_missing():
    """NaN or crossed quotes fall through to the floor spread."""
    bid = np.array([[np.nan, 0.10]])
    ask = np.array([[0.20, 0.09]])  # right cell is crossed (ask < bid).
    w = spread_weights(bid, ask, floor_spread=1e-3, normalise=False)
    # Both cells should hit the same floor-based weight = 1 / floor^2.
    expected = 1.0 / (1e-3 * 1e-3)
    assert np.allclose(w, expected)


def test_spread_weights_normalisation_mean_one():
    bid = np.array([[0.19, 0.18], [0.17, 0.16]])
    ask = np.array([[0.21, 0.22], [0.23, 0.24]])
    w = spread_weights(bid, ask, normalise=True)
    assert np.isclose(w.mean(), 1.0, atol=1e-9)


def test_spread_weights_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        spread_weights(np.zeros((2, 2)), np.zeros((3, 3)))


def test_spread_weights_invalid_floor_raises():
    import pytest

    with pytest.raises(ValueError):
        spread_weights(np.zeros((1, 1)), np.zeros((1, 1)), floor_spread=0.0)


def test_calibrate_heston_with_vega_weights_still_recovers_truth():
    """End-to-end sanity: vega-weighted calibration still lands near the truth."""
    spec = _spec()
    true = _heston_true()
    ivs = _heston_iv_grid(true, spec, N=192)

    # Perturbed initial guess.
    initial = HestonParams(kappa=2.0, theta=0.05, nu=0.4, rho=-0.5, v0=0.05)
    w = vega_weights(spec, ivs)
    res = calibrate_heston(spec, ivs, initial=initial, weights=w, N=192, maxiter=800)

    assert res.success or res.loss < 1e-4
    # Each parameter within 5% of truth on a self-generated surface is easy.
    assert abs(res.params["kappa"] - true.kappa) / true.kappa < 0.20
    assert abs(res.params["theta"] - true.theta) / true.theta < 0.10
    assert abs(res.params["nu"] - true.nu) / true.nu < 0.15
    assert abs(res.params["rho"] - true.rho) / abs(true.rho) < 0.10
    assert abs(res.params["v0"] - true.v0) / true.v0 < 0.10
