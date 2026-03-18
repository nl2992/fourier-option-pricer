"""Property-based numerical invariants for representative Fourier models.

These tests complement the paper-backed fixtures by checking generated
parameter points inside a numerically stable region. The goal is to guard
core invariants that should hold beyond a handful of hand-picked examples.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from foureng.models.base import ForwardSpec
from foureng.models.bates import BatesParams, bates_cf
from foureng.models.bsm import BsmParams, bsm_cf
from foureng.models.heston import HestonParams, heston_cf_form2, heston_cumulants
from foureng.pipeline import price_strip
from foureng.pricers.cos import cos_auto_grid, cos_prices


@st.composite
def forward_specs(draw) -> ForwardSpec:
    return ForwardSpec(
        S0=draw(st.floats(min_value=80.0, max_value=120.0, allow_nan=False, allow_infinity=False)),
        r=draw(st.floats(min_value=0.0, max_value=0.08, allow_nan=False, allow_infinity=False)),
        q=draw(st.floats(min_value=0.0, max_value=0.04, allow_nan=False, allow_infinity=False)),
        T=draw(st.floats(min_value=0.10, max_value=2.50, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def bsm_params(draw) -> BsmParams:
    return BsmParams(
        sigma=draw(st.floats(min_value=0.05, max_value=1.00, allow_nan=False, allow_infinity=False))
    )


@st.composite
def heston_params(draw) -> HestonParams:
    return HestonParams(
        kappa=draw(
            st.floats(min_value=0.50, max_value=5.00, allow_nan=False, allow_infinity=False)
        ),
        theta=draw(
            st.floats(min_value=0.02, max_value=0.30, allow_nan=False, allow_infinity=False)
        ),
        nu=draw(st.floats(min_value=0.10, max_value=1.00, allow_nan=False, allow_infinity=False)),
        rho=draw(st.floats(min_value=-0.90, max_value=0.20, allow_nan=False, allow_infinity=False)),
        v0=draw(st.floats(min_value=0.02, max_value=0.30, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def bates_params(draw) -> BatesParams:
    return BatesParams(
        kappa=draw(
            st.floats(min_value=0.50, max_value=5.00, allow_nan=False, allow_infinity=False)
        ),
        theta=draw(
            st.floats(min_value=0.02, max_value=0.30, allow_nan=False, allow_infinity=False)
        ),
        nu=draw(st.floats(min_value=0.10, max_value=1.00, allow_nan=False, allow_infinity=False)),
        rho=draw(st.floats(min_value=-0.90, max_value=0.20, allow_nan=False, allow_infinity=False)),
        v0=draw(st.floats(min_value=0.02, max_value=0.30, allow_nan=False, allow_infinity=False)),
        lam_j=0.0,
        mu_j=draw(
            st.floats(min_value=-0.20, max_value=0.10, allow_nan=False, allow_infinity=False)
        ),
        sigma_j=draw(
            st.floats(min_value=0.05, max_value=0.40, allow_nan=False, allow_infinity=False)
        ),
    )


def _strike_grid(spot: float) -> np.ndarray:
    return np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=float) * spot


@settings(max_examples=20, deadline=None)
@given(fwd=forward_specs(), params=bsm_params())
def test_bsm_generated_strip_obeys_bounds_and_shape(fwd: ForwardSpec, params: BsmParams) -> None:
    strikes = _strike_grid(fwd.S0)
    prices = price_strip("bsm", "cos", strikes, fwd, params)

    lower = fwd.disc * np.maximum(fwd.F0 - strikes, 0.0)
    upper = np.full_like(strikes, fwd.disc * fwd.F0)
    convexity = prices[:-2] - 2.0 * prices[1:-1] + prices[2:]

    assert np.all(np.isfinite(prices))
    assert np.all(prices >= lower - 1e-9)
    assert np.all(prices <= upper + 1e-9)
    assert np.all(np.diff(prices) <= 1e-9)
    assert np.all(convexity >= -1e-8)


@settings(max_examples=15, deadline=None)
@given(fwd=forward_specs(), params=heston_params())
def test_heston_generated_strip_obeys_bounds_and_shape(
    fwd: ForwardSpec, params: HestonParams
) -> None:
    strikes = _strike_grid(fwd.S0)
    prices = price_strip("heston", "cos", strikes, fwd, params)

    lower = fwd.disc * np.maximum(fwd.F0 - strikes, 0.0)
    upper = np.full_like(strikes, fwd.disc * fwd.F0)
    convexity = prices[:-2] - 2.0 * prices[1:-1] + prices[2:]

    assert np.all(np.isfinite(prices))
    assert np.all(prices >= lower - 1e-7)
    assert np.all(prices <= upper + 1e-7)
    assert np.all(np.diff(prices) <= 1e-7)
    assert np.all(convexity >= -5e-5)


@settings(max_examples=15, deadline=None)
@given(fwd=forward_specs(), params=heston_params())
def test_heston_cos_two_path_consistency_generated(fwd: ForwardSpec, params: HestonParams) -> None:
    strikes = np.array([0.9, 1.0, 1.1], dtype=float) * fwd.S0
    grid = cos_auto_grid(heston_cumulants(fwd, params), N=256, L=10.0)

    def phi(u: np.ndarray) -> np.ndarray:
        return heston_cf_form2(u, fwd, params)

    put_parity = cos_prices(phi, fwd, strikes, grid, payoff_mode="put_parity").call_prices
    call_direct = cos_prices(phi, fwd, strikes, grid, payoff_mode="call_direct").call_prices
    assert np.max(np.abs(put_parity - call_direct)) < 2e-5


@settings(max_examples=20, deadline=None)
@given(fwd=forward_specs(), bsm=bsm_params(), heston=heston_params(), bates=bates_params())
def test_characteristic_function_normalization_generated(
    fwd: ForwardSpec,
    bsm: BsmParams,
    heston: HestonParams,
    bates: BatesParams,
) -> None:
    assert abs(bsm_cf(np.array([0.0]), fwd, bsm)[0] - 1.0) < 1e-12
    assert abs(heston_cf_form2(np.array([0.0]), fwd, heston)[0] - 1.0) < 1e-12
    assert abs(bates_cf(np.array([0.0]), fwd, bates)[0] - 1.0) < 1e-12


@settings(max_examples=15, deadline=None)
@given(fwd=forward_specs(), params=bates_params())
def test_bates_zero_jump_intensity_reduces_to_heston_generated(
    fwd: ForwardSpec,
    params: BatesParams,
) -> None:
    strikes = _strike_grid(fwd.S0)
    heston = HestonParams(
        kappa=params.kappa,
        theta=params.theta,
        nu=params.nu,
        rho=params.rho,
        v0=params.v0,
    )

    bates_prices = price_strip("bates", "cos", strikes, fwd, params)
    heston_prices = price_strip("heston", "cos", strikes, fwd, heston)
    assert np.max(np.abs(bates_prices - heston_prices)) < 1e-9
