"""Verify that price_strip still works as before, and that price() wraps it correctly.

These tests are the Phase 1 'definition of done' for the product layer:
price_strip(...) still works, but all future payoffs route through ProductSpec.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import foureng as fe
from foureng.pipeline import price
from foureng.products import EuropeanOption


@pytest.fixture
def bsm_setup():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    return fwd, params, strikes


def test_price_strip_bsm_cos_returns_array(bsm_setup):
    fwd, params, strikes = bsm_setup
    result = fe.price_strip("bsm", "cos", strikes, fwd, params)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)


def test_price_strip_bsm_all_nonnegative(bsm_setup):
    fwd, params, strikes = bsm_setup
    result = fe.price_strip("bsm", "cos", strikes, fwd, params)
    assert np.all(result >= 0)


def test_price_strip_bsm_decreasing_in_strike(bsm_setup):
    fwd, params, strikes = bsm_setup
    result = fe.price_strip("bsm", "cos", strikes, fwd, params)
    assert np.all(np.diff(result) < 0), "Call prices should decrease with strike"


def test_price_strip_unknown_model_raises():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    with pytest.raises(ValueError, match="unknown model"):
        fe.price_strip("nonexistent_model", "cos", [100.0], fwd, params)


def test_price_strip_unknown_method_raises():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    with pytest.raises(ValueError, match="unknown method"):
        fe.price_strip("bsm", "nonexistent_method", [100.0], fwd, params)


def test_price_strip_empty_strikes_raises():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    with pytest.raises(ValueError):
        fe.price_strip("bsm", "cos", [], fwd, params)


# ── price() product-level dispatcher ──────────────────────────────────────

def test_price_european_option_scalar_result():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.0)  # T overridden by product
    params = fe.BsmParams(sigma=0.20)
    opt = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    result = price(opt, "bsm", "cos", fwd, params)
    assert isinstance(result, float)
    assert result > 0.0


def test_price_european_uses_product_maturity_not_fwd_T():
    """price() overrides fwd.T with the product's maturity."""
    fwd_wrong_T = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=99.0)
    params = fe.BsmParams(sigma=0.20)
    opt = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    result_product = price(opt, "bsm", "cos", fwd_wrong_T, params)

    fwd_correct = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    result_strip = float(
        fe.price_strip("bsm", "cos", [100.0], fwd_correct, params)[0]
    )
    assert result_product == approx(result_strip, rel=1e-6)


def test_price_strip_equivalent_to_vector_of_european_products():
    """price() on each European should match price_strip vector output."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    strikes = [90.0, 100.0, 110.0]

    strip = fe.price_strip("bsm", "cos", strikes, fwd, params)
    scalars = [
        price(EuropeanOption(strike=k, maturity=1.0, cp=1), "bsm", "cos", fwd, params)
        for k in strikes
    ]
    for s, p in zip(strip, scalars):
        assert float(s) == approx(p, rel=1e-8)


def test_price_unimplemented_product_type_raises_not_implemented():
    from foureng.products import BarrierOption
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    opt = BarrierOption(strike=100.0, barrier=80.0, maturity=1.0, cp=1, barrier_type="down_out")
    with pytest.raises(NotImplementedError, match="barrier"):
        price(opt, "bsm", "cos", fwd, params)


def test_price_non_product_spec_raises_type_error():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    with pytest.raises(TypeError, match="ProductSpec"):
        price("not_a_product", "bsm", "cos", fwd, params)  # type: ignore
