from __future__ import annotations

import pytest
from pytest import approx

import foureng as fe
from foureng.models.bsm import bsm_asset_or_nothing, bsm_cash_or_nothing
from foureng.pipeline import price
from foureng.pricers.cos_digital import cos_digital_price
from foureng.products import DigitalOption


def test_digital_bsm_dispatch_matches_cash_closed_form():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=3.0)
    params = fe.BsmParams(sigma=0.2)
    product = DigitalOption(
        strike=100.0,
        maturity=1.0,
        cp=1,
        payoff_type="cash_or_nothing",
        cash_amount=2.5,
    )

    actual = price(product, "bsm", "digital_bsm", fwd, params)
    expected = bsm_cash_or_nothing(
        fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0),
        params,
        100.0,
        cp=1,
        cash_amount=2.5,
    )

    assert actual == approx(expected, rel=1e-12, abs=1e-12)


def test_digital_bsm_dispatch_matches_asset_closed_form():
    fwd = fe.ForwardSpec(S0=100.0, r=0.04, q=0.01, T=2.0)
    params = fe.BsmParams(sigma=0.22)
    product = DigitalOption(
        strike=95.0,
        maturity=0.75,
        cp=-1,
        payoff_type="asset_or_nothing",
    )

    actual = price(product, "bsm", "digital_bsm", fwd, params)
    expected = bsm_asset_or_nothing(
        fe.ForwardSpec(S0=100.0, r=0.04, q=0.01, T=0.75),
        params,
        95.0,
        cp=-1,
    )

    assert actual == approx(expected, rel=1e-12, abs=1e-12)


def test_cos_digital_dispatch_matches_public_pricer_for_non_bsm_model():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=4.0)
    params = fe.VGParams(sigma=0.12, nu=0.2, theta=-0.1)
    product = DigitalOption(
        strike=105.0,
        maturity=0.5,
        cp=1,
        payoff_type="cash_or_nothing",
    )

    actual = price(product, "vg", "cos_digital", fwd, params)
    expected = cos_digital_price(
        "vg",
        fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.5),
        params,
        product,
    )

    assert actual == approx(expected, rel=1e-12, abs=1e-12)


def test_digital_bsm_rejects_non_bsm_model():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04)
    product = DigitalOption(strike=100.0, maturity=1.0, cp=1)

    with pytest.raises(NotImplementedError, match="model='bsm'"):
        price(product, "heston", "digital_bsm", fwd, params)


def test_digital_unknown_method_is_explicit():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    product = DigitalOption(strike=100.0, maturity=1.0, cp=1)

    with pytest.raises(NotImplementedError, match="digital_bsm") as exc_info:
        price(product, "bsm", "cos", fwd, params)
    assert "cos_digital" in str(exc_info.value)
