from __future__ import annotations

import pytest
from pytest import approx

import foureng as fe
from foureng.analytics.bsm_exotics import margrabe_exchange
from foureng.pipeline import price
from foureng.products import ExchangeOption


def test_exchange_bsm_dispatch_matches_margrabe_formula():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=4.0)
    params = fe.BsmParams(sigma=0.2)
    product = ExchangeOption(maturity=1.0, spot2=95.0, q2=0.01, sigma2=0.25, rho=0.4)

    actual = price(product, "bsm", "exchange_bsm", fwd, params)
    expected = margrabe_exchange(100.0, 95.0, 0.02, 0.01, 1.0, 0.2, 0.25, 0.4)

    assert actual == approx(expected, rel=1e-12, abs=1e-12)


def test_exchange_bsm_uses_product_maturity_not_forward_spec_maturity():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=3.0)
    params = fe.BsmParams(sigma=0.22)
    product = ExchangeOption(maturity=0.75, spot2=98.0, q2=0.02, sigma2=0.18, rho=-0.2)

    actual = price(product, "bsm", "exchange_bsm", fwd, params)
    correct = margrabe_exchange(100.0, 98.0, 0.01, 0.02, 0.75, 0.22, 0.18, -0.2)
    wrong = margrabe_exchange(100.0, 98.0, 0.01, 0.02, 3.0, 0.22, 0.18, -0.2)

    assert actual == approx(correct, rel=1e-12, abs=1e-12)
    assert abs(actual - wrong) > 1e-3


def test_exchange_bsm_rejects_non_bsm_model():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04)
    product = ExchangeOption(maturity=1.0, spot2=95.0, sigma2=0.2, rho=0.0)

    with pytest.raises(NotImplementedError, match="model='bsm'"):
        price(product, "heston", "exchange_bsm", fwd, params)


def test_exchange_unknown_method_is_explicit():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    product = ExchangeOption(maturity=1.0)

    with pytest.raises(NotImplementedError, match="exchange_bsm"):
        price(product, "bsm", "cos", fwd, params)
