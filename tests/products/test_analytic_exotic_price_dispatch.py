"""Product-dispatch tests for analytic BSM exotics."""

from __future__ import annotations

import pytest

import foureng as fe
from foureng.pipeline import price
from foureng.products import ForwardStartOption, LookbackOption


def test_price_dispatches_forward_start_bsm():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    product = ForwardStartOption(start_time=0.5, maturity=1.0, cp=1, alpha=1.0)

    actual = price(product, "bsm", "forward_start_bsm", fwd, params)
    expected = fe.bsm_forward_start(
        fwd.S0,
        product.alpha,
        product.start_time,
        product.maturity,
        fwd.r,
        fwd.q,
        params.sigma,
        cp=product.cp,
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_price_dispatches_lookback_bsm():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    product = LookbackOption(maturity=1.0, cp=1, strike_type="floating", monitoring="continuous")

    actual = price(product, "bsm", "lookback_bsm", fwd, params)
    expected = fe.bsm_lookback_floating(
        fwd.S0,
        S_min=fwd.S0,
        S_max=fwd.S0,
        r=fwd.r,
        q=fwd.q,
        T=product.maturity,
        sigma=params.sigma,
        cp=product.cp,
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_forward_start_dispatch_rejects_non_bsm_model():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04)
    product = ForwardStartOption(start_time=0.5, maturity=1.0, cp=1, alpha=1.0)

    with pytest.raises(NotImplementedError, match="only for model='bsm'"):
        price(product, "heston", "forward_start_bsm", fwd, params)


def test_lookback_dispatch_rejects_fixed_strike():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    product = LookbackOption(maturity=1.0, cp=1, strike_type="fixed", strike=100.0)

    with pytest.raises(NotImplementedError, match="floating-strike"):
        price(product, "bsm", "lookback_bsm", fwd, params)


def test_lookback_dispatch_rejects_discrete_monitoring():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    product = LookbackOption(maturity=1.0, cp=1, strike_type="floating", monitoring="discrete")

    with pytest.raises(NotImplementedError, match="continuous monitoring"):
        price(product, "bsm", "lookback_bsm", fwd, params)
