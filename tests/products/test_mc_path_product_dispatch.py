"""Product-dispatch tests for Monte Carlo path-dependent products."""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.mc.engine import MCSpec, mc_price
from foureng.pipeline import price
from foureng.products import CliquetOption, LookbackOption, VarianceOption, VarianceSwap


def test_price_dispatches_lookback_mc_fixed_strike():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    product = LookbackOption(maturity=1.0, cp=1, strike_type="fixed", strike=100.0)
    mc = MCSpec(n_paths=20_000, n_steps=64, seed=7, antithetic=True)

    actual = price(product, "bsm", "lookback_mc", fwd, params, grid=mc)
    expected = mc_price(fwd, params.sigma, product, mc).price

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_price_dispatches_variance_swap_mc():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    product = VarianceSwap(maturity=1.0, sampling_times=np.linspace(1 / 12, 1.0, 12))
    mc = MCSpec(n_paths=20_000, n_steps=12, seed=11, antithetic=True)

    actual = price(product, "bsm", "variance_mc", fwd, params, grid=mc)
    expected = mc_price(fwd, params.sigma, product, mc).price

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_price_dispatches_variance_option_mc():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    product = VarianceOption(
        strike=0.03,
        maturity=1.0,
        cp=1,
        sampling_times=np.linspace(1 / 12, 1.0, 12),
        variance_type="integrated",
    )
    mc = MCSpec(n_paths=20_000, n_steps=12, seed=13, antithetic=True)

    actual = price(product, "bsm", "variance_mc", fwd, params, grid=mc)
    expected = mc_price(fwd, params.sigma, product, mc).price

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_price_dispatches_cliquet_mc():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    product = CliquetOption(
        maturity=1.0,
        reset_times=np.array([0.25, 0.5, 0.75, 1.0]),
        local_floor=-0.03,
        local_cap=0.05,
    )
    mc = MCSpec(n_paths=20_000, n_steps=4, seed=17, antithetic=True)

    actual = price(product, "bsm", "cliquet_mc", fwd, params, grid=mc)
    expected = mc_price(fwd, params.sigma, product, mc).price

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_price_variance_mc_rejects_non_bsm_model():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04)
    product = VarianceSwap(maturity=1.0, sampling_times=np.linspace(1 / 12, 1.0, 12))

    with pytest.raises(NotImplementedError, match="only for model='bsm'"):
        price(product, "heston", "variance_mc", fwd, params)
