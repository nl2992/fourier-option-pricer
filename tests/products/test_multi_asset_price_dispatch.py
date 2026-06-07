from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import foureng as fe
from foureng.analytics.bsm_exotics import kirk_spread, margrabe_exchange
from foureng.mc.engine import MCSpec
from foureng.pipeline import price
from foureng.products import BasketOption, BestOfOption, ExchangeOption, SpreadOption


def test_spread_bsm_dispatch_matches_kirk_formula():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=5.0)
    params = fe.BsmParams(sigma=0.22)
    product = SpreadOption(
        strike=5.0,
        maturity=1.5,
        cp=1,
        spot2=92.0,
        q2=0.015,
        sigma2=0.18,
        rho=0.35,
    )

    actual = price(product, "bsm", "spread_bsm", fwd, params)
    expected = kirk_spread(100.0, 92.0, 5.0, 0.03, 0.01, 0.015, 1.5, 0.22, 0.18, 0.35, cp=1)

    assert actual == approx(expected, rel=1e-12, abs=1e-12)


def test_exchange_multi_asset_mc_matches_deterministic_zero_vol_case():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.01, T=2.0)
    params = fe.BsmParams(sigma=1e-12)
    product = ExchangeOption(maturity=1.25, spot2=90.0, q2=0.02, sigma2=1e-12, rho=0.4)

    actual = price(
        product,
        "bsm",
        "multi_asset_mc",
        fwd,
        params,
        grid=MCSpec(n_paths=2_000, seed=7, antithetic=True),
    )
    expected = margrabe_exchange(100.0, 90.0, 0.01, 0.02, 1.25, 1e-12, 1e-12, 0.4)

    assert actual == approx(expected, rel=1e-9, abs=1e-9)


def test_basket_multi_asset_mc_matches_deterministic_zero_vol_case():
    fwd = fe.ForwardSpec(S0=100.0, r=0.04, q=0.01, T=3.0)
    params = fe.BsmParams(sigma=1e-12)
    product = BasketOption(
        strike=180.0,
        maturity=1.5,
        cp=1,
        weights=np.array([0.6, 0.4]),
        other_spots=np.array([120.0]),
        other_dividend_yields=np.array([0.02]),
        other_volatilities=np.array([1e-12]),
        corr_matrix=np.eye(2),
    )

    actual = price(
        product,
        "bsm",
        "multi_asset_mc",
        fwd,
        params,
        grid=MCSpec(n_paths=2_000, seed=11, antithetic=True),
    )
    s1_t = 100.0 * np.exp((0.04 - 0.01) * 1.5)
    s2_t = 120.0 * np.exp((0.04 - 0.02) * 1.5)
    expected = np.exp(-0.04 * 1.5) * max(0.6 * s1_t + 0.4 * s2_t - 180.0, 0.0)

    assert actual == approx(expected, rel=1e-9, abs=1e-9)


def test_spread_multi_asset_mc_matches_deterministic_zero_vol_case():
    fwd = fe.ForwardSpec(S0=110.0, r=0.03, q=0.01, T=2.0)
    params = fe.BsmParams(sigma=1e-12)
    product = SpreadOption(
        strike=7.0,
        maturity=1.0,
        cp=-1,
        spot2=95.0,
        q2=0.015,
        sigma2=1e-12,
        rho=-0.3,
    )

    actual = price(
        product,
        "bsm",
        "multi_asset_mc",
        fwd,
        params,
        grid=MCSpec(n_paths=2_000, seed=19, antithetic=True),
    )
    s1_t = 110.0 * np.exp((0.03 - 0.01) * 1.0)
    s2_t = 95.0 * np.exp((0.03 - 0.015) * 1.0)
    expected = np.exp(-0.03 * 1.0) * max(-(s1_t - s2_t - 7.0), 0.0)

    assert actual == approx(expected, rel=1e-9, abs=1e-9)


def test_best_of_multi_asset_mc_matches_deterministic_zero_vol_case():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=2.0)
    params = fe.BsmParams(sigma=1e-12)
    product = BestOfOption(
        strike=108.0,
        maturity=1.0,
        cp=1,
        n_assets=3,
        other_spots=np.array([95.0, 120.0]),
        other_dividend_yields=np.array([0.01, 0.03]),
        other_volatilities=np.array([1e-12, 1e-12]),
        corr_matrix=np.eye(3),
    )

    actual = price(
        product,
        "bsm",
        "multi_asset_mc",
        fwd,
        params,
        grid=MCSpec(n_paths=2_000, seed=23, antithetic=True),
    )
    terminals = np.array(
        [
            100.0 * np.exp((0.05 - 0.02) * 1.0),
            95.0 * np.exp((0.05 - 0.01) * 1.0),
            120.0 * np.exp((0.05 - 0.03) * 1.0),
        ]
    )
    expected = np.exp(-0.05 * 1.0) * max(np.max(terminals) - 108.0, 0.0)

    assert actual == approx(expected, rel=1e-9, abs=1e-9)


def test_spread_bsm_rejects_non_bsm_model():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.4, rho=-0.5, v0=0.04)
    product = SpreadOption(strike=0.0, maturity=1.0, cp=1)

    with pytest.raises(NotImplementedError, match="model='bsm'"):
        price(product, "heston", "spread_bsm", fwd, params)


def test_basket_unknown_method_is_explicit():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    product = BasketOption(strike=100.0, maturity=1.0, cp=1, weights=np.array([0.5, 0.5]))

    with pytest.raises(NotImplementedError, match="multi_asset_mc"):
        price(product, "bsm", "cos", fwd, params)
