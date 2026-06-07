from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.mc.engine import MCSpec
from foureng.pipeline import price
from foureng.pricers.lattice import LatticeGrid, bsm_lattice_price
from foureng.products import (
    AmericanOption,
    AsianOption,
    BermudanOption,
    EuropeanOption,
    ExchangeOption,
)


def test_price_dispatches_european_monte_carlo():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=2.0)
    params = fe.BsmParams(sigma=0.2)
    product = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    mc = MCSpec(n_paths=20_000, n_steps=1, seed=7, antithetic=True)

    actual = price(product, "bsm", "monte_carlo", fwd, params, grid=mc)
    repeat = price(product, "bsm", "monte_carlo", fwd, params, grid=mc)

    assert actual == pytest.approx(repeat, rel=0.0, abs=0.0)


def test_price_dispatches_american_monte_carlo_close_to_lattice():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=2.0)
    params = fe.BsmParams(sigma=0.2)
    product = AmericanOption(strike=100.0, maturity=1.0, cp=-1)
    mc = MCSpec(n_paths=80_000, n_steps=64, seed=11, antithetic=True)

    actual = price(product, "bsm", "monte_carlo", fwd, params, grid=mc)
    reference = bsm_lattice_price(
        fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0),
        params,
        100.0,
        cp=-1,
        exercise="american",
        grid=LatticeGrid(steps=512),
    )

    assert abs(actual - reference) < 0.35


def test_price_dispatches_bermudan_monte_carlo_between_european_and_american():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    product = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=np.array([0.5, 1.0]))
    mc = MCSpec(n_paths=80_000, n_steps=2, seed=13, antithetic=True)

    actual = price(product, "bsm", "monte_carlo", fwd, params, grid=mc)
    european = price(
        EuropeanOption(strike=100.0, maturity=1.0, cp=-1),
        "bsm",
        "monte_carlo",
        fwd,
        params,
        grid=mc,
    )
    american = bsm_lattice_price(
        fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0),
        params,
        100.0,
        cp=-1,
        exercise="american",
        grid=LatticeGrid(steps=512),
    )

    assert european <= actual <= american + 0.1


def test_price_dispatches_asian_monte_carlo_as_alias():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    product = AsianOption(
        strike=100.0,
        maturity=1.0,
        cp=1,
        average_type="arithmetic",
        monitoring_times=np.linspace(0.25, 1.0, 4),
    )
    mc = MCSpec(n_paths=20_000, n_steps=32, seed=17, antithetic=True)

    alias_price = price(product, "bsm", "monte_carlo", fwd, params, grid=mc)
    existing_price = price(product, "bsm", "asian_mc", fwd, params, grid=mc)

    assert alias_price == pytest.approx(existing_price, rel=1e-12, abs=1e-12)


def test_price_dispatches_exchange_monte_carlo_as_alias():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.01, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    product = ExchangeOption(maturity=1.0, spot2=95.0, q2=0.02, sigma2=0.25, rho=0.3)
    mc = MCSpec(n_paths=20_000, seed=19, antithetic=True)

    alias_price = price(product, "bsm", "monte_carlo", fwd, params, grid=mc)
    existing_price = price(product, "bsm", "multi_asset_mc", fwd, params, grid=mc)

    assert alias_price == pytest.approx(existing_price, rel=1e-12, abs=1e-12)


def test_price_monte_carlo_rejects_non_bsm_model():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04)
    product = AmericanOption(strike=100.0, maturity=1.0, cp=-1)

    with pytest.raises(NotImplementedError, match="only for model='bsm'"):
        price(product, "heston", "monte_carlo", fwd, params)
