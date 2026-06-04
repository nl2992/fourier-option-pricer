from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

import foureng as fe
from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd
from foureng.pipeline import price
from foureng.products import AmericanOption


def _closed_form_strip(fwd, params, strikes, *, cp=1):
    return np.array(
        [
            bs_price_from_fwd(
                params.sigma,
                BSInputs(fwd.F0, float(k), fwd.T, fwd.r, fwd.q, is_call=(cp == 1)),
            )
            for k in strikes
        ]
    )


def test_bsm_lattice_european_strip_matches_closed_form():
    fwd = fe.ForwardSpec(S0=100.0, r=0.04, q=0.01, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    grid = fe.LatticeGrid(steps=1200)

    actual = fe.price_strip("bsm", "lattice", strikes, fwd, params, grid=grid)
    expected = _closed_form_strip(fwd, params, strikes)

    assert_allclose(actual, expected, atol=3e-3, rtol=3e-4)


def test_bsm_pde_fd_european_strip_matches_closed_form():
    fwd = fe.ForwardSpec(S0=100.0, r=0.04, q=0.01, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    strikes = np.array([90.0, 100.0, 110.0])
    grid = fe.PDEGrid(spot_steps=500, time_steps=500, s_max_mult=4.0)

    actual = fe.price_strip("bsm", "pde_fd", strikes, fwd, params, grid=grid)
    expected = _closed_form_strip(fwd, params, strikes)

    assert_allclose(actual, expected, atol=2.5e-2, rtol=2e-3)


def test_american_put_pde_and_lattice_are_consistent_and_above_european():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.25)
    product = AmericanOption(strike=105.0, maturity=1.0, cp=-1)

    lattice = price(product, "bsm", "lattice", fwd, params, grid=fe.LatticeGrid(steps=900))
    pde = price(
        product,
        "bsm",
        "pde_fd",
        fwd,
        params,
        grid=fe.PDEGrid(spot_steps=450, time_steps=450, s_max_mult=4.0),
    )
    euro = bs_price_from_fwd(
        params.sigma, BSInputs(fwd.F0, product.strike, fwd.T, fwd.r, fwd.q, is_call=False)
    )

    assert lattice >= euro
    assert pde >= euro
    assert_allclose(pde, lattice, atol=5e-2, rtol=5e-3)
