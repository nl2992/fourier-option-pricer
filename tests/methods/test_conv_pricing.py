from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

import foureng as fe
from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd


def test_conv_bsm_matches_closed_form_calls_and_puts():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.25)
    params = fe.BsmParams(sigma=0.22)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 125.0])
    grid = fe.CONVGrid(N=4096, u_max=180.0)

    calls = fe.price_strip("bsm", "conv", strikes, fwd, params, grid=grid, cp=1)
    puts = fe.price_strip("bsm", "conv", strikes, fwd, params, grid=grid, cp=-1)

    expected_calls = np.array(
        [
            bs_price_from_fwd(params.sigma, BSInputs(fwd.F0, float(k), fwd.T, fwd.r, fwd.q, True))
            for k in strikes
        ]
    )
    expected_puts = np.array(
        [
            bs_price_from_fwd(params.sigma, BSInputs(fwd.F0, float(k), fwd.T, fwd.r, fwd.q, False))
            for k in strikes
        ]
    )

    assert_allclose(calls, expected_calls, atol=2e-5, rtol=2e-6)
    assert_allclose(puts, expected_puts, atol=2e-5, rtol=2e-6)


def test_conv_heston_is_close_to_lewis_reference():
    fwd = fe.ForwardSpec(S0=100.0, r=0.02, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.35, rho=-0.6, v0=0.04)
    strikes = np.array([90.0, 100.0, 110.0])
    phi = lambda u: fe.heston_cf_form2(u, fwd, params)

    conv = fe.price_strip(
        "heston", "conv", strikes, fwd, params, grid=fe.CONVGrid(N=4096, u_max=180.0)
    )
    ref = fe.lewis_call_prices(
        phi,
        strikes,
        spot=fwd.S0,
        texp=fwd.T,
        intr=fwd.r,
        divr=fwd.q,
        u_max=180.0,
        n_u=4096,
    )

    assert_allclose(conv, ref, atol=2e-4, rtol=2e-5)
