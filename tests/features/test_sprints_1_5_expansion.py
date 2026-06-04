from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

import foureng as fe
from foureng.pipeline import price
from foureng.products import AsianOption, DoubleBarrierOption


def test_discrete_geometric_asian_dispatch_matches_public_formula():
    fwd = fe.ForwardSpec(S0=100.0, r=0.04, q=0.01, T=2.0)
    params = fe.BsmParams(sigma=0.2)
    times = np.array([0.25, 0.5, 0.75, 1.0])
    product = AsianOption(
        strike=100.0,
        maturity=1.0,
        cp=1,
        average_type="geometric",
        monitoring_times=times,
    )

    actual = price(product, "bsm", "asian_bsm", fwd, params)
    expected = fe.bsm_discrete_geometric_asian(
        fwd.S0, product.strike, fwd.r, fwd.q, times, params.sigma, cp=1
    )

    assert actual == expected


def test_asian_mc_geometric_agrees_with_exact_within_confidence_band():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.18)
    times = np.linspace(1.0 / 12.0, 1.0, 12)
    product = AsianOption(
        strike=100.0,
        maturity=1.0,
        cp=1,
        average_type="geometric",
        monitoring_times=times,
    )
    mc = fe.MCSpec(n_paths=40_000, n_steps=12, seed=123, antithetic=True)

    actual = price(product, "bsm", "asian_mc", fwd, params, grid=mc)
    expected = fe.bsm_discrete_geometric_asian(
        fwd.S0, product.strike, fwd.r, fwd.q, times, params.sigma, cp=1
    )

    assert abs(actual - expected) < 0.08


def test_double_barrier_mc_in_out_parity_on_same_seed():
    fwd = fe.ForwardSpec(S0=100.0, r=0.02, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    mc = fe.MCSpec(n_paths=30_000, n_steps=80, seed=7, antithetic=True)
    out_product = DoubleBarrierOption(
        strike=100.0,
        lower_barrier=70.0,
        upper_barrier=140.0,
        maturity=1.0,
        cp=1,
        knockout=True,
    )
    in_product = DoubleBarrierOption(
        strike=100.0,
        lower_barrier=70.0,
        upper_barrier=140.0,
        maturity=1.0,
        cp=1,
        knockout=False,
    )
    out_price = price(out_product, "bsm", "double_barrier_mc", fwd, params, grid=mc)
    in_price = price(in_product, "bsm", "double_barrier_mc", fwd, params, grid=mc)
    vanilla = fe.bs_price_from_fwd(
        params.sigma,
        fe.BSInputs(fwd.F0, out_product.strike, fwd.T, fwd.r, fwd.q),
    )

    assert out_price >= 0.0
    assert in_price >= 0.0
    assert_allclose(out_price + in_price, vanilla, atol=0.25, rtol=0.02)


def test_proj_european_matches_cos_baseline_for_bsm():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    strikes = np.array([90.0, 100.0, 110.0])

    proj = fe.price_strip("bsm", "proj", strikes, fwd, params)
    cos = fe.price_strip("bsm", "cos", strikes, fwd, params, grid=fe.COSGrid(N=512, a=-2.0, b=2.0))

    assert_allclose(proj, cos, atol=2e-4, rtol=2e-5)


def test_mellin_facade_matches_conv_for_vg():
    fwd = fe.ForwardSpec(S0=100.0, r=0.01, q=0.0, T=0.75)
    params = fe.VGParams(sigma=0.12, nu=0.2, theta=-0.1)
    strikes = np.array([90.0, 100.0, 110.0])
    grid = fe.CONVGrid(N=4096, u_max=180.0)

    mellin = fe.price_strip("vg", "mellin", strikes, fwd, params, grid=grid)
    conv = fe.price_strip("vg", "conv", strikes, fwd, params, grid=grid)

    assert_allclose(mellin, conv, atol=1e-12, rtol=1e-12)


def test_sabr_hagan_beta_one_zero_nu_reduces_to_bsm():
    fwd = fe.ForwardSpec(S0=100.0, r=0.02, q=0.0, T=1.0)
    params = fe.SabrParams(alpha=0.2, beta=1.0, rho=-0.3, nu=0.0)
    strikes = np.array([90.0, 100.0, 110.0])

    sabr = fe.price_strip("sabr", "sabr_hagan", strikes, fwd, params)
    bsm = np.array(
        [
            fe.bs_price_from_fwd(0.2, fe.BSInputs(fwd.F0, float(k), fwd.T, fwd.r, fwd.q))
            for k in strikes
        ]
    )

    assert_allclose(sabr, bsm, atol=1e-10, rtol=1e-10)
