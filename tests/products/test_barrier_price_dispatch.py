from __future__ import annotations

import pytest
from pytest import approx

import foureng as fe
from foureng.analytics.bsm_barrier import bsm_barrier_price, bsm_call
from foureng.pipeline import price
from foureng.products import BarrierOption


def test_price_dispatches_continuous_zero_rebate_bsm_barrier():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=99.0)
    params = fe.BsmParams(sigma=0.25)
    product = BarrierOption(
        strike=90.0,
        barrier=80.0,
        maturity=1.0,
        cp=1,
        barrier_type="down_out",
    )

    actual = price(product, "bsm", "barrier_bsm", fwd, params)
    expected = bsm_barrier_price(
        100.0,
        90.0,
        80.0,
        0.05,
        0.02,
        1.0,
        0.25,
        "down_out",
        cp=1,
    )

    assert actual == approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("cp", [1, -1])
@pytest.mark.parametrize("base,barrier,strike", [("down", 80.0, 90.0), ("up", 120.0, 110.0)])
def test_price_dispatch_preserves_in_out_parity(cp, base, barrier, strike):
    fwd = fe.ForwardSpec(S0=100.0, r=0.04, q=0.01, T=1.0)
    params = fe.BsmParams(sigma=0.22)
    out_product = BarrierOption(
        strike=strike,
        barrier=barrier,
        maturity=1.0,
        cp=cp,
        barrier_type=f"{base}_out",
    )
    in_product = BarrierOption(
        strike=strike,
        barrier=barrier,
        maturity=1.0,
        cp=cp,
        barrier_type=f"{base}_in",
    )

    out_price = price(out_product, "bsm", "barrier_bsm", fwd, params)
    in_price = price(in_product, "bsm", "barrier_bsm", fwd, params)
    vanilla = fe.bs_price_from_fwd(
        params.sigma,
        fe.BSInputs(fwd.F0, strike, fwd.T, fwd.r, fwd.q, is_call=(cp == 1)),
    )

    assert out_price >= 0.0
    assert in_price >= 0.0
    assert out_price + in_price == approx(vanilla, rel=1e-10, abs=1e-10)


def test_barrier_bsm_uses_product_maturity_not_forward_spec_maturity():
    fwd_wrong_t = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=3.0)
    params = fe.BsmParams(sigma=0.2)
    product = BarrierOption(
        strike=90.0,
        barrier=80.0,
        maturity=0.75,
        cp=1,
        barrier_type="down_out",
    )

    actual = price(product, "bsm", "barrier_bsm", fwd_wrong_t, params)
    wrong = bsm_barrier_price(100.0, 90.0, 80.0, 0.05, 0.0, 3.0, 0.2, "down_out", cp=1)
    correct = bsm_barrier_price(100.0, 90.0, 80.0, 0.05, 0.0, 0.75, 0.2, "down_out", cp=1)

    assert actual == approx(correct, rel=1e-12, abs=1e-12)
    assert abs(actual - wrong) > 1e-3


def test_barrier_bsm_rejects_non_bsm_model():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04)
    product = BarrierOption(
        strike=90.0,
        barrier=80.0,
        maturity=1.0,
        cp=1,
        barrier_type="down_out",
    )

    with pytest.raises(NotImplementedError, match="model='bsm'"):
        price(product, "heston", "barrier_bsm", fwd, params)


def test_barrier_bsm_rejects_rebates_and_discrete_monitoring():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    rebate_product = BarrierOption(
        strike=90.0,
        barrier=80.0,
        maturity=1.0,
        cp=1,
        barrier_type="down_out",
        rebate=1.0,
    )
    discrete_product = BarrierOption(
        strike=90.0,
        barrier=80.0,
        maturity=1.0,
        cp=1,
        barrier_type="down_out",
        monitoring="discrete",
    )

    with pytest.raises(NotImplementedError, match="zero rebates"):
        price(rebate_product, "bsm", "barrier_bsm", fwd, params)
    with pytest.raises(NotImplementedError, match="continuous monitoring"):
        price(discrete_product, "bsm", "barrier_bsm", fwd, params)


def test_barrier_bsm_knockout_is_bounded_by_vanilla():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
    params = fe.BsmParams(sigma=0.25)
    product = BarrierOption(
        strike=90.0,
        barrier=80.0,
        maturity=1.0,
        cp=1,
        barrier_type="down_out",
    )

    barrier_price = price(product, "bsm", "barrier_bsm", fwd, params)
    vanilla = bsm_call(fwd.S0, product.strike, fwd.r, fwd.q, product.maturity, params.sigma)

    assert 0.0 <= barrier_price <= vanilla
