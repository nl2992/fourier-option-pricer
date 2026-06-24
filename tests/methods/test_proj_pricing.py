"""PROJ (frame-projection) European pricer tests.

PROJ is a faithful port of Kirkby's ``PROJ_European.m``. We validate it against
the BSM closed form and against the in-house COS pricer across the 1-D Lévy
model family, for both calls and puts and several maturities.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import foureng as fe
from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd
from foureng.models.base import ForwardSpec
from foureng.models.registry import MODEL_REGISTRY
from foureng.pipeline import price_strip
from foureng.pricers.cos_bermudan import cos_bermudan_price_strip


def test_proj_bsm_matches_closed_form_calls_and_puts():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.25)
    params = fe.BsmParams(sigma=0.22)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 125.0])

    calls = price_strip("bsm", "proj", strikes, fwd, params, cp=1)
    puts = price_strip("bsm", "proj", strikes, fwd, params, cp=-1)

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

    assert_allclose(calls, expected_calls, atol=1e-8, rtol=0)
    assert_allclose(puts, expected_puts, atol=1e-8, rtol=0)


def _levy_cases():
    return [
        ("bsm", fe.BsmParams(sigma=0.2)),
        ("vg", fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14)),
        ("cgmy", fe.CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.8)),
        ("kou", fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)),
        ("merton_jd", fe.MertonJDParams(sigma=0.15, lam=0.5, muj=-0.1, sigj=0.15)),
        ("nig", fe.NigParams(sigma=0.2, nu=0.2, theta=-0.1)),
    ]


@pytest.mark.parametrize("model,params", _levy_cases())
@pytest.mark.parametrize("T", [0.5, 1.0, 2.0])
def test_proj_matches_cos_across_levy_models(model, params, T):
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=T)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    cos = price_strip(model, "cos_improved", strikes, fwd, params)
    proj = price_strip(model, "proj", strikes, fwd, params)

    # PROJ vs COS agree to ~1e-6 across the Lévy family.
    assert_allclose(proj, cos, atol=5e-6, rtol=1e-5)


@pytest.mark.parametrize("model,params", _levy_cases())
def test_proj_put_call_parity(model, params):
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    calls = price_strip(model, "proj", strikes, fwd, params, cp=1)
    puts = price_strip(model, "proj", strikes, fwd, params, cp=-1)

    parity = (calls - puts) - (fwd.S0 * np.exp(-fwd.q * fwd.T) - strikes * np.exp(-fwd.r * fwd.T))
    assert_allclose(parity, 0.0, atol=1e-12)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_proj_spline_orders_converge_to_bsm(order):
    """All four B-spline orders converge; cubic is the most accurate."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.04, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.25)
    strikes = np.array([90.0, 100.0, 110.0])
    grid = fe.proj_auto_grid(fe.bsm_cumulants(fwd, params), N=2**14, order=order)

    proj = fe.proj_price_at_strikes(
        lambda u: fe.bsm_cf(u, fwd, params),
        fwd,
        grid,
        strikes,
        cp=1,
        c1=fe.bsm_cumulants(fwd, params)[0],
    )
    expected = np.array(
        [
            bs_price_from_fwd(params.sigma, BSInputs(fwd.F0, float(k), fwd.T, fwd.r, fwd.q, True))
            for k in strikes
        ]
    )
    # Higher spline order → tighter tolerance; even Haar is within 1e-2.
    tol = {0: 1e-2, 1: 1e-4, 2: 1e-5, 3: 1e-6}[order]
    assert_allclose(proj, expected, atol=tol, rtol=0)


def _step_cf(model, params, S0, r, q, dt):
    """One-step risk-neutral CF of log(S_{dt}/S_0) for a Lévy model."""
    fwd_dt = ForwardSpec(S0=S0, r=r, q=q, T=dt)
    cf = MODEL_REGISTRY[model].cf
    drift = (r - q) * dt
    return lambda u: np.exp(1j * u * drift) * np.asarray(cf(u, fwd_dt, params), dtype=complex)


@pytest.mark.parametrize(
    "model,params,tol",
    [
        ("bsm", fe.BsmParams(sigma=0.2), 1e-3),
        ("vg", fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14), 3e-3),
        ("kou", fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0), 2e-3),
        ("cgmy", fe.CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.8), 3e-3),
    ],
)
@pytest.mark.parametrize("M", [10, 50])
def test_proj_bermudan_put_matches_cos_bermudan(model, params, tol, M):
    """PROJ Bermudan put agrees with the independent COS Bermudan engine.

    The two methods use different bases (linear B-spline projection vs cosine
    series), so agreement at the 1e-3 level cross-validates both.
    """
    S0, r, q, T, W = 100.0, 0.05, 0.0, 1.0, 100.0
    fwd = ForwardSpec(S0=S0, r=r, q=q, T=T)
    dt = T / M
    ex_times = np.arange(1, M + 1) * dt

    cums = MODEL_REGISTRY[model].cumulants(fwd, params)
    alph = 10.0 * np.sqrt(abs(cums[1]) + np.sqrt(abs(cums[2])))
    step_cf = _step_cf(model, params, S0, r, q, dt)

    proj = fe.proj_bermudan_put(step_cf, S0=S0, r=r, T=T, W=W, M=M, N=2**15, alph=alph)
    cos = cos_bermudan_price_strip(model, fwd, params, np.array([W]), T, ex_times, cp=-1)[0]

    assert abs(proj - cos) < tol


def test_proj_bermudan_dominates_european():
    """A Bermudan put is worth at least its European counterpart."""
    S0, r, q, T, W, M = 100.0, 0.05, 0.0, 1.0, 100.0, 25
    fwd = ForwardSpec(S0=S0, r=r, q=q, T=T)
    params = fe.BsmParams(sigma=0.25)
    dt = T / M
    cums = MODEL_REGISTRY["bsm"].cumulants(fwd, params)
    alph = 10.0 * np.sqrt(abs(cums[1]) + np.sqrt(abs(cums[2])))

    berm = fe.proj_bermudan_put(
        _step_cf("bsm", params, S0, r, q, dt), S0=S0, r=r, T=T, W=W, M=M, N=2**15, alph=alph
    )
    euro = price_strip("bsm", "proj", np.array([W]), fwd, params, cp=-1)[0]
    assert berm >= euro - 1e-6


@pytest.mark.parametrize(
    "model,params,tol",
    [
        ("bsm", fe.BsmParams(sigma=0.2), 1e-3),
        ("vg", fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14), 3e-3),
        ("kou", fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0), 2e-3),
    ],
)
def test_price_proj_bermudan_dispatch_matches_cos(model, params, tol):
    """The product-level price(..., method='proj') Bermudan route matches cos_bermudan."""
    from foureng.pricers.cos_bermudan import cos_bermudan_price
    from foureng.products.bermudan import BermudanOption

    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    M = 50
    ex = np.arange(1, M + 1) * (1.0 / M)
    prod = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=ex)

    proj = fe.price(prod, model, "proj", fwd, params)
    cos = cos_bermudan_price(model, fwd, params, prod)
    assert abs(proj - cos) < tol


@pytest.mark.parametrize(
    "model,params",
    [
        ("bsm", fe.BsmParams(sigma=0.2)),
        ("vg", fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14)),
        ("kou", fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)),
    ],
)
def test_price_proj_bermudan_call_no_dividend_matches_european(model, params):
    """For q=0, Bermudan calls equal European calls and the PROJ route should use that."""
    from foureng.products.bermudan import BermudanOption

    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    M = 50
    ex = np.arange(1, M + 1) * (1.0 / M)
    prod = BermudanOption(strike=100.0, maturity=1.0, cp=1, exercise_times=ex)

    proj = fe.price(prod, model, "proj", fwd, params)
    euro = price_strip(model, "proj", np.array([100.0]), fwd, params, cp=1)[0]
    assert abs(proj - euro) < 1e-10


def test_price_proj_bermudan_guards():
    """PROJ Bermudan dispatch rejects dividend-sensitive calls, non-Lévy models, and non-uniform grids."""
    from foureng.products.bermudan import BermudanOption

    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    M = 20
    ex = np.arange(1, M + 1) * (1.0 / M)
    bsm = fe.BsmParams(sigma=0.2)

    with pytest.raises(NotImplementedError):  # dividend-sensitive calls still deferred
        fwd_div = ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
        fe.price(BermudanOption(100.0, 1.0, 1, ex), "bsm", "proj", fwd_div, bsm)
    with pytest.raises(NotImplementedError):  # non-Lévy model
        heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.6, v0=0.04)
        fe.price(BermudanOption(100.0, 1.0, -1, ex), "heston", "proj", fwd, heston)
    with pytest.raises(NotImplementedError):  # non-uniform schedule
        fe.price(BermudanOption(100.0, 1.0, -1, np.array([0.3, 0.5, 1.0])), "bsm", "proj", fwd, bsm)
