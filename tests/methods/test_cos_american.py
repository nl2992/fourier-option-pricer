"""American options under 1-D Levy models via Richardson-extrapolated COS Bermudans.

References
----------
Fang, F. & Oosterlee, C.W. (2009). Pricing early-exercise and discrete barrier
options by Fourier-cosine series expansions. Numerische Mathematik 114, 27-62.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import foureng as fe
from foureng.core.capabilities import explain_capability
from foureng.models.base import ForwardSpec
from foureng.models.registry import MODEL_REGISTRY
from foureng.pricers.cos_bermudan import cos_american_price, cos_bermudan_price
from foureng.products.american import AmericanOption
from foureng.products.bermudan import BermudanOption

KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)
CGMY = fe.CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.8)
NIG = fe.NigParams(sigma=0.2, nu=0.3, theta=-0.1)


# Derived references: independent CRR binomial trees, averaged over n and n+1
# steps (cancelling the odd/even oscillation) at n = 20k and 40k, then
# Richardson-extrapolated in 1/n. Their own error is ~1e-5.
@pytest.mark.derived_reference
@pytest.mark.parametrize(
    "S0,K,r,q,sigma,T,cp,ref,tol",
    [
        (100.0, 110.0, 0.10, 0.00, 0.2, 1.0, -1, 10.71919116, 5e-5),
        (40.0, 36.0, 0.06, 0.00, 0.4, 0.5, -1, 2.20315529, 2e-5),
        (100.0, 100.0, 0.05, 0.03, 0.3, 2.0, 1, 17.47631463, 2e-5),
    ],
)
def test_bsm_american_matches_extrapolated_binomial(S0, K, r, q, sigma, T, cp, ref, tol):
    fwd = ForwardSpec(S0=S0, r=r, q=q, T=T)
    got = fe.price(
        AmericanOption(strike=K, maturity=T, cp=cp), "bsm", "cos_american", fwd, fe.BsmParams(sigma)
    )
    assert got == pytest.approx(ref, abs=tol)


def test_american_call_without_dividends_equals_european():
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    for model, params in (("bsm", fe.BsmParams(sigma=0.25)), ("kou", KOU)):
        am = cos_american_price(model, fwd, params, AmericanOption(105.0, 1.0, 1), base_dates=16)
        eu = fe.price_strip(model, "cos_improved", np.array([105.0]), fwd, params)[0]
        assert am == pytest.approx(eu, abs=1e-8)


def _step_cf(model, params, S0, r, q, dt):
    fwd_dt = ForwardSpec(S0=S0, r=r, q=q, T=dt)
    cf = MODEL_REGISTRY[model].cf
    return lambda u: (
        np.exp(1j * u * (r - q) * dt) * np.asarray(cf(u, fwd_dt, params), dtype=complex)
    )


@pytest.mark.parametrize("model,params,tol", [("kou", KOU, 2e-6), ("cgmy", CGMY, 2e-5)])
def test_levy_american_matches_richardson_on_independent_proj_engine(model, params, tol):
    """Same extrapolation, independent Bermudan engine (Kirkby's PROJ recursion)."""
    S0, r, q, T, W, m = 100.0, 0.05, 0.0, 1.0, 100.0, 16
    fwd = ForwardSpec(S0=S0, r=r, q=q, T=T)
    c1, c2, c4 = MODEL_REGISTRY[model].cumulants(fwd, params)
    alph = 30.0 * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    proj = [
        fe.proj_bermudan_put(
            _step_cf(model, params, S0, r, q, T / M), S0=S0, r=r, T=T, W=W, M=M, N=2**15, alph=alph
        )
        for M in (m, 2 * m, 4 * m, 8 * m)
    ]
    proj_american = (64 * proj[3] - 56 * proj[2] + 14 * proj[1] - proj[0]) / 21
    cos_american = cos_american_price(model, fwd, params, AmericanOption(W, T, -1), base_dates=m)
    assert cos_american == pytest.approx(proj_american, abs=tol)


def test_no_arbitrage_ordering_and_immediate_exercise():
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    am = cos_american_price("nig", fwd, NIG, AmericanOption(100.0, 1.0, -1), base_dates=16)
    berm = cos_bermudan_price(
        "nig", fwd, NIG, BermudanOption(100.0, 1.0, -1, np.linspace(1 / 128, 1.0, 128)), N=1024
    )
    eu = fe.price_strip("nig", "cos_improved", np.array([100.0]), fwd, NIG, cp=-1)[0]
    assert am >= berm >= eu
    # Deep in the money with r > 0: immediate exercise is optimal.
    deep = cos_american_price("bsm", fwd, fe.BsmParams(0.2), AmericanOption(200.0, 1.0, -1))
    assert deep == pytest.approx(100.0, abs=1e-10)


def test_american_put_is_decreasing_and_convex_in_spot():
    vals = [
        cos_american_price(
            "kou",
            ForwardSpec(S0=s, r=0.05, q=0.0, T=0.5),
            KOU,
            AmericanOption(100.0, 0.5, -1),
            base_dates=16,
        )
        for s in (85.0, 95.0, 105.0, 115.0)
    ]
    d = np.diff(vals)
    assert np.all(d < 0.0) and np.all(np.diff(d) > 0.0)


def test_pipeline_dispatch_and_sv_models_rejected():
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    prod = AmericanOption(100.0, 1.0, -1)
    bsm = fe.BsmParams(0.2)
    assert fe.price(prod, "bsm", "cos_american", fwd, bsm) == pytest.approx(
        cos_american_price("bsm", fwd, bsm, prod)
    )
    heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.6, v0=0.04)
    with pytest.raises(NotImplementedError, match="1-D COS Bermudan"):
        fe.price(prod, "heston", "cos_american", fwd, heston)
    with pytest.raises(NotImplementedError, match="cos_american"):
        fe.price(prod, "kou", "lattice", fwd, KOU)
    assert "stochastic-volatility" in explain_capability("heston", "american", "cos_american")


def test_fo2009_bermudan_is_exact_in_the_european_limit():
    """One exercise date at maturity must reproduce the European COS price."""
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
    for model, params in (("bsm", fe.BsmParams(0.2)), ("kou", KOU), ("nig", NIG)):
        for cp in (-1, 1):
            berm = cos_bermudan_price(
                model, fwd, params, BermudanOption(95.0, 1.0, cp, np.array([1.0])), N=1024
            )
            eu = fe.price_strip(model, "cos_improved", np.array([95.0]), fwd, params, cp=cp)[0]
            assert berm == pytest.approx(eu, abs=1e-9)


def test_fo2009_bermudan_converges_exponentially_in_n():
    fwd = ForwardSpec(S0=100.0, r=0.1, q=0.0, T=1.0)
    prod = BermudanOption(110.0, 1.0, -1, np.linspace(1 / 64, 1.0, 64))
    v = {n: cos_bermudan_price("bsm", fwd, fe.BsmParams(0.2), prod, N=n) for n in (256, 512, 1024)}
    assert abs(v[1024] - v[512]) < 1e-9
    assert abs(v[512] - v[256]) < 1e-3


def test_n_spatial_is_deprecated():
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    prod = BermudanOption(100.0, 1.0, -1, np.array([0.5, 1.0]))
    with pytest.warns(DeprecationWarning, match="n_spatial"):
        cos_bermudan_price("bsm", fwd, fe.BsmParams(0.2), prod, n_spatial=2048)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cos_bermudan_price("bsm", fwd, fe.BsmParams(0.2), prod)
