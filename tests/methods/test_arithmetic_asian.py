"""Deterministic arithmetic Asians under Levy models (Carverhill-Clewlow / ASCOS)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

import foureng as fe
from foureng.products.asian import AsianOption

S0, R, Q = 100.0, 0.05, 0.02
FWD = fe.ForwardSpec(S0=S0, r=R, q=Q, T=1.0)
KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)


@pytest.mark.parametrize("model,params", [("bsm", fe.BsmParams(0.25)), ("kou", KOU)])
def test_single_date_is_the_european(model, params):
    for cp in (1, -1):
        got = fe.levy_arithmetic_asian_price(
            model, FWD, params, strike=95.0, monitoring_times=[1.0], cp=cp
        )
        ref = fe.price_strip(model, "contour", [95.0], FWD, params, cp=cp)[0]
        assert got == pytest.approx(ref, abs=1e-10)


@pytest.mark.derived_reference
@pytest.mark.parametrize("t1", [0.5, 0.3])
def test_two_dates_match_exact_quadrature(t1):
    """BSM, N = 2: given S_1 the payoff is half a Black call on S_2 struck at 2K - S_1."""
    sig, K, t2 = 0.25, 100.0, 1.0

    def integrand(y1):
        S1 = S0 * np.exp((R - Q - 0.5 * sig * sig) * t1 + sig * np.sqrt(t1) * y1)
        k_eff, F2 = 2 * K - S1, S1 * np.exp((R - Q) * (t2 - t1))
        inner = 0.5 * (F2 - k_eff) if k_eff <= 0 else 0.5 * fe.black_price(F2, k_eff, t2 - t1, sig)
        return inner * np.exp(-0.5 * y1 * y1) / np.sqrt(2 * np.pi)

    kink = (np.log(2 * K / S0) - (R - Q - 0.5 * sig * sig) * t1) / (sig * np.sqrt(t1))
    quad = sum(
        integrate.quad(integrand, lo, hi, epsabs=1e-14, epsrel=1e-13, limit=400)[0]
        for lo, hi in ((-12.0, kink), (kink, 12.0))
    )
    ref = np.exp(-R * t2) * quad
    got = fe.levy_arithmetic_asian_price(
        "bsm", FWD, fe.BsmParams(sig), strike=K, monitoring_times=[t1, t2]
    )
    assert got == pytest.approx(ref, abs=1e-11)


def test_converged_in_cosine_terms_and_quadrature():
    t = np.arange(1, 53) / 52
    coarse = fe.levy_arithmetic_asian_price("kou", FWD, KOU, strike=100.0, monitoring_times=t)
    fine = fe.levy_arithmetic_asian_price(
        "kou", FWD, KOU, strike=100.0, monitoring_times=t, n_cos=512, n_quad=2048
    )
    assert coarse == pytest.approx(fine, abs=1e-9)


def _mc_with_geometric_cv(model, params, t, K, n=200_000, seed=5):
    rng = np.random.default_rng(seed)
    N = t.size
    dt = 1.0 / N
    if model == "bsm":
        sig = params.sigma
        incr = (R - Q - 0.5 * sig * sig) * dt + sig * np.sqrt(dt) * rng.standard_normal((n, N))
        geo_ref = fe.bsm_discrete_geometric_asian(S0, K, R, Q, t, sig, cp=1)
    else:
        k = params
        zeta = k.p * k.eta1 / (k.eta1 - 1) + (1 - k.p) * k.eta2 / (k.eta2 + 1) - 1
        mu = (R - Q - 0.5 * k.sigma**2 - k.lam * zeta) * dt
        counts = rng.poisson(k.lam * dt, (n, N))
        jumps = np.zeros((n, N))
        for i in range(int(counts.max())):
            up = rng.random((n, N)) < k.p
            size = np.where(
                up, rng.exponential(1 / k.eta1, (n, N)), -rng.exponential(1 / k.eta2, (n, N))
            )
            jumps += np.where(counts > i, size, 0.0)
        incr = mu + k.sigma * np.sqrt(dt) * rng.standard_normal((n, N)) + jumps
        geo_ref = fe.levy_geometric_asian_price(
            model, FWD, params, strikes=[K], monitoring_times=t
        )[0]
    S = S0 * np.exp(np.cumsum(incr, axis=1))
    disc = np.exp(-R * FWD.T)
    arith = disc * np.maximum(S.mean(axis=1) - K, 0.0)
    geo = disc * np.maximum(np.exp(np.log(S).mean(axis=1)) - K, 0.0)
    beta = np.cov(arith, geo)[0, 1] / geo.var()
    est = arith - beta * (geo - geo_ref)
    return est.mean(), est.std() / np.sqrt(n)


@pytest.mark.parametrize("model,params", [("bsm", fe.BsmParams(0.25)), ("kou", KOU)])
def test_matches_monte_carlo_with_exact_geometric_control(model, params):
    t = np.arange(1, 13) / 12
    mc, se = _mc_with_geometric_cv(model, params, t, 100.0)
    got = fe.levy_arithmetic_asian_price(model, FWD, params, strike=100.0, monitoring_times=t)
    assert abs(got - mc) < 4.0 * se


def test_parity_jensen_and_pipeline_dispatch():
    t = np.arange(1, 13) / 12
    call = fe.levy_arithmetic_asian_price("kou", FWD, KOU, strike=100.0, monitoring_times=t)
    put = fe.levy_arithmetic_asian_price("kou", FWD, KOU, strike=100.0, monitoring_times=t, cp=-1)
    mean_a = S0 * np.mean(np.exp((R - Q) * t))
    assert call - put == pytest.approx(np.exp(-R) * (mean_a - 100.0), abs=1e-12)
    geo = fe.levy_geometric_asian_price("kou", FWD, KOU, strikes=[100.0], monitoring_times=t)[0]
    assert call > geo  # AM >= GM

    prod = AsianOption(strike=100.0, maturity=1.0, cp=1, monitoring_times=t)
    assert fe.price(prod, "kou", "asian_cos", FWD, KOU) == pytest.approx(call, abs=1e-14)
    geo_prod = AsianOption(strike=100.0, maturity=1.0, monitoring_times=t, average_type="geometric")
    with pytest.raises(NotImplementedError, match="asian_cf"):
        fe.price(geo_prod, "kou", "asian_cos", FWD, KOU)
    heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.4, rho=-0.6, v0=0.04)
    with pytest.raises(NotImplementedError, match="1-D Levy"):
        fe.price(prod, "heston", "asian_cos", FWD, heston)
