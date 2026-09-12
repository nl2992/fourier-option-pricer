"""Barndorff-Nielsen & Shephard (2001) Gamma-OU stochastic volatility."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

import foureng as fe

FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.8)
P = fe.BNSParams(v0=0.03, lam=0.6, a=0.8, b=18.0, rho=-2.0)


def test_no_jumps_is_black_scholes_with_decaying_variance():
    p0 = fe.BNSParams(v0=0.05, lam=1.5, a=0.0, b=10.0, rho=-1.0)
    total_var = 0.05 * (1 - np.exp(-1.5 * FWD.T)) / 1.5
    u = np.linspace(0.0, 60.0, 31)
    np.testing.assert_allclose(
        fe.bns_cf(u, FWD, p0),
        fe.bsm_cf(u, FWD, fe.BsmParams(np.sqrt(total_var / FWD.T))),
        atol=1e-14,
    )


def test_martingale_and_first_cumulant():
    assert fe.bns_cf(np.array([-1j]), FWD, P)[0] == pytest.approx(1.0, abs=1e-14)
    T, lam, a, b, rho, v0 = FWD.T, P.lam, P.a, P.b, P.rho, P.v0
    e_iv = v0 * (1 - np.exp(-lam * T)) / lam + (a / b) * (T - (1 - np.exp(-lam * T)) / lam)
    c1 = -a * lam * rho * T / (b - rho) - 0.5 * e_iv + rho * lam * T * a / b
    assert fe.bns_cumulants(FWD, P)[0] == pytest.approx(c1, abs=1e-10)


def test_prices_match_exact_conditional_simulation():
    """Jumps of z arrive at rate lam*a in real time; given them the integrated
    variance and the jump sum are exact and log S_T is Gaussian."""
    rng = np.random.default_rng(3)
    n = 300_000
    T, lam, a, b, rho, v0 = FWD.T, P.lam, P.a, P.b, P.rho, P.v0
    counts = rng.poisson(lam * a * T, n)
    iv = np.full(n, v0 * (1 - np.exp(-lam * T)) / lam)
    jump_sum = np.zeros(n)
    for i in range(int(counts.max())):
        tau, jump, hit = rng.uniform(0, T, n), rng.exponential(1 / b, n), counts > i
        iv += np.where(hit, jump * (1 - np.exp(-lam * (T - tau))) / lam, 0.0)
        jump_sum += np.where(hit, jump, 0.0)
    mean = -a * lam * rho * T / (b - rho) - 0.5 * iv + rho * jump_sum
    f_cond, sd = FWD.F0 * np.exp(mean + 0.5 * iv), np.sqrt(iv)
    K = np.array([85.0, 100.0, 115.0])
    d1 = (np.log(f_cond[None, :] / K[:, None]) + 0.5 * sd * sd) / sd
    pay = FWD.disc * (f_cond * ndtr(d1) - K[:, None] * ndtr(d1 - sd))
    cv = f_cond - FWD.F0
    est = pay - np.array([np.cov(r, cv)[0, 1] / cv.var() for r in pay])[:, None] * cv
    se = est.std(axis=1) / np.sqrt(n)
    got = fe.price_strip("bns", "contour", K, FWD, P)
    assert np.all(np.abs(got - est.mean(axis=1)) < 4.0 * se)
    np.testing.assert_allclose(fe.price_strip("bns", "cos_improved", K, FWD, P), got, atol=1e-9)


def test_leverage_steepens_the_downside():
    fwd = fe.ForwardSpec(S0=100.0, r=0.0, q=0.0, T=0.25)
    K = np.array([85.0])
    flat = fe.price_strip("bns", "contour", K, fwd, fe.BNSParams(0.03, 0.6, 0.8, 18.0, 0.0), cp=-1)
    skew = fe.price_strip("bns", "contour", K, fwd, fe.BNSParams(0.03, 0.6, 0.8, 18.0, -3.0), cp=-1)
    assert skew[0] > flat[0]


def test_validation():
    with pytest.raises(ValueError, match="rho"):
        fe.BNSParams(v0=0.03, lam=0.6, a=0.8, b=2.0, rho=3.0)
    with pytest.raises(ValueError, match="lam"):
        fe.BNSParams(v0=0.03, lam=0.0, a=0.8, b=2.0, rho=-1.0)
