"""Tests for the vectorised "Let's Be Rational" implied-vol solver (Jäckel 2015)."""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
import foureng.iv.lets_be_rational as lbr
from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd, implied_vol_brent


def _otm_grid(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    F = np.full(n, 100.0)
    K = F * np.exp(rng.uniform(-1.5, 1.5, n))
    T = rng.uniform(0.02, 5.0, n)
    sigma = rng.uniform(0.03, 1.5, n)
    cp = np.where(K > F, 1.0, -1.0)  # OTM quotes carry the full time value
    disc = np.exp(-0.03 * T)
    return F, K, T, sigma, cp, disc


def test_roundtrip_machine_precision_on_otm_grid():
    F, K, T, sigma, cp, disc = _otm_grid(20_000)
    p = fe.black_price(F, K, T, sigma, disc=disc, cp=cp)
    keep = p > 1e-300
    iv = fe.implied_vol_lets_be_rational(p, F, K, T, disc=disc, cp=cp)
    rel = np.abs(iv[keep] / sigma[keep] - 1.0)
    assert np.all(np.isfinite(iv[keep]))
    assert rel.max() < 5e-14
    assert np.median(rel) < 1e-15


def test_two_householder_iterations_suffice(monkeypatch):
    """Jäckel's headline property: the rational guess plus two order-3 steps."""
    F, K, T, sigma, cp, disc = _otm_grid(5_000, seed=11)
    p = fe.black_price(F, K, T, sigma, disc=disc, cp=cp)
    keep = p > 1e-300
    monkeypatch.setattr(lbr, "_MAX_ITER", 2)
    iv = fe.implied_vol_lets_be_rational(p, F, K, T, disc=disc, cp=cp)
    assert np.abs(iv[keep] / sigma[keep] - 1.0).max() < 5e-14


@pytest.mark.numerical_stability
def test_reprices_high_precision_quotes_in_the_wings():
    """Invert 50-digit reference prices, including 1e-250-sized far-wing quotes.

    The attainable accuracy is set by conditioning, so the check is the backward
    error: the recovered vol must reproduce the input price (evaluated in high
    precision) to near machine precision.
    """
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50

    def nb(x, s):
        x, s = mp.mpf(x), mp.mpf(s)
        return mp.exp(x / 2) * mp.ncdf(x / s + s / 2) - mp.exp(-x / 2) * mp.ncdf(x / s - s / 2)

    worst = 0.0
    for x in (0.0, -1e-10, -1e-3, -0.2, -1.0, -3.0, -8.0, -20.0):
        for s in (1e-3, 0.01, 0.1, 0.2, 0.4, 1.0, 2.5, 6.0):
            beta = nb(x, s)
            if beta < mp.mpf("1e-280") or beta >= mp.exp(mp.mpf(x) / 2):
                continue
            s_hat = lbr._implied_total_vol_otm(np.array([float(beta)]), np.array([x]))[0]
            back = abs(nb(x, s_hat) / mp.mpf(float(beta)) - 1)
            worst = max(worst, float(back))
    assert worst < 1e-12


@pytest.mark.numerical_stability
def test_normalised_black_matches_high_precision():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    for x in (0.0, -1e-6, -0.5, -2.0, -10.0):
        for s in (1e-3, 0.05, 0.3, 0.5, 1.5, 4.0):
            ref = mp.exp(mp.mpf(x) / 2) * mp.ncdf(mp.mpf(x) / s + mp.mpf(s) / 2) - mp.exp(
                -mp.mpf(x) / 2
            ) * mp.ncdf(mp.mpf(x) / s - mp.mpf(s) / 2)
            if ref < mp.mpf("1e-290"):
                continue
            b, c = lbr._nb_otm(np.array([x]), np.array([s]))
            assert abs(mp.mpf(float(b[0])) / ref - 1) < 2e-13
            assert abs(mp.mpf(float(c[0])) / (mp.exp(mp.mpf(x) / 2) - ref) - 1) < 2e-13


def test_matches_brent_on_ordinary_quotes():
    for K in (70.0, 95.0, 100.0, 108.0, 140.0):
        for is_call in (True, False):
            inp = BSInputs(F0=101.0, K=K, T=0.75, r=0.02, q=0.0, is_call=is_call)
            price = bs_price_from_fwd(0.27, inp)
            ref = implied_vol_brent(price, inp)
            got = fe.implied_vol_lets_be_rational(
                price, 101.0, K, 0.75, disc=np.exp(-0.02 * 0.75), cp=1 if is_call else -1
            )
            assert got == pytest.approx(ref, abs=1e-10)
            assert got == pytest.approx(0.27, rel=1e-13)


def test_in_the_money_quotes_use_parity_reduction():
    K = np.array([60.0, 150.0])
    cp = np.array([1.0, -1.0])
    p = fe.black_price(100.0, K, 2.0, 0.37, disc=0.95, cp=cp)
    np.testing.assert_allclose(
        fe.implied_vol_lets_be_rational(p, 100.0, K, 2.0, disc=0.95, cp=cp), 0.37, rtol=1e-12
    )


def test_broadcasting_and_scalar_shapes():
    K = np.array([[90.0], [110.0]])
    T = np.array([0.5, 1.0, 2.0])
    p = fe.black_price(100.0, K, T, 0.2)
    iv = fe.implied_vol_lets_be_rational(p, 100.0, K, T)
    assert iv.shape == (2, 3)
    np.testing.assert_allclose(iv, 0.2, rtol=1e-13)
    scalar = fe.implied_vol_lets_be_rational(
        fe.black_price(100.0, 110.0, 1.0, 0.2), 100.0, 110.0, 1.0
    )
    assert np.ndim(scalar) == 0 and float(scalar) == pytest.approx(0.2, rel=1e-14)


def test_boundary_and_invalid_quotes():
    got = fe.implied_vol_lets_be_rational(
        price=[0.0, 20.0 - 1e-9, 100.0, 1.0, np.nan, 1.0, 1.0],
        F=100.0,
        K=[120.0, 80.0, 100.0, 100.0, 100.0, 100.0, -5.0],
        T=[1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 1.0],
    )
    assert got[0] == 0.0  # OTM call worth exactly nothing: zero vol
    assert np.isnan(got[1])  # below intrinsic
    assert np.isnan(got[2])  # at the upper bound disc * F
    assert np.all(np.isnan(got[3:]))  # T <= 0, non-finite price, bad strike
    with pytest.raises(ValueError, match="cp"):
        fe.implied_vol_lets_be_rational(1.0, 100.0, 100.0, 1.0, cp=0)


def test_black_price_agrees_with_closed_form_and_parity():
    K = np.linspace(60.0, 160.0, 11)
    calls = fe.black_price(100.0, K, 1.3, 0.31, disc=0.97, cp=1)
    puts = fe.black_price(100.0, K, 1.3, 0.31, disc=0.97, cp=-1)
    ref = [
        bs_price_from_fwd(0.31, BSInputs(F0=100.0, K=k, T=1.3, r=-np.log(0.97) / 1.3, q=0.0))
        for k in K
    ]
    np.testing.assert_allclose(calls, ref, rtol=1e-12)
    np.testing.assert_allclose(calls - puts, 0.97 * (100.0 - K), atol=1e-12)
    assert fe.black_price(100.0, 250.0, 0.1, 0.0) == 0.0


def test_model_iv_surface_uses_vectorised_solver():
    spec = fe.SurfaceSpec(
        S0=100.0,
        r=0.02,
        q=0.01,
        maturities=np.array([0.25, 1.0]),
        strikes=np.linspace(80.0, 120.0, 9),
    )
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.4, rho=-0.6, v0=0.05)
    ivs = fe.model_iv_surface(
        spec,
        lambda fwd: lambda u: fe.heston_cf_form2(u, fwd, params),
        lambda fwd: fe.heston_cumulants(fwd, params),
    )
    prices = fe.model_price_surface(
        spec,
        lambda fwd: lambda u: fe.heston_cf_form2(u, fwd, params),
        lambda fwd: fe.heston_cumulants(fwd, params),
    )
    T = spec.maturities[:, None]
    F = spec.S0 * np.exp((spec.r - spec.q) * T)
    repriced = fe.black_price(F, spec.strikes[None, :], T, ivs, disc=np.exp(-spec.r * T))
    np.testing.assert_allclose(repriced, prices, rtol=1e-12)
    assert ivs[0, 0] > ivs[0, -1]  # negative rho: downward skew
