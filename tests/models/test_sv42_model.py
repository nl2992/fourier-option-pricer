"""4/2 stochastic volatility model (Grasselli 2017): CF reductions, oracle, and MC."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

import foureng as fe

FWD = fe.ForwardSpec(S0=100.0, r=0.02, q=0.01, T=0.7)
U = np.concatenate([np.linspace(0.0, 5.0, 21), np.linspace(5.0, 300.0, 40)])


def _p(a, b, rho=-0.6, v0=0.04, kappa=1.8, theta=0.05, nu=0.25):
    return fe.Sv42Params(v0=v0, kappa=kappa, theta=theta, nu=nu, rho=rho, a=a, b=b)


def test_b_zero_is_heston_even_without_feller():
    hp = fe.HestonParams(kappa=1.1, theta=0.03, nu=0.5, rho=-0.7, v0=0.05)  # 2k*th < nu^2
    got = fe.sv42_cf(U, FWD, _p(1.0, 0.0, rho=-0.7, v0=0.05, kappa=1.1, theta=0.03, nu=0.5))
    np.testing.assert_allclose(got, fe.heston_cf_form2(U, FWD, hp), atol=1e-14)


def test_a_scales_heston_variance():
    """a sqrt(v) with CIR v is Heston for a^2 v: (v0, theta, nu) -> a^2 (v0, theta), a nu."""
    a = 1.7
    hp = fe.HestonParams(kappa=1.8, theta=a * a * 0.05, nu=a * 0.25, rho=-0.6, v0=a * a * 0.04)
    np.testing.assert_allclose(
        fe.sv42_cf(U, FWD, _p(a, 0.0)), fe.heston_cf_form2(U, FWD, hp), atol=1e-14
    )


def test_a_zero_is_the_three_halves_model():
    """V = b^2 / v is a 3/2 process: kappa' = (kappa theta - nu^2)/b^2,
    theta' = kappa b^2 / (kappa theta - nu^2), vol-of-vol nu/b, correlation -rho."""
    kappa, theta, nu, rho, v0, b = 1.8, 0.05, 0.25, -0.6, 0.04, 0.2
    sv32 = fe.Sv32Params(
        v0=b * b / v0,
        kappa=(kappa * theta - nu * nu) / b**2,
        theta=kappa * b * b / (kappa * theta - nu * nu),
        nu=nu / b,
        rho=-rho,
    )
    np.testing.assert_allclose(fe.sv42_cf(U, FWD, _p(0.0, b)), fe.sv32_cf(U, FWD, sv32), atol=1e-14)


@pytest.mark.parametrize("a,b,rho", [(1.0, 0.05, -0.6), (0.5, 0.2, -0.9), (0.8, 0.1, 0.4)])
def test_martingale_and_normalisation(a, b, rho):
    phi = fe.sv42_cf(np.array([0.0, -1j]), FWD, _p(a, b, rho=rho))
    np.testing.assert_allclose(phi, [1.0, 1.0], atol=1e-13)
    assert np.all(np.abs(fe.sv42_cf(U, FWD, _p(a, b, rho=rho))) <= 1.0 + 1e-13)


@pytest.mark.numerical_stability
@pytest.mark.parametrize("T", [0.01, 0.3, 2.0])
@pytest.mark.parametrize("a,b,rho", [(1.0, 0.05, -0.6), (0.5, 0.2, -0.9), (0.0, 0.3, 0.3)])
def test_cf_matches_high_precision_evaluation(T, a, b, rho):
    """The log-space Kummer summation against 30-digit mpmath hyp1f1 on the same formula."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    v0, kappa, theta, nu = 0.04, 1.8, 0.05, 0.25

    def cf_mp(u):
        u = mp.mpc(u)
        i = mp.mpc(0, 1)
        kth = kappa * theta - nu**2 / 2
        psi = (i * u + u * u * (1 - rho**2)) / 2
        mu = psi * a * a - i * u * rho * a * kappa / nu
        eta = psi * b * b + i * u * rho * b * kth / nu
        lam, gam = -i * u * rho * a / nu, i * u * rho * b / nu
        const = (
            -2 * a * b * psi * T
            - i * u * rho * (a / nu) * (v0 + kappa * theta * T)
            + i * u * rho * (b / nu) * (kappa * T - mp.log(v0))
        )
        kt = mp.sqrt(kappa**2 + 2 * nu**2 * mu)
        k1 = (kt - kappa) / nu**2
        D = mp.sqrt(kth**2 + 2 * eta * nu**2)
        b1 = (D - kth) / nu**2
        c = nu**2 * (1 - mp.exp(-kt * T)) / (4 * kt)
        hd = 1 + 2 * D / nu**2
        zeta = v0 * mp.exp(-kt * T) / c
        s = (lam - k1) * c
        P = gam - b1
        mom = (
            (2 * c) ** P
            * (1 + 2 * s) ** (-(hd + P))
            * mp.exp(-zeta / 2)
            * mp.gamma(hd + P)
            / mp.gamma(hd)
            * mp.hyp1f1(hd + P, hd, zeta / (2 * (1 + 2 * s)))
        )
        return complex(mp.exp(const - k1 * (v0 + kappa * theta * T) - b1 * kt * T) * v0**b1 * mom)

    us = np.array([0.0, 0.7, 2.5, 9.0, 40.0, 150.0])
    fwd = fe.ForwardSpec(S0=100.0, r=0.02, q=0.01, T=T)
    got = fe.sv42_cf(us, fwd, _p(a, b, rho=rho))
    np.testing.assert_allclose(got, [cf_mp(u) for u in us], atol=5e-13)


def _conditional_mc(p, fwd, strikes, n=100_000, steps=200, seed=3):
    """Exact CIR transitions; given the variance path log S_T is Gaussian, so price
    by the conditional Black formula, with E[F_cond] = F as a control variate."""
    rng = np.random.default_rng(seed)
    T, dt = fwd.T, fwd.T / steps
    k, th, s, rho, a, b = p.kappa, p.theta, p.nu, p.rho, p.a, p.b
    c = s * s * (1.0 - np.exp(-k * dt)) / (4.0 * k)
    d = 4.0 * k * th / (s * s)
    v = np.full(n, p.v0)
    i1 = np.zeros(n)
    i2 = np.zeros(n)
    for _ in range(steps):
        vn = c * rng.noncentral_chisquare(d, v * np.exp(-k * dt) / c)
        i1 += 0.5 * (v + vn) * dt
        i2 += 0.5 * (1.0 / v + 1.0 / vn) * dt
        v = vn
    g2 = a * a * i1 + 2.0 * a * b * T + b * b * i2
    int_g_dz = (a / s) * (v - p.v0 - k * th * T + k * i1) + (b / s) * (
        np.log(v / p.v0) + (s * s / 2.0 - k * th) * i2 + k * T
    )
    m = -0.5 * g2 + rho * int_g_dz
    var = (1.0 - rho * rho) * g2
    f_cond = fwd.F0 * np.exp(m + 0.5 * var)
    sd = np.sqrt(var)
    K = np.asarray(strikes)[:, None]
    d1 = (np.log(f_cond / K) + 0.5 * var) / sd
    pay = fwd.disc * (f_cond * ndtr(d1) - K * ndtr(d1 - sd))
    cv = f_cond - fwd.F0
    beta = np.array([np.cov(row, cv)[0, 1] / cv.var() for row in pay])
    adj = pay - beta[:, None] * cv[None, :]
    return adj.mean(axis=1), adj.std(axis=1) / np.sqrt(n)


@pytest.mark.parametrize(
    "a,b,rho", [(1.0, 0.02, -0.7), (0.6, 0.06, -0.7), (0.0, 0.04, -0.5), (0.8, 0.04, 0.3)]
)
def test_prices_match_exact_path_monte_carlo(a, b, rho):
    """Validates the derivation itself (not just its evaluation) for general (a, b)."""
    p = _p(a, b, rho=rho, theta=0.04, nu=0.2)
    fwd = fe.ForwardSpec(S0=100.0, r=0.02, q=0.01, T=0.5)
    K = np.array([85.0, 100.0, 115.0])
    mc, se = _conditional_mc(p, fwd, K)
    cos = fe.price_strip("sv42", "cos_improved", K, fwd, p)
    assert np.all(np.abs(cos - mc) < 4.0 * se + 1e-4), (cos, mc, se)


def test_fourier_engines_agree_and_cumulants_are_consistent():
    p = _p(0.9, 0.08, rho=-0.7)
    K = np.array([80.0, 95.0, 100.0, 105.0, 125.0])
    cos = fe.price_strip("sv42", "cos_improved", K, FWD, p)
    np.testing.assert_allclose(fe.price_strip("sv42", "hilbert", K, FWD, p), cos, atol=1e-8)
    np.testing.assert_allclose(fe.price_strip("sv42", "cos", K, FWD, p), cos, atol=1e-7)
    c1, c2, c4 = fe.sv42_cumulants(FWD, p)
    # Martingale X_T = log(S_T/F): E[e^X] = 1 means c1 ~ -c2/2 to leading order.
    assert c2 > 0 and abs(c1 + 0.5 * c2) < 0.25 * c2


def test_b_adds_a_volatility_floor_to_short_dated_smiles():
    """With b > 0 the instantaneous vol is >= 2 sqrt(ab): deep OTM short-dated
    options keep more value than under Heston with the same a."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.0, q=0.0, T=0.05)
    K = np.array([120.0])
    heston_like = fe.price_strip("sv42", "cos_improved", K, fwd, _p(1.0, 0.0, v0=0.002))
    four_two = fe.price_strip("sv42", "cos_improved", K, fwd, _p(1.0, 0.03, v0=0.002))
    assert four_two[0] > heston_like[0]


def test_parameter_validation():
    with pytest.raises(ValueError, match="Feller"):
        fe.Sv42Params(v0=0.04, kappa=1.0, theta=0.02, nu=0.5, rho=-0.5, a=1.0, b=0.1)
    with pytest.raises(ValueError, match="both be zero"):
        _p(0.0, 0.0)
    with pytest.raises(ValueError, match="rho"):
        _p(1.0, 0.1, rho=1.0)
    with pytest.raises(ValueError, match="b must be >= 0"):
        _p(1.0, -0.1)
    assert fe.price_strip("sv42", "cos", np.array([100.0]), FWD, _p(1.0, 0.05)).shape == (1,)
