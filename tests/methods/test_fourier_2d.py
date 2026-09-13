"""Two-asset Fourier pricer (Hurd & Zhou 2010) and the joint-CF models.

References used here are independent of the Fourier code:
- BSM: condition on the second Brownian motion, so asset 1 is lognormal and the
  price is a 1-D integral of Black-Scholes prices (adaptive quadrature).
- VG on a common gamma clock: condition on the clock as well (nested quadrature).
- Three-factor SV: Monte Carlo that is exact given the variance path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import integrate, stats

import foureng as fe
from foureng.analytics.bsm_exotics import kirk_spread, margrabe_exchange
from foureng.core.capabilities import explain_capability
from foureng.models.base import ForwardSpec
from foureng.models.heston import heston_cf
from foureng.models.joint import JOINT_MODELS, Bsm2dParams, Heston2dParams, Vg2dParams, joint_cf
from foureng.pricers.fourier_2d import (
    Fourier2DGrid,
    fourier_exchange_price,
    fourier_rainbow_price,
    fourier_spread_price,
)
from foureng.products import BestOfOption, ExchangeOption, RainbowOption, SpreadOption

S1, S2, R, Q1, Q2, T = 100.0, 96.0, 0.1, 0.05, 0.05, 1.0
FWD = ForwardSpec(S1, R, Q1, T)
BSM2 = Bsm2dParams(0.2, 0.1, 0.5)
VG2 = Vg2dParams(0.2, 0.1, -0.14, -0.05, 0.3, 0.5)
H2 = Heston2dParams(
    v0=0.04, kappa=1.5, theta=0.05, nu=0.4, sigma1=1.0, sigma2=0.8, rho=0.5, rho1=-0.6, rho2=-0.3
)


def _bs_call(F, K, sig, disc):
    d1 = (np.log(F / K) + 0.5 * sig**2) / sig
    return disc * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d1 - sig))


def _conditional_bsm(fwd, spot2, q2, s1, s2, rho, payoff):
    """Integrand over the asset-2 Brownian motion of a price that is closed form given it."""
    T_ = fwd.T
    F1, F2 = fwd.F0, spot2 * np.exp((fwd.r - q2) * T_)
    sc = s1 * np.sqrt((1.0 - rho**2) * T_)

    def f(z):
        s2T = F2 * np.exp(-0.5 * s2**2 * T_ + s2 * np.sqrt(T_) * z)
        F1c = F1 * np.exp(-0.5 * s1**2 * T_ + rho * s1 * np.sqrt(T_) * z + 0.5 * sc**2)
        return payoff(F1c, s2T, sc) * stats.norm.pdf(z)

    return f


def _ref_bsm(kind, K, fwd=FWD, spot2=S2, q2=Q2, s1=0.2, s2=0.1, rho=0.5):
    disc = fwd.disc
    if kind == "spread":

        def pay(F1c, s2T, sc):
            return _bs_call(F1c, s2T + K, sc, disc)

    else:

        def pay(F1c, s2T, sc):
            if s2T <= K:
                return 0.0
            return _bs_call(F1c, K, sc, disc) - _bs_call(F1c, s2T, sc, disc)

    f = _conditional_bsm(fwd, spot2, q2, s1, s2, rho, pay)
    F2 = spot2 * np.exp((fwd.r - q2) * fwd.T)
    zK = (np.log(K / F2) + 0.5 * s2**2 * fwd.T) / (s2 * np.sqrt(fwd.T))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return integrate.quad(
            f,
            -12,
            12,
            points=[zK] if kind == "min" else None,
            epsabs=1e-14,
            epsrel=1e-13,
            limit=500,
        )[0]


# ---------------------------------------------------------------- joint CFs


@pytest.mark.parametrize("model,params", [("bsm2d", BSM2), ("vg2d", VG2), ("heston2d", H2)])
def test_joint_cf_normalisation_and_martingales(model, params):
    one = joint_cf(model, np.array([0.0]), np.array([0.0]), T, params)
    m1 = joint_cf(model, np.array([-1j]), np.array([0.0]), T, params)
    m2 = joint_cf(model, np.array([0.0]), np.array([-1j]), T, params)
    np.testing.assert_allclose([one[0], m1[0], m2[0]], [1.0, 1.0, 1.0], atol=1e-14)
    u1, u2 = np.array([0.7 + 0.2j, -3.0 - 0.4j]), np.array([-1.1 + 0.1j, 2.5 - 0.3j])
    lhs = np.conj(joint_cf(model, u1, u2, T, params))
    rhs = joint_cf(model, -np.conj(u1), -np.conj(u2), T, params)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-13)


def test_joint_marginals_are_registry_models():
    u = np.linspace(-40.0, 40.0, 81)
    zero = 0.0 * u
    np.testing.assert_allclose(
        joint_cf("vg2d", u, zero, T, VG2),
        fe.vg_cf(u, FWD, fe.VGParams(0.2, 0.3, -0.14)),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        joint_cf("vg2d", zero, u, T, VG2),
        fe.vg_cf(u, FWD, fe.VGParams(0.1, 0.3, -0.05)),
        atol=1e-14,
    )
    heston = fe.HestonParams(kappa=1.5, theta=0.05, nu=0.4, rho=-0.6, v0=0.04)
    np.testing.assert_allclose(
        joint_cf("heston2d", u, zero, T, H2), heston_cf(u, FWD, heston), atol=1e-14
    )
    np.testing.assert_allclose(
        joint_cf("bsm2d", u, zero, T, BSM2), fe.bsm_cf(u, FWD, fe.BsmParams(sigma=0.2)), atol=1e-14
    )


def test_joint_params_validation():
    with pytest.raises(ValueError):
        Bsm2dParams(0.2, 0.1, 1.5)
    with pytest.raises(ValueError):
        Vg2dParams(0.2, 0.1, 5.0, 0.0, 0.3, 0.0)  # no finite mean
    with pytest.raises(ValueError):
        Heston2dParams(0.04, 1.0, 0.04, 0.3, 1.0, 1.0, rho=0.9, rho1=0.9, rho2=-0.9)
    with pytest.raises(TypeError):
        joint_cf("vg2d", np.array([0.0]), np.array([0.0]), T, BSM2)
    with pytest.raises(ValueError):
        joint_cf("heston3d", np.array([0.0]), np.array([0.0]), T, BSM2)
    assert set(JOINT_MODELS) == {"bsm2d", "vg2d", "heston2d"}


# ---------------------------------------------------------------- BSM references


@pytest.mark.parametrize("K", [0.5, 4.0, 10.0, 30.0])
def test_bsm_spread_matches_conditional_quadrature(K):
    v = fourier_spread_price("bsm2d", FWD, BSM2, spot2=S2, q2=Q2, strike=K)
    assert v == pytest.approx(_ref_bsm("spread", K), abs=1e-10)


def test_bsm_exchange_is_margrabe_and_spread_limit():
    exact = margrabe_exchange(S1, S2, Q1, Q2, T, 0.2, 0.1, 0.5)
    assert fourier_exchange_price("bsm2d", FWD, BSM2, spot2=S2, q2=Q2) == pytest.approx(
        exact, abs=1e-10
    )
    assert fourier_spread_price("bsm2d", FWD, BSM2, spot2=S2, q2=Q2, strike=0.0) == pytest.approx(
        exact, abs=1e-10
    )
    tiny = fourier_spread_price("bsm2d", FWD, BSM2, spot2=S2, q2=Q2, strike=1e-6)
    assert tiny == pytest.approx(exact, abs=2e-6)


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_bsm_rainbow_matches_quadrature_and_replication(K):
    call_min = fourier_rainbow_price("bsm2d", FWD, BSM2, spot2=S2, q2=Q2, strike=K, kind="min")
    assert call_min == pytest.approx(_ref_bsm("min", K), abs=1e-10)
    call_max = fourier_rainbow_price("bsm2d", FWD, BSM2, spot2=S2, q2=Q2, strike=K, kind="max")
    F2 = S2 * np.exp((R - Q2) * T)
    c1 = _bs_call(FWD.F0, K, 0.2, FWD.disc)
    c2 = _bs_call(F2, K, 0.1, FWD.disc)
    assert call_max + call_min == pytest.approx(c1 + c2, abs=1e-10)
    # parity: put - call = disc (K - E[extreme])
    exch = margrabe_exchange(S1, S2, Q1, Q2, T, 0.2, 0.1, 0.5)
    mean_min = FWD.disc * FWD.F0 - exch
    put_min = fourier_rainbow_price(
        "bsm2d", FWD, BSM2, spot2=S2, q2=Q2, strike=K, kind="min", cp=-1
    )
    assert put_min - call_min == pytest.approx(FWD.disc * K - mean_min, abs=1e-10)
    put_max = fourier_rainbow_price(
        "bsm2d", FWD, BSM2, spot2=S2, q2=Q2, strike=K, kind="max", cp=-1
    )
    mean_max = FWD.disc * (FWD.F0 + F2) - mean_min
    assert put_max - call_max == pytest.approx(FWD.disc * K - mean_max, abs=1e-10)


@pytest.mark.parametrize(
    "T_,s1,s2,rho",
    [(0.05, 0.1, 0.15, 0.9), (0.5, 0.1, 0.15, 0.9), (5.0, 0.4, 0.3, -0.5), (2.0, 0.2, 0.2, 0.0)],
)
def test_bsm_stress_short_long_and_correlated(T_, s1, s2, rho):
    fwd = ForwardSpec(100.0, 0.03, 0.01, T_)
    p = Bsm2dParams(s1, s2, rho)
    for K in (10.0, 40.0):
        v = fourier_spread_price("bsm2d", fwd, p, spot2=90.0, q2=0.0, strike=K)
        ref = _ref_bsm("spread", K, fwd=fwd, spot2=90.0, q2=0.0, s1=s1, s2=s2, rho=rho)
        assert v == pytest.approx(ref, abs=1e-9)
    Km = 90.0
    v = fourier_rainbow_price("bsm2d", fwd, p, spot2=90.0, q2=0.0, strike=Km, kind="min")
    ref = _ref_bsm("min", Km, fwd=fwd, spot2=90.0, q2=0.0, s1=s1, s2=s2, rho=rho)
    assert v == pytest.approx(ref, abs=1e-9)


def test_node_cap_warns_when_accuracy_is_lost():
    fwd = ForwardSpec(100.0, 0.03, 0.01, 0.05)
    p = Bsm2dParams(0.1, 0.15, 0.9)
    with pytest.warns(RuntimeWarning, match="n_max"):
        fourier_spread_price(
            "bsm2d", fwd, p, spot2=90.0, q2=0.0, strike=1.0, grid=Fourier2DGrid(n_max=256)
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fourier_spread_price("bsm2d", FWD, BSM2, spot2=S2, q2=Q2, strike=4.0)


# ---------------------------------------------------------------- VG and SV


def _vg_mixture_reference(kind, K):
    """Condition on the gamma clock: the pair is then a correlated Gaussian."""
    p, nu = VG2, VG2.nu
    om = [
        np.log(1 - nu * th - 0.5 * nu * s**2) / nu
        for th, s in ((p.theta1, p.sigma1), (p.theta2, p.sigma2))
    ]
    F1, F2 = FWD.F0, S2 * np.exp((R - Q2) * T)
    disc = FWD.disc
    sc_unit = p.sigma1 * np.sqrt(1 - p.rho**2)

    def given_clock(g):
        m1 = np.log(F1) + om[0] * T + p.theta1 * g
        m2 = np.log(F2) + om[1] * T + p.theta2 * g
        sc = sc_unit * np.sqrt(g)

        def f(z):
            s2T = np.exp(m2 + p.sigma2 * np.sqrt(g) * z)
            F1c = np.exp(m1 + p.rho * p.sigma1 * np.sqrt(g) * z + 0.5 * sc**2)
            if kind == "spread":
                val = _bs_call(F1c, s2T + K, sc, disc)
            else:
                val = (
                    (_bs_call(F1c, K, sc, disc) - _bs_call(F1c, s2T, sc, disc)) if s2T > K else 0.0
                )
            return val * stats.norm.pdf(z)

        pts = [(np.log(K) - m2) / (p.sigma2 * np.sqrt(g))] if kind == "min" else None
        return integrate.quad(f, -10, 10, points=pts, epsabs=1e-13, epsrel=1e-12, limit=200)[0]

    dens = stats.gamma(a=T / nu, scale=nu).pdf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return integrate.quad(
            lambda g: given_clock(g) * dens(g), 0, 12 * T, epsabs=1e-12, epsrel=1e-11, limit=200
        )[0]


@pytest.mark.slow
@pytest.mark.parametrize("kind,K", [("spread", 4.0), ("min", 100.0)])
def test_vg2d_matches_gamma_mixture(kind, K):
    if kind == "spread":
        v = fourier_spread_price("vg2d", FWD, VG2, spot2=S2, q2=Q2, strike=K)
    else:
        v = fourier_rainbow_price("vg2d", FWD, VG2, spot2=S2, q2=Q2, strike=K, kind="min")
    assert v == pytest.approx(_vg_mixture_reference(kind, K), abs=1e-9)


def test_vg2d_exchange_matches_one_dimensional_mixture():
    # given the clock the exchange option is Margrabe, so one quadrature suffices
    p, nu = VG2, VG2.nu
    om1 = np.log(1 - nu * p.theta1 - 0.5 * nu * p.sigma1**2) / nu
    om2 = np.log(1 - nu * p.theta2 - 0.5 * nu * p.sigma2**2) / nu
    F1, F2 = FWD.F0, S2 * np.exp((R - Q2) * T)
    vol = np.sqrt(p.sigma1**2 + p.sigma2**2 - 2 * p.rho * p.sigma1 * p.sigma2)

    def given_clock(g):
        a1 = F1 * np.exp(om1 * T + p.theta1 * g + 0.5 * p.sigma1**2 * g)
        a2 = F2 * np.exp(om2 * T + p.theta2 * g + 0.5 * p.sigma2**2 * g)
        return _bs_call(a1, a2, vol * np.sqrt(g), FWD.disc)  # Margrabe given the clock

    dens = stats.gamma(a=T / nu, scale=nu).pdf
    ref = integrate.quad(lambda g: given_clock(g) * dens(g), 0, 12 * T, epsabs=1e-13, limit=200)[0]
    v = fourier_exchange_price("vg2d", FWD, VG2, spot2=S2, q2=Q2)
    assert v == pytest.approx(ref, abs=1e-10)


def _heston2d_terminal(fwd, spot2, q2, p, n_paths, n_steps, seed):
    """Exact given the variance path: int sqrt(v) dW_v follows from the CIR identity."""
    rng = np.random.default_rng(seed)
    dt = fwd.T / n_steps
    v = np.full(n_paths, p.v0)
    V = np.zeros(n_paths)
    for _ in range(n_steps):
        vp = np.maximum(v, 0.0)
        vn = (
            v
            + p.kappa * (p.theta - vp) * dt
            + p.nu * np.sqrt(vp * dt) * rng.standard_normal(n_paths)
        )
        V += 0.5 * (vp + np.maximum(vn, 0.0)) * dt
        v = vn
    stoch_int = (np.maximum(v, 0) - p.v0 - p.kappa * p.theta * fwd.T + p.kappa * V) / p.nu
    r12 = (p.rho - p.rho1 * p.rho2) / np.sqrt((1 - p.rho1**2) * (1 - p.rho2**2))
    z1 = rng.standard_normal(n_paths)
    z2 = r12 * z1 + np.sqrt(1 - r12**2) * rng.standard_normal(n_paths)
    X1 = -0.5 * p.sigma1**2 * V + p.sigma1 * (
        p.rho1 * stoch_int + np.sqrt((1 - p.rho1**2) * V) * z1
    )
    X2 = -0.5 * p.sigma2**2 * V + p.sigma2 * (
        p.rho2 * stoch_int + np.sqrt((1 - p.rho2**2) * V) * z2
    )
    return fwd.F0 * np.exp(X1), spot2 * np.exp((fwd.r - q2) * fwd.T) * np.exp(X2)


def test_heston2d_matches_conditional_monte_carlo():
    fwd = ForwardSpec(100.0, 0.05, 0.0, 1.0)
    a, b = _heston2d_terminal(fwd, 95.0, 0.0, H2, 200_000, 100, seed=7)
    d = fwd.disc
    checks = [
        (
            fourier_spread_price("heston2d", fwd, H2, spot2=95.0, strike=5.0),
            np.maximum(a - b - 5.0, 0),
        ),
        (fourier_exchange_price("heston2d", fwd, H2, spot2=95.0), np.maximum(a - b, 0)),
        (
            fourier_rainbow_price("heston2d", fwd, H2, spot2=95.0, strike=95.0, kind="min"),
            np.maximum(np.minimum(a, b) - 95.0, 0),
        ),
        (
            fourier_rainbow_price("heston2d", fwd, H2, spot2=95.0, strike=100.0, kind="max", cp=-1),
            np.maximum(100.0 - np.maximum(a, b), 0),
        ),
    ]
    for price, pay in checks:
        se = d * pay.std() / np.sqrt(pay.size)
        assert abs(price - d * pay.mean()) < 4.0 * se


def test_heston2d_vanilla_leg_is_registry_heston():
    # C_max + C_min = C_1 + C_2, and with sigma1 = 1 leg 1 is the registry Heston
    fwd = ForwardSpec(100.0, 0.05, 0.0, 1.0)
    K = 100.0
    cmax = fourier_rainbow_price("heston2d", fwd, H2, spot2=95.0, strike=K, kind="max")
    cmin = fourier_rainbow_price("heston2d", fwd, H2, spot2=95.0, strike=K, kind="min")
    h1 = fe.HestonParams(kappa=1.5, theta=0.05, nu=0.4, rho=-0.6, v0=0.04)
    c1 = fe.price_strip("heston", "contour", np.array([K]), fwd, h1)[0]
    # leg 2: sigma2 = 0.8 scales the variance, i.e. Heston with v0, theta and nu rescaled
    s = H2.sigma2**2
    h2 = fe.HestonParams(kappa=1.5, theta=0.05 * s, nu=0.4 * H2.sigma2, rho=-0.3, v0=0.04 * s)
    fwd2 = ForwardSpec(95.0, 0.05, 0.0, 1.0)
    c2 = fe.price_strip("heston", "contour", np.array([K]), fwd2, h2)[0]
    assert cmax + cmin == pytest.approx(c1 + c2, abs=1e-9)


# ---------------------------------------------------------------- dispatcher


def test_price_routes_products_through_fourier_2d():
    fwd = ForwardSpec(S1, R, Q1, 2.0)  # the product maturity overrides fwd.T
    bsm = fe.BsmParams(sigma=0.2)
    spread = SpreadOption(strike=4.0, maturity=1.0, spot2=S2, q2=Q2, sigma2=0.1, rho=0.5)
    v = fe.price(spread, "bsm", "fourier_2d", fwd, bsm)
    assert v == pytest.approx(_ref_bsm("spread", 4.0), abs=1e-10)
    assert v == pytest.approx(kirk_spread(S1, S2, 4.0, R, Q1, Q2, T, 0.2, 0.1, 0.5), abs=1e-4)
    put = SpreadOption(strike=4.0, maturity=1.0, cp=-1, spot2=S2, q2=Q2, sigma2=0.1, rho=0.5)
    F2 = S2 * np.exp((R - Q2) * T)
    assert fe.price(put, "bsm", "fourier_2d", fwd, bsm) - v == pytest.approx(
        -FWD.disc * (FWD.F0 - F2 - 4.0), abs=1e-10
    )
    exch = ExchangeOption(maturity=1.0, spot2=S2, q2=Q2, sigma2=0.1, rho=0.5)
    assert fe.price(exch, "bsm", "fourier_2d", fwd, bsm) == pytest.approx(
        margrabe_exchange(S1, S2, Q1, Q2, T, 0.2, 0.1, 0.5), abs=1e-10
    )
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    best = BestOfOption(
        strike=100.0,
        maturity=1.0,
        other_spots=np.array([S2]),
        other_dividend_yields=np.array([Q2]),
        other_volatilities=np.array([0.1]),
        corr_matrix=corr,
    )
    rain = RainbowOption(
        strike=100.0, maturity=1.0, kind="max", spot2=S2, q2=Q2, sigma2=0.1, rho=0.5
    )
    assert fe.price(best, "bsm", "fourier_2d", fwd, bsm) == pytest.approx(
        fe.price(rain, "bsm", "fourier_2d", fwd, bsm), abs=1e-12
    )
    # a two-asset model ignores the product's sigma2 and rho
    rain_vg = fe.price(rain, "vg2d", "fourier_2d", fwd, VG2)
    assert rain_vg == pytest.approx(
        fourier_rainbow_price("vg2d", FWD, VG2, spot2=S2, q2=Q2, strike=100.0), abs=1e-12
    )


def test_rainbow_monte_carlo_route_agrees():
    fwd = ForwardSpec(S1, R, Q1, T)
    bsm = fe.BsmParams(sigma=0.2)
    rain = RainbowOption(
        strike=100.0, maturity=1.0, kind="min", cp=-1, spot2=S2, q2=Q2, sigma2=0.1, rho=0.5
    )
    mc = fe.price(rain, "bsm", "multi_asset_mc", fwd, bsm)
    exact = fe.price(rain, "bsm", "fourier_2d", fwd, bsm)
    assert mc == pytest.approx(exact, abs=0.05)


def test_fourier_2d_rejects_unsupported_inputs():
    bsm = fe.BsmParams(sigma=0.2)
    with pytest.raises(NotImplementedError, match="two-asset"):
        fe.price(
            fe.products.EuropeanOption(strike=100.0, maturity=1.0), "bsm", "fourier_2d", FWD, bsm
        )
    three = BestOfOption(
        strike=100.0,
        maturity=1.0,
        n_assets=3,
        other_spots=np.array([90.0, 95.0]),
        other_dividend_yields=np.array([0.0, 0.0]),
        other_volatilities=np.array([0.2, 0.2]),
        corr_matrix=np.eye(3),
    )
    with pytest.raises(NotImplementedError, match="two assets"):
        fe.price(three, "bsm", "fourier_2d", FWD, bsm)
    spread = SpreadOption(strike=4.0, maturity=1.0, spot2=S2)
    with pytest.raises(NotImplementedError, match="two-asset model"):
        fe.price(spread, "heston", "fourier_2d", FWD, fe.HestonParams(1.0, 0.04, 0.3, -0.5, 0.04))
    with pytest.raises(TypeError):
        fe.price(spread, "bsm", "fourier_2d", FWD, bsm, grid=64)
    with pytest.raises(ValueError, match="eps"):
        fourier_spread_price(
            "bsm2d", FWD, BSM2, spot2=S2, strike=4.0, grid=Fourier2DGrid(eps=(-1.0, 1.0))
        )
    heavy = Vg2dParams(0.4, 0.4, 0.0, 0.0, 2.0, 0.0)  # moments of order ~2.3 only
    with pytest.raises(ValueError, match="not finite"):
        fourier_spread_price("vg2d", FWD, heavy, spot2=S2, strike=4.0)
    with pytest.raises(ValueError, match="kind"):
        RainbowOption(strike=100.0, maturity=1.0, kind="median")
    assert "Supported" in explain_capability("vg2d", "spread", "fourier_2d")
    assert "Not supported" in explain_capability("vg2d", "spread", "cos")
