"""Optimal-contour Fourier pricer (Lord & Kahl 2007) with exp-sinh quadrature."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

import foureng as fe
from foureng.models.registry import MODEL_REGISTRY
from foureng.pricers.contour import contour_price_at_strikes

FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.5)
MODELS = {
    "heston": fe.HestonParams(kappa=2.0, theta=0.04, nu=0.6, rho=-0.7, v0=0.04),
    "vg": fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14),
    "cgmy": fe.CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.8),
    "kou": fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0),
    "nig": fe.NigParams(sigma=0.2, nu=0.3, theta=-0.1),
    "sv42": fe.Sv42Params(v0=0.04, kappa=1.8, theta=0.04, nu=0.2, rho=-0.7, a=0.8, b=0.04),
}


@pytest.mark.parametrize("T,sigma", [(0.01, 0.3), (0.25, 0.2), (5.0, 0.4)])
@pytest.mark.parametrize("cp", [1, -1])
def test_bsm_full_relative_precision_to_six_standard_deviations(T, sigma, cp):
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=T)
    K = fwd.F0 * np.exp(np.linspace(-6.0, 6.0, 25) * sigma * np.sqrt(T))
    got = fe.price_strip("bsm", "contour", K, fwd, fe.BsmParams(sigma), cp=cp)
    ref = fe.black_price(fwd.F0, K, T, sigma, disc=fwd.disc, cp=cp)
    np.testing.assert_allclose(got, ref, rtol=5e-14, atol=0)


def test_bsm_extreme_wings_keep_relative_precision():
    """Prices of order 1e-34 and 1e-39: the optimal contour sits near c ~ 140."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.25)
    K = np.array([30.0, 400.0])
    for cp in (1, -1):
        got = fe.price_strip("bsm", "contour", K, fwd, fe.BsmParams(0.2), cp=cp)
        ref = fe.black_price(fwd.F0, K, fwd.T, 0.2, disc=fwd.disc, cp=cp)
        np.testing.assert_allclose(got, ref, rtol=1e-13, atol=0)


def _vg_gamma_mixture_price(K, cp, fwd, sigma, nu, theta):
    """Independent VG reference: Black price integrated against the Gamma clock."""
    T = fwd.T
    omega = np.log(1.0 - theta * nu - 0.5 * sigma * sigma * nu) / nu

    def f(g):
        Fg = fwd.F0 * np.exp(omega * T + theta * g + 0.5 * sigma * sigma * g)
        vol = sigma * np.sqrt(g)
        return fe.black_price(Fg, K, 1.0, vol, disc=fwd.disc, cp=cp) * stats.gamma.pdf(
            g, T / nu, scale=nu
        )

    return integrate.quad(f, 0.0, np.inf, epsabs=0.0, epsrel=1e-13, limit=500)[0]


@pytest.mark.derived_reference
@pytest.mark.parametrize("nu", [0.2, 0.25])  # 0.25 gives T/nu = 2: MGF finite garbage past its pole
def test_vg_deep_wings_match_gamma_mixture_reference(nu):
    sigma, theta = 0.12, -0.14
    K = np.array([20.0, 50.0, 100.0, 200.0, 400.0])
    params = fe.VGParams(sigma=sigma, nu=nu, theta=theta)
    for cp in (1, -1):
        got = fe.price_strip("vg", "contour", K, FWD, params, cp=cp)
        ref = [_vg_gamma_mixture_price(k, cp, FWD, sigma, nu, theta) for k in K]
        np.testing.assert_allclose(got, ref, rtol=5e-13, atol=0)


@pytest.mark.parametrize("model", list(MODELS))
def test_price_is_independent_of_the_contour(model):
    """Residue bookkeeping: four contour heights in three regions give one price."""
    params = MODELS[model]
    K = np.array([85.0, 120.0])
    prices = np.array(
        [
            fe.price_strip(model, "contour", K, FWD, params, grid=fe.ContourGrid(c=c))
            for c in (-1.5, 0.5, 1.8, 3.0)
        ]
    )
    assert np.ptp(prices, axis=0).max() < 1e-12


@pytest.mark.parametrize("model", ["heston", "nig", "cgmy"])
def test_agrees_with_refined_cos_and_hilbert(model):
    params = MODELS[model]
    K = np.linspace(70.0, 140.0, 15)
    got = fe.price_strip(model, "contour", K, FWD, params)
    cums = MODEL_REGISTRY[model].cumulants(FWD, params)
    cos = fe.price_strip(model, "cos", K, FWD, params, grid=fe.cos_auto_grid(cums, N=4096, L=16))
    hil = fe.price_strip(model, "hilbert", K, FWD, params, grid=fe.HilbertGrid(h=0.01, N=1 << 16))
    np.testing.assert_allclose(got, cos, atol=5e-13)
    np.testing.assert_allclose(got, hil, atol=5e-13)


def test_heston_moment_explosion_is_detected():
    """rho > 0 and large vol-of-vol: M(c) explodes at moderate c."""
    hp = fe.HestonParams(kappa=0.5, theta=0.09, nu=1.2, rho=0.4, v0=0.09)
    K = np.array([40.0, 70.0, 100.0, 150.0])
    got = fe.price_strip("heston", "contour", K, FWD, hp)
    ref = fe.price_strip("heston", "hilbert", K, FWD, hp, grid=fe.HilbertGrid(h=0.005, N=1 << 18))
    np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_put_call_parity_and_direct_function():
    params = MODELS["kou"]
    K = np.linspace(60.0, 160.0, 11)
    calls = fe.price_strip("kou", "contour", K, FWD, params, cp=1)
    puts = fe.price_strip("kou", "contour", K, FWD, params, cp=-1)
    np.testing.assert_allclose(calls - puts, FWD.disc * (FWD.F0 - K), atol=1e-12)
    phi = lambda u: MODEL_REGISTRY["kou"].cf(u, FWD, params)
    np.testing.assert_array_equal(contour_price_at_strikes(phi, FWD, K), calls)


def test_invalid_inputs():
    phi = lambda u: fe.bsm_cf(u, FWD, fe.BsmParams(0.2))
    with pytest.raises(ValueError, match="poles"):
        contour_price_at_strikes(phi, FWD, [100.0], grid=fe.ContourGrid(c=1.0))
    with pytest.raises(ValueError, match="cp"):
        contour_price_at_strikes(phi, FWD, [100.0], cp=0)
    with pytest.raises(ValueError, match="strikes"):
        contour_price_at_strikes(phi, FWD, [-1.0])
