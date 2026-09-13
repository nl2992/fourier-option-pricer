"""Registry-driven calibration with CF gradients (``foureng.calibrate``)."""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.models.base import ForwardSpec
from foureng.models.cf_gradients import (
    ANALYTIC_GRADIENT_MODELS,
    cf_and_gradient,
    param_names,
    params_to_vector,
)
from foureng.models.registry import MODEL_REGISTRY
from foureng.surface.calibration import calibrate_heston
from foureng.surface.model_calibration import MarketQuotes, _Maturity, calibrate
from foureng.surface.vol_surface import SurfaceSpec

S0, R, Q = 100.0, 0.02, 0.01
MATS = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
STRIKES = np.linspace(70.0, 130.0, 9)

PARAMS = {
    "bsm": fe.BsmParams(sigma=0.25),
    "heston": fe.HestonParams(kappa=2.0, theta=0.05, nu=0.6, rho=-0.7, v0=0.03),
    "bates": fe.BatesParams(
        kappa=2.0, theta=0.05, nu=0.6, rho=-0.7, v0=0.03, lam_j=0.5, mu_j=-0.1, sigma_j=0.15
    ),
    "vg": fe.VGParams(0.2, 0.3, -0.14),
    "nig": fe.NigParams(0.2, 0.3, -0.14),
    "cgmy": fe.CgmyParams(C=1.0, G=5.0, M=10.0, Y=0.5),
    "kou": fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0),
    "merton_jd": fe.MertonJDParams(sigma=0.15, lam=0.5, muj=-0.1, sigj=0.15),
}


def _surface(model, params, mats=MATS, strikes=STRIKES, method="contour"):
    ivs = []
    for T in mats:
        fwd = ForwardSpec(S0, R, Q, T)
        calls = fe.price_strip(model, method, strikes, fwd, params)
        ivs.append(fe.implied_vol_lets_be_rational(calls, fwd.F0, strikes, T, disc=fwd.disc))
    Ts = np.repeat(mats, len(strikes))
    Ks = np.tile(strikes, len(mats))
    return Ts, Ks, np.concatenate(ivs)


# ------------------------------------------------------------------ CF gradients


def test_analytic_gradient_models_are_the_expected_set():
    assert set(ANALYTIC_GRADIENT_MODELS) == set(PARAMS)


@pytest.mark.parametrize("model", sorted(PARAMS))
def test_analytic_cf_matches_registry_and_gradient_matches_differences(model):
    fwd = ForwardSpec(S0, R, Q, 0.7)
    u = np.linspace(-60.0, 60.0, 241)
    phi, grad = cf_and_gradient(model, u, fwd, PARAMS[model])
    np.testing.assert_allclose(phi, MODEL_REGISTRY[model].cf(u, fwd, PARAMS[model]), atol=1e-14)
    _, grad_fd = cf_and_gradient(model, u, fwd, PARAMS[model], analytic=False)
    assert grad.shape == (len(param_names(model)), len(u))
    np.testing.assert_allclose(grad, grad_fd, atol=5e-9)


def test_cf_gradient_is_zero_for_the_martingale_point():
    # phi(-i) = 1 for every parameter value, so its gradient vanishes
    fwd = ForwardSpec(S0, R, Q, 1.0)
    for model, p in PARAMS.items():
        _, grad = cf_and_gradient(model, np.array([-1j, 0.0]), fwd, p)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)


# ------------------------------------------------------------------ pricing kernel


@pytest.mark.parametrize("T", [0.02, 0.5, 5.0])
def test_kernel_matches_black_scholes(T):
    fwd = ForwardSpec(S0, 0.03, 0.01, T)
    K = np.array([50.0, 80.0, 95.0, 100.0, 105.0, 130.0, 200.0])
    is_call = K >= fwd.F0
    p = fe.BsmParams(sigma=0.25)
    mat = _Maturity("bsm", fwd, K, is_call, p, 4096)
    v = mat.price(MODEL_REGISTRY["bsm"].cf(mat.u, fwd, p))
    ref = fe.black_price(fwd.F0, K, T, 0.25, disc=fwd.disc, cp=np.where(is_call, 1, -1))
    np.testing.assert_allclose(v, ref, atol=1e-12)


def test_kernel_gradient_matches_price_differences():
    fwd = ForwardSpec(S0, R, Q, 0.5)
    K = STRIKES
    is_call = K >= fwd.F0
    p = PARAMS["heston"]
    mat = _Maturity("heston", fwd, K, is_call, p, 4096)
    _, dphi = cf_and_gradient("heston", mat.u, fwd, p)
    grad = mat.gradient(dphi)
    x = params_to_vector("heston", p)
    for j, name in enumerate(param_names("heston")):
        h = 1e-6 * max(abs(x[j]), 0.1)
        kw = {n: float(v) for n, v in zip(param_names("heston"), x)}
        kw_up, kw_dn = dict(kw), dict(kw)
        kw_up[name] += h
        kw_dn[name] -= h
        cf = MODEL_REGISTRY["heston"].cf
        up = mat.price(cf(mat.u, fwd, fe.HestonParams(**kw_up)))
        dn = mat.price(cf(mat.u, fwd, fe.HestonParams(**kw_dn)))
        np.testing.assert_allclose(grad[j], (up - dn) / (2 * h), atol=1e-6)


# ------------------------------------------------------------------ round trips


def test_heston_round_trip_from_a_poor_start_beats_nelder_mead():
    true = fe.HestonParams(kappa=2.5, theta=0.06, nu=0.7, rho=-0.65, v0=0.04)
    start = fe.HestonParams(kappa=1.0, theta=0.03, nu=0.3, rho=-0.2, v0=0.02)
    Ts, Ks, ivs = _surface("heston", true)
    fit = calibrate("heston", MarketQuotes(S0, R, Q, Ts, Ks, ivs=ivs), start)
    assert fit.success and fit.gradient == "analytic"
    np.testing.assert_allclose(
        params_to_vector("heston", fit.params), params_to_vector("heston", true), rtol=1e-6
    )
    assert fit.rmse_iv < 1e-9
    # the Nelder-Mead calibrator needs far more evaluations for a worse fit
    spec = SurfaceSpec(S0, R, Q, MATS, STRIKES)
    old = calibrate_heston(spec, ivs.reshape(len(MATS), -1), start)
    assert fit.nfev * 10 < old.nfev
    assert fit.rmse_iv < np.sqrt(np.mean(old.residuals**2))


@pytest.mark.parametrize(
    "model,true,start",
    [
        (
            "bates",
            PARAMS["bates"],
            fe.BatesParams(
                kappa=1.0, theta=0.03, nu=0.3, rho=-0.3, v0=0.02, lam_j=0.2, mu_j=0.0, sigma_j=0.2
            ),
        ),
        (
            "kou",
            fe.KouParams(sigma=0.12, lam=0.8, p=0.3, eta1=12.0, eta2=6.0),
            fe.KouParams(sigma=0.2, lam=0.3, p=0.5, eta1=20.0, eta2=10.0),
        ),
        (
            "merton_jd",
            fe.MertonJDParams(sigma=0.12, lam=0.6, muj=-0.12, sigj=0.1),
            fe.MertonJDParams(sigma=0.2, lam=0.2, muj=0.0, sigj=0.2),
        ),
        (
            "cgmy",
            fe.CgmyParams(C=0.8, G=6.0, M=12.0, Y=0.6),
            fe.CgmyParams(C=0.4, G=10.0, M=20.0, Y=0.3),
        ),
    ],
)
def test_round_trips_with_analytic_gradients(model, true, start):
    Ts, Ks, ivs = _surface(model, true)
    fit = calibrate(model, MarketQuotes(S0, R, Q, Ts, Ks, ivs=ivs), start)
    assert fit.rmse_iv < 5e-8
    np.testing.assert_allclose(
        params_to_vector(model, fit.params), params_to_vector(model, true), rtol=1e-4
    )


def test_vg_short_maturity_is_limited_by_cos_convergence_only():
    # T < nu gives VG a singular density peak, so COS converges algebraically;
    # the fit is still far inside any bid-ask spread
    true, start = fe.VGParams(0.18, 0.25, -0.15), fe.VGParams(0.3, 0.5, 0.0)
    Ts, Ks, ivs = _surface("vg", true)
    fit = calibrate("vg", MarketQuotes(S0, R, Q, Ts, Ks, ivs=ivs), start)
    assert fit.rmse_iv < 1e-6
    np.testing.assert_allclose(fit.values["sigma"], 0.18, rtol=1e-4)


def test_difference_gradient_fallback_for_other_models():
    true = fe.MeixnerParams(a=0.3, b=-0.8, delta=0.6)
    start = fe.MeixnerParams(a=0.5, b=-0.2, delta=1.0)
    mats = np.array([0.25, 0.5, 1.0, 2.0])
    Ts, Ks, ivs = _surface("meixner", true, mats=mats)
    fit = calibrate("meixner", MarketQuotes(S0, R, Q, Ts, Ks, ivs=ivs), start)
    assert fit.gradient == "fd"
    assert fit.rmse_iv < 1e-8
    np.testing.assert_allclose(
        params_to_vector("meixner", fit.params), params_to_vector("meixner", true), rtol=1e-5
    )


# ------------------------------------------------------------------ options


def test_noisy_quotes_fit_to_the_noise_level():
    true = fe.HestonParams(kappa=2.5, theta=0.06, nu=0.7, rho=-0.65, v0=0.04)
    Ts, Ks, ivs = _surface("heston", true)
    noise = 0.0005 * np.random.default_rng(3).standard_normal(ivs.shape)  # 5 bp
    fit = calibrate("heston", MarketQuotes(S0, R, Q, Ts, Ks, ivs=ivs + noise), true)
    assert fit.rmse_iv < 1.05 * np.sqrt(np.mean(noise**2))
    assert abs(fit.values["rho"] - true.rho) < 0.05


def test_fixed_parameters_prices_input_and_weights():
    true = fe.HestonParams(kappa=2.5, theta=0.06, nu=0.7, rho=-0.65, v0=0.04)
    Ts, Ks, ivs = _surface("heston", true)
    start = fe.HestonParams(kappa=1.0, theta=0.03, nu=0.3, rho=-0.65, v0=0.02)
    fit = calibrate("heston", MarketQuotes(S0, R, Q, Ts, Ks, ivs=ivs), start, fixed=["rho"])
    assert fit.values["rho"] == -0.65
    assert fit.values["kappa"] == pytest.approx(2.5, rel=1e-6)
    # prices with a mix of calls and puts give the same fit as vols
    F = S0 * np.exp((R - Q) * Ts)
    D = np.exp(-R * Ts)
    cp = np.where(np.arange(len(Ts)) % 2 == 0, 1, -1)
    prices = fe.black_price(F, Ks, Ts, ivs, disc=D, cp=cp)
    fit_p = calibrate("heston", MarketQuotes(S0, R, Q, Ts, Ks, prices=prices, cp=cp), start)
    assert fit_p.values["kappa"] == pytest.approx(2.5, rel=1e-6)
    # a corrupted quote with zero weight is ignored
    bad = ivs.copy()
    bad[7] += 0.05
    w = np.ones_like(ivs)
    w[7] = 0.0
    fit_w = calibrate("heston", MarketQuotes(S0, R, Q, Ts, Ks, ivs=bad, weights=w), start)
    assert fit_w.values["theta"] == pytest.approx(0.06, rel=1e-6)
    assert abs(fit_w.iv_residuals[7] + 0.05) < 1e-6


def test_from_surface_matches_flat_quotes():
    spec = SurfaceSpec(S0, R, Q, MATS, STRIKES)
    ivs = np.full((len(MATS), len(STRIKES)), 0.2)
    q = MarketQuotes.from_surface(spec, ivs)
    assert q.maturities.shape == (45,) and q.strikes[:9].tolist() == STRIKES.tolist()
    fit = calibrate("bsm", q, fe.BsmParams(sigma=0.3))
    assert fit.values["sigma"] == pytest.approx(0.2, abs=1e-9)


def test_calibrate_rejects_bad_inputs():
    Ts, Ks = np.array([1.0]), np.array([100.0])
    q = MarketQuotes(S0, R, Q, Ts, Ks, ivs=np.array([0.2]))
    with pytest.raises(ValueError, match="unknown model"):
        calibrate("heston3", q, PARAMS["heston"])
    with pytest.raises(ValueError, match="fixed"):
        calibrate("bsm", q, PARAMS["bsm"], fixed=["kappa"])
    with pytest.raises(ValueError, match="every parameter"):
        calibrate("bsm", q, PARAMS["bsm"], fixed=["sigma"])
    with pytest.raises(ValueError, match="gradient"):
        calibrate("bsm", q, PARAMS["bsm"], gradient="adjoint")
    with pytest.raises(ValueError, match="exactly one"):
        MarketQuotes(S0, R, Q, Ts, Ks)
    with pytest.raises(ValueError, match="cp"):
        MarketQuotes(S0, R, Q, Ts, Ks, prices=np.array([10.0]))
    with pytest.raises(ValueError, match="same length"):
        MarketQuotes(S0, R, Q, Ts, Ks, ivs=np.array([0.2, 0.3]))
    arb = MarketQuotes(S0, R, Q, Ts, Ks, prices=np.array([200.0]), cp=np.array([1]))
    with pytest.raises(ValueError, match="no implied vol"):
        calibrate("bsm", arb, PARAMS["bsm"])
