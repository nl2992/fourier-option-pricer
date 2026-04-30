"""Additional small edge-case tests for project robustness."""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams
from foureng.iv.implied_vol import BSInputs, implied_vol_newton_safeguarded
from foureng.mc.black_scholes_mc import MCSpec, european_call_mc
from foureng.pipeline import price_strip
from foureng.models.bates import BatesParams, bates_cf, bates_cumulants
from foureng.models.heston import HestonParams, heston_cf, heston_cumulants
from foureng.models.heston_cgmy import HestonCGMYParams, heston_cgmy_cf
from foureng.models.heston_kou import HestonKouParams, heston_kou_cf, heston_kou_cumulants
from foureng.utils.grids import FFTGrid, FRFTGrid


def _heston_base_case():
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    params = HestonParams(kappa=2.5, theta=0.05, nu=0.35, rho=-0.45, v0=0.04)
    strikes = np.array([85.0, 95.0, 100.0, 105.0, 115.0])
    return fwd, params, strikes


def _bates_no_jump_params(heston: HestonParams):
    return BatesParams(
        kappa=heston.kappa,
        theta=heston.theta,
        nu=heston.nu,
        rho=heston.rho,
        v0=heston.v0,
        lam_j=0.0,
        mu_j=-0.08,
        sigma_j=0.22,
    )


def _heston_kou_no_jump_params(heston: HestonParams):
    return HestonKouParams(
        kappa=heston.kappa,
        theta=heston.theta,
        nu=heston.nu,
        rho=heston.rho,
        v0=heston.v0,
        lam_j=0.0,
        p_j=0.35,
        eta1=12.0,
        eta2=8.0,
    )


def _heston_cgmy_no_jump_params(heston: HestonParams):
    return HestonCGMYParams(
        kappa=heston.kappa,
        theta=heston.theta,
        nu=heston.nu,
        rho=heston.rho,
        v0=heston.v0,
        C=0.0,
        G=5.0,
        M=9.0,
        Y=0.6,
    )


def test_carr_madan_rejects_nonpositive_strikes():
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    params = BsmParams(sigma=0.2)

    with pytest.raises(ValueError, match="strikes must be > 0"):
        price_strip(
            "bsm",
            "carr_madan",
            np.array([90.0, 0.0, 110.0]),
            fwd,
            params,
            grid=FFTGrid(N=4096, eta=0.25, alpha=1.5),
        )


def test_bsm_cos_prices_are_monotone_and_above_intrinsic_value():
    fwd = ForwardSpec(S0=100.0, r=0.02, q=0.01, T=1.0)
    params = BsmParams(sigma=0.25)
    strikes = np.linspace(70.0, 130.0, 13)

    prices = price_strip("bsm", "cos", strikes, fwd, params)
    lower_bound = np.maximum(fwd.disc * (fwd.F0 - strikes), 0.0)

    assert np.all(np.isfinite(prices))
    assert np.all(prices >= lower_bound - 1e-9)
    assert np.all(np.diff(prices) <= 1e-9)


def test_bsm_monte_carlo_is_reproducible_with_fixed_seed():
    strikes = np.array([90.0, 100.0, 110.0])
    spec = MCSpec(n_paths=25_000, seed=1234)

    first = european_call_mc(100.0, strikes, 1.0, 0.03, 0.01, 0.2, spec)
    second = european_call_mc(100.0, strikes, 1.0, 0.03, 0.01, 0.2, spec)

    assert np.array_equal(first, second)


def test_bsm_implied_vol_round_trip_across_small_surface():
    sigma = 0.28
    params = BsmParams(sigma=sigma)
    strikes = np.array([90.0, 100.0, 110.0])

    for T in [0.5, 1.0, 2.0]:
        fwd = ForwardSpec(S0=100.0, r=0.025, q=0.01, T=T)
        prices = price_strip("bsm", "cos", strikes, fwd, params)
        ivs = np.array(
            [
                implied_vol_newton_safeguarded(
                    float(price),
                    BSInputs(F0=fwd.F0, K=float(K), T=T, r=fwd.r, q=fwd.q),
                )
                for price, K in zip(prices, strikes)
            ]
        )

        assert np.max(np.abs(ivs - sigma)) < 1e-8


def test_bates_zero_jump_intensity_cf_reduces_to_heston():
    fwd, heston, _ = _heston_base_case()
    bates = _bates_no_jump_params(heston)
    u = np.array([-3.0, -1.0, 0.0, 0.75, 2.5], dtype=float)

    assert np.allclose(bates_cf(u, fwd, bates), heston_cf(u, fwd, heston), atol=1e-13, rtol=1e-13)


def test_bates_zero_jump_intensity_cumulants_reduce_to_heston():
    fwd, heston, _ = _heston_base_case()
    bates = _bates_no_jump_params(heston)

    assert np.allclose(bates_cumulants(fwd, bates), heston_cumulants(fwd, heston), atol=1e-12, rtol=1e-12)


def test_bates_zero_jump_intensity_cos_prices_match_heston():
    fwd, heston, strikes = _heston_base_case()
    bates = _bates_no_jump_params(heston)

    bates_prices = price_strip("bates", "cos", strikes, fwd, bates)
    heston_prices = price_strip("heston", "cos", strikes, fwd, heston)

    assert np.allclose(bates_prices, heston_prices, atol=1e-10, rtol=1e-10)


def test_bates_zero_jump_intensity_carr_madan_prices_match_heston():
    fwd, heston, strikes = _heston_base_case()
    bates = _bates_no_jump_params(heston)
    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)

    bates_prices = price_strip("bates", "carr_madan", strikes, fwd, bates, grid=grid)
    heston_prices = price_strip("heston", "carr_madan", strikes, fwd, heston, grid=grid)

    assert np.allclose(bates_prices, heston_prices, atol=1e-12, rtol=1e-12)


def test_bates_zero_jump_intensity_frft_prices_match_heston():
    fwd, heston, strikes = _heston_base_case()
    bates = _bates_no_jump_params(heston)
    grid = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)

    bates_prices = price_strip("bates", "frft", strikes, fwd, bates, grid=grid)
    heston_prices = price_strip("heston", "frft", strikes, fwd, heston, grid=grid)

    assert np.allclose(bates_prices, heston_prices, atol=1e-12, rtol=1e-12)


def test_heston_kou_zero_jump_intensity_cf_reduces_to_heston():
    fwd, heston, _ = _heston_base_case()
    heston_kou = _heston_kou_no_jump_params(heston)
    u = np.array([-4.0, -1.5, 0.0, 1.25, 3.0], dtype=float)

    assert np.allclose(heston_kou_cf(u, fwd, heston_kou), heston_cf(u, fwd, heston), atol=1e-13, rtol=1e-13)


def test_heston_kou_zero_jump_intensity_cumulants_reduce_to_heston():
    fwd, heston, _ = _heston_base_case()
    heston_kou = _heston_kou_no_jump_params(heston)

    assert np.allclose(heston_kou_cumulants(fwd, heston_kou), heston_cumulants(fwd, heston), atol=1e-12, rtol=1e-12)


def test_heston_kou_zero_jump_intensity_cos_prices_match_heston():
    fwd, heston, strikes = _heston_base_case()
    heston_kou = _heston_kou_no_jump_params(heston)

    heston_kou_prices = price_strip("heston_kou", "cos", strikes, fwd, heston_kou)
    heston_prices = price_strip("heston", "cos", strikes, fwd, heston)

    assert np.allclose(heston_kou_prices, heston_prices, atol=1e-10, rtol=1e-10)


def test_heston_cgmy_zero_activity_cf_reduces_to_heston():
    fwd, heston, _ = _heston_base_case()
    heston_cgmy = _heston_cgmy_no_jump_params(heston)
    u = np.array([-2.5, -0.5, 0.0, 1.0, 3.5], dtype=float)

    assert np.allclose(heston_cgmy_cf(u, fwd, heston_cgmy), heston_cf(u, fwd, heston), atol=1e-13, rtol=1e-13)
