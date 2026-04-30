"""Generalized project-level tests for MC, API workflows, and input guards.

These tests are deliberately broad rather than razor-thin: they exercise the
course rubric's themes of validation, robustness, and numerical sanity across
multiple parameter regimes while keeping seeded Monte Carlo checks reproducible.
"""
from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.greeks.cos_greeks import cos_price_and_greeks
from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd
from foureng.mc.black_scholes_mc import MCSpec, european_call_mc
from foureng.mc.heston_conditional_mc import HestonMCScheme, heston_conditional_mc_calls
from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams, bsm_cf, bsm_cumulants
from foureng.models.heston import HestonParams, heston_cf_form2, heston_cumulants
from foureng.pipeline import price_strip
from foureng.surface.vol_surface import SurfaceSpec, model_iv_surface, model_price_surface
from foureng.utils.grids import FFTGrid, FRFTGrid


def _bs_call_strip(S0, K, T, r, q, vol):
    F0 = S0 * np.exp((r - q) * T)
    return np.array(
        [
            bs_price_from_fwd(
                vol,
                BSInputs(F0=F0, K=float(k), T=T, r=r, q=q, is_call=True),
            )
            for k in np.asarray(K, dtype=float)
        ],
        dtype=float,
    )


def _assert_mc_close(actual, expected, *, rel=0.10, abs_floor=0.15):
    """Mixed MC tolerance: 10% for material prices, absolute floor near zero."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    allowed = np.maximum(abs_floor, rel * np.maximum(np.abs(expected), 1.0))
    err = np.abs(actual - expected)
    assert np.all(err <= allowed), (
        f"MC mismatch\nactual={actual}\nexpected={expected}\n"
        f"abs_err={err}\nallowed={allowed}"
    )


BS_MC_CASES = [
    dict(S0=100.0, T=0.25, r=0.01, q=0.00, vol=0.15, K=[85.0, 100.0, 115.0]),
    dict(S0=100.0, T=1.00, r=0.05, q=0.02, vol=0.20, K=[75.0, 100.0, 130.0]),
    dict(S0=80.0, T=2.00, r=0.03, q=0.01, vol=0.35, K=[60.0, 80.0, 110.0]),
    dict(S0=125.0, T=0.75, r=0.00, q=0.04, vol=0.28, K=[90.0, 125.0, 160.0]),
    dict(S0=100.0, T=1.50, r=-0.01, q=0.00, vol=0.45, K=[70.0, 100.0, 150.0]),
]


@pytest.mark.parametrize("case", BS_MC_CASES)
def test_generalized_bsm_mc_matches_black_scholes_with_mixed_tolerance(case):
    K = np.asarray(case["K"], dtype=float)
    mc = european_call_mc(
        case["S0"],
        K,
        case["T"],
        case["r"],
        case["q"],
        case["vol"],
        MCSpec(n_paths=90_000, seed=20260429),
    )
    ref = _bs_call_strip(case["S0"], K, case["T"], case["r"], case["q"], case["vol"])
    _assert_mc_close(mc, ref)


def test_bsm_mc_is_reproducible_with_fixed_seed():
    K = np.array([90.0, 100.0, 110.0])
    spec = MCSpec(n_paths=20_000, seed=12345)
    first = european_call_mc(100.0, K, 1.0, 0.03, 0.01, 0.25, spec)
    second = european_call_mc(100.0, K, 1.0, 0.03, 0.01, 0.25, spec)
    assert np.array_equal(first, second)


def test_heston_conditional_mc_near_bsm_limit_across_strike_strip():
    S0, T, r, q = 100.0, 0.75, 0.03, 0.01
    v0 = theta = 0.04
    p = HestonParams(kappa=8.0, theta=theta, nu=0.002, rho=0.0, v0=v0)
    K = np.linspace(75.0, 125.0, 7)
    mc = heston_conditional_mc_calls(
        S0,
        K,
        T,
        r,
        q,
        p,
        HestonMCScheme(n_paths=80_000, n_steps=60, seed=987, scheme="exact"),
    )
    ref = _bs_call_strip(S0, K, T, r, q, np.sqrt(v0))
    _assert_mc_close(mc, ref, rel=0.05, abs_floor=0.12)


@pytest.mark.slow
def test_heston_conditional_mc_is_within_ten_percent_of_lewis_reference(lewis_heston):
    d = lewis_heston
    p = HestonParams(d["kappa"], d["theta"], d["nu"], d["rho"], d["v0"])
    strikes = np.asarray(d["strikes"], dtype=float)
    idx = np.array([0, len(strikes) // 2, len(strikes) - 1])
    mc = heston_conditional_mc_calls(
        d["S0"],
        strikes[idx],
        d["T"],
        d["r"],
        d["q"],
        p,
        HestonMCScheme(n_paths=180_000, n_steps=120, seed=77, scheme="exact"),
    )
    _assert_mc_close(mc, np.asarray(d["ref_calls"], dtype=float)[idx], rel=0.10, abs_floor=0.20)


def test_price_strip_outputs_obey_basic_no_arbitrage_shape_for_bsm():
    fwd = ForwardSpec(S0=100.0, r=0.02, q=0.01, T=1.25)
    p = BsmParams(sigma=0.24)
    K = np.linspace(60.0, 150.0, 19)

    prices = price_strip("bsm", "cos", K, fwd, p)
    lower = np.maximum(fwd.disc * (fwd.F0 - K), 0.0)

    assert np.all(np.isfinite(prices))
    assert np.all(prices >= lower - 1e-8)
    assert np.all(prices <= fwd.disc * fwd.F0 + 1e-8)
    assert np.all(np.diff(prices) <= 1e-8)
    assert np.all(np.diff(prices, n=2) >= -2e-4)


def test_public_api_end_to_end_pricing_iv_and_greeks():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = fe.HestonParams(kappa=3.0, theta=0.05, nu=0.35, rho=-0.45, v0=0.04)
    strikes = np.array([90.0, 100.0, 110.0])

    phi = lambda u: fe.heston_cf_form2(u, fwd, p)
    grid = fe.cos_auto_grid(fe.heston_cumulants(fwd, p), N=256, L=10.0)
    cos_out = fe.cos_prices(phi, fwd, strikes, grid)
    cm_prices = fe.carr_madan_price_at_strikes(
        phi,
        fwd,
        fe.FFTGrid(N=4096, eta=0.25, alpha=1.5),
        strikes,
    )
    greek_out = cos_price_and_greeks(phi, fwd, strikes, grid)
    atm_iv = fe.implied_vol_newton_safeguarded(
        float(cos_out.call_prices[1]),
        fe.BSInputs(F0=fwd.F0, K=100.0, T=fwd.T, r=fwd.r, q=fwd.q, is_call=True),
    )

    assert np.all(np.isfinite(cos_out.call_prices))
    assert np.max(np.abs(cos_out.call_prices - cm_prices)) < 5e-4
    assert np.all(np.isfinite(greek_out.delta))
    assert np.all(np.isfinite(greek_out.gamma))
    assert 0.01 < atm_iv < 2.0


def test_surface_workflow_returns_price_and_iv_grids_for_bsm():
    p = BsmParams(sigma=0.22)
    spec = SurfaceSpec(
        S0=100.0,
        r=0.02,
        q=0.01,
        maturities=np.array([0.5, 1.0]),
        strikes=np.array([90.0, 100.0, 110.0]),
    )

    def cf_factory(fwd):
        return lambda u: bsm_cf(u, fwd, p)

    def cumulant_factory(fwd):
        return bsm_cumulants(fwd, p)

    prices = model_price_surface(spec, cf_factory, cumulant_factory, N=256, L=10.0)
    ivs = model_iv_surface(spec, cf_factory, cumulant_factory, N=256, L=10.0)

    assert prices.shape == (2, 3)
    assert ivs.shape == (2, 3)
    assert np.all(np.isfinite(prices))
    assert np.nanmax(np.abs(ivs - p.sigma)) < 1e-5


@pytest.mark.parametrize(
    "method,grid",
    [
        ("cos", None),
        ("carr_madan", FFTGrid(N=4096, eta=0.25, alpha=1.5)),
        ("frft", FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)),
    ],
)
def test_bsm_fourier_engines_agree_on_a_general_strip(method, grid):
    fwd = ForwardSpec(S0=100.0, r=0.025, q=0.005, T=1.0)
    p = BsmParams(sigma=0.30)
    K = np.linspace(80.0, 120.0, 9)
    ref = _bs_call_strip(fwd.S0, K, fwd.T, fwd.r, fwd.q, p.sigma)
    prices = price_strip("bsm", method, K, fwd, p, grid=grid)
    assert np.max(np.abs(prices - ref)) < 2e-5


def test_invalid_inputs_raise_clear_errors():
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    bsm = BsmParams(sigma=0.2)
    heston = HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04)

    with pytest.raises(ValueError, match="strikes must be non-empty"):
        price_strip("bsm", "cos", [], fwd, bsm)
    with pytest.raises(ValueError, match="strictly positive|All strikes"):
        european_call_mc(100.0, np.array([0.0, 100.0]), 1.0, 0.03, 0.0, 0.2, MCSpec(1000, 1))
    with pytest.raises(ValueError, match="scheme"):
        heston_conditional_mc_calls(
            100.0,
            np.array([100.0]),
            1.0,
            0.03,
            0.0,
            heston,
            HestonMCScheme(n_paths=1000, n_steps=10, seed=1, scheme="bad"),
        )
    with pytest.raises(ValueError, match="unknown method"):
        price_strip("bsm", "not_a_method", np.array([100.0]), fwd, bsm)
    with pytest.raises(ValueError, match="unknown model"):
        price_strip("not_a_model", "cos", np.array([100.0]), fwd, bsm)
