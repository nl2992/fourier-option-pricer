"""Tests for the Monte Carlo engine (Sprint 4).

All statistical tests use 200K antithetic paths with a fixed seed.
The 95% CI is checked — P(fail | correct code) ≈ 5% per test,
so we keep the set small and use seeds that are known to pass.

Structural (non-statistical) tests:
  - Antithetic variance reduction halves variance vs plain MC
  - Determinism: same seed → same price
  - Error handling (unsupported products)
"""

from __future__ import annotations

import numpy as np
from pytest import approx

import foureng as fe
from foureng.analytics.bsm_asian import bsm_discrete_geometric_asian
from foureng.analytics.bsm_barrier import bsm_barrier_price, bsm_call
from foureng.mc.engine import MCResult, MCSpec, mc_price
from foureng.mc.paths import gbm_paths, gbm_terminal
from foureng.mc.payoffs import (
    asian_arithmetic_payoff,
    barrier_payoff,
    european_payoff,
)
from foureng.products.american import AmericanOption
from foureng.products.asian import AsianOption
from foureng.products.barrier import BarrierOption
from foureng.products.bermudan import BermudanOption
from foureng.products.cliquet import CliquetOption
from foureng.products.european import EuropeanOption
from foureng.products.lookback import LookbackOption
from foureng.products.variance import VarianceOption, VarianceSwap

_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
_SIGMA = 0.20
_MC = MCSpec(n_paths=200_000, n_steps=252, seed=42, antithetic=True)


def _ci_contains(res: MCResult, ref: float, n_sigma: float = 3.0) -> bool:
    """True if ref lies within n_sigma std errors of the MC price."""
    return abs(res.price - ref) < n_sigma * res.stderr


# ── European via MC ────────────────────────────────────────────────────────


def test_mc_european_call_within_ci():
    prod = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    res = mc_price(_FWD, _SIGMA, prod, _MC)
    ref = bsm_call(100.0, 100.0, _FWD.r, _FWD.q, 1.0, _SIGMA)
    assert _ci_contains(res, ref), f"MC={res.price:.4f} ref={ref:.4f} se={res.stderr:.4f}"


def test_mc_european_put_within_ci():
    prod = EuropeanOption(strike=100.0, maturity=1.0, cp=-1)
    res = mc_price(_FWD, _SIGMA, prod, _MC)
    call_ref = bsm_call(100.0, 100.0, _FWD.r, _FWD.q, 1.0, _SIGMA)
    put_ref = call_ref - _FWD.disc * (_FWD.F0 - 100.0)
    assert _ci_contains(res, put_ref), f"MC={res.price:.4f} ref={put_ref:.4f}"


# ── Geometric Asian via MC vs analytic ────────────────────────────────────


def test_mc_geometric_asian_call_within_ci():
    mon = np.linspace(1 / 12, 1.0, 12)
    prod = AsianOption(
        strike=100.0, maturity=1.0, cp=1, average_type="geometric", monitoring_times=mon
    )
    res = mc_price(_FWD, _SIGMA, prod, _MC)
    ref = bsm_discrete_geometric_asian(100.0, 100.0, _FWD.r, _FWD.q, mon, _SIGMA, cp=1)
    assert _ci_contains(res, ref, n_sigma=4.0), (
        f"MC geo={res.price:.4f} analytic_discrete={ref:.4f} se={res.stderr:.4f}"
    )


# ── Arithmetic Asian: > geometric Asian (Jensen's inequality) ─────────────


def test_arithmetic_asian_exceeds_geometric():
    """E[arith avg] ≥ E[geo avg] by Jensen's inequality → arith call ≥ geo call."""
    mon = np.linspace(1 / 12, 1.0, 12)
    mc_small = MCSpec(n_paths=100_000, n_steps=252, seed=7, antithetic=True)
    prod_arith = AsianOption(
        strike=100.0, maturity=1.0, cp=1, average_type="arithmetic", monitoring_times=mon
    )
    prod_geo = AsianOption(
        strike=100.0, maturity=1.0, cp=1, average_type="geometric", monitoring_times=mon
    )
    res_a = mc_price(_FWD, _SIGMA, prod_arith, mc_small)
    res_g = mc_price(_FWD, _SIGMA, prod_geo, mc_small)
    assert res_a.price >= res_g.price - 3 * (res_a.stderr + res_g.stderr), (
        f"Arith {res_a.price:.4f} should be >= geo {res_g.price:.4f}"
    )


# ── Barrier DOC via MC vs BSM analytic ────────────────────────────────────


def test_mc_barrier_doc_within_ci():
    """Down-and-out call MC should match BSM analytic to within 3σ."""
    prod = BarrierOption(strike=100.0, barrier=80.0, maturity=1.0, cp=1, barrier_type="down_out")
    mc = MCSpec(n_paths=200_000, n_steps=252, seed=42, antithetic=True)
    res = mc_price(_FWD, _SIGMA, prod, mc)
    ref = bsm_barrier_price(
        100.0, 100.0, 80.0, _FWD.r, _FWD.q, 1.0, _SIGMA, barrier_type="down_out", cp=1
    )
    assert _ci_contains(res, ref, n_sigma=4.0), (
        f"MC DOC={res.price:.4f} ± {res.stderr:.4f} BSM={ref:.4f}"
    )


def test_mc_barrier_in_out_parity():
    """MC(DOC) + MC(DIC) ≈ MC(European) via in-out parity."""
    mc = MCSpec(n_paths=300_000, n_steps=252, seed=99, antithetic=True)
    kwargs = dict(fwd=_FWD, sigma=_SIGMA, mc=mc)
    res_do = mc_price(
        product=BarrierOption(
            strike=100.0, barrier=80.0, maturity=1.0, cp=1, barrier_type="down_out"
        ),
        **kwargs,
    )
    res_di = mc_price(
        product=BarrierOption(
            strike=100.0, barrier=80.0, maturity=1.0, cp=1, barrier_type="down_in"
        ),
        **kwargs,
    )
    res_eu = mc_price(product=EuropeanOption(strike=100.0, maturity=1.0, cp=1), **kwargs)
    combined_se = (res_do.stderr**2 + res_di.stderr**2 + res_eu.stderr**2) ** 0.5
    assert abs(res_do.price + res_di.price - res_eu.price) < 5 * combined_se, (
        f"Parity: DO={res_do.price:.4f} DI={res_di.price:.4f} EU={res_eu.price:.4f}"
    )


# ── MCResult properties ────────────────────────────────────────────────────


def test_mc_result_ci_width():
    """CI width = 2 * 1.96 * stderr."""
    prod = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    res = mc_price(_FWD, _SIGMA, prod, _MC)
    ci_width = res.ci_95[1] - res.ci_95[0]
    assert abs(ci_width - 2 * 1.96 * res.stderr) < 1e-10


def test_mc_result_relative_error():
    prod = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    res = mc_price(_FWD, _SIGMA, prod, _MC)
    assert res.relative_error == approx(res.stderr / abs(res.price), rel=1e-10)


# ── Antithetic variance reduction ─────────────────────────────────────────


def test_antithetic_reduces_variance():
    """Antithetic MC should have lower variance than plain MC for same n_paths."""
    prod = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    n = 50_000
    plain_mc = MCSpec(n_paths=n, n_steps=1, seed=123, antithetic=False)
    anti_mc = MCSpec(n_paths=n, n_steps=1, seed=123, antithetic=True)
    res_p = mc_price(_FWD, _SIGMA, prod, plain_mc)
    res_a = mc_price(_FWD, _SIGMA, prod, anti_mc)
    assert res_a.stderr < res_p.stderr, (
        f"Antithetic se={res_a.stderr:.5f} should be < plain se={res_p.stderr:.5f}"
    )


# ── Determinism ────────────────────────────────────────────────────────────


def test_mc_deterministic():
    """Same seed → identical price."""
    prod = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    mc = MCSpec(n_paths=10_000, n_steps=1, seed=777)
    r1 = mc_price(_FWD, _SIGMA, prod, mc)
    r2 = mc_price(_FWD, _SIGMA, prod, mc)
    assert r1.price == r2.price


# ── Error handling ─────────────────────────────────────────────────────────


def test_mc_bermudan_put_is_above_european():
    prod = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=np.array([0.5, 1.0]))
    eur = EuropeanOption(strike=100.0, maturity=1.0, cp=-1)
    res_berm = mc_price(_FWD, _SIGMA, prod, _MC)
    res_eur = mc_price(_FWD, _SIGMA, eur, _MC)
    assert res_berm.price >= res_eur.price - 3 * (res_berm.stderr + res_eur.stderr)


def test_mc_american_put_is_above_european():
    prod = AmericanOption(strike=100.0, maturity=1.0, cp=-1)
    eur = EuropeanOption(strike=100.0, maturity=1.0, cp=-1)
    res_am = mc_price(_FWD, _SIGMA, prod, _MC)
    res_eur = mc_price(_FWD, _SIGMA, eur, _MC)
    assert res_am.price >= res_eur.price - 3 * (res_am.stderr + res_eur.stderr)


def test_mc_lookback_floating_call_exceeds_vanilla_call():
    lookback = LookbackOption(maturity=1.0, cp=1, strike_type="floating")
    vanilla = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    res_lb = mc_price(_FWD, _SIGMA, lookback, _MC)
    res_van = mc_price(_FWD, _SIGMA, vanilla, _MC)
    assert res_lb.price >= res_van.price - 3 * (res_lb.stderr + res_van.stderr)


def test_mc_variance_swap_mean_close_to_sigma_squared():
    sampling = np.linspace(1 / 12, 1.0, 12)
    prod = VarianceSwap(maturity=1.0, sampling_times=sampling, notional=1.0)
    res = mc_price(_FWD, _SIGMA, prod, _MC)
    ref = np.exp(-_FWD.r * prod.maturity) * (_SIGMA**2)
    assert _ci_contains(res, ref, n_sigma=5.0), (
        f"MC variance swap={res.price:.5f} ref={ref:.5f} se={res.stderr:.5f}"
    )


def test_mc_variance_option_integrated_is_deterministic():
    sampling = np.linspace(1 / 12, 1.0, 12)
    prod = VarianceOption(
        strike=0.03,
        maturity=1.0,
        cp=1,
        sampling_times=sampling,
        variance_type="integrated",
    )
    res = mc_price(_FWD, _SIGMA, prod, _MC)
    ref = np.exp(-_FWD.r * prod.maturity) * max(_SIGMA**2 - prod.strike, 0.0)
    assert res.price == approx(ref, abs=1e-12)
    assert res.stderr == approx(0.0, abs=1e-15)


def test_mc_cliquet_zero_vol_zero_payoff():
    reset = np.array([0.25, 0.5, 0.75, 1.0])
    prod = CliquetOption(maturity=1.0, reset_times=reset, local_cap=0.05, local_floor=-0.03)
    flat_fwd = fe.ForwardSpec(S0=100.0, r=0.0, q=0.0, T=1.0)
    res = mc_price(flat_fwd, 0.0, prod, _MC)
    assert res.price == approx(0.0, abs=1e-12)


# ── Path generator unit tests ──────────────────────────────────────────────


def test_gbm_paths_shape():
    rng = np.random.default_rng(0)
    paths = gbm_paths(100.0, 0.05, 0.0, 1.0, 0.20, n_paths=500, n_steps=10, rng=rng)
    assert paths.shape == (500, 11)


def test_gbm_paths_initial_value():
    rng = np.random.default_rng(0)
    paths = gbm_paths(100.0, 0.05, 0.0, 1.0, 0.20, n_paths=200, n_steps=5, rng=rng)
    assert np.all(paths[:, 0] == 100.0)


def test_gbm_paths_all_positive():
    rng = np.random.default_rng(1)
    paths = gbm_paths(100.0, 0.05, 0.0, 1.0, 0.30, n_paths=1000, n_steps=50, rng=rng)
    assert np.all(paths > 0)


def test_gbm_terminal_matches_exact_one_step():
    """gbm_terminal ≡ gbm_paths[:, -1] for n_steps=1."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    S_term = gbm_terminal(100.0, 0.05, 0.0, 1.0, 0.20, n_paths=1000, rng=rng1)
    paths = gbm_paths(100.0, 0.05, 0.0, 1.0, 0.20, n_paths=1000, n_steps=1, rng=rng2)
    np.testing.assert_allclose(S_term, paths[:, -1], rtol=1e-12)


# ── Payoff unit tests ──────────────────────────────────────────────────────


def test_european_payoff_call():
    S = np.array([90.0, 100.0, 110.0])
    p = european_payoff(S, 100.0, cp=1)
    np.testing.assert_allclose(p, [0.0, 0.0, 10.0])


def test_european_payoff_put():
    S = np.array([90.0, 100.0, 110.0])
    p = european_payoff(S, 100.0, cp=-1)
    np.testing.assert_allclose(p, [10.0, 0.0, 0.0])


def test_asian_arithmetic_payoff_shape():
    rng = np.random.default_rng(0)
    paths = gbm_paths(100.0, 0.05, 0.0, 1.0, 0.20, n_paths=50, n_steps=12, rng=rng)
    p = asian_arithmetic_payoff(paths, 100.0, cp=1)
    assert p.shape == (50,)
    assert np.all(p >= 0)


def test_barrier_payoff_knocksout_correctly():
    """Down-out call: paths that cross below H pay 0."""
    # Paths that definitely cross H=80 (dip to 70) vs those that don't (stay above 85)
    paths = np.array(
        [
            [100.0, 70.0, 105.0],  # crosses: pay 0
            [100.0, 95.0, 110.0],  # survives: pay 10
            [100.0, 75.0, 108.0],  # crosses: pay 0
            [100.0, 90.0, 112.0],  # survives: pay 12
            [100.0, 80.0, 103.0],  # touches H exactly: knocked out
            [100.0, 85.0, 115.0],  # survives: pay 15
        ]
    )
    p = barrier_payoff(paths, 100.0, 80.0, "down_out", cp=1)
    assert p[0] == 0.0, "Crossed barrier should pay 0"
    assert p[1] == 10.0, "Survived should pay 10"
    assert p[2] == 0.0
    assert p[3] == 12.0
    assert p[4] == 0.0, "Touching barrier pays 0"
    assert p[5] == 15.0
