"""Stochastic-volatility exotics by COS in the log-price and a CTMC in the variance."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

import foureng as fe
from foureng.core.capabilities import explain_capability
from foureng.models.base import ForwardSpec
from foureng.pricers.cos_bermudan import cos_bermudan_price
from foureng.pricers.cos_ctmc import (
    CTMCVarianceGrid,
    cir_chain,
    cos_ctmc_american_price,
    cos_ctmc_barrier_price,
    cos_ctmc_bermudan_price,
    cos_ctmc_european_price,
)
from foureng.products import AmericanOption, BarrierOption, BermudanOption

# Ikonen & Toivanen (2004) American put benchmark under Heston
IT = fe.HestonParams(kappa=5.0, theta=0.16, nu=0.9, rho=0.1, v0=0.0625)
IT_T, IT_K, IT_R = 0.25, 10.0, 0.1
IT_SPOTS = [8.0, 9.0, 10.0, 11.0, 12.0]
IT_AMERICAN = [2.000000, 1.107629, 0.520038, 0.213681, 0.082046]

FWD = ForwardSpec(100.0, 0.03, 0.01, 1.0)
HESTON = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.7, v0=0.04)  # Feller fails
BATES = fe.BatesParams(
    kappa=2.0, theta=0.04, nu=0.5, rho=-0.7, v0=0.04, lam_j=0.5, mu_j=-0.1, sigma_j=0.15
)


def _euro(model, fwd, params, K, cp=-1):
    return fe.price_strip(model, "contour", np.array([K]), fwd, params, cp=cp)[0]


# ------------------------------------------------------------------ the chain


@pytest.mark.parametrize("nu", [0.3, 0.5, 0.9])
def test_cir_chain_is_a_generator_with_the_right_mean(nu):
    v0, kappa, theta, T = 0.04, 2.0, 0.05, 1.0
    v, Q, i0 = cir_chain(v0, kappa, theta, nu, T, 96)
    assert v[0] == 0.0 and v[i0] == v0 and np.all(np.diff(v) > 0)
    np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-10)
    assert np.all(Q - np.diag(np.diag(Q)) >= 0.0)
    mean = np.eye(len(v))[i0] @ expm(T * Q) @ v
    exact = theta + (v0 - theta) * np.exp(-kappa * T)
    assert mean == pytest.approx(exact, abs=2e-4)


# ------------------------------------------------------------------ Europeans through the chain


@pytest.mark.parametrize("S0", [8.0, 10.0, 12.0])
def test_heston_european_through_the_chain(S0):
    fwd = ForwardSpec(S0, IT_R, 0.0, IT_T)
    v = cos_ctmc_european_price("heston", fwd, IT, strike=IT_K, cp=-1)
    assert v == pytest.approx(_euro("heston", fwd, IT, IT_K), abs=5e-6)


def test_feller_violating_heston_and_bates_europeans():
    for K in (80.0, 100.0, 120.0):
        v = cos_ctmc_european_price("heston", FWD, HESTON, strike=K, cp=-1)
        assert v == pytest.approx(_euro("heston", FWD, HESTON, K), abs=2e-5)
        v = cos_ctmc_european_price("bates", FWD, BATES, strike=K, cp=-1)
        assert v == pytest.approx(_euro("bates", FWD, BATES, K), abs=2e-5)


def test_chain_error_is_second_order_and_extrapolation_removes_it():
    ref = _euro("heston", FWD, HESTON, 100.0)
    errs = [
        abs(
            cos_ctmc_european_price(
                "heston",
                FWD,
                HESTON,
                strike=100.0,
                cp=-1,
                grid=CTMCVarianceGrid(n_states=m, richardson=False),
            )
            - ref
        )
        for m in (32, 64, 128)
    ]
    assert errs[0] > errs[1] > errs[2]
    assert errs[1] / errs[2] > 3.0  # about 4 for a second-order error
    extrapolated = cos_ctmc_european_price("heston", FWD, HESTON, strike=100.0, cp=-1)
    assert abs(extrapolated - ref) < errs[2]


def test_regime_switching_is_exact():
    rs = fe.RegimeSwitchingBsmParams(
        sigmas=[0.15, 0.35], generator=[[-1.0, 1.0], [2.0, -2.0]], initial_probs=[1.0, 0.0]
    )
    for K in (80.0, 100.0, 120.0):
        v = cos_ctmc_european_price("regime_switching", FWD, rs, strike=K, cp=-1)
        assert v == pytest.approx(_euro("regime_switching", FWD, rs, K), abs=1e-10)
    # identical regimes: a BSM Bermudan
    same = fe.RegimeSwitchingBsmParams(
        sigmas=[0.25, 0.25], generator=[[-1.0, 1.0], [2.0, -2.0]], initial_probs=[1.0, 0.0]
    )
    prod = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=np.arange(1, 13) / 12.0)
    assert cos_ctmc_bermudan_price("regime_switching", FWD, same, prod) == pytest.approx(
        cos_bermudan_price("bsm", FWD, fe.BsmParams(sigma=0.25), prod), abs=1e-12
    )


# ------------------------------------------------------------------ early exercise


def test_heston_bermudans_increase_to_the_american():
    fwd = ForwardSpec(10.0, IT_R, 0.0, IT_T)
    one = BermudanOption(strike=IT_K, maturity=IT_T, cp=-1, exercise_times=np.array([IT_T]))
    assert cos_ctmc_bermudan_price("heston", fwd, IT, one) == pytest.approx(
        _euro("heston", fwd, IT, IT_K), abs=5e-6
    )
    values = [
        cos_ctmc_bermudan_price(
            "heston",
            fwd,
            IT,
            BermudanOption(
                strike=IT_K, maturity=IT_T, cp=-1, exercise_times=IT_T * np.arange(1, M + 1) / M
            ),
        )
        for M in (4, 16, 64)
    ]
    assert values[0] < values[1] < values[2] < IT_AMERICAN[2]


def test_heston_american_matches_ikonen_toivanen_at_the_money():
    fwd = ForwardSpec(10.0, IT_R, 0.0, IT_T)
    v = cos_ctmc_american_price(
        "heston", fwd, IT, AmericanOption(strike=IT_K, maturity=IT_T, cp=-1)
    )
    assert v == pytest.approx(IT_AMERICAN[2], abs=2e-5)


@pytest.mark.slow
def test_heston_american_ikonen_toivanen_table():
    for S0, ref in zip(IT_SPOTS, IT_AMERICAN):
        fwd = ForwardSpec(S0, IT_R, 0.0, IT_T)
        prod = AmericanOption(strike=IT_K, maturity=IT_T, cp=-1)
        assert cos_ctmc_american_price("heston", fwd, IT, prod) == pytest.approx(ref, abs=2e-5)


def test_bates_early_exercise_premium_and_heston_limit():
    prod = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=np.arange(1, 13) / 12)
    berm = cos_ctmc_bermudan_price("bates", FWD, BATES, prod)
    assert berm > _euro("bates", FWD, BATES, 100.0)
    no_jumps = fe.BatesParams(
        kappa=2.0, theta=0.04, nu=0.5, rho=-0.7, v0=0.04, lam_j=0.0, mu_j=-0.1, sigma_j=0.15
    )
    assert cos_ctmc_bermudan_price("bates", FWD, no_jumps, prod) == pytest.approx(
        cos_ctmc_bermudan_price("heston", FWD, HESTON, prod), abs=1e-10
    )


# ------------------------------------------------------------------ barriers


@pytest.mark.parametrize("L", [90.0, 110.0])
def test_single_monitoring_date_is_a_european_replication(L):
    v = cos_ctmc_barrier_price(
        "heston", FWD, HESTON, strike=100.0, barrier=L, maturity=1.0, cp=1, n_monitor=1
    )
    h = 1e-3
    digital = -(_euro("heston", FWD, HESTON, L + h, 1) - _euro("heston", FWD, HESTON, L - h, 1)) / (
        2 * h
    )
    ref = _euro("heston", FWD, HESTON, max(100.0, L), 1) + (max(L, 100.0) - 100.0) * digital
    assert v == pytest.approx(ref, abs=2e-5)


def test_knock_in_plus_knock_out_is_vanilla():
    kw = dict(strike=100.0, barrier=90.0, maturity=1.0, cp=1, n_monitor=12)
    out = cos_ctmc_barrier_price("heston", FWD, HESTON, barrier_type="down_out", **kw)
    inn = cos_ctmc_barrier_price("heston", FWD, HESTON, barrier_type="down_in", **kw)
    assert out + inn == pytest.approx(_euro("heston", FWD, HESTON, 100.0, 1), abs=2e-5)


def _heston_mc_barrier(fwd, p, barrier, K, n_mon, sub, n_paths, seed, cp, down):
    rng = np.random.default_rng(seed)
    n = n_mon * sub
    dt = fwd.T / n
    x = np.zeros(n_paths)
    v = np.full(n_paths, p.v0)
    alive = np.ones(n_paths, bool)
    for k in range(n):
        z1 = rng.standard_normal(n_paths)
        z2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        x += -0.5 * vp * dt + np.sqrt(vp * dt) * z2
        v += p.kappa * (p.theta - vp) * dt + p.nu * np.sqrt(vp * dt) * z1
        if (k + 1) % sub == 0:
            S = fwd.S0 * np.exp((fwd.r - fwd.q) * (k + 1) * dt + x)
            alive &= (S > barrier) if down else (S < barrier)
    S = fwd.S0 * np.exp((fwd.r - fwd.q) * fwd.T + x)
    pay = np.maximum(cp * (S - K), 0.0) * alive * fwd.disc
    return pay.mean(), pay.std() / np.sqrt(n_paths)


@pytest.mark.parametrize(
    "barrier_type,barrier,cp,seed", [("down_out", 90.0, 1, 1), ("up_out", 115.0, -1, 2)]
)
def test_discrete_barriers_match_monte_carlo(barrier_type, barrier, cp, seed):
    v = cos_ctmc_barrier_price(
        "heston",
        FWD,
        HESTON,
        strike=100.0,
        barrier=barrier,
        maturity=1.0,
        barrier_type=barrier_type,
        cp=cp,
        n_monitor=12,
    )
    mc, se = _heston_mc_barrier(
        FWD, HESTON, barrier, 100.0, 12, 20, 100_000, seed, cp, barrier_type == "down_out"
    )
    assert abs(v - mc) < 4.0 * se


# ------------------------------------------------------------------ dispatcher


def test_price_routes_cos_ctmc_products():
    fwd = ForwardSpec(10.0, IT_R, 0.0, IT_T)
    berm = BermudanOption(
        strike=IT_K, maturity=IT_T, cp=-1, exercise_times=IT_T * np.arange(1, 5) / 4
    )
    assert fe.price(berm, "heston", "cos_ctmc", fwd, IT) == pytest.approx(
        cos_ctmc_bermudan_price("heston", fwd, IT, berm), abs=1e-14
    )
    bar = BarrierOption(
        strike=100.0, barrier=90.0, maturity=1.0, barrier_type="down_out", monitoring="discrete"
    )
    assert fe.price(bar, "heston", "cos_ctmc", FWD, HESTON, grid=12) == pytest.approx(
        cos_ctmc_barrier_price(
            "heston", FWD, HESTON, strike=100.0, barrier=90.0, maturity=1.0, n_monitor=12
        ),
        abs=1e-14,
    )
    with pytest.raises(NotImplementedError, match="cos_ctmc"):
        fe.price(berm, "vg", "cos_ctmc", fwd, fe.VGParams(0.2, 0.3, -0.1))
    rebate = BarrierOption(strike=100.0, barrier=90.0, maturity=1.0, rebate=1.0)
    with pytest.raises(NotImplementedError, match="rebate"):
        fe.price(rebate, "heston", "cos_ctmc", FWD, HESTON)
    with pytest.raises(ValueError, match="barrier_type"):
        cos_ctmc_barrier_price(
            "heston", FWD, HESTON, strike=100.0, barrier=90.0, maturity=1.0, barrier_type="knock"
        )
    assert "Supported" in explain_capability("heston", "bermudan", "cos_ctmc")
    assert "cos_ctmc" in explain_capability("heston", "bermudan", "cos_bermudan")
