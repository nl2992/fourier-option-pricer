"""Exact Levy geometric-Asian pricer validation.

The CF-product identity is exact, so three independent references pin it
down:
- under BSM it must match the discrete Kemna-Vorst closed form;
- with a single monitoring date the geometric average IS the terminal spot
  at t_1, so any Levy model must reproduce its European price at T = t_1;
- under Kou it must sit inside the Monte Carlo confidence band.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.analytics.bsm_asian import bsm_discrete_geometric_asian
from foureng.pricers.geometric_asian import levy_geometric_asian_price
from foureng.products.asian import AsianOption

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.01, T=1.0)
_TIMES = np.linspace(0.1, 1.0, 10)
_STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
_KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.6, eta1=25.0, eta2=15.0)
_VG = fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14)


@pytest.mark.parametrize("cp", [1, -1])
def test_bsm_matches_discrete_kemna_vorst(cp):
    params = fe.BsmParams(sigma=0.25)
    cf_prices = levy_geometric_asian_price(
        "bsm", _FWD, params, strikes=_STRIKES, monitoring_times=_TIMES, cp=cp
    )
    ref = np.array(
        [
            bsm_discrete_geometric_asian(
                _FWD.S0, float(K), _FWD.r, _FWD.q, _TIMES, params.sigma, cp=cp
            )
            for K in _STRIKES
        ]
    )
    np.testing.assert_allclose(cf_prices, ref, atol=1e-8, rtol=0.0)


def test_bsm_unequal_monitoring_matches_closed_form():
    params = fe.BsmParams(sigma=0.3)
    times = np.array([0.05, 0.30, 0.35, 0.80, 1.00])
    cf_prices = levy_geometric_asian_price(
        "bsm", _FWD, params, strikes=_STRIKES, monitoring_times=times
    )
    ref = np.array(
        [
            bsm_discrete_geometric_asian(_FWD.S0, float(K), _FWD.r, _FWD.q, times, params.sigma)
            for K in _STRIKES
        ]
    )
    np.testing.assert_allclose(cf_prices, ref, atol=1e-8, rtol=0.0)


@pytest.mark.parametrize(
    "model, params",
    [
        ("kou", _KOU),
        ("vg", _VG),
        ("merton_jd", fe.MertonJDParams(sigma=0.15, lam=0.5, muj=-0.1, sigj=0.2)),
    ],
)
def test_single_fixing_reduces_to_european(model, params):
    """M=1 geometric average == terminal spot at t_1, so the price must equal
    the European COS price with maturity t_1."""
    t1 = 0.6
    fwd_t1 = fe.ForwardSpec(S0=_FWD.S0, r=_FWD.r, q=_FWD.q, T=t1)
    asian = levy_geometric_asian_price(
        model, _FWD, params, strikes=_STRIKES, monitoring_times=np.array([t1])
    )
    european = fe.price_strip(model, "cos", _STRIKES, fwd_t1, params)
    np.testing.assert_allclose(asian, european, atol=5e-7, rtol=0.0)


def test_kou_matches_monte_carlo():
    rng = np.random.default_rng(7)
    n_paths, M = 200_000, 10
    dt = float(_TIMES[1] - _TIMES[0])
    sig, lam, p_up, e1, e2 = _KOU.sigma, _KOU.lam, _KOU.p, _KOU.eta1, _KOU.eta2
    zeta = p_up * e1 / (e1 - 1.0) + (1.0 - p_up) * e2 / (e2 + 1.0) - 1.0
    drift = (_FWD.r - _FWD.q - 0.5 * sig * sig - lam * zeta) * dt

    increments = drift + sig * np.sqrt(dt) * rng.standard_normal((n_paths, M))
    n_jumps = rng.poisson(lam * dt, size=(n_paths, M))
    max_j = int(n_jumps.max())
    for j in range(1, max_j + 1):
        mask = n_jumps >= j
        n_active = int(mask.sum())
        up = rng.random(n_active) < p_up
        jump = np.where(
            up,
            rng.exponential(1.0 / e1, size=n_active),
            -rng.exponential(1.0 / e2, size=n_active),
        )
        add = np.zeros((n_paths, M))
        add[mask] = jump
        increments += add

    log_s = np.log(_FWD.S0) + (_FWD.r - _FWD.q) * 0.0 + np.cumsum(increments, axis=1)
    geo = np.exp(log_s.mean(axis=1))
    disc = np.exp(-_FWD.r * float(_TIMES[-1]))

    K = 100.0
    payoff = disc * np.maximum(geo - K, 0.0)
    mc, se = float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_paths))
    cf = float(
        levy_geometric_asian_price("kou", _FWD, _KOU, strikes=K, monitoring_times=_TIMES, cp=1)[0]
    )
    assert abs(cf - mc) < 4.0 * se, f"CF {cf:.4f} vs MC {mc:.4f} +/- {se:.4f}"


def test_parity_and_monotonicity():
    calls = levy_geometric_asian_price(
        "kou", _FWD, _KOU, strikes=_STRIKES, monitoring_times=_TIMES, cp=1
    )
    puts = levy_geometric_asian_price(
        "kou", _FWD, _KOU, strikes=_STRIKES, monitoring_times=_TIMES, cp=-1
    )
    # call - put is linear in K with slope -disc(T_M)
    diff = calls - puts
    disc = np.exp(-_FWD.r * float(_TIMES[-1]))
    slopes = np.diff(diff) / np.diff(_STRIKES)
    np.testing.assert_allclose(slopes, -disc, atol=1e-10, rtol=0.0)
    assert np.all(np.diff(calls) < 0.0)
    assert np.all(np.diff(puts) > 0.0)


def test_pipeline_dispatch():
    product = AsianOption(
        strike=100.0,
        maturity=float(_TIMES[-1]),
        cp=1,
        average_type="geometric",
        monitoring_times=_TIMES,
        strike_type="fixed",
    )
    via_price = fe.price(product, "kou", "asian_cf", _FWD, _KOU)
    direct = float(
        levy_geometric_asian_price("kou", _FWD, _KOU, strikes=100.0, monitoring_times=_TIMES, cp=1)[
            0
        ]
    )
    assert via_price == pytest.approx(direct, abs=0.0)


def test_rejects_unsupported_cases():
    with pytest.raises(ValueError):
        levy_geometric_asian_price(
            "heston",
            _FWD,
            fe.HestonParams(kappa=2.0, theta=0.04, nu=0.35, rho=-0.6, v0=0.04),
            strikes=100.0,
            monitoring_times=_TIMES,
        )
    with pytest.raises(ValueError):
        levy_geometric_asian_price("kou", _FWD, _KOU, strikes=100.0, monitoring_times=_TIMES, cp=0)
    product = AsianOption(
        strike=100.0,
        maturity=1.0,
        cp=1,
        average_type="arithmetic",
        monitoring_times=_TIMES,
        strike_type="fixed",
    )
    with pytest.raises(NotImplementedError):
        fe.price(product, "kou", "asian_cf", _FWD, _KOU)
