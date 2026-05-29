"""Tests for the COS Bermudan pricer (FO2009 backward induction) — BSM model.

Phase 3.3: Fang & Oosterlee (2009) Bermudan COS.

Key numerical properties verified:
  - M=1 at maturity ≡ European put/call (consistent with COS European)
  - Bermudan ≥ European (no exercise-opportunity discount)
  - Monotone in exercise frequency (more dates → higher price)
  - Bermudan put/call positive and finite
  - Strip pricing consistent with scalar loop
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx
from scipy.stats import norm

import foureng as fe
from foureng.pricers.cos_bermudan import cos_bermudan_price, cos_bermudan_price_strip
from foureng.products.bermudan import BermudanOption

# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def bsm_atm():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    return fwd, params


def _bsm_put(S, K, r, q, T, sig):
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def _bsm_call(S, K, r, q, T, sig):
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ── M=1 matches European ───────────────────────────────────────────────────


def test_bsm_bermudan_m1_matches_european_put(bsm_atm):
    fwd, params = bsm_atm
    product = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=np.array([1.0]))
    price = cos_bermudan_price("bsm", fwd, params, product)
    ref = _bsm_put(fwd.S0, 100.0, fwd.r, fwd.q, fwd.T, params.sigma)
    assert price == approx(ref, abs=1e-3), (
        f"M=1 Bermudan put {price:.6f} should match BSM European put {ref:.6f}"
    )


def test_bsm_bermudan_m1_matches_european_call(bsm_atm):
    fwd, params = bsm_atm
    product = BermudanOption(strike=100.0, maturity=1.0, cp=1, exercise_times=np.array([1.0]))
    price = cos_bermudan_price("bsm", fwd, params, product)
    ref = _bsm_call(fwd.S0, 100.0, fwd.r, fwd.q, fwd.T, params.sigma)
    assert price == approx(ref, abs=1e-3), (
        f"M=1 Bermudan call {price:.6f} should match BSM European call {ref:.6f}"
    )


# ── Bermudan ≥ European ────────────────────────────────────────────────────


def test_bsm_bermudan_put_exceeds_european(bsm_atm):
    fwd, params = bsm_atm
    eur = _bsm_put(fwd.S0, 100.0, fwd.r, fwd.q, fwd.T, params.sigma)
    for m, ex in [
        (2, np.array([0.5, 1.0])),
        (4, np.linspace(0.25, 1.0, 4)),
        (12, np.linspace(1 / 12, 1.0, 12)),
    ]:
        product = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=ex)
        price = cos_bermudan_price("bsm", fwd, params, product)
        assert price >= eur - 1e-6, f"M={m} Bermudan put {price:.6f} < European put {eur:.6f}"


def test_bsm_bermudan_call_exceeds_european(bsm_atm):
    fwd, params = bsm_atm
    eur = _bsm_call(fwd.S0, 100.0, fwd.r, fwd.q, fwd.T, params.sigma)
    for m, ex in [(4, np.linspace(0.25, 1.0, 4)), (12, np.linspace(1 / 12, 1.0, 12))]:
        product = BermudanOption(strike=100.0, maturity=1.0, cp=1, exercise_times=ex)
        price = cos_bermudan_price("bsm", fwd, params, product)
        assert price >= eur - 1e-6, f"M={m} Bermudan call {price:.6f} < European call {eur:.6f}"


# ── Monotone in exercise frequency ────────────────────────────────────────


def test_bsm_bermudan_put_monotone_in_exercise_frequency(bsm_atm):
    fwd, params = bsm_atm
    Ms = [1, 2, 4, 12, 50]
    prices = []
    for m in Ms:
        ex = np.linspace(1 / m, 1.0, m)
        product = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=ex)
        prices.append(cos_bermudan_price("bsm", fwd, params, product))
    for i in range(len(prices) - 1):
        assert prices[i] <= prices[i + 1] + 1e-6, (
            f"M={Ms[i]}: {prices[i]:.6f} > M={Ms[i + 1]}: {prices[i + 1]:.6f}"
        )


# ── Positive and finite ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "K,cp",
    [
        (90.0, -1),
        (100.0, -1),
        (110.0, -1),
        (90.0, 1),
        (100.0, 1),
        (110.0, 1),
    ],
)
def test_bsm_bermudan_prices_positive_finite(K, cp, bsm_atm):
    fwd, params = bsm_atm
    ex = np.linspace(0.25, 1.0, 4)
    product = BermudanOption(strike=K, maturity=1.0, cp=cp, exercise_times=ex)
    price = cos_bermudan_price("bsm", fwd, params, product)
    assert np.isfinite(price) and price >= 0.0, f"K={K} cp={cp}: price={price}"


# ── Strip consistency ──────────────────────────────────────────────────────


def test_bsm_bermudan_strip_matches_scalar_loop(bsm_atm):
    fwd, params = bsm_atm
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    ex = np.linspace(0.25, 1.0, 4)

    strip = cos_bermudan_price_strip("bsm", fwd, params, strikes, 1.0, ex, cp=-1)

    for i, K in enumerate(strikes):
        product = BermudanOption(strike=float(K), maturity=1.0, cp=-1, exercise_times=ex)
        scalar = cos_bermudan_price("bsm", fwd, params, product)
        assert strip[i] == approx(scalar, rel=1e-10), f"Strip mismatch at K={K}"


def test_bsm_bermudan_put_strip_monotone_in_strike(bsm_atm):
    fwd, params = bsm_atm
    strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
    ex = np.linspace(0.25, 1.0, 4)
    strip = cos_bermudan_price_strip("bsm", fwd, params, strikes, 1.0, ex, cp=-1)
    assert np.all(np.diff(strip) > 0), "Bermudan puts must increase with strike"


def test_bsm_bermudan_call_strip_monotone_in_strike(bsm_atm):
    fwd, params = bsm_atm
    strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
    ex = np.linspace(0.25, 1.0, 4)
    strip = cos_bermudan_price_strip("bsm", fwd, params, strikes, 1.0, ex, cp=1)
    assert np.all(np.diff(strip) < 0), "Bermudan calls must decrease with strike"


# ── Error handling ─────────────────────────────────────────────────────────


def test_bsm_bermudan_rejects_sv_model():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04)
    product = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=np.array([1.0]))
    with pytest.raises(NotImplementedError, match="1-D COS Bermudan"):
        cos_bermudan_price("heston", fwd, params, product)


def test_bsm_bermudan_rejects_unknown_model():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    product = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=np.array([1.0]))
    with pytest.raises(ValueError, match="unknown model"):
        cos_bermudan_price("unicorn_model", fwd, None, product)
