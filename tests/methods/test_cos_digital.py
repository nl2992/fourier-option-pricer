"""Tests for COS digital option pricing (Phase 4.1).

Covers:
  - Cash-or-nothing call/put vs BSM analytic reference
  - Asset-or-nothing call/put vs BSM analytic reference
  - Vanilla decomposition: call = AoN_call - K * CoN_call (all models)
  - Monotonicity and boundedness
  - Put–call completeness: CoN_call + CoN_put = disc
  - Strip pricing consistency and monotonicity
  - Error handling (unknown model)
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import foureng as fe
from foureng.models.bsm import bsm_asset_or_nothing, bsm_cash_or_nothing
from foureng.pricers.cos_digital import cos_digital_price, cos_digital_price_strip
from foureng.products.digital import DigitalOption

# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def bsm_fwd():
    return fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)


@pytest.fixture
def bsm_params():
    return fe.BsmParams(sigma=0.20)


# ── BSM analytic reference — cash-or-nothing ──────────────────────────────


@pytest.mark.parametrize(
    "K,cp",
    [
        (85.0, 1),
        (100.0, 1),
        (115.0, 1),
        (85.0, -1),
        (100.0, -1),
        (115.0, -1),
    ],
)
def test_bsm_cash_or_nothing_matches_analytic(K, cp, bsm_fwd, bsm_params):
    """COS cash-or-nothing price matches BSM closed-form to 1e-5."""
    ref = bsm_cash_or_nothing(bsm_fwd, bsm_params, K, cp)
    product = DigitalOption(strike=K, maturity=1.0, cp=cp, payoff_type="cash_or_nothing")
    cos = cos_digital_price("bsm", bsm_fwd, bsm_params, product)
    assert cos == approx(ref, abs=1e-5), f"K={K} cp={cp}: COS={cos:.8f} BSM={ref:.8f}"


# ── BSM analytic reference — asset-or-nothing ─────────────────────────────


@pytest.mark.parametrize(
    "K,cp",
    [
        (85.0, 1),
        (100.0, 1),
        (115.0, 1),
        (85.0, -1),
        (100.0, -1),
        (115.0, -1),
    ],
)
def test_bsm_asset_or_nothing_matches_analytic(K, cp, bsm_fwd, bsm_params):
    """COS asset-or-nothing price matches BSM closed-form to 1e-5."""
    ref = bsm_asset_or_nothing(bsm_fwd, bsm_params, K, cp)
    product = DigitalOption(strike=K, maturity=1.0, cp=cp, payoff_type="asset_or_nothing")
    cos = cos_digital_price("bsm", bsm_fwd, bsm_params, product)
    assert cos == approx(ref, abs=1e-5), f"K={K} cp={cp}: COS={cos:.8f} BSM={ref:.8f}"


# ── Vanilla decomposition across all models ───────────────────────────────

LEVY_MODELS = [
    ("bsm", fe.BsmParams(sigma=0.20)),
    ("vg", fe.VGParams(sigma=0.12, nu=0.3, theta=-0.14)),
    ("cgmy", fe.CgmyParams(C=1.0, G=5.0, M=5.0, Y=1.5)),
    ("nig", fe.NigParams(sigma=0.15, nu=0.3, theta=-0.14)),
    ("kou", fe.KouParams(sigma=0.15, lam=0.5, p=0.4, eta1=10.0, eta2=5.0)),
    ("merton_jd", fe.MertonJDParams(sigma=0.15, lam=0.5, muj=-0.1, sigj=0.10)),
    ("heston", fe.HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04)),
    (
        "bates",
        fe.BatesParams(
            kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04, lam_j=0.5, mu_j=-0.1, sigma_j=0.1
        ),
    ),
]


@pytest.mark.parametrize("model,params", LEVY_MODELS)
@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_vanilla_decomposition_from_digitals(model, params, K):
    """AoN_call − K · CoN_call must equal the vanilla call (all models)."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
    aon = cos_digital_price(
        model,
        fwd,
        params,
        DigitalOption(strike=K, maturity=1.0, cp=1, payoff_type="asset_or_nothing"),
    )
    con = cos_digital_price(
        model,
        fwd,
        params,
        DigitalOption(strike=K, maturity=1.0, cp=1, payoff_type="cash_or_nothing"),
    )
    vanilla_from_dig = aon - K * con
    vanilla_ref = float(fe.price_strip(model, "cos_improved", [K], fwd, params)[0])
    assert abs(vanilla_from_dig - vanilla_ref) < 1e-4, (
        f"{model} K={K}: {vanilla_from_dig:.6f} != {vanilla_ref:.6f}"
    )


# ── Put–call completeness: CoN_call + CoN_put = disc ─────────────────────


@pytest.mark.parametrize("model,params", LEVY_MODELS)
def test_cash_or_nothing_put_call_sum_equals_discount(model, params):
    """CoN call + CoN put = disc (probability of up + down = 1)."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
    disc = fwd.disc
    for K in [90.0, 100.0, 110.0]:
        con_call = cos_digital_price(
            model,
            fwd,
            params,
            DigitalOption(strike=K, maturity=1.0, cp=1, payoff_type="cash_or_nothing"),
        )
        con_put = cos_digital_price(
            model,
            fwd,
            params,
            DigitalOption(strike=K, maturity=1.0, cp=-1, payoff_type="cash_or_nothing"),
        )
        assert abs(con_call + con_put - disc) < 1e-5, (
            f"{model} K={K}: CoN_call+CoN_put={con_call + con_put:.8f} != disc={disc:.8f}"
        )


# ── Monotonicity ──────────────────────────────────────────────────────────


def test_cash_or_nothing_call_decreases_with_strike(bsm_fwd, bsm_params):
    """CoN call price is strictly decreasing in strike (higher K → less likely to exercise)."""
    strikes = np.array([80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    prices = cos_digital_price_strip("bsm", bsm_fwd, bsm_params, strikes, 1.0, cp=1)
    assert np.all(np.diff(prices) < 0), "CoN call must strictly decrease with strike"


def test_cash_or_nothing_put_increases_with_strike(bsm_fwd, bsm_params):
    """CoN put price is strictly increasing in strike."""
    strikes = np.array([80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    prices = cos_digital_price_strip("bsm", bsm_fwd, bsm_params, strikes, 1.0, cp=-1)
    assert np.all(np.diff(prices) > 0), "CoN put must strictly increase with strike"


# ── Boundedness ──────────────────────────────────────────────────────────


def test_cash_or_nothing_bounded_by_discount(bsm_fwd, bsm_params):
    """Every CoN price lies in [0, disc]."""
    strikes = np.linspace(70.0, 130.0, 13)
    disc = bsm_fwd.disc
    for cp in (1, -1):
        prices = cos_digital_price_strip("bsm", bsm_fwd, bsm_params, strikes, 1.0, cp=cp)
        assert np.all(prices >= 0), f"cp={cp}: negative CoN price"
        assert np.all(prices <= disc + 1e-8), f"cp={cp}: CoN price exceeds discount"


def test_asset_or_nothing_bounded_by_spot(bsm_fwd, bsm_params):
    """AoN price lies in [0, S0·exp(-qT)] (upper bound = undiscounted spot)."""
    fwd = bsm_fwd
    upper = fwd.S0 * np.exp(-fwd.q * fwd.T)
    strikes = np.linspace(70.0, 130.0, 13)
    for cp in (1, -1):
        prices = cos_digital_price_strip(
            "bsm", fwd, bsm_params, strikes, 1.0, cp=cp, payoff_type="asset_or_nothing"
        )
        assert np.all(prices >= 0), f"cp={cp}: negative AoN price"
        assert np.all(prices <= upper + 1e-6), f"cp={cp}: AoN price exceeds upper bound"


# ── Cash amount scaling ───────────────────────────────────────────────────


def test_cash_amount_scales_price(bsm_fwd, bsm_params):
    """CoN price scales linearly with cash_amount."""
    product_1 = DigitalOption(
        strike=100.0, maturity=1.0, cp=1, payoff_type="cash_or_nothing", cash_amount=1.0
    )
    product_5 = DigitalOption(
        strike=100.0, maturity=1.0, cp=1, payoff_type="cash_or_nothing", cash_amount=5.0
    )
    p1 = cos_digital_price("bsm", bsm_fwd, bsm_params, product_1)
    p5 = cos_digital_price("bsm", bsm_fwd, bsm_params, product_5)
    assert p5 == approx(5.0 * p1, rel=1e-10)


# ── Strip consistency ─────────────────────────────────────────────────────


def test_strip_matches_scalar_loop(bsm_fwd, bsm_params):
    """Strip function gives identical results to scalar loop."""
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    strip = cos_digital_price_strip("bsm", bsm_fwd, bsm_params, strikes, 1.0, cp=1)
    for i, K in enumerate(strikes):
        product = DigitalOption(strike=float(K), maturity=1.0, cp=1, payoff_type="cash_or_nothing")
        scalar = cos_digital_price("bsm", bsm_fwd, bsm_params, product)
        assert strip[i] == approx(scalar, rel=1e-12), f"Mismatch at K={K}"


# ── Error handling ────────────────────────────────────────────────────────


def test_unknown_model_raises():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    product = DigitalOption(strike=100.0, maturity=1.0, cp=1)
    with pytest.raises(ValueError, match="unknown model"):
        cos_digital_price("no_such_model", fwd, None, product)
