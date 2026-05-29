"""Bermudan COS consistency tests across supported Lévy models.

Checks that for each supported 1-D Lévy model:
  - M=1 matches the COS European put price
  - M>1 Bermudan >= European
  - Prices are positive and finite
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.pricers.cos_bermudan import cos_bermudan_price
from foureng.products.bermudan import BermudanOption


@pytest.fixture
def fwd():
    return fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)


# ── parametrize Lévy models ────────────────────────────────────────────────

LEVY_CASES = [
    ("vg",    fe.VGParams(sigma=0.12, nu=0.3, theta=-0.14)),
    ("cgmy",  fe.CgmyParams(C=1.0, G=5.0, M=5.0, Y=1.5)),
    ("nig",   fe.NigParams(sigma=0.15, nu=0.3, theta=-0.14)),
    ("kou",   fe.KouParams(sigma=0.15, lam=0.5, p=0.4, eta1=10.0, eta2=5.0)),
    ("merton_jd", fe.MertonJDParams(sigma=0.15, lam=0.5, muj=-0.1, sigj=0.10)),
]


@pytest.mark.parametrize("model,params", LEVY_CASES)
def test_levy_bermudan_m1_matches_european_put(model, params, fwd):
    """M=1 Bermudan at T should match the COS European put price."""
    fwd_T = fe.ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=1.0)
    eur_call = float(fe.price_strip(model, "cos_improved", [100.0], fwd_T, params)[0])
    # Derive put via put-call parity
    disc = np.exp(-fwd.r * fwd.T)
    eur_put = eur_call - disc * (fwd.F0 - 100.0)

    product = BermudanOption(strike=100.0, maturity=1.0, cp=-1,
                             exercise_times=np.array([1.0]))
    berm = cos_bermudan_price(model, fwd, params, product)
    assert abs(berm - eur_put) < 5e-3, (
        f"{model}: M=1 Bermudan put {berm:.6f} != European put {eur_put:.6f}"
    )


@pytest.mark.parametrize("model,params", LEVY_CASES)
def test_levy_bermudan_exceeds_european(model, params, fwd):
    """M=4 Bermudan must be >= European (more exercise → higher value)."""
    product_eur = BermudanOption(strike=100.0, maturity=1.0, cp=-1,
                                 exercise_times=np.array([1.0]))
    product_4 = BermudanOption(strike=100.0, maturity=1.0, cp=-1,
                               exercise_times=np.linspace(0.25, 1.0, 4))
    p_eur = cos_bermudan_price(model, fwd, params, product_eur)
    p_4 = cos_bermudan_price(model, fwd, params, product_4)
    assert p_4 >= p_eur - 1e-6, (
        f"{model}: M=4 Bermudan {p_4:.6f} < European {p_eur:.6f}"
    )


@pytest.mark.parametrize("model,params", LEVY_CASES)
def test_levy_bermudan_positive_finite(model, params, fwd):
    """M=4 Bermudan prices must be finite and non-negative."""
    for K in [90.0, 100.0, 110.0]:
        product = BermudanOption(strike=K, maturity=1.0, cp=-1,
                                 exercise_times=np.linspace(0.25, 1.0, 4))
        price = cos_bermudan_price(model, fwd, params, product)
        assert np.isfinite(price) and price >= 0.0, (
            f"{model} K={K}: price={price}"
        )


@pytest.mark.parametrize("model,params", LEVY_CASES)
def test_levy_bermudan_monotone_in_frequency(model, params, fwd):
    """More exercise dates → higher Bermudan value."""
    prices = []
    for m in [1, 4, 12]:
        ex = np.linspace(1 / m, 1.0, m)
        product = BermudanOption(strike=100.0, maturity=1.0, cp=-1,
                                 exercise_times=ex)
        prices.append(cos_bermudan_price(model, fwd, params, product))
    assert prices[0] <= prices[1] + 1e-6, (
        f"{model}: M=1 {prices[0]:.6f} > M=4 {prices[1]:.6f}"
    )
    assert prices[1] <= prices[2] + 1e-6, (
        f"{model}: M=4 {prices[1]:.6f} > M=12 {prices[2]:.6f}"
    )


# ── SV and SV+jump models rejected ────────────────────────────────────────

@pytest.mark.parametrize("model,params", [
    ("heston", fe.HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04)),
    ("bates",  fe.BatesParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04,
                               lam_j=0.5, mu_j=-0.1, sigma_j=0.1)),
])
def test_sv_models_raise_not_implemented(model, params):
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    product = BermudanOption(strike=100.0, maturity=1.0, cp=-1,
                             exercise_times=np.array([1.0]))
    with pytest.raises(NotImplementedError):
        cos_bermudan_price(model, fwd, params, product)
