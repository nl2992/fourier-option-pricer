"""Dispatcher-level guarantees for :func:`foureng.pipeline.price`.

Two things are asserted here that the strip-level tests cannot cover, because
they are promises the *product* API makes and the strip API does not:

1. A ``BermudanOption`` is routed to the COS Bermudan engine rather than
   raising ``NotImplementedError``. The engine itself was already tested; it
   was simply unreachable through ``price()``.
2. A product with ``cp=-1`` is priced as a **put**. ``price_strip`` documents
   that its in-house pricers always return calls and leave parity to the
   caller, which is fine for a strip API — but ``price()`` receives a product
   that *is* a put, so delegating that convention would return a silently
   wrong number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import foureng as fe
from foureng.pipeline import price
from foureng.products import BermudanOption, EuropeanOption

S0, K, R, Q, T, SIGMA = 100.0, 100.0, 0.05, 0.0, 1.0, 0.20

# In-house COS pricers; pyfeng_fft is excluded because it honours cp itself.
_INHOUSE = ["cos", "cos_improved", "cos_filtered"]


def _fwd():
    return fe.ForwardSpec(S0=S0, r=R, q=Q, T=T)


def _bsm_closed_form(cp: int) -> float:
    from scipy.stats import norm

    d1 = (math.log(S0 / K) + (R - Q + 0.5 * SIGMA**2) * T) / (SIGMA * math.sqrt(T))
    d2 = d1 - SIGMA * math.sqrt(T)
    call = S0 * math.exp(-Q * T) * norm.cdf(d1) - K * math.exp(-R * T) * norm.cdf(d2)
    if cp == 1:
        return call
    return call - S0 * math.exp(-Q * T) + K * math.exp(-R * T)


# --------------------------------------------------------------------------
# 1. Bermudan routing
# --------------------------------------------------------------------------


def test_price_routes_bermudan_instead_of_raising():
    opt = BermudanOption(strike=K, maturity=T, cp=-1, exercise_times=np.array([0.5, 1.0]))
    value = price(opt, "bsm", "cos_bermudan", _fwd(), fe.BsmParams(sigma=SIGMA))
    assert isinstance(value, float)
    assert value > 0.0


def test_bermudan_m1_matches_european_put_through_dispatcher():
    """A single exercise date at maturity is a European put."""
    berm = BermudanOption(strike=K, maturity=T, cp=-1, exercise_times=np.array([T]))
    value = price(berm, "bsm", "cos_bermudan", _fwd(), fe.BsmParams(sigma=SIGMA))
    assert value == pytest.approx(_bsm_closed_form(-1), abs=1e-3)


def test_bermudan_value_increases_with_exercise_opportunities():
    params = fe.BsmParams(sigma=SIGMA)
    values = []
    for m in (1, 2, 4, 12):
        times = np.linspace(T / m, T, m)
        opt = BermudanOption(strike=K, maturity=T, cp=-1, exercise_times=times)
        values.append(price(opt, "bsm", "cos_bermudan", _fwd(), params))
    assert all(values[i] <= values[i + 1] + 1e-9 for i in range(len(values) - 1))


def test_bermudan_rejects_methods_without_early_exercise():
    opt = BermudanOption(strike=K, maturity=T, cp=-1, exercise_times=np.array([T]))
    for method in ("cos", "cos_improved", "carr_madan"):
        with pytest.raises(NotImplementedError, match="does not support Bermudan"):
            price(opt, "bsm", method, _fwd(), fe.BsmParams(sigma=SIGMA))


def test_bermudan_still_rejects_stochastic_vol_models():
    """The engine's own 1-D Lévy gate must survive being routed."""
    opt = BermudanOption(strike=K, maturity=T, cp=-1, exercise_times=np.array([T]))
    params = fe.HestonParams(v0=0.04, kappa=1.5, theta=0.04, nu=0.3, rho=-0.7)
    with pytest.raises(NotImplementedError):
        price(opt, "heston", "cos_bermudan", _fwd(), params)


# --------------------------------------------------------------------------
# 2. Put-call handling at the product layer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", _INHOUSE)
@pytest.mark.parametrize("cp", [1, -1])
def test_price_matches_closed_form_for_both_option_types(method, cp):
    opt = EuropeanOption(strike=K, maturity=T, cp=cp)
    value = price(opt, "bsm", method, _fwd(), fe.BsmParams(sigma=SIGMA))
    assert value == pytest.approx(_bsm_closed_form(cp), abs=1e-6)


@pytest.mark.parametrize("method", _INHOUSE)
def test_put_and_call_differ_at_the_product_layer(method):
    """Regression: price() used to return the call value for cp=-1."""
    fwd, params = _fwd(), fe.BsmParams(sigma=SIGMA)
    call = price(EuropeanOption(strike=K, maturity=T, cp=1), "bsm", method, fwd, params)
    put = price(EuropeanOption(strike=K, maturity=T, cp=-1), "bsm", method, fwd, params)
    assert abs(call - put) > 1.0


@pytest.mark.parametrize("method", _INHOUSE)
def test_put_call_parity_holds_through_dispatcher(method):
    fwd, params = _fwd(), fe.BsmParams(sigma=SIGMA)
    call = price(EuropeanOption(strike=K, maturity=T, cp=1), "bsm", method, fwd, params)
    put = price(EuropeanOption(strike=K, maturity=T, cp=-1), "bsm", method, fwd, params)
    forward_value = S0 * math.exp(-Q * T) - K * math.exp(-R * T)
    assert (call - put) == pytest.approx(forward_value, abs=1e-6)


def test_pyfeng_fft_put_unaffected_by_parity_adjustment():
    """pyfeng_fft honours cp natively and must not be double-adjusted."""
    fwd, params = _fwd(), fe.BsmParams(sigma=SIGMA)
    put = price(EuropeanOption(strike=K, maturity=T, cp=-1), "bsm", "pyfeng_fft", fwd, params)
    assert put == pytest.approx(_bsm_closed_form(-1), abs=1e-4)
