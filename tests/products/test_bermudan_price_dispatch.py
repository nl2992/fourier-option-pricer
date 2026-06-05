"""Product-dispatch tests for Bermudan pricing."""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.pipeline import price
from foureng.pricers.cos_bermudan import cos_bermudan_price
from foureng.products import BermudanOption


def test_price_dispatches_to_cos_bermudan_for_bsm():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    product = BermudanOption(
        strike=100.0,
        maturity=1.0,
        cp=-1,
        exercise_times=np.linspace(0.25, 1.0, 4),
    )

    actual = price(product, "bsm", "cos_bermudan", fwd, params)
    expected = cos_bermudan_price("bsm", fwd, params, product)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_price_dispatches_to_cos_bermudan_for_vg():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.VGParams(sigma=0.12, nu=0.3, theta=-0.14)
    product = BermudanOption(
        strike=100.0,
        maturity=1.0,
        cp=-1,
        exercise_times=np.linspace(0.25, 1.0, 4),
    )

    actual = price(product, "vg", "cos_bermudan", fwd, params)
    expected = cos_bermudan_price("vg", fwd, params, product)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_price_bermudan_rejects_wrong_method():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    product = BermudanOption(
        strike=100.0,
        maturity=1.0,
        cp=-1,
        exercise_times=np.linspace(0.25, 1.0, 4),
    )

    with pytest.raises(NotImplementedError, match="cos_bermudan"):
        price(product, "bsm", "cos", fwd, params)


def test_price_bermudan_rejects_sv_model_with_clear_error():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04)
    product = BermudanOption(
        strike=100.0,
        maturity=1.0,
        cp=-1,
        exercise_times=np.linspace(0.25, 1.0, 4),
    )

    with pytest.raises(NotImplementedError, match="1-D COS Bermudan"):
        price(product, "heston", "cos_bermudan", fwd, params)
