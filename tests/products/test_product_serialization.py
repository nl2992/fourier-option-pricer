"""Tests for product dataclass serialization / repr / equality."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from foureng.products import (
    AsianOption,
    BermudanOption,
    EuropeanOption,
    ForwardStartOption,
)


def test_european_option_equality():
    a = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    b = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    assert a == b


def test_european_option_inequality():
    a = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    b = EuropeanOption(strike=110.0, maturity=1.0, cp=1)
    assert a != b


def test_european_option_repr_contains_strike():
    opt = EuropeanOption(strike=95.0, maturity=1.0, cp=-1)
    assert "95.0" in repr(opt)


def test_european_option_asdict():
    opt = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    d = dataclasses.asdict(opt)
    assert d["strike"] == 100.0
    assert d["cp"] == 1


def test_forward_start_asdict():
    opt = ForwardStartOption(start_time=0.5, maturity=1.0, cp=1, alpha=1.1)
    d = dataclasses.asdict(opt)
    assert d["alpha"] == pytest.approx(1.1)


def test_bermudan_exercise_times_stored_correctly():
    t = np.array([0.25, 0.5, 0.75, 1.0])
    opt = BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=t)
    np.testing.assert_array_equal(opt.exercise_times, t)


def test_asian_monitoring_times_stored_correctly():
    t = np.linspace(0.1, 1.0, 10)
    opt = AsianOption(strike=100.0, maturity=1.0, cp=1,
                      monitoring_times=t, average_type="geometric")
    np.testing.assert_array_almost_equal(opt.monitoring_times, t)


def test_product_type_not_in_init():
    """product_type should not be settable via constructor."""
    # EuropeanOption has product_type as a non-init field
    opt = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    assert opt.product_type == "european"


def test_product_is_hashable():
    opt = EuropeanOption(strike=100.0, maturity=1.0, cp=1)
    d = {opt: "value"}
    assert d[opt] == "value"
