"""Boundary and invalid-input tests for all product dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from foureng.products import (
    AmericanOption,
    AsianOption,
    BarrierOption,
    BermudanOption,
    DigitalOption,
    EuropeanOption,
    ForwardStartOption,
    LookbackOption,
)


@pytest.mark.parametrize("strike", [-100.0, -1.0, -1e-10, 0.0])
def test_negative_or_zero_strike_european_rejected(strike):
    with pytest.raises(ValueError, match="strike"):
        EuropeanOption(strike=strike, maturity=1.0, cp=1)


@pytest.mark.parametrize("maturity", [-1.0, 0.0, -1e-10])
def test_nonpositive_maturity_rejected(maturity):
    with pytest.raises(ValueError, match="maturity"):
        EuropeanOption(strike=100.0, maturity=maturity, cp=1)


@pytest.mark.parametrize("cp", [0, 2, -2, 100])
def test_invalid_cp_rejected(cp):
    with pytest.raises(ValueError, match="cp"):
        EuropeanOption(strike=100.0, maturity=1.0, cp=cp)


def test_barrier_type_all_valid_values_accepted():
    for bt, barrier in [("down_out", 80.0), ("down_in", 80.0), ("up_out", 120.0), ("up_in", 120.0)]:
        opt = BarrierOption(strike=100.0, barrier=barrier, maturity=1.0, cp=1, barrier_type=bt)
        assert opt.barrier_type == bt


def test_barrier_type_invalid_string_rejected():
    with pytest.raises(ValueError, match="barrier_type"):
        BarrierOption(strike=100.0, barrier=80.0, maturity=1.0, cp=1, barrier_type="knock_out")  # type: ignore


def test_asian_empty_monitoring_times_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        AsianOption(strike=100.0, maturity=1.0, cp=1, monitoring_times=np.array([]))


def test_asian_negative_monitoring_time_rejected():
    with pytest.raises(ValueError, match="> 0"):
        AsianOption(strike=100.0, maturity=1.0, cp=1, monitoring_times=np.array([-0.5, 0.5, 1.0]))


def test_bermudan_empty_exercise_times_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        BermudanOption(strike=100.0, maturity=1.0, cp=-1, exercise_times=np.array([]))


def test_american_basic():
    opt = AmericanOption(strike=100.0, maturity=1.0, cp=-1)
    assert opt.product_type == "american"


def test_lookback_invalid_monitoring_rejected():
    with pytest.raises(ValueError, match="monitoring"):
        LookbackOption(maturity=1.0, cp=1, monitoring="weekly")  # type: ignore


def test_digital_negative_strike_rejected():
    with pytest.raises(ValueError, match="strike"):
        DigitalOption(strike=-5.0, maturity=1.0, cp=1)


def test_forward_start_negative_start_time_rejected():
    with pytest.raises(ValueError, match="start_time"):
        ForwardStartOption(start_time=-0.1, maturity=1.0, cp=1)
