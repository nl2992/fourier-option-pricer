"""Sv32 backend stability smoke test using the known-stable PyFENG docstring params.

This is a backend stability test, not a Baldeaux-Badran paper replication.
The parameters are the ones PyFENG uses in its own docstring example (v0=0.06,
kappa=20.48, theta=0.218, nu=3.20, rho=-0.99) at a one-month maturity, which
lie within the numerical stability range of Sv32Fft.

The test verifies no-arbitrage shape conditions (finite, non-negative, monotone
decreasing, convex in strike). No exact prices are asserted.

For the original Baldeaux-Badran figure params see:
    tests/papers/test_sv32_baldeaux_badran_original_smoke.py
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.derived_reference, pytest.mark.numerical_stability]

pyfeng = pytest.importorskip("pyfeng", reason="pyfeng not installed  -  sv32 tests require it")

import foureng as fe

_FWD = fe.ForwardSpec(S0=100.0, r=0.0, q=0.0, T=1.0 / 12.0)
_PARAMS = fe.Sv32Params(v0=0.06, kappa=20.48, theta=0.218, nu=3.20, rho=-0.99)
_STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])


def test_sv32_compat_params_finite_non_negative():
    prices = fe.price_strip("sv32", "pyfeng_fft", _STRIKES, _FWD, _PARAMS)
    assert np.all(np.isfinite(prices)), f"Non-finite prices: {prices}"
    assert np.all(prices >= 0.0), f"Negative price(s): {prices}"


def test_sv32_compat_params_monotone_decreasing():
    prices = fe.price_strip("sv32", "pyfeng_fft", _STRIKES, _FWD, _PARAMS)
    assert np.all(np.diff(prices) <= 1e-8), (
        f"Call prices not monotone decreasing in strike: {prices}"
    )


def test_sv32_compat_params_convex_in_strike():
    prices = fe.price_strip("sv32", "pyfeng_fft", _STRIKES, _FWD, _PARAMS)
    convexity = prices[:-2] - 2 * prices[1:-1] + prices[2:]
    assert np.all(convexity >= -1e-8), (
        f"Call prices not convex in strike: {prices}"
    )
