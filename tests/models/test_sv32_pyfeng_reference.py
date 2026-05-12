"""Sv32 frozen regression tests against sv32_pyfeng_reference.json.

Complements (does NOT duplicate) the existing test_sv32_pyfeng_paper.py.
That file covers paper benchmarks, CF identities, cross-engine agreement
against live PyFENG calls, rho=0 symmetry, and other structural tests.

This file covers:
1. pyfeng_fft prices vs the frozen JSON reference (regression test)
2. cos_improved prices vs the frozen JSON reference (new engine)
3. CF martingale/normalization loaded from JSON params (belt-and-suspenders)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.paper, pytest.mark.derived_reference]

pyfeng = pytest.importorskip("pyfeng", reason="pyfeng not installed — sv32 tests require it")

import foureng as fe

# ---------------------------------------------------------------------------
# Load reference
# ---------------------------------------------------------------------------

_REF_PATH = Path(__file__).resolve().parents[1] / "refs" / "sv32_pyfeng_reference.json"

with _REF_PATH.open() as _f:
    _REF = json.load(_f)

_STRIKES = np.array(_REF["strikes"])
_REF_PRICES = np.array(_REF["call_prices"])
_ATOL_TIGHT = _REF["absolute_tolerance_published_prices"]   # 5e-4
_ATOL_LOOSE = 1e-3


def _make_fwd_params():
    fwd = fe.ForwardSpec(
        S0=_REF["spot"],
        r=_REF["risk_free_rate"],
        q=_REF["dividend_yield"],
        T=_REF["maturity"],
    )
    params = fe.Sv32Params(
        v0=_REF["v0"],
        kappa=_REF["kappa"],
        theta=_REF["theta"],
        nu=_REF["nu"],
        rho=_REF["rho"],
    )
    return fwd, params


# ---------------------------------------------------------------------------
# Test 1: pyfeng_fft vs frozen reference
# ---------------------------------------------------------------------------


def test_sv32_pyfeng_fft_matches_frozen_reference():
    """pyfeng_fft prices must match the frozen JSON reference to atol=5e-4."""
    fwd, params = _make_fwd_params()
    prices = fe.price_strip("sv32", "pyfeng_fft", _STRIKES, fwd, params)
    np.testing.assert_allclose(
        prices, _REF_PRICES, atol=_ATOL_TIGHT,
        err_msg="sv32 pyfeng_fft vs frozen reference",
    )


# ---------------------------------------------------------------------------
# Test 2: cos_improved vs frozen reference
# ---------------------------------------------------------------------------


def test_sv32_cos_improved_agrees_with_frozen_reference():
    """cos_improved prices must agree with the frozen reference to atol=1e-3."""
    fwd, params = _make_fwd_params()
    prices = fe.price_strip("sv32", "cos_improved", _STRIKES, fwd, params)
    np.testing.assert_allclose(
        prices, _REF_PRICES, atol=_ATOL_LOOSE,
        err_msg="sv32 cos_improved vs frozen reference",
    )


# ---------------------------------------------------------------------------
# Test 3: CF identities loaded from JSON params
# ---------------------------------------------------------------------------


def test_sv32_cf_martingale_from_json_params():
    """phi(0) = 1 and phi(-i) = 1 for the JSON parameter set."""
    fwd, params = _make_fwd_params()
    phi = lambda u: fe.sv32_cf(u, fwd, params)

    val_zero = phi(np.array([0.0]))[0]
    assert abs(val_zero - 1.0) < 1e-13, f"sv32 phi(0) = {val_zero} != 1"

    val_neg_i = phi(np.array([-1j]))[0]
    assert abs(val_neg_i - 1.0) < 1e-10, (
        f"sv32 phi(-i) = {val_neg_i} != 1 (martingale condition)"
    )
