"""Markov regime-switching BSM model validation.

Degeneracy structure gives exact references:
- one regime  -> the CF and prices must reduce to plain BSM;
- zero generator -> the CF is the initial-probability mixture of BSM CFs;
- conditional on occupation times the price is BSM at the occupation-weighted
  variance, so every price lies between the all-low-vol and all-high-vol BSM
  prices.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.models.regime_switching import (
    RegimeSwitchingBsmParams,
    regime_switching_cf,
    regime_switching_cumulants,
)

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.01, T=1.0)
_STRIKES = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
_U = np.linspace(-40.0, 40.0, 161)

_TWO_REGIME = RegimeSwitchingBsmParams(
    sigmas=(0.15, 0.35),
    generator=((-1.0, 1.0), (2.0, -2.0)),
    initial_probs=(0.7, 0.3),
)


def test_single_regime_reduces_to_bsm_cf():
    rs = RegimeSwitchingBsmParams(sigmas=(0.2,), generator=((0.0,),), initial_probs=(1.0,))
    phi_rs = regime_switching_cf(_U, _FWD, rs)
    phi_bsm = fe.bsm_cf(_U, _FWD, fe.BsmParams(sigma=0.2))
    np.testing.assert_allclose(phi_rs, phi_bsm, atol=1e-12, rtol=0.0)


def test_zero_generator_is_bsm_mixture():
    rs = RegimeSwitchingBsmParams(
        sigmas=(0.1, 0.4),
        generator=((0.0, 0.0), (0.0, 0.0)),
        initial_probs=(0.6, 0.4),
    )
    phi_rs = regime_switching_cf(_U, _FWD, rs)
    mixture = 0.6 * fe.bsm_cf(_U, _FWD, fe.BsmParams(sigma=0.1)) + 0.4 * fe.bsm_cf(
        _U, _FWD, fe.BsmParams(sigma=0.4)
    )
    np.testing.assert_allclose(phi_rs, mixture, atol=1e-12, rtol=0.0)


def test_martingale_normalization():
    phi_minus_i = regime_switching_cf(np.array([-1j]), _FWD, _TWO_REGIME)
    np.testing.assert_allclose(phi_minus_i, [1.0 + 0.0j], atol=1e-12)


def test_cumulants_single_regime_match_bsm():
    rs = RegimeSwitchingBsmParams(sigmas=(0.25,), generator=((0.0,),), initial_probs=(1.0,))
    c1, c2, c4 = regime_switching_cumulants(_FWD, rs)
    b1, b2, b4 = fe.bsm_cumulants(_FWD, fe.BsmParams(sigma=0.25))
    assert abs(c1 - b1) < 1e-8
    assert abs(c2 - b2) < 1e-6
    assert abs(c4 - b4) < 1e-4


def test_two_regime_price_between_single_regime_bsm_prices():
    rs_prices = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, _TWO_REGIME)
    lo = fe.price_strip("bsm", "cos", _STRIKES, _FWD, fe.BsmParams(sigma=0.15))
    hi = fe.price_strip("bsm", "cos", _STRIKES, _FWD, fe.BsmParams(sigma=0.35))
    assert np.all(rs_prices >= np.minimum(lo, hi) - 1e-10)
    assert np.all(rs_prices <= np.maximum(lo, hi) + 1e-10)


def test_cos_agrees_with_carr_madan_and_hilbert():
    cos = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, _TWO_REGIME)
    cm = fe.price_strip(
        "regime_switching",
        "carr_madan",
        _STRIKES,
        _FWD,
        _TWO_REGIME,
        grid=fe.FFTGrid(N=16384, eta=0.05, alpha=1.5),
    )
    hil = fe.price_strip("regime_switching", "hilbert", _STRIKES, _FWD, _TWO_REGIME)
    np.testing.assert_allclose(cos, cm, atol=2e-4, rtol=0.0)
    np.testing.assert_allclose(cos, hil, atol=1e-6, rtol=0.0)


def test_put_call_parity():
    calls = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, _TWO_REGIME, cp=1)
    puts = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, _TWO_REGIME, cp=-1)
    np.testing.assert_allclose(calls - puts, _FWD.disc * (_FWD.F0 - _STRIKES), atol=1e-8)


def test_fast_switching_approaches_stationary_average_variance():
    """With very fast switching, integrated variance concentrates at the
    stationary mean, so the price approaches BSM at that effective vol."""
    speed = 400.0
    rs = RegimeSwitchingBsmParams(
        sigmas=(0.15, 0.35),
        generator=((-speed, speed), (speed, -speed)),
        initial_probs=(0.5, 0.5),
    )
    # symmetric chain -> stationary distribution (1/2, 1/2)
    var_eff = 0.5 * 0.15**2 + 0.5 * 0.35**2
    bsm_eff = fe.price_strip("bsm", "cos", _STRIKES, _FWD, fe.BsmParams(sigma=np.sqrt(var_eff)))
    rs_prices = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, rs)
    np.testing.assert_allclose(rs_prices, bsm_eff, atol=5e-2, rtol=0.0)


def test_param_validation():
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(sigmas=(), generator=(), initial_probs=())
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(
            sigmas=(0.2, -0.1), generator=((0, 0), (0, 0)), initial_probs=(0.5, 0.5)
        )
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(
            sigmas=(0.2, 0.3), generator=((-1.0, 0.5), (1.0, -1.0)), initial_probs=(0.5, 0.5)
        )
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(
            sigmas=(0.2, 0.3), generator=((-1.0, 1.0), (-2.0, 2.0)), initial_probs=(0.5, 0.5)
        )
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(
            sigmas=(0.2, 0.3), generator=((-1.0, 1.0), (1.0, -1.0)), initial_probs=(0.9, 0.3)
        )
