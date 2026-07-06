"""CTMC diffusion-approximation validation.

Independent references:
- European vanillas against the BSM closed form, with grid convergence;
- American puts against the CRR lattice American engine;
- American calls with q = 0 must equal Europeans (no early exercise);
- a constant-vol callable must reproduce the constant-vol path exactly,
  and a genuinely local CEV-style vol must move the smile the right way.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd
from foureng.pricers.ctmc import (
    CTMCGrid,
    ctmc_american_price,
    ctmc_european_price,
    ctmc_european_price_at_strikes,
)

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.01, T=1.0)
_SIGMA = 0.2
_STRIKES = np.array([70.0, 85.0, 100.0, 115.0, 130.0])


def _bs_ref(K, cp=1):
    call = bs_price_from_fwd(_SIGMA, BSInputs(F0=_FWD.F0, K=float(K), T=_FWD.T, r=_FWD.r, q=_FWD.q))
    if cp == 1:
        return call
    return call - _FWD.disc * (_FWD.F0 - K)


@pytest.mark.parametrize("cp", [1, -1])
def test_european_matches_bsm_closed_form(cp):
    grid = CTMCGrid(n_states=801, width=6.0)
    prices = ctmc_european_price_at_strikes(
        _FWD.S0, _STRIKES, _FWD.r, _FWD.q, _FWD.T, _SIGMA, cp=cp, grid=grid
    )
    ref = np.array([_bs_ref(K, cp) for K in _STRIKES])
    np.testing.assert_allclose(prices, ref, atol=5e-4, rtol=0.0)


def test_grid_convergence():
    ref = _bs_ref(100.0)
    errs = []
    for m in (101, 201, 401):
        v = ctmc_european_price(
            _FWD.S0, 100.0, _FWD.r, _FWD.q, _FWD.T, _SIGMA, grid=CTMCGrid(n_states=m)
        )
        errs.append(abs(v - ref))
    # second-order convergence: each doubling cuts the error ~4x
    assert errs[1] < 0.35 * errs[0]
    assert errs[2] < 0.35 * errs[1]
    assert errs[2] < 2e-3


def test_american_put_matches_lattice():
    from foureng.products.american import AmericanOption

    product = AmericanOption(strike=105.0, maturity=_FWD.T, cp=-1)
    lattice = fe.price(product, "bsm", "lattice", _FWD, fe.BsmParams(sigma=_SIGMA))
    ctmc = fe.price(product, "bsm", "ctmc", _FWD, fe.BsmParams(sigma=_SIGMA))
    assert ctmc == pytest.approx(lattice, rel=2e-3)


def test_american_call_no_dividends_equals_european():
    fwd0 = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    amer = ctmc_american_price(fwd0.S0, 100.0, fwd0.r, 0.0, fwd0.T, _SIGMA)
    euro = ctmc_european_price(fwd0.S0, 100.0, fwd0.r, 0.0, fwd0.T, _SIGMA)
    assert amer == pytest.approx(euro, abs=1e-8)


def test_american_dominates_european_and_intrinsic():
    amer = ctmc_american_price(_FWD.S0, 110.0, _FWD.r, _FWD.q, _FWD.T, _SIGMA, cp=-1)
    euro = ctmc_european_price(_FWD.S0, 110.0, _FWD.r, _FWD.q, _FWD.T, _SIGMA, cp=-1)
    assert amer >= euro - 1e-12
    assert amer >= 10.0  # intrinsic of the ITM put


def test_constant_vol_callable_matches_constant():
    const = ctmc_european_price(_FWD.S0, 100.0, _FWD.r, _FWD.q, _FWD.T, _SIGMA)
    via_fn = ctmc_european_price(
        _FWD.S0, 100.0, _FWD.r, _FWD.q, _FWD.T, lambda S: np.full_like(S, _SIGMA)
    )
    assert via_fn == pytest.approx(const, abs=0.0)


def test_local_vol_skew_direction():
    """A CEV-style downward-sloping local vol must produce a downward IV
    skew: OTM puts richer, OTM calls cheaper than flat-vol BSM."""

    def cev_vol(S, sigma0=_SIGMA, beta=0.6, S_ref=100.0):
        return sigma0 * (np.asarray(S) / S_ref) ** (beta - 1.0)

    grid = CTMCGrid(n_states=601)
    put_lv = ctmc_european_price(_FWD.S0, 80.0, _FWD.r, _FWD.q, _FWD.T, cev_vol, cp=-1, grid=grid)
    put_flat = _bs_ref(80.0, cp=-1)
    call_lv = ctmc_european_price(_FWD.S0, 120.0, _FWD.r, _FWD.q, _FWD.T, cev_vol, cp=1, grid=grid)
    call_flat = _bs_ref(120.0, cp=1)
    assert put_lv > put_flat
    assert call_lv < call_flat


def test_price_strip_dispatch_and_parity():
    params = fe.BsmParams(sigma=_SIGMA)
    calls = fe.price_strip("bsm", "ctmc", _STRIKES, _FWD, params)
    puts = fe.price_strip("bsm", "ctmc", _STRIKES, _FWD, params, cp=-1)
    np.testing.assert_allclose(calls - puts, _FWD.disc * (_FWD.F0 - _STRIKES), atol=5e-3)
    with pytest.raises(ValueError):
        fe.price_strip("heston", "ctmc", _STRIKES, _FWD, params)


def test_validation_errors():
    with pytest.raises(ValueError):
        CTMCGrid(n_states=3)
    with pytest.raises(ValueError):
        CTMCGrid(width=-1.0)
    with pytest.raises(ValueError):
        ctmc_european_price(_FWD.S0, 100.0, _FWD.r, _FWD.q, _FWD.T, _SIGMA, cp=0)
    with pytest.raises(ValueError):
        ctmc_european_price(_FWD.S0, -5.0, _FWD.r, _FWD.q, _FWD.T, _SIGMA)
    with pytest.raises(ValueError):
        ctmc_american_price(_FWD.S0, 100.0, _FWD.r, _FWD.q, _FWD.T, _SIGMA, n_steps=0)
    with pytest.raises(ValueError):
        ctmc_european_price(_FWD.S0, 100.0, _FWD.r, _FWD.q, _FWD.T, -0.2)
