"""Hilbert-transform pricer (Feng & Linetsky 2008) validation.

The discrete Hilbert transform on the half-integer sinc grid is exponentially
convergent for strip-analytic characteristic functions, so under BSM it must
hit the closed form to near machine precision, and under Heston/Kou/VG it must
agree with the COS engine to tight tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

import foureng as fe
from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd
from foureng.pricers.hilbert import hilbert_itm_probabilities, hilbert_price_at_strikes
from foureng.utils.grids import HilbertGrid

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.01, T=1.0)
_BSM = fe.BsmParams(sigma=0.20)
_STRIKES = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0])


def _bs_reference(cp: int = 1) -> np.ndarray:
    calls = np.array(
        [
            bs_price_from_fwd(
                _BSM.sigma, BSInputs(F0=_FWD.F0, K=float(K), T=_FWD.T, r=_FWD.r, q=_FWD.q)
            )
            for K in _STRIKES
        ]
    )
    if cp == 1:
        return calls
    return calls - _FWD.disc * (_FWD.F0 - _STRIKES)


def test_hilbert_bsm_calls_match_closed_form():
    prices = fe.price_strip("bsm", "hilbert", _STRIKES, _FWD, _BSM)
    np.testing.assert_allclose(prices, _bs_reference(), atol=1e-8, rtol=0.0)


def test_hilbert_bsm_puts_match_closed_form():
    prices = fe.price_strip("bsm", "hilbert", _STRIKES, _FWD, _BSM, cp=-1)
    np.testing.assert_allclose(prices, _bs_reference(cp=-1), atol=1e-8, rtol=0.0)


def test_hilbert_bsm_probabilities_are_nd1_nd2():
    phi = lambda u: fe.bsm_cf(u, _FWD, _BSM)
    pi1, pi2 = hilbert_itm_probabilities(phi, _FWD, _STRIKES)
    sig_sqrt_t = _BSM.sigma * np.sqrt(_FWD.T)
    d1 = (np.log(_FWD.F0 / _STRIKES) + 0.5 * sig_sqrt_t**2) / sig_sqrt_t
    d2 = d1 - sig_sqrt_t
    np.testing.assert_allclose(pi1, norm.cdf(d1), atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(pi2, norm.cdf(d2), atol=1e-10, rtol=0.0)


def test_hilbert_put_call_parity():
    calls = fe.price_strip("bsm", "hilbert", _STRIKES, _FWD, _BSM, cp=1)
    puts = fe.price_strip("bsm", "hilbert", _STRIKES, _FWD, _BSM, cp=-1)
    np.testing.assert_allclose(calls - puts, _FWD.disc * (_FWD.F0 - _STRIKES), atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "model, params",
    [
        ("heston", fe.HestonParams(kappa=2.0, theta=0.04, nu=0.35, rho=-0.6, v0=0.04)),
        ("kou", fe.KouParams(sigma=0.2, lam=1.0, p=0.6, eta1=25.0, eta2=10.0)),
        ("vg", fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14)),
    ],
)
def test_hilbert_agrees_with_cos(model, params):
    hilbert = fe.price_strip(model, "hilbert", _STRIKES, _FWD, params)
    cos = fe.price_strip(model, "cos", _STRIKES, _FWD, params)
    np.testing.assert_allclose(hilbert, cos, atol=2e-6, rtol=0.0)


def test_hilbert_custom_grid_dispatch():
    grid = HilbertGrid(h=0.02, N=1 << 14)
    prices = fe.price_strip("bsm", "hilbert", _STRIKES, _FWD, _BSM, grid=grid)
    np.testing.assert_allclose(prices, _bs_reference(), atol=1e-9, rtol=0.0)


def test_hilbert_direct_call_matches_dispatch():
    phi = lambda u: fe.bsm_cf(u, _FWD, _BSM)
    direct = hilbert_price_at_strikes(phi, _FWD, _STRIKES, cp=1)
    routed = fe.price_strip("bsm", "hilbert", _STRIKES, _FWD, _BSM)
    np.testing.assert_allclose(direct, routed, atol=0.0, rtol=0.0)


def test_hilbert_grid_validation():
    with pytest.raises(ValueError):
        HilbertGrid(h=-0.1).u()
    with pytest.raises(ValueError):
        HilbertGrid(N=1).u()
    with pytest.raises(ValueError):
        hilbert_price_at_strikes(lambda u: u, _FWD, _STRIKES, cp=2)
    with pytest.raises(ValueError):
        hilbert_price_at_strikes(lambda u: u, _FWD, np.array([-1.0]))
