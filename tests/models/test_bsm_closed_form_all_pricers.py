"""BSM all-pricer closed-form baseline.

Every pricing method in the package must agree with the Black-Scholes closed-form
price to within its stated numerical precision. This is the most basic sanity
check: if any method fails here, the wiring is broken.

Tolerances (single-maturity, 7-strike strip, no stress parameters):
    cos, cos_improved    1e-8
    lewis                1e-7
    carr_madan           1e-4
    frft                 1e-4
    pyfeng_fft           1e-6
"""
from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd
from foureng.pricers.lewis import lewis_call_prices

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.01, T=1.0)
_PARAMS = fe.BsmParams(sigma=0.20)
_STRIKES = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0])


def _bs_reference():
    """Closed-form Black-Scholes call prices from bs_price_from_fwd."""
    F0 = _FWD.F0
    r, T = _FWD.r, _FWD.T
    sigma = _PARAMS.sigma
    return np.array([
        bs_price_from_fwd(sigma, BSInputs(F0=F0, K=float(K), T=T, r=r, q=_FWD.q))
        for K in _STRIKES
    ])


_REF = _bs_reference()


def test_bsm_cos():
    prices = fe.price_strip("bsm", "cos", _STRIKES, _FWD, _PARAMS)
    np.testing.assert_allclose(prices, _REF, atol=1e-8, rtol=0.0)


def test_bsm_cos_improved():
    prices = fe.price_strip("bsm", "cos_improved", _STRIKES, _FWD, _PARAMS)
    np.testing.assert_allclose(prices, _REF, atol=1e-8, rtol=0.0)


def test_bsm_lewis():
    phi = lambda u: fe.bsm_cf(u, _FWD, _PARAMS)
    prices = lewis_call_prices(
        phi,
        _STRIKES,
        spot=_FWD.S0,
        texp=_FWD.T,
        intr=_FWD.r,
        divr=_FWD.q,
        method="trapz",
        u_max=200.0,
        n_u=8192,
    )
    np.testing.assert_allclose(prices, _REF, atol=1e-7, rtol=0.0)


def test_bsm_carr_madan():
    grid = fe.FFTGrid(N=16384, eta=0.05, alpha=1.5)
    prices = fe.price_strip("bsm", "carr_madan", _STRIKES, _FWD, _PARAMS, grid=grid)
    np.testing.assert_allclose(prices, _REF, atol=1e-4, rtol=0.0)


def test_bsm_frft():
    grid = fe.FRFTGrid(N=4096, eta=0.05, lam=0.01, alpha=1.5)
    prices = fe.price_strip("bsm", "frft", _STRIKES, _FWD, _PARAMS, grid=grid)
    np.testing.assert_allclose(prices, _REF, atol=1e-4, rtol=0.0)


def test_bsm_pyfeng_fft():
    pytest.importorskip("pyfeng", reason="pyfeng not installed")
    prices = fe.price_strip("bsm", "pyfeng_fft", _STRIKES, _FWD, _PARAMS)
    np.testing.assert_allclose(prices, _REF, atol=1e-6, rtol=0.0)
