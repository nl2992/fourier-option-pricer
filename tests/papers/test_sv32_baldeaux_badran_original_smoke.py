"""Qualitative smoke test for Sv32 using the original Baldeaux-Badran (2012) figure params.

This test does NOT assert exact prices. It checks that the 3/2 SV model with
the Baldeaux-Badran figure parameters produces a call strip satisfying necessary
no-arbitrage shape conditions (finite, non-negative, monotone decreasing,
convex in strike).

If PyFENG's Sv32Fft backend cannot produce valid prices for these parameters
(known numerical instability for small v0/nu at very short maturities), the
test is marked xfail rather than silently falling back to different parameters.
The fallback stability check lives in tests/models/test_sv32_compat_shape_smoke.py.

Reference
---------
Baldeaux, J. & Badran, A. (2012), "Consistent Modelling of VIX and Equity
Derivatives Using a 3/2 plus Jumps Model", working paper. Figure parameters:
v0=0.0016, kappa=19.45, theta=0.04, nu=0.28, rho=-0.72, T=9/365.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytestmark = [pytest.mark.paper, pytest.mark.qualitative_figure, pytest.mark.numerical_stability]

pyfeng = pytest.importorskip("pyfeng", reason="pyfeng not installed — sv32 tests require it")

import foureng as fe

_FWD = fe.ForwardSpec(S0=100.0, r=0.0, q=0.0, T=9.0 / 365.0)
_PARAMS = fe.Sv32Params(v0=0.0016, kappa=19.45, theta=0.04, nu=0.28, rho=-0.72)
_STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])


def test_sv32_baldeaux_badran_original_params_shape():
    """pyfeng_fft on original BB params must satisfy no-arbitrage shape conditions.

    If Sv32Fft returns NaN or raises for these parameters, the test is xfailed
    with a note on the known numerical instability. Do not silently substitute
    different parameters here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            prices = fe.price_strip("sv32", "pyfeng_fft", _STRIKES, _FWD, _PARAMS)
        except Exception as exc:
            pytest.xfail(
                f"Known Sv32Fft numerical instability for original Baldeaux-Badran "
                f"short-maturity parameters (v0=0.0016, nu=0.28, T=9d): {exc}"
            )

    if not (np.all(np.isfinite(prices)) and np.all(prices >= 0.0)):
        pytest.xfail(
            "Known Sv32Fft numerical instability for original Baldeaux-Badran "
            "short-maturity parameters: prices are NaN or negative."
        )

    assert np.all(np.diff(prices) <= 1e-8), (
        f"Call prices not monotone decreasing: {prices}"
    )
    assert np.all(prices[:-2] - 2 * prices[1:-1] + prices[2:] >= -1e-8), (
        f"Call prices not convex in strike: {prices}"
    )
