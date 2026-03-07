"""Paper benchmark: Double Heston calls vs Kelly (2025).

Reference
---------
Kelly (2025) reports vanilla call prices for a two-factor Heston model
evaluated via Fourier inversion. We use the parameters and prices from
the paper's Table to validate our implementation to 4 decimal places
(the stated rounding precision of the reference values).

Parameters
----------
    S0 = 100,  r = 0.1,  q = 0,  T = 1
    kappa1 = 0.9,  theta1 = 0.10,  nu1 = 0.1,  rho1 = -0.5,  v01 = 0.36
    kappa2 = 1.2,  theta2 = 0.15,  nu2 = 0.2,  rho2 = -0.5,  v02 = 0.49

Reference calls (4 d.p.)
------------------------
    K  = [60, 80, 100, 110, 130]
    C  = [52.7623, 42.2321, 33.9625, 30.5234, 24.7718]

Tolerance: 5e-4 (one unit in the last stated decimal).
"""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.double_heston import DoubleHestonParams
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

_FWD = ForwardSpec(S0=100.0, r=0.1, q=0.0, T=1.0)

_PARAMS = DoubleHestonParams(
    kappa1=0.9,  theta1=0.10, nu1=0.1, rho1=-0.5, v01=0.36,
    kappa2=1.2,  theta2=0.15, nu2=0.2, rho2=-0.5, v02=0.49,
)

_STRIKES = np.array([60.0, 80.0, 100.0, 110.0, 130.0])

_REF_CALLS = np.array([52.7623, 42.2321, 33.9625, 30.5234, 24.7718])

_TOL = 5e-4


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_double_heston_calls_cos():
    calls = price_strip("double_heston", "cos", _STRIKES, _FWD, _PARAMS)
    err = float(np.max(np.abs(calls - _REF_CALLS)))
    assert err < _TOL, (
        f"Double Heston COS calls vs Kelly (2025): max|err| = {err:.2e} > {_TOL:.2e}\n"
        f"  calls = {calls}\n  ref   = {_REF_CALLS}"
    )


@pytest.mark.slow
def test_double_heston_calls_frft():
    grid = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)
    calls = price_strip("double_heston", "frft", _STRIKES, _FWD, _PARAMS, grid=grid)
    err = float(np.max(np.abs(calls - _REF_CALLS)))
    assert err < _TOL, (
        f"Double Heston FRFT calls vs Kelly (2025): max|err| = {err:.2e} > {_TOL:.2e}"
    )


@pytest.mark.slow
def test_double_heston_calls_carr_madan():
    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    calls = price_strip("double_heston", "carr_madan", _STRIKES, _FWD, _PARAMS, grid=grid)
    err = float(np.max(np.abs(calls - _REF_CALLS)))
    assert err < _TOL, (
        f"Double Heston CM calls vs Kelly (2025): max|err| = {err:.2e} > {_TOL:.2e}"
    )


def test_double_heston_puts_put_call_parity():
    """Puts via put-call parity must match reference within tolerance."""
    calls = price_strip("double_heston", "cos", _STRIKES, _FWD, _PARAMS)
    df = np.exp(-_FWD.r * _FWD.T)
    puts = calls - df * (_FWD.F0 - _STRIKES)
    _REF_PUTS = np.array([7.0526, 14.6191, 24.4463, 30.0556, 42.4007])
    err = float(np.max(np.abs(puts - _REF_PUTS)))
    assert err < _TOL, (
        f"Double Heston puts (PCP) vs Kelly (2025): max|err| = {err:.2e} > {_TOL:.2e}\n"
        f"  puts = {puts}\n  ref  = {_REF_PUTS}"
    )
