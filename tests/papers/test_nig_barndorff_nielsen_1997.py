"""Paper benchmark: NIG call prices — Barndorff-Nielsen (1997).

Reference
---------
Barndorff-Nielsen, O. E. (1997), "Normal inverse Gaussian distributions and
stochastic volatility modelling", *Scand. J. Statist.*, 24(1), 1-13.

Also:
Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
CRC Press, Table 14.2 (NIG calibrated to S&P500 option data).

The NIG model (sigma, nu, theta parameterization) is backed by PyFENG's
:class:`ExpNigFft`.  Two representative parameter sets are tested:

Set 1 (typical equity calibration, Cont-Tankov Table 14.2 style)
    sigma=0.12, nu=1.2, theta=-0.14;  T=1, S0=100, r=0.05, q=0

Set 2 (moderate parameters, cross-check set)
    sigma=0.20, nu=0.50, theta=-0.10;  T=1, S0=100, r=0.05, q=0

Reference calls (Lewis integration, 8 d.p.)
--------------------------------------------
K = [85, 90, 95, 100, 105, 110, 115]

set1: [20.77216377, 16.65562212, 12.77302902,  9.21213483,
        6.09372200,  3.57103348,  1.78888216]
set2: [20.88474727, 17.02392471, 13.51204716, 10.42310712,
        7.81284024,  5.70271081,  4.07152579]

Tolerance: 1e-3 (COS, CM, FRFT vs Lewis reference).
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.nig import NigParams
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid

pytestmark = [pytest.mark.paper, pytest.mark.derived_reference]

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

_FWD = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
_STRIKES = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
_TOL = 1e-3

_SETS = {
    "ct2004_calibrated": (
        NigParams(sigma=0.12, nu=1.2, theta=-0.14),
        np.array([20.77216377, 16.65562212, 12.77302902,  9.21213483,
                   6.09372200,  3.57103348,  1.78888216]),
    ),
    "moderate": (
        NigParams(sigma=0.20, nu=0.50, theta=-0.10),
        np.array([20.88474727, 17.02392471, 13.51204716, 10.42310712,
                   7.81284024,  5.70271081,  4.07152579]),
    ),
}

_GRID_CM = FFTGrid(N=4096, eta=0.25, alpha=1.5)
_GRID_FRFT = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)


# ---------------------------------------------------------------------------
# Cross-method tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_nig_calls_cos(name, params, ref):
    """COS prices match Lewis reference within 1e-3."""
    calls = price_strip("nig", "cos", _STRIKES, _FWD, params)
    err = float(np.max(np.abs(calls - ref)))
    assert err < _TOL, (
        f"NIG({name}) COS max|err|={err:.2e} > {_TOL:.2e}\n"
        f"  calls={calls}\n  ref  ={ref}"
    )


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_nig_calls_carr_madan(name, params, ref):
    """Carr-Madan FFT matches Lewis reference within 1e-3."""
    calls = price_strip("nig", "carr_madan", _STRIKES, _FWD, params, grid=_GRID_CM)
    err = float(np.max(np.abs(calls - ref)))
    assert err < _TOL, f"NIG({name}) CM max|err|={err:.2e} > {_TOL:.2e}"


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_nig_calls_frft(name, params, ref):
    """FRFT matches Lewis reference within 1e-3."""
    calls = price_strip("nig", "frft", _STRIKES, _FWD, params, grid=_GRID_FRFT)
    err = float(np.max(np.abs(calls - ref)))
    assert err < _TOL, f"NIG({name}) FRFT max|err|={err:.2e} > {_TOL:.2e}"


# ---------------------------------------------------------------------------
# NIG-specific properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_nig_put_call_parity(name, params, ref):
    """Puts from PCP must be non-negative."""
    calls = price_strip("nig", "cos", _STRIKES, _FWD, params)
    df = np.exp(-_FWD.r * _FWD.T)
    puts = calls - df * (_FWD.F0 - _STRIKES)
    assert np.all(puts >= -1e-6), f"NIG({name}) negative puts from PCP: {puts}"


def test_nig_call_prices_positive_and_monotone():
    """All NIG call prices must be positive and monotone decreasing in strike."""
    p = NigParams(sigma=0.20, nu=0.5, theta=-0.10)
    calls = price_strip("nig", "cos", _STRIKES, _FWD, p)
    assert np.all(calls > 0), f"NIG calls must be positive: {calls}"
    assert np.all(np.diff(calls) < 0), f"NIG calls must decrease with strike: {calls}"


def test_nig_sigma_raises_atm():
    """Higher sigma → higher ATM call (direct volatility effect)."""
    K_atm = np.array([_FWD.F0])
    p_lo = NigParams(sigma=0.10, nu=0.5, theta=-0.05)
    p_hi = NigParams(sigma=0.30, nu=0.5, theta=-0.05)
    c_lo = price_strip("nig", "cos", K_atm, _FWD, p_lo)[0]
    c_hi = price_strip("nig", "cos", K_atm, _FWD, p_hi)[0]
    assert c_hi > c_lo, (
        f"ATM call should increase with sigma: sigma=0.10→{c_lo:.4f}, sigma=0.30→{c_hi:.4f}"
    )
