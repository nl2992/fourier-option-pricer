"""Paper benchmark: OUSV (Schöbel-Zhu) call prices.

Reference
---------
Schöbel, R. & Zhu, J. (1999), "Stochastic Volatility With an
Ornstein-Uhlenbeck Process: An Extension", *European Finance Review*, 3,
23-46.

The OUSV model (also called the Schöbel-Zhu model) is the ``ousv`` entry in
this project's registry.  Unlike Heston, the *instantaneous volatility*
(not variance) follows the OU process:

    d sigma_t = kappa * (theta - sigma_t) dt + nu dW_v

Two representative parameter sets are tested:

Set 1 (standard benchmark, well-centred)
    sigma0=0.20, kappa=2.0, theta=0.20, nu=0.3, rho=-0.5
    T=1, S0=100, r=0.05, q=0

Set 2 (high-skew parameterisation)
    sigma0=0.20, kappa=1.5, theta=0.15, nu=0.3, rho=-0.7
    T=1, S0=100, r=0.05, q=0

Reference calls (Lewis integration, 8 d.p.)
--------------------------------------------
K = [85, 90, 95, 100, 105, 110, 115]

set1: [21.77964753, 18.04888961, 14.61092135, 11.51807067,
        8.82076618,  6.55822854,  4.74465290]
set2: [21.69339919, 17.86026535, 14.28606016, 11.02696613,
        8.14700627,  5.71282600,  3.77863658]

Tolerance: 1e-3 (COS, CM, FRFT vs Lewis reference).
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.ousv import OusvParams
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
    "sz1999_bench": (
        OusvParams(sigma0=0.20, kappa=2.0, theta=0.20, nu=0.3, rho=-0.5),
        np.array(
            [21.77964753, 18.04888961, 14.61092135, 11.51807067, 8.82076618, 6.55822854, 4.74465290]
        ),
    ),
    "sz1999_hiskew": (
        OusvParams(sigma0=0.20, kappa=1.5, theta=0.15, nu=0.3, rho=-0.7),
        np.array(
            [21.69339919, 17.86026535, 14.28606016, 11.02696613, 8.14700627, 5.71282600, 3.77863658]
        ),
    ),
}

_GRID_CM = FFTGrid(N=4096, eta=0.25, alpha=1.5)
_GRID_FRFT = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)


# ---------------------------------------------------------------------------
# Cross-method tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_ousv_calls_cos(name, params, ref):
    """COS prices match Lewis reference within 1e-3."""
    calls = price_strip("ousv", "cos", _STRIKES, _FWD, params)
    err = float(np.max(np.abs(calls - ref)))
    assert err < _TOL, (
        f"OUSV({name}) COS max|err|={err:.2e} > {_TOL:.2e}\n  calls={calls}\n  ref  ={ref}"
    )


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_ousv_calls_carr_madan(name, params, ref):
    """Carr-Madan FFT matches Lewis reference within 1e-3."""
    calls = price_strip("ousv", "carr_madan", _STRIKES, _FWD, params, grid=_GRID_CM)
    err = float(np.max(np.abs(calls - ref)))
    assert err < _TOL, f"OUSV({name}) CM max|err|={err:.2e} > {_TOL:.2e}"


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_ousv_calls_frft(name, params, ref):
    """FRFT matches Lewis reference within 1e-3."""
    calls = price_strip("ousv", "frft", _STRIKES, _FWD, params, grid=_GRID_FRFT)
    err = float(np.max(np.abs(calls - ref)))
    assert err < _TOL, f"OUSV({name}) FRFT max|err|={err:.2e} > {_TOL:.2e}"


# ---------------------------------------------------------------------------
# OUSV-specific properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_ousv_put_call_parity(name, params, ref):
    """Puts from PCP must be non-negative."""
    calls = price_strip("ousv", "cos", _STRIKES, _FWD, params)
    df = np.exp(-_FWD.r * _FWD.T)
    puts = calls - df * (_FWD.F0 - _STRIKES)
    assert np.all(puts >= -1e-6), f"OUSV({name}) negative puts from PCP: {puts}"


def test_ousv_skew_effect_on_otm():
    """More negative rho → lower OTM call (left-skew from leverage effect)."""
    K_otm = np.array([115.0])
    p_lo_skew = OusvParams(sigma0=0.2, kappa=2.0, theta=0.2, nu=0.3, rho=-0.3)
    p_hi_skew = OusvParams(sigma0=0.2, kappa=2.0, theta=0.2, nu=0.3, rho=-0.7)
    c_lo = price_strip("ousv", "cos", K_otm, _FWD, p_lo_skew)[0]
    c_hi = price_strip("ousv", "cos", K_otm, _FWD, p_hi_skew)[0]
    assert c_hi < c_lo, (
        f"OTM call should decrease with more negative rho: rho=-0.3→{c_lo:.4f}, rho=-0.7→{c_hi:.4f}"
    )


def test_ousv_vs_pyfeng_fft_agreement():
    """OUSV COS agrees with PyFENG's own FFT pricer to 1e-3."""
    p = OusvParams(sigma0=0.20, kappa=2.0, theta=0.20, nu=0.3, rho=-0.5)
    calls_cos = price_strip("ousv", "cos", _STRIKES, _FWD, p)
    calls_pf = price_strip("ousv", "pyfeng_fft", _STRIKES, _FWD, p)
    err = float(np.max(np.abs(calls_cos - calls_pf)))
    assert err < 1e-3, f"OUSV COS vs pyfeng_fft max|err|={err:.2e}"
