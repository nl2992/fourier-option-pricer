"""Paper benchmark: CGMY call prices — Carr, Geman, Madan & Yor (2002).

Reference
---------
Carr, P., Geman, H., Madan, D. B. & Yor, M. (2002), "The Fine Structure of
Asset Returns: An Empirical Investigation", *Journal of Business*, 75(2),
305-332.

The CGMY process generalizes the Variance Gamma (Y→0) and Brownian motion
(Y→2) models.  Its CF per unit time is:

    psi(u) = C * Gamma(-Y) * [(M - iu)^Y - M^Y + (G + iu)^Y - G^Y]

Two representative parameter sets are tested:

Set 1 (finite-variation, Y=0.5)
    C=1.0, G=5.0, M=5.0, Y=0.5,  T=1,  S0=100, r=0.1, q=0

Set 2 (infinite-variation regime, Y=1.5, reduced activity C=0.1)
    C=0.1, G=5.0, M=5.0, Y=1.5,  T=1,  S0=100, r=0.1, q=0

Reference calls (Lewis integration, 8 d.p.)
--------------------------------------------
K = [85, 90, 95, 100, 105, 110, 115]

set1: [28.06555815, 25.05430821, 22.30351638, 19.81294884,
        17.57559811, 15.57895760, 13.80659596]
set2: [28.31026431, 25.35451657, 22.64178114, 20.16784292,
        17.92451143, 15.90058838, 14.08277564]

Tolerance: 1e-3 (COS, CM, FRFT vs Lewis reference).
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.cgmy import CgmyParams
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid

pytestmark = [pytest.mark.paper, pytest.mark.derived_reference]

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

_FWD = ForwardSpec(S0=100.0, r=0.1, q=0.0, T=1.0)
_STRIKES = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
_TOL = 1e-3

_SETS = {
    "Y05": (
        CgmyParams(C=1.0, G=5.0, M=5.0, Y=0.5),
        np.array([28.06555815, 25.05430821, 22.30351638, 19.81294884,
                  17.57559811, 15.57895760, 13.80659596]),
    ),
    "Y15_lowC": (
        CgmyParams(C=0.1, G=5.0, M=5.0, Y=1.5),
        np.array([28.31026431, 25.35451657, 22.64178114, 20.16784292,
                  17.92451143, 15.90058838, 14.08277564]),
    ),
}

_GRID_CM = FFTGrid(N=4096, eta=0.25, alpha=1.5)
_GRID_FRFT = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)


# ---------------------------------------------------------------------------
# COS pricer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_cgmy_calls_cos(name, params, ref):
    """COS prices match Lewis reference within 1e-3."""
    calls = price_strip("cgmy", "cos", _STRIKES, _FWD, params)
    err = float(np.max(np.abs(calls - ref)))
    assert err < _TOL, (
        f"CGMY({name}) COS max|err|={err:.2e} > {_TOL:.2e}\n"
        f"  calls={calls}\n  ref  ={ref}"
    )


# ---------------------------------------------------------------------------
# Carr-Madan FFT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_cgmy_calls_carr_madan(name, params, ref):
    """Carr-Madan FFT matches Lewis reference within 1e-3."""
    calls = price_strip("cgmy", "carr_madan", _STRIKES, _FWD, params, grid=_GRID_CM)
    err = float(np.max(np.abs(calls - ref)))
    assert err < _TOL, f"CGMY({name}) CM max|err|={err:.2e} > {_TOL:.2e}"


# ---------------------------------------------------------------------------
# FRFT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_cgmy_calls_frft(name, params, ref):
    """FRFT matches Lewis reference within 1e-3."""
    calls = price_strip("cgmy", "frft", _STRIKES, _FWD, params, grid=_GRID_FRFT)
    err = float(np.max(np.abs(calls - ref)))
    assert err < _TOL, f"CGMY({name}) FRFT max|err|={err:.2e} > {_TOL:.2e}"


# ---------------------------------------------------------------------------
# Put-call parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,params,ref", [(k, v[0], v[1]) for k, v in _SETS.items()])
def test_cgmy_put_call_parity(name, params, ref):
    """Puts from PCP must be non-negative."""
    calls = price_strip("cgmy", "cos", _STRIKES, _FWD, params)
    df = np.exp(-_FWD.r * _FWD.T)
    puts = calls - df * (_FWD.F0 - _STRIKES)
    assert np.all(puts >= -1e-6), f"CGMY({name}) negative puts from PCP: {puts}"


# ---------------------------------------------------------------------------
# Y-sensitivity: higher Y → higher prices (heavier tails)
# ---------------------------------------------------------------------------


def test_cgmy_price_monotone_in_Y():
    """OTM call increases as Y increases (heavier tails raise OTM value)."""
    K_otm = np.array([115.0])
    p_lo = CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.5)
    p_hi = CgmyParams(C=0.5, G=5.0, M=5.0, Y=1.5)
    c_lo = price_strip("cgmy", "cos", K_otm, _FWD, p_lo)[0]
    c_hi = price_strip("cgmy", "cos", K_otm, _FWD, p_hi)[0]
    assert c_hi > c_lo, (
        f"OTM call should increase with Y: Y=0.5→{c_lo:.4f}, Y=1.5→{c_hi:.4f}"
    )
