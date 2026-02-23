"""Regression benchmark: VGSA call prices across four representative parameter sets.

Reference
---------
Carr, Geman, Madan & Yor (2003), "Stochastic Volatility for Lévy Processes",
Mathematical Finance, 13(3), 345–382 (CGMY 2003).
Carr, Geman, Madan & Yor (2002), "The Fine Structure of Asset Returns",
Journal of Business, 75(2), 305–332 (CGMY 2002, Table 9.2: VGSA estimates).

Four representative quarterly VGSA parameter sets, with call prices computed
via FRFT (N=4096) serving as the within-tolerance reference. Carr-Madan FFT
must agree to 1e-3. The four sets span a range of activity-clustering levels
(lam) and tail shapes (G, M).

Parameter sets (quarterly T = 0.25)
------------------------------------
    S0 = 100,  r = 0.05,  q = 0,  T = 0.25
    K  = [90, 95, 100, 105, 110]

    set1: C=0.4, G=4.0, M=15.0, kappa=1.5, eta=0.9, lam=0.8
    set2: C=0.6, G=6.0, M=12.0, kappa=2.0, eta=1.2, lam=1.2
    set3: C=1.0, G=5.0, M=10.0, kappa=1.5, eta=1.0, lam=1.0
    set4: C=0.8, G=7.0, M=20.0, kappa=1.0, eta=0.8, lam=0.5

FRFT reference calls (6 d.p.)
-------------------------------
    set1: [11.867905,  7.315162,  3.012896,  0.375016,  0.135240]
    set2: [11.676740,  7.134971,  2.935046,  0.698175,  0.320330]
    set3: [12.177284,  7.840511,  3.894353,  1.350993,  0.706396]
    set4: [11.558081,  6.980102,  2.749104,  0.356768,  0.102587]

Tolerance: 1e-3 (Carr-Madan FFT vs FRFT reference).
"""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.vgsa import VGSAParams
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid

pytestmark = [pytest.mark.paper, pytest.mark.derived_reference]

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

_FWD = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.25)
_STRIKES = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
_TOL = 1e-3

_SETS = {
    "set1": (
        VGSAParams(C=0.4, G=4.0, M=15.0, kappa=1.5, eta=0.9, lam=0.8),
        np.array([11.867905,  7.315162,  3.012896,  0.375016,  0.135240]),
    ),
    "set2": (
        VGSAParams(C=0.6, G=6.0, M=12.0, kappa=2.0, eta=1.2, lam=1.2),
        np.array([11.676740,  7.134971,  2.935046,  0.698175,  0.320330]),
    ),
    "set3": (
        VGSAParams(C=1.0, G=5.0, M=10.0, kappa=1.5, eta=1.0, lam=1.0),
        np.array([12.177284,  7.840511,  3.894353,  1.350993,  0.706396]),
    ),
    "set4": (
        VGSAParams(C=0.8, G=7.0, M=20.0, kappa=1.0, eta=0.8, lam=0.5),
        np.array([11.558081,  6.980102,  2.749104,  0.356768,  0.102587]),
    ),
}

_GRID_FRFT = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)
_GRID_CM   = FFTGrid(N=4096, eta=0.25, alpha=1.5)


# ---------------------------------------------------------------------------
# Tests — one parametrized test per method per set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(_SETS.keys()))
def test_vgsa_frft_calls(name):
    params, ref = _SETS[name]
    calls = price_strip("vgsa", "frft", _STRIKES, _FWD, params, grid=_GRID_FRFT)
    err = float(np.max(np.abs(calls - ref)))
    assert err < _TOL, (
        f"VGSA FRFT {name}: max|err| = {err:.3e} > {_TOL:.3e}\n"
        f"  calls = {calls}\n  ref   = {ref}"
    )


@pytest.mark.parametrize("name", list(_SETS.keys()))
@pytest.mark.slow
def test_vgsa_carr_madan_calls(name):
    params, ref = _SETS[name]
    calls = price_strip("vgsa", "carr_madan", _STRIKES, _FWD, params, grid=_GRID_CM)
    err = float(np.max(np.abs(calls - ref)))
    assert err < _TOL, (
        f"VGSA CM {name}: max|err| = {err:.3e} > {_TOL:.3e}\n"
        f"  calls = {calls}\n  ref   = {ref}"
    )


@pytest.mark.parametrize("name", list(_SETS.keys()))
def test_vgsa_phi_boundary(name):
    """phi(0)=1 and phi(-i)=1 for all parameter sets."""
    from foureng.models.vgsa import vgsa_cf

    params, _ = _SETS[name]
    u_check = np.array([0.0 + 0j, -1j])
    phi = vgsa_cf(u_check, _FWD, params)
    assert abs(phi[0] - 1.0) < 1e-12, f"{name}: phi(0) = {phi[0]}"
    assert abs(phi[1] - 1.0) < 1e-12, f"{name}: phi(-i) = {phi[1]}"
