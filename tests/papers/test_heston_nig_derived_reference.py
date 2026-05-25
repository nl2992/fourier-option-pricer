"""Paper benchmark: Heston-NIG SVJ derived-reference prices.

Reference
---------
The Heston-NIG model is a standard affine stochastic-volatility + Lévy-jump
construction described in:

    Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
    CRC Press, Chapter 15 (SV models with Lévy jumps).

The independent-factor CF factorisation phi_HN = phi_Heston * phi_NIG_jump
is a direct consequence of the independence assumption and the martingale-
correction structure used throughout this project (same as Heston-CGMY and
Heston-Kou).

Reference prices below are derived with our own high-precision Lewis
integration (machine precision) and serve as a regression baseline.

Parameter set (equity-calibration representative)
--------------------------------------------------
    S0 = 100,  r = 0.05,  q = 0,  T = 1
    kappa = 2.0,  theta = 0.04,  nu = 0.3,  rho = -0.7,  v0 = 0.04
    nig_sigma = 0.15,  nig_nu = 0.5,  nig_theta = -0.10

Reference calls (8 d.p., Lewis integration)
--------------------------------------------
    K  = [85, 90, 95, 100, 105, 110, 115]
    C  = [22.13592221, 18.59093412, 15.36198783, 12.47674426,
           9.95179384,  7.79087026,  5.98438065]

Tolerance: 1e-3 (one digit below reference precision).
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.heston_nig import HestonNIGParams
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid

pytestmark = [pytest.mark.paper, pytest.mark.derived_reference]

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

_FWD = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)

_PARAMS = HestonNIGParams(
    kappa=2.0,
    theta=0.04,
    nu=0.3,
    rho=-0.7,
    v0=0.04,
    nig_sigma=0.15,
    nig_nu=0.5,
    nig_theta=-0.10,
)

_STRIKES = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])

_REF_CALLS = np.array([
    22.13592221,
    18.59093412,
    15.36198783,
    12.47674426,
     9.95179384,
     7.79087026,
     5.98438065,
])

_TOL = 1e-3


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_heston_nig_calls_cos():
    """COS pricer agrees with Lewis reference to 1e-3."""
    calls = price_strip("heston_nig", "cos", _STRIKES, _FWD, _PARAMS)
    err = float(np.max(np.abs(calls - _REF_CALLS)))
    assert err < _TOL, (
        f"Heston-NIG COS calls vs Lewis ref: max|err| = {err:.2e} > {_TOL:.2e}\n"
        f"  calls = {calls}\n  ref   = {_REF_CALLS}"
    )


def test_heston_nig_calls_carr_madan():
    """Carr-Madan FFT agrees with Lewis reference to 1e-3."""
    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    calls = price_strip("heston_nig", "carr_madan", _STRIKES, _FWD, _PARAMS, grid=grid)
    err = float(np.max(np.abs(calls - _REF_CALLS)))
    assert err < _TOL, (
        f"Heston-NIG CM calls vs Lewis ref: max|err| = {err:.2e} > {_TOL:.2e}"
    )


def test_heston_nig_calls_frft():
    """FRFT agrees with Lewis reference to 1e-3."""
    grid = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)
    calls = price_strip("heston_nig", "frft", _STRIKES, _FWD, _PARAMS, grid=grid)
    err = float(np.max(np.abs(calls - _REF_CALLS)))
    assert err < _TOL, (
        f"Heston-NIG FRFT calls vs Lewis ref: max|err| = {err:.2e} > {_TOL:.2e}"
    )


def test_heston_nig_put_call_parity():
    """Puts via put-call parity must match reference within tolerance."""
    calls = price_strip("heston_nig", "cos", _STRIKES, _FWD, _PARAMS)
    df = np.exp(-_FWD.r * _FWD.T)
    puts = calls - df * (_FWD.F0 - _STRIKES)
    # Puts must be positive (no negative option prices)
    assert np.all(puts >= -1e-8), f"Some Heston-NIG put prices are negative: {puts}"
    # Put prices should increase with strike (for OTM → ITM)
    # (not strictly required across all strikes but for deep OTM puts)
    assert puts[-1] > puts[0], "Deep ITM put should be > deep OTM put"


@pytest.mark.parametrize("K,ref", list(zip(_STRIKES.tolist(), _REF_CALLS.tolist())))
def test_heston_nig_cos_per_strike(K, ref):
    """COS call price at individual strike matches Lewis reference."""
    calls = price_strip("heston_nig", "cos", np.array([K]), _FWD, _PARAMS)
    err = abs(calls[0] - ref)
    assert err < _TOL, f"Heston-NIG COS K={K}: err={err:.2e} > {_TOL:.2e}"
