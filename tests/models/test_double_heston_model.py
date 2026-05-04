"""Structural tests for the Double Heston (Christoffersen 2009) CF.

These tests validate model-level properties independently of any
paper benchmark:

1. CF boundary conditions  phi(0) = 1, phi(-i) = 1
2. Factor-swap symmetry    phi(kappa1,…,v01; kappa2,…,v02) == phi(kappa2,…,v02; kappa1,…,v01)
3. COS / FRFT / CM consistency  (cross-method, ≤ 1e-5 max absolute error)
4. No-arbitrage bounds     0 < call < F0 * discount, put > 0
"""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.double_heston import DoubleHestonParams, double_heston_cf
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dh_params():
    """Christoffersen (2009) two-factor parameters."""
    return DoubleHestonParams(
        kappa1=0.9, theta1=0.1, nu1=0.1, rho1=-0.5, v01=0.36,
        kappa2=1.2, theta2=0.15, nu2=0.2, rho2=-0.5, v02=0.49,
    )


@pytest.fixture
def dh_fwd():
    return ForwardSpec(S0=100.0, r=0.1, q=0.0, T=1.0)


@pytest.fixture
def dh_strikes():
    return np.array([60.0, 80.0, 100.0, 110.0, 130.0])


# ---------------------------------------------------------------------------
# CF boundary conditions
# ---------------------------------------------------------------------------

def test_phi_at_zero_is_one(dh_params, dh_fwd):
    phi = double_heston_cf(np.array([0.0 + 0j]), dh_fwd, dh_params)
    assert abs(phi[0] - 1.0) < 1e-14, f"phi(0) = {phi[0]}, expected 1"


def test_phi_at_minus_i_is_one(dh_params, dh_fwd):
    phi = double_heston_cf(np.array([-1j]), dh_fwd, dh_params)
    assert abs(phi[0] - 1.0) < 1e-12, f"phi(-i) = {phi[0]}, expected 1"


# ---------------------------------------------------------------------------
# Factor-swap symmetry
# ---------------------------------------------------------------------------

def test_factor_swap_symmetry(dh_fwd):
    p1 = DoubleHestonParams(
        kappa1=0.9, theta1=0.1, nu1=0.1, rho1=-0.5, v01=0.36,
        kappa2=1.2, theta2=0.15, nu2=0.2, rho2=-0.3, v02=0.04,
    )
    p2 = DoubleHestonParams(
        kappa1=1.2, theta1=0.15, nu1=0.2, rho1=-0.3, v01=0.04,
        kappa2=0.9, theta2=0.1, nu2=0.1, rho2=-0.5, v02=0.36,
    )
    u = np.linspace(0.1, 5.0, 20)
    phi1 = double_heston_cf(u, dh_fwd, p1)
    phi2 = double_heston_cf(u, dh_fwd, p2)
    err = float(np.max(np.abs(phi1 - phi2)))
    assert err < 1e-14, f"Factor-swap CF max|diff| = {err:.3e}"


# ---------------------------------------------------------------------------
# Cross-method consistency
# ---------------------------------------------------------------------------

def test_cos_frft_consistency(dh_params, dh_fwd, dh_strikes):
    grid = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)
    calls_cos  = price_strip("double_heston", "cos",  dh_strikes, dh_fwd, dh_params)
    calls_frft = price_strip("double_heston", "frft", dh_strikes, dh_fwd, dh_params, grid=grid)
    err = float(np.max(np.abs(calls_cos - calls_frft)))
    assert err < 1e-5, f"COS vs FRFT max|err| = {err:.3e}"


def test_cos_carr_madan_consistency(dh_params, dh_fwd, dh_strikes):
    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    calls_cos = price_strip("double_heston", "cos",          dh_strikes, dh_fwd, dh_params)
    calls_cm  = price_strip("double_heston", "carr_madan",   dh_strikes, dh_fwd, dh_params, grid=grid)
    err = float(np.max(np.abs(calls_cos - calls_cm)))
    assert err < 1e-5, f"COS vs CM max|err| = {err:.3e}"


# ---------------------------------------------------------------------------
# No-arbitrage price bounds
# ---------------------------------------------------------------------------

def test_call_price_bounds(dh_params, dh_fwd, dh_strikes):
    calls = price_strip("double_heston", "cos", dh_strikes, dh_fwd, dh_params)
    df = np.exp(-dh_fwd.r * dh_fwd.T)
    assert np.all(calls > 0), "Some Double Heston call prices are non-positive"
    assert np.all(calls < dh_fwd.F0 * df + 1e-8), "Some Double Heston calls exceed forward PV"


def test_put_call_parity(dh_params, dh_fwd, dh_strikes):
    calls = price_strip("double_heston", "cos", dh_strikes, dh_fwd, dh_params)
    df = np.exp(-dh_fwd.r * dh_fwd.T)
    puts_pcp = calls - df * (dh_fwd.F0 - dh_strikes)
    assert np.all(puts_pcp > -1e-8), "Put-call parity yields negative puts"


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def test_invalid_kappa_raises():
    with pytest.raises(ValueError, match="kappa"):
        DoubleHestonParams(kappa1=-1.0, theta1=0.1, nu1=0.1, rho1=-0.5, v01=0.04,
                           kappa2=1.0, theta2=0.1, nu2=0.1, rho2=-0.5, v02=0.04)


def test_invalid_nu_raises():
    with pytest.raises(ValueError, match="nu"):
        DoubleHestonParams(kappa1=1.0, theta1=0.1, nu1=0.0, rho1=-0.5, v01=0.04,
                           kappa2=1.0, theta2=0.1, nu2=0.1, rho2=-0.5, v02=0.04)
