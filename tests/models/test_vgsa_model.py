"""Structural tests for the VGSA (Carr-Geman-Madan-Yor 2003) CF.

These tests validate model-level properties independently of any
paper benchmark:

1. CF boundary conditions  phi(0) = 1, phi(-i) = 1
2. Deterministic-clock reduction   lam=0, C=eta → exp(T*(psi(u)-iu*psi(-i)))
3. VG reduction   lam=0, C=eta=1 → standard VG(G,M) CF (machine precision)
4. FRFT / CM cross-method consistency (≤ 1e-3)
5. No-arbitrage price bounds
6. Parameter validation
"""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.vgsa import VGSAParams, _vg_levy_exponent, vgsa_cf
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vgsa_params():
    """Representative VGSA parameters (moderate activity clustering)."""
    return VGSAParams(C=1.0, G=5.0, M=10.0, kappa=1.5, eta=1.0, lam=1.0)


@pytest.fixture
def vgsa_fwd():
    return ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.5)


@pytest.fixture
def vgsa_strikes():
    return np.array([90.0, 95.0, 100.0, 105.0, 110.0])


# ---------------------------------------------------------------------------
# CF boundary conditions
# ---------------------------------------------------------------------------

def test_phi_at_zero_is_one(vgsa_params, vgsa_fwd):
    phi = vgsa_cf(np.array([0.0 + 0j]), vgsa_fwd, vgsa_params)
    assert abs(phi[0] - 1.0) < 1e-14, f"phi(0) = {phi[0]}, expected 1"


def test_phi_at_minus_i_is_one(vgsa_params, vgsa_fwd):
    phi = vgsa_cf(np.array([-1j]), vgsa_fwd, vgsa_params)
    assert abs(phi[0] - 1.0) < 1e-12, f"phi(-i) = {phi[0]}, expected 1"


def test_phi_at_zero_is_one_deterministic(vgsa_fwd):
    p = VGSAParams(C=1.0, G=5.0, M=10.0, kappa=1.5, eta=1.0, lam=0.0)
    phi = vgsa_cf(np.array([0.0 + 0j]), vgsa_fwd, p)
    assert abs(phi[0] - 1.0) < 1e-14


def test_phi_at_minus_i_is_one_deterministic(vgsa_fwd):
    p = VGSAParams(C=1.0, G=5.0, M=10.0, kappa=1.5, eta=1.0, lam=0.0)
    phi = vgsa_cf(np.array([-1j]), vgsa_fwd, p)
    assert abs(phi[0] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# VG reduction: lam=0, C=eta=1 → phi = exp(T*(psi_VG(u) - iu*psi_VG(-i)))
# ---------------------------------------------------------------------------

def test_vgsa_reduces_to_vg_when_deterministic_clock():
    """With lam=0 and C=eta=1 the VGSA CF equals the VG(G,M) CF to machine precision."""
    G, M, T = 5.0, 10.0, 1.0
    fwd = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=T)
    p = VGSAParams(C=1.0, G=G, M=M, kappa=1.5, eta=1.0, lam=0.0)

    u = np.array([1.0, 2.0, 3.0, -0.5, 0.5, 0.1, 4.0])
    psi_u  = _vg_levy_exponent(u, G, M)
    psi_mi = _vg_levy_exponent(np.array([-1j]), G, M)[0]
    phi_vg_direct = np.exp(T * (psi_u - 1j * u * psi_mi))
    phi_vgsa = vgsa_cf(u, fwd, p)

    err = float(np.max(np.abs(phi_vgsa - phi_vg_direct)))
    assert err < 1e-14, (
        f"VGSA(lam=0, C=eta=1) vs VG(G,M) direct CF: max|err| = {err:.3e}"
    )


# ---------------------------------------------------------------------------
# Cross-method consistency (FRFT vs Carr-Madan)
# ---------------------------------------------------------------------------

def test_frft_carr_madan_consistency(vgsa_params, vgsa_fwd, vgsa_strikes):
    grid_frft = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)
    grid_cm   = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    calls_frft = price_strip("vgsa", "frft",        vgsa_strikes, vgsa_fwd, vgsa_params, grid=grid_frft)
    calls_cm   = price_strip("vgsa", "carr_madan",  vgsa_strikes, vgsa_fwd, vgsa_params, grid=grid_cm)
    err = float(np.max(np.abs(calls_frft - calls_cm)))
    assert err < 1e-3, f"VGSA FRFT vs CM max|err| = {err:.3e}"


# ---------------------------------------------------------------------------
# No-arbitrage bounds
# ---------------------------------------------------------------------------

def test_call_price_bounds(vgsa_params, vgsa_fwd, vgsa_strikes):
    grid = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)
    calls = price_strip("vgsa", "frft", vgsa_strikes, vgsa_fwd, vgsa_params, grid=grid)
    df = np.exp(-vgsa_fwd.r * vgsa_fwd.T)
    assert np.all(calls > 0), "Some VGSA call prices are non-positive"
    assert np.all(calls < vgsa_fwd.F0 * df + 1e-6), "Some VGSA calls exceed forward PV"


def test_put_positivity(vgsa_params, vgsa_fwd, vgsa_strikes):
    grid = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)
    calls = price_strip("vgsa", "frft", vgsa_strikes, vgsa_fwd, vgsa_params, grid=grid)
    df = np.exp(-vgsa_fwd.r * vgsa_fwd.T)
    puts = calls - df * (vgsa_fwd.F0 - vgsa_strikes)
    assert np.all(puts > -1e-6), "Some VGSA puts are negative"


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def test_invalid_C_raises():
    with pytest.raises(ValueError, match="C"):
        VGSAParams(C=0.0, G=5.0, M=10.0, kappa=1.5, eta=1.0, lam=0.5)


def test_invalid_M_raises():
    with pytest.raises(ValueError, match="M"):
        VGSAParams(C=1.0, G=5.0, M=0.5, kappa=1.5, eta=1.0, lam=0.5)


def test_invalid_G_raises():
    with pytest.raises(ValueError, match="G"):
        VGSAParams(C=1.0, G=0.0, M=10.0, kappa=1.5, eta=1.0, lam=0.5)


def test_invalid_kappa_raises():
    with pytest.raises(ValueError, match="kappa"):
        VGSAParams(C=1.0, G=5.0, M=10.0, kappa=-1.0, eta=1.0, lam=0.5)
