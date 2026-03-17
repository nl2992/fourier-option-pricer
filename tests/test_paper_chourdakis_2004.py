"""Chourdakis (2004) fractional-FFT tests."""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cf_form2
from foureng.models.variance_gamma import VGParams, vg_cf
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.frft import frft_price_at_strikes
from foureng.utils.grids import FFTGrid, FRFTGrid


pytestmark = [pytest.mark.paper, pytest.mark.derived_reference, pytest.mark.numerical_stability]


def _heston_lewis_phi(d):
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"]
    )
    return fwd, lambda u: heston_cf_form2(u, fwd, params)


def test_frft_matches_fft_at_nyquist_zeta(lewis_heston):
    """With the Nyquist strike step, FRFT should collapse to CM-FFT."""
    d = lewis_heston
    fwd, phi = _heston_lewis_phi(d)
    N, eta, alpha = 4096, 0.25, 1.5
    lam = 2.0 * np.pi / (N * eta)

    fft_grid = FFTGrid(N=N, eta=eta, alpha=alpha)
    frft_grid = FRFTGrid(N=N, eta=eta, lam=lam, alpha=alpha)

    calls_fft = carr_madan_price_at_strikes(phi, fwd, fft_grid, d["strikes"])
    calls_frft = frft_price_at_strikes(phi, fwd, frft_grid, d["strikes"])
    err = np.abs(calls_fft - calls_frft).max()
    assert err < 1e-8, f"FRFT(zeta=1/N) vs FFT max err = {err:.3e}"


def test_frft_lewis_N128_hits_1e4(lewis_heston):
    """Lewis Heston: FRFT with N=128 should hit 1e-4."""
    d = lewis_heston
    fwd, phi = _heston_lewis_phi(d)
    grid = FRFTGrid(N=128, eta=0.25, lam=0.005, alpha=1.5)
    calls = frft_price_at_strikes(phi, fwd, grid, d["strikes"])
    err = np.abs(calls - d["ref_calls"]).max()
    assert err < 1e-4, f"FRFT N=128 vs Lewis max err = {err:.3e}"


def test_frft_beats_cmfft_at_matched_N(lewis_heston):
    """At matched N, FRFT should beat CM-FFT on the Lewis Heston strip."""
    d = lewis_heston
    fwd, phi = _heston_lewis_phi(d)
    strikes = d["strikes"]

    calls_fft = carr_madan_price_at_strikes(
        phi,
        fwd,
        FFTGrid(N=64, eta=0.25, alpha=1.5),
        strikes,
    )
    calls_frft = frft_price_at_strikes(
        phi,
        fwd,
        FRFTGrid(N=64, eta=0.5, lam=0.02, alpha=1.5),
        strikes,
    )

    err_fft = np.abs(calls_fft - d["ref_calls"]).max()
    err_frft = np.abs(calls_frft - d["ref_calls"]).max()
    assert err_frft < err_fft / 10.0, (
        f"FRFT N=64 err={err_frft:.3e} not 10x better than CM-FFT N=64 err={err_fft:.3e}"
    )
    assert err_frft < 1e-2, f"FRFT N=64 absolute err = {err_frft:.3e}"


def test_frft_cm1999_vg_case4(cm1999_vg):
    """CM1999 Case 4 VG: FRFT should reach table precision by N=1024."""
    d = cm1999_vg
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = VGParams(sigma=d["sigma"], nu=d["nu"], theta=d["theta"])
    phi = lambda u: vg_cf(u, fwd, params)

    grid = FRFTGrid(N=1024, eta=0.25, lam=0.005, alpha=1.5)
    calls = frft_price_at_strikes(phi, fwd, grid, d["strikes"])
    puts = calls - d["S0"] * np.exp(-d["q"] * d["T"]) + d["strikes"] * np.exp(-d["r"] * d["T"])
    err = np.abs(puts - d["ref_puts"]).max()
    assert err < 1e-3, f"FRFT VG N=1024 vs CM1999 max err = {err:.3e}"
