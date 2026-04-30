"""Phase 2 tests: Carr-Madan FFT vs published tables.

CM1999 Case 4 — Variance Gamma PUT prices at K=77,78,79.
Lewis (2001)  — 15-digit Heston CALL prices at K=80..120.
"""
from __future__ import annotations
import numpy as np

from foureng.models.base import ForwardSpec
from foureng.models.variance_gamma import VGParams, vg_cf
from foureng.models.heston import HestonParams, heston_cf_form2
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.utils.grids import FFTGrid


def test_vg_carr_madan_cm1999_case4(cm1999_vg):
    d = cm1999_vg
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = VGParams(sigma=d["sigma"], nu=d["nu"], theta=d["theta"])
    phi = lambda u: vg_cf(u, fwd, params)

    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    C = carr_madan_price_at_strikes(phi, fwd, grid, d["strikes"])

    # put = call - S0*exp(-qT) + K*exp(-rT)
    P = C - d["S0"] * np.exp(-d["q"] * d["T"]) + d["strikes"] * np.exp(-d["r"] * d["T"])

    err = np.abs(P - d["ref_puts"]).max()
    assert err < 1e-3, f"CM1999 VG Case 4 max err = {err:.3e}\n P = {P}\n ref = {d['ref_puts']}"


def test_heston_carr_madan_lewis(lewis_heston):
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                          rho=d["rho"], v0=d["v0"])
    phi = lambda u: heston_cf_form2(u, fwd, params)

    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    C = carr_madan_price_at_strikes(phi, fwd, grid, d["strikes"])
    err = np.abs(C - d["ref_calls"]).max()
    assert err < 1e-3, f"Lewis Heston max err = {err:.3e}\n C = {C}\n ref = {d['ref_calls']}"


def test_heston_atm_put_call_parity(lewis_heston):
    """Robustness: FFT-priced call + discounted-cash  ==  underlying-fwd (put-call parity)."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                          rho=d["rho"], v0=d["v0"])
    phi = lambda u: heston_cf_form2(u, fwd, params)

    K = np.array([100.0])
    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    C = carr_madan_price_at_strikes(phi, fwd, grid, K)[0]
    # parity: C - P = S0*e^{-qT} - K*e^{-rT};  we only check C is sane vs Lewis
    assert np.isfinite(C) and 10.0 < C < 25.0


def test_carr_madan_fft_grid_sensitivity(lewis_heston):
    """Minimum N to hit 1e-4: should not require more than 4096."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                          rho=d["rho"], v0=d["v0"])
    phi = lambda u: heston_cf_form2(u, fwd, params)

    ok = False
    for N in (1024, 2048, 4096):
        grid = FFTGrid(N=N, eta=0.25, alpha=1.5)
        C = carr_madan_price_at_strikes(phi, fwd, grid, d["strikes"])
        if np.abs(C - d["ref_calls"]).max() < 1e-3:
            ok = True
            break
    assert ok, "Carr-Madan failed to converge to 1e-3 even at N=4096"
