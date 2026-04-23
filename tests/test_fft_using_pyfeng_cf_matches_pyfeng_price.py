"""Our Fourier engines fed PyFENG's CF should agree with PyFENG's own price.

Why this is strong evidence:
  * The *model* (Heston, VG) comes from PyFENG.
  * The *CF evaluation* comes from PyFENG.
  * The *Fourier engine* (Carr-Madan / FRFT / COS) is ours.

If our strip prices agree with PyFENG's ``model.price(...)`` then our
Fourier engines are correct independently of any CF-derivation work
done in this project. That isolates the "engine" from the "model" in a
way the paper-anchored tests can't.

Skipped if PyFENG isn't installed.
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cumulants
from foureng.models.variance_gamma import VGParams, vg_cumulants
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.frft import frft_price_at_strikes
from foureng.pricers.cos import cos_prices, cos_auto_grid
from foureng.utils.grids import FFTGrid, FRFTGrid


pyfeng = pytest.importorskip(
    "pyfeng",
    reason="pyfeng not installed; engine-vs-PyFENG-price check skipped",
)


def _heston_bundle(lewis_heston):
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    m = pyfeng.HestonFft(sigma=p.v0, vov=p.nu, rho=p.rho, mr=p.kappa,
                          theta=p.theta, intr=fwd.r, divr=fwd.q)
    # PyFENG CF as ``phi(u)`` — independent of our own heston_cf wrapper.
    phi = lambda u: np.asarray(m.charfunc_logprice(np.asarray(u), texp=fwd.T),
                               dtype=np.complex128)
    return fwd, p, m, phi, np.asarray(d["strikes"], dtype=float)


def _vg_bundle(cm1999_vg):
    d = cm1999_vg
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = VGParams(sigma=d["sigma"], nu=d["nu"], theta=d["theta"])
    m = pyfeng.VarGammaFft(sigma=p.sigma, vov=p.nu, theta=p.theta,
                            intr=fwd.r, divr=fwd.q)
    phi = lambda u: np.asarray(m.charfunc_logprice(np.asarray(u), texp=fwd.T),
                               dtype=np.complex128)
    return fwd, p, m, phi, np.asarray(d["strikes"], dtype=float)


# --- Heston ---------------------------------------------------------------

def test_carr_madan_with_pyfeng_cf_matches_pyfeng_price_heston(lewis_heston):
    fwd, p, m, phi, K = _heston_bundle(lewis_heston)
    C_ours = carr_madan_price_at_strikes(phi, fwd, FFTGrid(4096, 0.25, 1.5), K)
    C_pf = np.asarray(m.price(K, spot=fwd.S0, texp=fwd.T, cp=1), dtype=float)
    err = float(np.max(np.abs(C_ours - C_pf)))
    assert err < 1e-4, f"CM+PyFENG CF vs PyFENG price (Heston): {err:.3e}"


def test_frft_with_pyfeng_cf_matches_pyfeng_price_heston(lewis_heston):
    fwd, p, m, phi, K = _heston_bundle(lewis_heston)
    grid = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)
    C_ours = frft_price_at_strikes(phi, fwd, grid, K)
    C_pf = np.asarray(m.price(K, spot=fwd.S0, texp=fwd.T, cp=1), dtype=float)
    err = float(np.max(np.abs(C_ours - C_pf)))
    assert err < 1e-4, f"FRFT+PyFENG CF vs PyFENG price (Heston): {err:.3e}"


def test_cos_with_pyfeng_cf_matches_pyfeng_price_heston(lewis_heston):
    fwd, p, m, phi, K = _heston_bundle(lewis_heston)
    grid = cos_auto_grid(heston_cumulants(fwd, p), N=256, L=10.0)
    C_ours = cos_prices(phi, fwd, K, grid).call_prices
    C_pf = np.asarray(m.price(K, spot=fwd.S0, texp=fwd.T, cp=1), dtype=float)
    err = float(np.max(np.abs(C_ours - C_pf)))
    assert err < 1e-4, f"COS+PyFENG CF vs PyFENG price (Heston): {err:.3e}"


# --- VG -------------------------------------------------------------------
# CM1999 Case 4 VG (T/nu = 0.125) is the tails-heavy regime. PyFENG's own
# FFT price floats around ~1e-3 vs the published table, and so does ours.
# Comparing our engine to PyFENG's price directly (same CF underneath)
# should be tighter since both are the same integrand. Keep 1e-3 here.

def test_carr_madan_with_pyfeng_cf_matches_pyfeng_price_vg(cm1999_vg):
    fwd, p, m, phi, K = _vg_bundle(cm1999_vg)
    C_ours = carr_madan_price_at_strikes(phi, fwd, FFTGrid(8192, 0.25, 1.5), K)
    C_pf = np.asarray(m.price(K, spot=fwd.S0, texp=fwd.T, cp=1), dtype=float)
    err = float(np.max(np.abs(C_ours - C_pf)))
    assert err < 1e-3, f"CM+PyFENG CF vs PyFENG price (VG): {err:.3e}"


def test_cos_with_pyfeng_cf_matches_pyfeng_price_vg(cm1999_vg):
    fwd, p, m, phi, K = _vg_bundle(cm1999_vg)
    grid = cos_auto_grid(vg_cumulants(fwd, p), N=2048, L=10.0)
    C_ours = cos_prices(phi, fwd, K, grid).call_prices
    C_pf = np.asarray(m.price(K, spot=fwd.S0, texp=fwd.T, cp=1), dtype=float)
    err = float(np.max(np.abs(C_ours - C_pf)))
    assert err < 1e-3, f"COS+PyFENG CF vs PyFENG price (VG): {err:.3e}"
