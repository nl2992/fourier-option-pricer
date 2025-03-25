"""PyFENG CF wrappers — check our CF is exactly PyFENG's, convention-aligned.

We delegate Heston and VG CFs entirely to PyFENG's ``charfunc_logprice``
(see ``foureng/char_func/heston.py`` and ``variance_gamma.py``). That
wrapper is supposed to be a thin adapter: our function ``heston_cf(u)``
should return exactly what a freshly-constructed PyFENG model's
``charfunc_logprice(u, texp=T)`` returns.

These tests lock that claim down. Failure here means either:
  * a PyFENG upgrade changed convention (log-forward vs log-spot) — in
    which case flip the documented no-op phase-shift hook in our wrapper;
  * or our wrapper picked up a bug (wrong kwarg name, stale cache).

Skipped if PyFENG isn't installed.
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.char_func.base import ForwardSpec
from foureng.char_func.heston import HestonParams, heston_cf
from foureng.char_func.variance_gamma import VGParams, vg_cf


pyfeng = pytest.importorskip(
    "pyfeng",
    reason="pyfeng not installed; cross-library CF check skipped",
)


def test_heston_cf_matches_pyfeng_charfunc_logprice(lewis_heston):
    """Our ``heston_cf`` is exactly ``pf.HestonFft.charfunc_logprice``."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])

    u = np.linspace(-10.0, 10.0, 41)
    phi_ours = heston_cf(u, fwd, p)

    m = pyfeng.HestonFft(sigma=p.v0, vov=p.nu, rho=p.rho, mr=p.kappa,
                         theta=p.theta, intr=fwd.r, divr=fwd.q)
    phi_pyfeng = np.asarray(m.charfunc_logprice(u, texp=fwd.T),
                            dtype=np.complex128)

    err = float(np.max(np.abs(phi_ours - phi_pyfeng)))
    # Bit-identical up to float round-off — our wrapper does no extra math.
    assert err < 1e-14, f"Heston CF wrapper vs PyFENG: max|err| = {err:.3e}"


def test_vg_cf_matches_pyfeng_charfunc_logprice(cm1999_vg):
    """Our ``vg_cf`` is exactly ``pf.VarGammaFft.charfunc_logprice``."""
    d = cm1999_vg
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = VGParams(sigma=d["sigma"], nu=d["nu"], theta=d["theta"])

    u = np.linspace(-10.0, 10.0, 41)
    phi_ours = vg_cf(u, fwd, p)

    m = pyfeng.VarGammaFft(sigma=p.sigma, vov=p.nu, theta=p.theta,
                            intr=fwd.r, divr=fwd.q)
    phi_pyfeng = np.asarray(m.charfunc_logprice(u, texp=fwd.T),
                            dtype=np.complex128)

    err = float(np.max(np.abs(phi_ours - phi_pyfeng)))
    assert err < 1e-14, f"VG CF wrapper vs PyFENG: max|err| = {err:.3e}"


def test_heston_cf_martingale(lewis_heston):
    """``phi(u=0) = 1`` exactly — pure consistency check on our wrapper."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])

    phi0 = heston_cf(np.array([0.0]), fwd, p)[0]
    assert abs(phi0 - 1.0) < 1e-14, f"phi(0) = {phi0}, expected 1+0j"


def test_vg_cf_martingale(cm1999_vg):
    """``phi(u=0) = 1`` exactly on VG as well."""
    d = cm1999_vg
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = VGParams(sigma=d["sigma"], nu=d["nu"], theta=d["theta"])

    phi0 = vg_cf(np.array([0.0]), fwd, p)[0]
    assert abs(phi0 - 1.0) < 1e-14, f"phi(0) = {phi0}, expected 1+0j"
