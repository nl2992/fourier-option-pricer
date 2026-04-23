"""OUSV (Schöbel-Zhu) adapter — CF parity, cross-method, frozen strip.

OUSV does not have a clean closed-form log-forward CF we can check
against, so the gates are:

  1. ``ousv_cf`` matches :meth:`pyfeng.OusvFft.charfunc_logprice`
     bit-exactly (~1e-14) — our wrapper is a pure adapter.
  2. ``phi(u=0) = 1`` — martingale.
  3. COS / FRFT / CM all agree with PyFENG's own ``price`` to the
     same ~1e-7 floor we see for Heston/VG on their default grids.
  4. A frozen 41-strike regression strip pins prices against any
     future refactor (see :data:`OUSV_REGRESSION_STRIP_V1`).
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.ousv import OusvParams, ousv_cf
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid


pyfeng = pytest.importorskip("pyfeng", reason="OUSV adapter is PyFENG-backed")


# -- Canonical test params (shared with OUSV_REGRESSION_STRIP_V1) -----------
_FWD    = ForwardSpec(S0=100.0, r=0.03, q=0.0, T=1.0)
_PARAMS = OusvParams(sigma0=0.2, kappa=2.0, theta=0.2, nu=0.3, rho=-0.5)


def test_ousv_cf_matches_pyfeng_charfunc_logprice():
    """Bit-identity wrapper check — see models/ousv.py for the translation."""
    u = np.linspace(-10.0, 10.0, 41)
    phi_ours = ousv_cf(u, _FWD, _PARAMS)

    m = pyfeng.OusvFft(
        sigma=_PARAMS.sigma0, mr=_PARAMS.kappa, theta=_PARAMS.theta,
        vov=_PARAMS.nu, rho=_PARAMS.rho, intr=_FWD.r, divr=_FWD.q,
    )
    phi_pf = np.asarray(m.charfunc_logprice(u, texp=_FWD.T),
                        dtype=np.complex128)
    err = float(np.max(np.abs(phi_ours - phi_pf)))
    assert err < 1e-14, f"OUSV CF wrapper vs PyFENG: max|err| = {err:.3e}"


def test_ousv_cf_martingale():
    phi0 = ousv_cf(np.array([0.0]), _FWD, _PARAMS)[0]
    assert abs(phi0 - 1.0) < 1e-14


def test_ousv_prices_match_pyfeng_fft_across_methods():
    K = np.linspace(80.0, 120.0, 21)
    C_pf = price_strip("ousv", "pyfeng_fft", K, _FWD, _PARAMS)

    C_cos  = price_strip("ousv", "cos", K, _FWD, _PARAMS)
    C_frft = price_strip("ousv", "frft", K, _FWD, _PARAMS,
                          grid=FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5))
    C_cm   = price_strip("ousv", "carr_madan", K, _FWD, _PARAMS,
                          grid=FFTGrid(N=4096, eta=0.25, alpha=1.5))

    # COS ~ 1e-9 (saturates), FFT engines ~ 1e-7 on the default grid.
    assert np.max(np.abs(C_cos  - C_pf)) < 1e-6, \
        f"OUSV COS vs pyfeng_fft: {np.max(np.abs(C_cos - C_pf)):.3e}"
    assert np.max(np.abs(C_frft - C_pf)) < 1e-6
    assert np.max(np.abs(C_cm   - C_pf)) < 1e-6


def test_ousv_regression_cm_oracle_grid(ousv_regression_v1):
    """CM at the oracle grid — numerical identity with the frozen array."""
    ref = ousv_regression_v1
    C = price_strip(ref.model, "carr_madan", ref.strikes, ref.fwd, ref.params,
                    grid=FFTGrid(N=32768, eta=0.10, alpha=1.5))
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-12, f"{ref.name}: CM@oracle max|err| = {err:.3e}"


def test_ousv_regression_cos(ousv_regression_v1):
    ref = ousv_regression_v1
    C = price_strip(ref.model, "cos", ref.strikes, ref.fwd, ref.params)
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-7, f"{ref.name}: COS max|err| = {err:.3e}"


def test_ousv_regression_frft(ousv_regression_v1):
    ref = ousv_regression_v1
    C = price_strip(ref.model, "frft", ref.strikes, ref.fwd, ref.params,
                    grid=FRFTGrid(N=16384, eta=0.10, lam=0.0025, alpha=1.5))
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-7, f"{ref.name}: FRFT max|err| = {err:.3e}"


def test_ousv_regression_pyfeng_fft(ousv_regression_v1):
    """PyFENG's own FFT pricer matches the frozen array — proves our oracle
    is consistent with PyFENG's engine, not just our own."""
    ref = ousv_regression_v1
    C = price_strip(ref.model, "pyfeng_fft", ref.strikes, ref.fwd, ref.params)
    err = float(np.max(np.abs(C - ref.prices)))
    # PyFENG's default grid is independent of our oracle's; allow 1e-7.
    assert err < 1e-7, f"{ref.name}: pyfeng_fft max|err| = {err:.3e}"
