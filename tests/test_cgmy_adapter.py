"""CGMY adapter — CF parity, cumulants, cross-method, frozen strip.

Parallel to ``test_ousv_adapter.py`` — same gate pattern adapted to the
pure-Lévy (no stochastic variance) CGMY model:

  1. ``cgmy_cf`` matches :meth:`pyfeng.CgmyFft.charfunc_logprice`
     bit-exactly (~1e-14) — our wrapper is a pure adapter.
  2. ``phi(u=0) = 1`` and ``phi(-i) = 1`` — martingale.
  3. Closed-form cumulants match a numerical Cauchy-integral reference
     computed off the PyFENG CF (cross-check against our own analytic
     formula in ``cgmy.py``).
  4. COS / FRFT / CM all agree with PyFENG's :meth:`CgmyFft.price` to
     the same ~1e-7 floor we see for the other PyFENG-backed pure-Lévy
     models on their default grids.
  5. A frozen 41-strike regression strip pins prices against future
     refactors (see :data:`CGMY_REGRESSION_STRIP_V1`).
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.char_func.base import ForwardSpec
from foureng.char_func.cgmy import CgmyParams, cgmy_cf, cgmy_cumulants
from foureng.pipeline import price_strip
from foureng.utils.cumulants import cumulants_from_cf
from foureng.utils.grids import FFTGrid, FRFTGrid


pyfeng = pytest.importorskip("pyfeng", reason="CGMY adapter is PyFENG-backed")


# -- Canonical test params (shared with CGMY_REGRESSION_STRIP_V1) -----------
_FWD    = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
_PARAMS = CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.7)


def test_cgmy_cf_matches_pyfeng_charfunc_logprice():
    """Bit-identity wrapper check — see char_func/cgmy.py for the translation."""
    u = np.linspace(-10.0, 10.0, 41)
    phi_ours = cgmy_cf(u, _FWD, _PARAMS)

    m = pyfeng.CgmyFft(
        C=_PARAMS.C, G=_PARAMS.G, M=_PARAMS.M, Y=_PARAMS.Y,
        intr=_FWD.r, divr=_FWD.q,
    )
    phi_pf = np.asarray(m.charfunc_logprice(u, texp=_FWD.T),
                        dtype=np.complex128)
    err = float(np.max(np.abs(phi_ours - phi_pf)))
    assert err < 1e-14, f"CGMY CF wrapper vs PyFENG: max|err| = {err:.3e}"


def test_cgmy_cf_martingale_at_zero():
    phi0 = cgmy_cf(np.array([0.0]), _FWD, _PARAMS)[0]
    assert abs(phi0 - 1.0) < 1e-14


def test_cgmy_cf_martingale_at_minus_i():
    """phi(-i) = E[exp(X_T)] = 1 when the martingale correction is baked in."""
    phi_mi = cgmy_cf(np.array([-1j]), _FWD, _PARAMS)[0]
    assert abs(phi_mi - 1.0) < 1e-12


def test_cgmy_cumulants_match_numerical():
    """Our closed-form cumulants match a Cauchy-integral readout of the CF."""
    phi = lambda u: cgmy_cf(u, _FWD, _PARAMS)
    c_num = cumulants_from_cf(phi, order=4, radius=0.25, M=64)
    c1, c2, c4 = cgmy_cumulants(_FWD, _PARAMS)
    assert abs(c1 - c_num[0]) < 1e-12
    assert abs(c2 - c_num[1]) < 1e-12
    # c4 is the noisiest digit of the Cauchy integral (dominated by the
    # M=64 truncation); 1e-10 is well inside that floor.
    assert abs(c4 - c_num[3]) < 1e-10


def test_cgmy_prices_match_pyfeng_fft_across_methods():
    K = np.linspace(80.0, 120.0, 21)
    C_pf = price_strip("cgmy", "pyfeng_fft", K, _FWD, _PARAMS)

    C_cos  = price_strip("cgmy", "cos", K, _FWD, _PARAMS)
    C_frft = price_strip("cgmy", "frft", K, _FWD, _PARAMS,
                          grid=FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5))
    C_cm   = price_strip("cgmy", "carr_madan", K, _FWD, _PARAMS,
                          grid=FFTGrid(N=4096, eta=0.25, alpha=1.5))

    # Same 1e-6 budget as OUSV/VG on default grids.
    assert np.max(np.abs(C_cos  - C_pf)) < 1e-6, \
        f"CGMY COS vs pyfeng_fft: {np.max(np.abs(C_cos - C_pf)):.3e}"
    assert np.max(np.abs(C_frft - C_pf)) < 1e-6
    assert np.max(np.abs(C_cm   - C_pf)) < 1e-6


def test_cgmy_regression_cm_oracle_grid(cgmy_regression_v1):
    """CM at the oracle grid — numerical identity with the frozen array."""
    ref = cgmy_regression_v1
    C = price_strip(ref.model, "carr_madan", ref.strikes, ref.fwd, ref.params,
                    grid=FFTGrid(N=32768, eta=0.10, alpha=1.5))
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-12, f"{ref.name}: CM@oracle max|err| = {err:.3e}"


def test_cgmy_regression_cos(cgmy_regression_v1):
    ref = cgmy_regression_v1
    C = price_strip(ref.model, "cos", ref.strikes, ref.fwd, ref.params)
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-7, f"{ref.name}: COS max|err| = {err:.3e}"


def test_cgmy_regression_frft(cgmy_regression_v1):
    ref = cgmy_regression_v1
    C = price_strip(ref.model, "frft", ref.strikes, ref.fwd, ref.params,
                    grid=FRFTGrid(N=16384, eta=0.10, lam=0.0025, alpha=1.5))
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-7, f"{ref.name}: FRFT max|err| = {err:.3e}"


def test_cgmy_regression_pyfeng_fft(cgmy_regression_v1):
    """PyFENG's own FFT pricer matches the frozen array to the default-grid floor."""
    ref = cgmy_regression_v1
    C = price_strip(ref.model, "pyfeng_fft", ref.strikes, ref.fwd, ref.params)
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-6, f"{ref.name}: pyfeng_fft max|err| = {err:.3e}"
