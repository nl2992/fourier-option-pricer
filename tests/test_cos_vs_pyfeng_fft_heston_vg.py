"""Third-party cross-check: COS (this repo) vs PyFENG's FFT pricers.

We already replicate FO2008/Lewis/CM1999 published values in
``test_phase4_cos_heston_fo2008.py``. Those pin COS to known constants. The
tests here instead check COS against an *independent* Fourier implementation
(PyFENG's ``HestonFft`` and ``VarGammaFft``), which is a different style of
evidence: not "does COS reproduce a paper number", but "does COS agree with
another library that solved the same integral with different code".

If PyFENG is not installed the tests skip (they don't fail). This keeps the
suite runnable in minimal environments while still catching regressions for
anyone who has PyFENG available.

PyFENG convention notes (verified against our fixtures):
  - ``HestonFft(sigma=v0, ...)``: first arg is **instantaneous variance**, not
    its square root. Tripped us up during development; kept as an explicit
    comment below so future readers don't redo the same experiment.
  - ``VarGammaFft(sigma, vov=nu, theta)``: same sigma/theta as ours, vov = nu.
  - Both take ``intr`` and ``divr`` in the constructor, not in ``price``.
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cf_form2, heston_cumulants
from foureng.models.variance_gamma import VGParams, vg_cf, vg_cumulants
from foureng.pricers.cos import cos_prices, cos_auto_grid


pyfeng = pytest.importorskip("pyfeng", reason="pyfeng not installed; "
                                               "this suite cross-checks against it")


def _cos_prices(phi, fwd: ForwardSpec, strikes: np.ndarray, cums, N: int, L: float) -> np.ndarray:
    grid = cos_auto_grid(cums, N=N, L=L)
    return cos_prices(phi, fwd, strikes, grid).call_prices


def test_cos_heston_agrees_with_pyfeng_fft(lewis_heston):
    """COS on Lewis params vs PyFENG HestonFft: agree to 1e-5."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    phi = lambda u: heston_cf_form2(u, fwd, p)

    strikes = d["strikes"]
    C_cos = _cos_prices(phi, fwd, strikes, heston_cumulants(fwd, p), N=256, L=10.0)

    # PyFENG HestonFft.sigma is v0 (variance), NOT sqrt(v0). Verified.
    h = pyfeng.HestonFft(sigma=p.v0, vov=p.nu, rho=p.rho, mr=p.kappa, theta=p.theta,
                         intr=fwd.r, divr=fwd.q)
    C_pf = h.price(strikes, spot=fwd.S0, texp=fwd.T, cp=1)

    err = float(np.max(np.abs(C_cos - C_pf)))
    assert err < 1e-5, f"COS vs PyFENG HestonFft (Lewis): {err:.3e}"


def test_cos_heston_agrees_with_pyfeng_fft_fo2008(fo2008_heston):
    """FO2008 ATM (Feller-violated) regime vs PyFENG HestonFft."""
    d = fo2008_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    phi = lambda u: heston_cf_form2(u, fwd, p)

    strikes = np.array([d["K"]], dtype=float)
    C_cos = _cos_prices(phi, fwd, strikes, heston_cumulants(fwd, p), N=256, L=10.0)

    h = pyfeng.HestonFft(sigma=p.v0, vov=p.nu, rho=p.rho, mr=p.kappa, theta=p.theta,
                         intr=fwd.r, divr=fwd.q)
    C_pf = h.price(strikes, spot=fwd.S0, texp=fwd.T, cp=1)

    err = float(np.abs(C_cos[0] - C_pf[0]))
    assert err < 1e-5, f"COS vs PyFENG HestonFft (FO2008): {err:.3e}"


def test_cos_vg_agrees_with_pyfeng_fft(cm1999_vg):
    """COS on CM1999 Case 4 VG vs PyFENG VarGammaFft (puts via parity)."""
    d = cm1999_vg
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = VGParams(sigma=d["sigma"], nu=d["nu"], theta=d["theta"])
    phi = lambda u: vg_cf(u, fwd, p)

    strikes = d["strikes"]
    C_cos = _cos_prices(phi, fwd, strikes, vg_cumulants(fwd, p), N=2048, L=10.0)

    m = pyfeng.VarGammaFft(sigma=p.sigma, vov=p.nu, theta=p.theta,
                            intr=fwd.r, divr=fwd.q)
    C_pf = m.price(strikes, spot=fwd.S0, texp=fwd.T, cp=1)

    # VG at T/nu small (heavy tails) is the hardest regime; allow 1e-3.
    err = float(np.max(np.abs(C_cos - C_pf)))
    assert err < 1e-3, f"COS vs PyFENG VarGammaFft (CM1999 c4): {err:.3e}"
