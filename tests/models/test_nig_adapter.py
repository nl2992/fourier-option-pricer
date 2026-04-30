"""NIG adapter — CF parity, cross-method, frozen strip.

Parallel to ``test_cgmy_adapter.py`` — same gate pattern for the
Normal-Inverse-Gaussian Lévy model, backed by :class:`pyfeng.ExpNigFft`:

  1. ``nig_cf`` matches :meth:`pyfeng.ExpNigFft.charfunc_logprice`
     bit-exactly (~1e-14).
  2. ``phi(u=0) = 1`` and ``phi(-i) = 1`` — the CF has the martingale
     correction ``mu`` baked in (see :mod:`foureng.models.nig`).
  3. COS / FRFT / CM all agree with PyFENG's :meth:`ExpNigFft.price` to
     the ~1e-7 default-grid floor.
  4. A frozen 41-strike regression strip pins prices against future
     refactors (see :data:`NIG_REGRESSION_STRIP_V1`).
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.nig import NigParams, nig_cf
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid


pyfeng = pytest.importorskip("pyfeng", reason="NIG adapter is PyFENG-backed")


# -- Canonical test params (shared with NIG_REGRESSION_STRIP_V1) ------------
_FWD    = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
_PARAMS = NigParams(sigma=0.2, nu=0.5, theta=-0.10)


def test_nig_cf_matches_pyfeng_charfunc_logprice():
    """Bit-identity wrapper check — see models/nig.py for the translation."""
    u = np.linspace(-10.0, 10.0, 41)
    phi_ours = nig_cf(u, _FWD, _PARAMS)

    m = pyfeng.ExpNigFft(
        sigma=_PARAMS.sigma, vov=_PARAMS.nu, theta=_PARAMS.theta,
        intr=_FWD.r, divr=_FWD.q,
    )
    phi_pf = np.asarray(m.charfunc_logprice(u, texp=_FWD.T),
                        dtype=np.complex128)
    err = float(np.max(np.abs(phi_ours - phi_pf)))
    assert err < 1e-14, f"NIG CF wrapper vs PyFENG: max|err| = {err:.3e}"


def test_nig_cf_martingale_at_zero():
    phi0 = nig_cf(np.array([0.0]), _FWD, _PARAMS)[0]
    assert abs(phi0 - 1.0) < 1e-14


def test_nig_cf_martingale_at_minus_i():
    """phi(-i) = E[exp(X_T)] = 1 — the ``mu`` drift is set for this."""
    phi_mi = nig_cf(np.array([-1j]), _FWD, _PARAMS)[0]
    assert abs(phi_mi - 1.0) < 1e-12


def test_nig_prices_match_pyfeng_fft_across_methods():
    K = np.linspace(80.0, 120.0, 21)
    C_pf = price_strip("nig", "pyfeng_fft", K, _FWD, _PARAMS)

    C_cos  = price_strip("nig", "cos", K, _FWD, _PARAMS)
    C_frft = price_strip("nig", "frft", K, _FWD, _PARAMS,
                          grid=FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5))
    C_cm   = price_strip("nig", "carr_madan", K, _FWD, _PARAMS,
                          grid=FFTGrid(N=4096, eta=0.25, alpha=1.5))

    assert np.max(np.abs(C_cos  - C_pf)) < 1e-6, \
        f"NIG COS vs pyfeng_fft: {np.max(np.abs(C_cos - C_pf)):.3e}"
    assert np.max(np.abs(C_frft - C_pf)) < 1e-6
    assert np.max(np.abs(C_cm   - C_pf)) < 1e-6


def test_nig_regression_cm_oracle_grid(nig_regression_v1):
    """CM at the oracle grid — numerical identity with the frozen array."""
    ref = nig_regression_v1
    C = price_strip(ref.model, "carr_madan", ref.strikes, ref.fwd, ref.params,
                    grid=FFTGrid(N=32768, eta=0.10, alpha=1.5))
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-12, f"{ref.name}: CM@oracle max|err| = {err:.3e}"


def test_nig_regression_cos(nig_regression_v1):
    ref = nig_regression_v1
    C = price_strip(ref.model, "cos", ref.strikes, ref.fwd, ref.params)
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-7, f"{ref.name}: COS max|err| = {err:.3e}"


def test_nig_regression_frft(nig_regression_v1):
    ref = nig_regression_v1
    C = price_strip(ref.model, "frft", ref.strikes, ref.fwd, ref.params,
                    grid=FRFTGrid(N=16384, eta=0.10, lam=0.0025, alpha=1.5))
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-7, f"{ref.name}: FRFT max|err| = {err:.3e}"


def test_nig_regression_pyfeng_fft(nig_regression_v1):
    """PyFENG's own FFT pricer matches the frozen array to the default-grid floor."""
    ref = nig_regression_v1
    C = price_strip(ref.model, "pyfeng_fft", ref.strikes, ref.fwd, ref.params)
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-6, f"{ref.name}: pyfeng_fft max|err| = {err:.3e}"
