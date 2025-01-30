"""Phase 4 post-gate: COS on Kou (2002) double-exponential jump diffusion.

Kou has a pole structure that restricts the strip of analyticity for the CF
(eta1 > 1 for finite jump mean; for the damped Carr-Madan call integrand,
the damping parameter must also satisfy eta1 > 1 + alpha, see Phase 2/3
validity work). COS does not need the damping parameter so it is unaffected
by the pole; it only needs cumulants to pick [a,b], and those are clean.

Three replication checks:
  - COS vs Carr-Madan high-N reference (1e-6 at N=128)
  - Monotone N-convergence at L=10
  - L-sensitivity at N=512
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.char_func.base import ForwardSpec
from foureng.char_func.kou import KouParams, kou_cf, kou_cumulants
from foureng.pricers.cos import cos_prices, cos_auto_grid
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.utils.grids import FFTGrid


@pytest.fixture
def kou_setup():
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.5)
    p = KouParams(sigma=0.16, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    phi = lambda u: kou_cf(u, fwd, p)
    # Carr-Madan reference at N=16384 — converged to ~1e-7
    ref = carr_madan_price_at_strikes(phi, fwd, FFTGrid(16384, 0.05, 1.5), strikes)
    return dict(fwd=fwd, p=p, phi=phi, strikes=strikes, ref=ref)


def test_cos_kou_vs_carr_madan(kou_setup):
    """COS at N=128 should match the high-N Carr-Madan reference to 1e-6."""
    d = kou_setup
    cums = kou_cumulants(d["fwd"], d["p"])
    grid = cos_auto_grid(cums, N=128, L=10.0)
    r = cos_prices(d["phi"], d["fwd"], d["strikes"], grid)
    err = np.abs(r.call_prices - d["ref"]).max()
    assert err < 1e-6, f"COS-Kou N=128 err = {err:.3e}"


def test_cos_kou_n_convergence(kou_setup):
    """N-convergence: error drops by >= 6 orders of magnitude from 32 -> 128."""
    d = kou_setup
    cums = kou_cumulants(d["fwd"], d["p"])
    errs = {}
    for N in (32, 64, 128):
        grid = cos_auto_grid(cums, N=N, L=10.0)
        r = cos_prices(d["phi"], d["fwd"], d["strikes"], grid)
        errs[N] = np.abs(r.call_prices - d["ref"]).max()
    assert errs[64] < errs[32], f"not monotone: {errs}"
    assert errs[128] < errs[64], f"not monotone: {errs}"
    assert errs[128] < errs[32] * 1e-6, f"too-slow convergence: {errs}"


def test_cos_kou_L_sensitivity(kou_setup):
    """L-sensitivity at N=512: prices stable to 1e-7 across L in {6, 8, 10, 12, 14}."""
    d = kou_setup
    cums = kou_cumulants(d["fwd"], d["p"])
    prices = {}
    for L in (6.0, 8.0, 10.0, 12.0, 14.0):
        grid = cos_auto_grid(cums, N=512, L=L)
        r = cos_prices(d["phi"], d["fwd"], d["strikes"], grid)
        prices[L] = r.call_prices.copy()
    # Max elementwise difference across L
    arr = np.stack(list(prices.values()), axis=0)
    spread = (arr.max(axis=0) - arr.min(axis=0)).max()
    assert spread < 1e-6, f"L-sensitivity too high: spread = {spread:.3e}"


def test_kou_cumulants_analytic_matches_cf(kou_setup):
    """Analytic Kou cumulants must match CF-derived to FP precision."""
    from foureng.utils.cumulants import cumulants_from_cf
    d = kou_setup
    c_num = cumulants_from_cf(d["phi"], order=4, radius=0.25, M=64)
    c_ana = kou_cumulants(d["fwd"], d["p"])
    assert abs(c_ana[0] - c_num[0]) < 1e-12
    assert abs(c_ana[1] - c_num[1]) < 1e-12
    assert abs(c_ana[2] - c_num[3]) < 1e-10  # c4 at FD cancellation floor
