"""Kou (2002) paper-facing COS tests."""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.kou import KouParams, kou_cf, kou_cumulants
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.utils.grids import FFTGrid


@pytest.fixture
def kou_setup():
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.5)
    params = KouParams(sigma=0.16, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    phi = lambda u: kou_cf(u, fwd, params)
    ref = carr_madan_price_at_strikes(phi, fwd, FFTGrid(16384, 0.05, 1.5), strikes)
    return dict(fwd=fwd, params=params, phi=phi, strikes=strikes, ref=ref)


def test_cos_kou_vs_carr_madan(kou_setup):
    """COS at N=128 should match the high-N Carr-Madan reference."""
    d = kou_setup
    cums = kou_cumulants(d["fwd"], d["params"])
    calls = cos_prices(d["phi"], d["fwd"], d["strikes"], cos_auto_grid(cums, N=128, L=10.0)).call_prices
    err = np.abs(calls - d["ref"]).max()
    assert err < 1e-6, f"COS-Kou N=128 err = {err:.3e}"


def test_cos_kou_n_convergence(kou_setup):
    """Error should fall rapidly as N doubles on the Kou strip."""
    d = kou_setup
    cums = kou_cumulants(d["fwd"], d["params"])
    errs = {}
    for N in (32, 64, 128):
        calls = cos_prices(d["phi"], d["fwd"], d["strikes"], cos_auto_grid(cums, N=N, L=10.0)).call_prices
        errs[N] = np.abs(calls - d["ref"]).max()
    assert errs[64] < errs[32], f"not monotone: {errs}"
    assert errs[128] < errs[64], f"not monotone: {errs}"
    assert errs[128] < errs[32] * 1e-6, f"too-slow convergence: {errs}"


def test_cos_kou_L_sensitivity(kou_setup):
    """The recommended-or-wider L band should be stable for Kou."""
    d = kou_setup
    cums = kou_cumulants(d["fwd"], d["params"])
    prices = {}
    for L in (8.0, 10.0, 12.0, 14.0):
        prices[L] = cos_prices(
            d["phi"], d["fwd"], d["strikes"], cos_auto_grid(cums, N=512, L=L)
        ).call_prices.copy()
    arr = np.stack(list(prices.values()), axis=0)
    spread = (arr.max(axis=0) - arr.min(axis=0)).max()
    assert spread < 1e-6, f"L-sensitivity too high: spread = {spread:.3e}"


def test_kou_cumulants_analytic_matches_cf(kou_setup):
    """Analytic Kou cumulants should match the CF-derived values."""
    from foureng.utils.cumulants import cumulants_from_cf

    d = kou_setup
    c_num = cumulants_from_cf(d["phi"], order=4, radius=0.25, M=64)
    c_ana = kou_cumulants(d["fwd"], d["params"])
    assert abs(c_ana[0] - c_num[0]) < 1e-12
    assert abs(c_ana[1] - c_num[1]) < 1e-12
    assert abs(c_ana[2] - c_num[3]) < 1e-10
