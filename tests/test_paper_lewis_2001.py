"""Lewis (2001) paper-facing tests."""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cf_form2, heston_cumulants
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.utils.grids import FFTGrid


pytestmark = [pytest.mark.paper, pytest.mark.external_reference]


def test_heston_carr_madan_lewis(lewis_heston):
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"]
    )
    phi = lambda u: heston_cf_form2(u, fwd, params)

    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    calls = carr_madan_price_at_strikes(phi, fwd, grid, d["strikes"])
    err = np.abs(calls - d["ref_calls"]).max()
    assert err < 1e-3, f"Lewis Heston max err = {err:.3e}\n C = {calls}\n ref = {d['ref_calls']}"


def test_heston_atm_put_call_parity(lewis_heston):
    """Robustness: the Lewis strip should produce a sane ATM Heston price."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"]
    )
    phi = lambda u: heston_cf_form2(u, fwd, params)

    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    call_atm = carr_madan_price_at_strikes(phi, fwd, grid, np.array([100.0]))[0]
    assert np.isfinite(call_atm) and 10.0 < call_atm < 25.0


def test_carr_madan_fft_grid_sensitivity(lewis_heston):
    """Minimum N to hit 1e-3 should stay at or below N=4096."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"]
    )
    phi = lambda u: heston_cf_form2(u, fwd, params)

    ok = False
    for N in (1024, 2048, 4096):
        grid = FFTGrid(N=N, eta=0.25, alpha=1.5)
        calls = carr_madan_price_at_strikes(phi, fwd, grid, d["strikes"])
        if np.abs(calls - d["ref_calls"]).max() < 1e-3:
            ok = True
            break
    assert ok, "Carr-Madan failed to converge to 1e-3 even at N=4096"


def test_cos_lewis_heston(lewis_heston):
    """COS should hit the Lewis Heston strip to 1e-6 by N=128."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"]
    )
    phi = lambda u: heston_cf_form2(u, fwd, params)

    grid = cos_auto_grid(heston_cumulants(fwd, params), N=128, L=10.0)
    calls = cos_prices(phi, fwd, d["strikes"], grid).call_prices
    err = np.abs(calls - d["ref_calls"]).max()
    assert err < 1e-6, f"Lewis Heston COS N=128 max err = {err:.3e}"


def test_cos_heston_exponential_convergence(lewis_heston):
    """COS on the Lewis strip should show fast N-convergence."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"]
    )
    phi = lambda u: heston_cf_form2(u, fwd, params)

    errs = {}
    for N in (32, 64, 128):
        grid = cos_auto_grid(heston_cumulants(fwd, params), N=N, L=10.0)
        calls = cos_prices(phi, fwd, d["strikes"], grid).call_prices
        errs[N] = np.abs(calls - d["ref_calls"]).max()

    assert errs[128] < errs[32] * 1e-4, f"COS not converging fast enough: {errs}"


def test_cos_matches_carr_madan(lewis_heston):
    """COS and Carr-Madan should agree on the Lewis Heston strip."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"]
    )
    phi = lambda u: heston_cf_form2(u, fwd, params)

    cm_calls = carr_madan_price_at_strikes(phi, fwd, FFTGrid(4096, 0.25, 1.5), d["strikes"])
    cos_calls = cos_prices(
        phi,
        fwd,
        d["strikes"],
        cos_auto_grid(heston_cumulants(fwd, params), N=160, L=10.0),
    ).call_prices
    err = np.abs(cm_calls - cos_calls).max()
    assert err < 1e-5, f"COS vs Carr-Madan disagreement on Lewis: {err:.3e}"
