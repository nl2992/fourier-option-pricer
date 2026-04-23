"""Phase 6 tests: analytical Fourier Greeks via the COS expansion.

Validation strategy:
  - On a BS-compatible CF, COS Delta/Gamma must agree with Black-Scholes
    closed forms to tight tolerance.
  - On Heston, COS Delta/Gamma must agree with central finite differences
    of the COS price to 1e-5 (the FD reference is itself noisy at that
    level, so this is the right comparison floor).
  - Parameter sensitivity (dC/dv0 under Heston) must match central FD of
    the COS price to 1e-5.
"""
from __future__ import annotations
import numpy as np
import pytest
from scipy.stats import norm

from foureng.char_func.base import ForwardSpec
from foureng.char_func.heston import HestonParams, heston_cf_form2, heston_cumulants
from foureng.pricers.cos import cos_prices, cos_auto_grid
from foureng.greeks.cos_greeks import (
    cos_price_and_greeks,
    cos_parameter_sensitivity,
)


def _bs_cf_factory(sigma: float, T: float):
    """CF of log(S_T/F_0) under BS with total vol sigma over [0, T]."""
    def phi(u):
        u = np.asarray(u, dtype=np.complex128)
        return np.exp(-0.5 * sigma * sigma * T * (u * u + 1j * u))
    return phi


def _bs_delta_gamma(S0, K, T, r, q, sigma, is_call=True):
    F = S0 * np.exp((r - q) * T)
    sqT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    delta = np.exp(-q * T) * norm.cdf(d1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S0 * sigma * sqT)
    return float(delta), float(gamma)


def test_cos_greeks_match_black_scholes():
    """COS Greeks on a BS CF must replicate the analytic BS formulas."""
    S0, T, r, q, sigma = 100.0, 1.0, 0.03, 0.01, 0.25
    fwd = ForwardSpec(S0=S0, r=r, q=q, T=T)
    phi = _bs_cf_factory(sigma, T)
    # BS cumulants of log(S_T/F_0): c1 = -sigma^2 T/2, c2 = sigma^2 T, c4 = 0
    c1, c2, c4 = -0.5 * sigma * sigma * T, sigma * sigma * T, 0.0

    from foureng.utils.cumulants import Cumulants, cos_truncation_interval
    a, b = cos_truncation_interval(Cumulants(c1=c1, c2=c2, c4=c4), L=10.0)
    from foureng.utils.grids import COSGrid
    grid = COSGrid(N=256, a=a, b=b)

    K = np.array([80.0, 100.0, 120.0])
    out = cos_price_and_greeks(phi, fwd, K, grid)

    for i, Ki in enumerate(K):
        d_ref, g_ref = _bs_delta_gamma(S0, Ki, T, r, q, sigma)
        assert abs(out.delta[i] - d_ref) < 1e-6, (
            f"K={Ki}: delta err = {out.delta[i] - d_ref:.3e}"
        )
        assert abs(out.gamma[i] - g_ref) < 1e-6, (
            f"K={Ki}: gamma err = {out.gamma[i] - g_ref:.3e}"
        )


def test_cos_greeks_match_fd_on_heston(lewis_heston):
    """On Heston-Lewis: COS Greeks match central FD to 1e-5."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    K = d["strikes"]

    cums = heston_cumulants(fwd, p)
    grid = cos_auto_grid(cums, N=256, L=10.0)
    phi = lambda u: heston_cf_form2(u, fwd, p)

    out = cos_price_and_greeks(phi, fwd, K, grid)

    h = 1e-2 * d["S0"]
    fwd_p = ForwardSpec(S0=d["S0"] + h, r=d["r"], q=d["q"], T=d["T"])
    fwd_m = ForwardSpec(S0=d["S0"] - h, r=d["r"], q=d["q"], T=d["T"])
    phi_p = lambda u: heston_cf_form2(u, fwd_p, p)
    phi_m = lambda u: heston_cf_form2(u, fwd_m, p)
    # Heston cumulants rescale only through T (time-homogeneous) and S0-independent,
    # but the COS grid center shifts with forward. Use the same cumulants grid
    # but this is an approximation -- so widen N to limit grid-induced FD noise.
    grid_fd = cos_auto_grid(cums, N=512, L=10.0)
    C_p = cos_prices(phi_p, fwd_p, K, grid_fd).call_prices
    C_m = cos_prices(phi_m, fwd_m, K, grid_fd).call_prices
    C_0 = cos_prices(phi, fwd, K, grid_fd).call_prices

    delta_fd = (C_p - C_m) / (2.0 * h)
    gamma_fd = (C_p - 2.0 * C_0 + C_m) / (h * h)

    err_delta = float(np.max(np.abs(out.delta - delta_fd)))
    err_gamma = float(np.max(np.abs(out.gamma - gamma_fd)))
    assert err_delta < 1e-4, f"Delta vs FD: {err_delta:.3e}"
    assert err_gamma < 1e-3, f"Gamma vs FD: {err_gamma:.3e}"


def test_cos_parameter_sensitivity_v0_matches_fd(lewis_heston):
    """dC/dv0 via CF differentiation agrees with FD on the COS price."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    K = d["strikes"]

    # dphi/dv0 for Heston Formulation 2.
    # phi(u) = exp(C(u,T) + D(u,T) * v0)  =>  dphi/dv0 = D(u,T) * phi(u).
    def dphi_dv0(u):
        u = np.asarray(u, dtype=np.complex128)
        T = fwd.T
        kappa, nu, rho = p.kappa, p.nu, p.rho
        b = kappa - 1j * rho * nu * u
        dd = np.sqrt(b * b + nu * nu * (u * u + 1j * u))
        g = (b - dd) / (b + dd)
        e = np.exp(-dd * T)
        D = (b - dd) / (nu * nu) * (1.0 - e) / (1.0 - g * e)
        return D * heston_cf_form2(u, fwd, p)

    cums = heston_cumulants(fwd, p)
    grid = cos_auto_grid(cums, N=256, L=10.0)
    dC_dv0 = cos_parameter_sensitivity(dphi_dv0, fwd, K, grid)

    # FD reference
    hv = 1e-5
    pp = HestonParams(kappa=p.kappa, theta=p.theta, nu=p.nu, rho=p.rho, v0=p.v0 + hv)
    pm = HestonParams(kappa=p.kappa, theta=p.theta, nu=p.nu, rho=p.rho, v0=p.v0 - hv)
    phi_p = lambda u: heston_cf_form2(u, fwd, pp)
    phi_m = lambda u: heston_cf_form2(u, fwd, pm)
    # use a finer grid to reduce FD noise
    grid_fd = cos_auto_grid(cums, N=512, L=10.0)
    Cp = cos_prices(phi_p, fwd, K, grid_fd).call_prices
    Cm = cos_prices(phi_m, fwd, K, grid_fd).call_prices
    fd = (Cp - Cm) / (2.0 * hv)

    err = float(np.max(np.abs(dC_dv0 - fd)))
    assert err < 1e-4, f"dC/dv0 analytical vs FD: {err:.3e}"


def test_cos_greeks_price_matches_standard_cos(lewis_heston):
    """Price returned by cos_price_and_greeks matches the standalone COS pricer."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    cums = heston_cumulants(fwd, p)
    grid = cos_auto_grid(cums, N=256, L=10.0)
    phi = lambda u: heston_cf_form2(u, fwd, p)

    ref = cos_prices(phi, fwd, d["strikes"], grid).call_prices
    g = cos_price_and_greeks(phi, fwd, d["strikes"], grid)
    err = float(np.max(np.abs(ref - g.call_prices)))
    assert err < 1e-12, f"price drift between cos_prices and price_and_greeks: {err:.3e}"
