"""Cross-model COS Greek tests (Sprint 6).

Verifies that cos_price_and_greeks produces Delta and Gamma consistent
with finite-difference approximations for all 12 major models.

Tolerance: 0.5% for Delta, 1% for Gamma (FD noise at h=0.01%).
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.greeks.cos_greeks import cos_price_and_greeks
from foureng.models.registry import MODEL_REGISTRY
from foureng.pricers.cos import cos_auto_grid

ALL_MODELS = [
    ("bsm", fe.BsmParams(sigma=0.20)),
    ("heston", fe.HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04)),
    ("vg", fe.VGParams(sigma=0.12, nu=0.3, theta=-0.14)),
    ("cgmy", fe.CgmyParams(C=1.0, G=5.0, M=5.0, Y=1.5)),
    ("nig", fe.NigParams(sigma=0.15, nu=0.3, theta=-0.14)),
    ("kou", fe.KouParams(sigma=0.15, lam=0.5, p=0.4, eta1=10.0, eta2=5.0)),
    ("merton_jd", fe.MertonJDParams(sigma=0.15, lam=0.5, muj=-0.1, sigj=0.10)),
    (
        "bates",
        fe.BatesParams(
            kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04, lam_j=0.5, mu_j=-0.1, sigma_j=0.1
        ),
    ),
    (
        "heston_kou",
        fe.HestonKouParams(
            kappa=2.0,
            theta=0.04,
            nu=0.5,
            rho=-0.5,
            v0=0.04,
            lam_j=0.5,
            p_j=0.4,
            eta1=10.0,
            eta2=5.0,
        ),
    ),
    (
        "heston_cgmy",
        fe.HestonCGMYParams(
            kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04, C=0.5, G=5.0, M=5.0, Y=1.5
        ),
    ),
    (
        "double_heston",
        fe.DoubleHestonParams(
            kappa1=2.0,
            theta1=0.02,
            nu1=0.3,
            rho1=-0.4,
            v01=0.02,
            kappa2=1.0,
            theta2=0.02,
            nu2=0.2,
            rho2=-0.3,
            v02=0.02,
        ),
    ),
    ("vgsa", fe.VGSAParams(C=1.0, G=5.0, M=5.0, kappa=3.0, eta=1.0, lam=1.0)),
]


def _price_strip(model, params, fwd):
    """Vectorised call prices using price_strip."""
    strikes = np.array([90.0, 100.0, 110.0])
    return fe.price_strip(model, "cos_improved", strikes, fwd, params)


def _greeks_and_fd(model, params, fwd, strikes):
    entry = MODEL_REGISTRY[model]
    cums = entry.cumulants(fwd, params)
    grid = cos_auto_grid(cums, N=256, L=12.0)
    phi = lambda u: entry.cf(u, fwd, params)

    # Analytic Greeks
    out = cos_price_and_greeks(phi, fwd, strikes, grid)

    # FD Delta: (C(S+h) - C(S-h)) / (2h)
    h = fwd.S0 * 1e-4
    fwd_up = fe.ForwardSpec(S0=fwd.S0 + h, r=fwd.r, q=fwd.q, T=fwd.T)
    fwd_dn = fe.ForwardSpec(S0=fwd.S0 - h, r=fwd.r, q=fwd.q, T=fwd.T)
    phi_up = lambda u: entry.cf(u, fwd_up, params)
    phi_dn = lambda u: entry.cf(u, fwd_dn, params)
    grid_up = cos_auto_grid(entry.cumulants(fwd_up, params), N=256, L=12.0)
    grid_dn = cos_auto_grid(entry.cumulants(fwd_dn, params), N=256, L=12.0)
    C_up = cos_price_and_greeks(phi_up, fwd_up, strikes, grid_up).call_prices
    C_dn = cos_price_and_greeks(phi_dn, fwd_dn, strikes, grid_dn).call_prices
    fd_delta = (C_up - C_dn) / (2 * h)

    # FD Gamma: (C(S+h) - 2C + C(S-h)) / h^2
    C_base = out.call_prices
    fd_gamma = (C_up - 2 * C_base + C_dn) / h**2

    return out.delta, out.gamma, fd_delta, fd_gamma


@pytest.mark.parametrize("model,params", ALL_MODELS)
def test_cos_delta_vs_fd(model, params):
    """COS Delta matches finite-difference to 0.5%."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
    strikes = np.array([90.0, 100.0, 110.0])
    delta, gamma, fd_delta, fd_gamma = _greeks_and_fd(model, params, fwd, strikes)

    # Skip near-zero delta (deep OTM options have very small delta)
    for i, K in enumerate(strikes):
        if abs(fd_delta[i]) < 1e-3:
            continue  # too small to test relative error
        rel = abs(delta[i] - fd_delta[i]) / abs(fd_delta[i])
        assert rel < 0.005, (
            f"{model} K={K}: Delta={delta[i]:.6f} FD={fd_delta[i]:.6f} rel={rel:.4f}"
        )


@pytest.mark.parametrize("model,params", ALL_MODELS)
def test_cos_gamma_vs_fd(model, params):
    """COS Gamma matches finite-difference to 1%."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
    strikes = np.array([95.0, 100.0, 105.0])  # focus on near-ATM
    delta, gamma, fd_delta, fd_gamma = _greeks_and_fd(model, params, fwd, strikes)

    for i, K in enumerate(strikes):
        if abs(fd_gamma[i]) < 1e-5:
            continue
        rel = abs(gamma[i] - fd_gamma[i]) / abs(fd_gamma[i])
        assert rel < 0.01, f"{model} K={K}: Gamma={gamma[i]:.6f} FD={fd_gamma[i]:.6f} rel={rel:.4f}"


@pytest.mark.parametrize("model,params", ALL_MODELS)
def test_delta_in_0_1(model, params):
    """Call delta ∈ [0, 1] for all strikes and models."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
    strikes = np.linspace(80.0, 120.0, 9)
    entry = MODEL_REGISTRY[model]
    cums = entry.cumulants(fwd, params)
    grid = cos_auto_grid(cums, N=256, L=12.0)
    phi = lambda u: entry.cf(u, fwd, params)
    out = cos_price_and_greeks(phi, fwd, strikes, grid)
    assert np.all(out.delta >= -0.01), f"{model}: negative delta detected"
    assert np.all(out.delta <= 1.01), f"{model}: delta > 1 detected"


@pytest.mark.parametrize("model,params", ALL_MODELS)
def test_gamma_non_negative(model, params):
    """Call gamma ≥ 0 (convexity of call in spot)."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
    strikes = np.linspace(85.0, 115.0, 7)
    entry = MODEL_REGISTRY[model]
    cums = entry.cumulants(fwd, params)
    grid = cos_auto_grid(cums, N=256, L=12.0)
    phi = lambda u: entry.cf(u, fwd, params)
    out = cos_price_and_greeks(phi, fwd, strikes, grid)
    assert np.all(out.gamma >= -1e-6), f"{model}: negative gamma detected"
