"""Tests for BSM analytical Greeks.

Strategy: compare each Greek against finite-difference bumps of bsm_call/bsm_put.
We also verify fundamental identities:
  - Put-call delta symmetry: delta_call - delta_put = e^{-qT}
  - Gamma put = Gamma call
  - Vega put = Vega call
  - Theta + Vega*sigma/(2T) + r*V - (r-q)*S*delta = 0  (BSM PDE)
  - Rho call > 0, Rho put < 0
  - Vanna: ∂delta/∂sigma ≈ finite diff
  - Volga: ∂vega/∂sigma  ≈ finite diff
  - Deep ITM call delta → e^{-qT}; deep OTM call delta → 0
  - bsm_all_greeks returns correct keys and values
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.analytics.bsm_barrier import bsm_call, bsm_put
from foureng.analytics.bsm_greeks import (
    bsm_all_greeks,
    bsm_delta,
    bsm_gamma,
    bsm_rho,
    bsm_theta,
    bsm_vanna,
    bsm_vega,
    bsm_volga,
)

# ── shared parameters ─────────────────────────────────────────────────────────

S, K, r, q, T, sigma = 100.0, 100.0, 0.05, 0.02, 1.0, 0.20
_PRICE = {"call": bsm_call, "put": bsm_put}
_CP = {"call": 1, "put": -1}
_EPS_S = 1e-4  # bump size for S finite-diff
_EPS_V = 1e-5  # bump size for sigma finite-diff
_EPS_R = 1e-6  # bump size for r finite-diff
_TOL = 1e-5  # absolute tolerance against finite differences


# ── delta ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cp,label", [(1, "call"), (-1, "put")])
def test_delta_finite_diff(cp, label):
    """∂V/∂S matches central finite difference."""
    price_fn = _PRICE[label]
    fd = (price_fn(S + _EPS_S, K, r, q, T, sigma) - price_fn(S - _EPS_S, K, r, q, T, sigma)) / (
        2 * _EPS_S
    )
    assert abs(bsm_delta(S, K, r, q, T, sigma, cp) - fd) < _TOL


def test_delta_put_call_parity():
    """delta_call - delta_put = e^{-qT}  (put-call delta symmetry)."""
    dc = bsm_delta(S, K, r, q, T, sigma, cp=1)
    dp = bsm_delta(S, K, r, q, T, sigma, cp=-1)
    assert abs(dc - dp - np.exp(-q * T)) < 1e-12


def test_delta_deep_itm_call():
    """Deep ITM call delta → e^{-qT}."""
    d = bsm_delta(1000.0, K, r, q, T, sigma, cp=1)
    assert abs(d - np.exp(-q * T)) < 1e-6


def test_delta_deep_otm_call():
    """Deep OTM call delta → 0."""
    d = bsm_delta(10.0, K, r, q, T, sigma, cp=1)
    assert abs(d) < 1e-6


def test_delta_deep_itm_put():
    """Deep ITM put delta → -e^{-qT}."""
    d = bsm_delta(5.0, K, r, q, T, sigma, cp=-1)
    assert abs(d + np.exp(-q * T)) < 1e-5


# ── gamma ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cp,label", [(1, "call"), (-1, "put")])
def test_gamma_finite_diff(cp, label):
    """∂²V/∂S² matches central finite difference."""
    price_fn = _PRICE[label]
    V_up = price_fn(S + _EPS_S, K, r, q, T, sigma)
    V_dn = price_fn(S - _EPS_S, K, r, q, T, sigma)
    V_0 = price_fn(S, K, r, q, T, sigma)
    fd = (V_up - 2 * V_0 + V_dn) / _EPS_S**2
    assert abs(bsm_gamma(S, K, r, q, T, sigma, cp) - fd) < _TOL


def test_gamma_call_equals_put():
    """Gamma is identical for calls and puts."""
    gc = bsm_gamma(S, K, r, q, T, sigma, cp=1)
    gp = bsm_gamma(S, K, r, q, T, sigma, cp=-1)
    assert abs(gc - gp) < 1e-14


def test_gamma_positive():
    """Gamma is always non-negative."""
    assert bsm_gamma(S, K, r, q, T, sigma) > 0
    assert bsm_gamma(50.0, K, r, q, T, sigma) > 0
    assert bsm_gamma(200.0, K, r, q, T, sigma) > 0


# ── vega ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cp,label", [(1, "call"), (-1, "put")])
def test_vega_finite_diff(cp, label):
    """∂V/∂σ matches central finite difference."""
    price_fn = _PRICE[label]
    fd = (price_fn(S, K, r, q, T, sigma + _EPS_V) - price_fn(S, K, r, q, T, sigma - _EPS_V)) / (
        2 * _EPS_V
    )
    assert abs(bsm_vega(S, K, r, q, T, sigma, cp) - fd) < _TOL


def test_vega_call_equals_put():
    """Vega is identical for calls and puts."""
    vc = bsm_vega(S, K, r, q, T, sigma, cp=1)
    vp = bsm_vega(S, K, r, q, T, sigma, cp=-1)
    assert abs(vc - vp) < 1e-12


def test_vega_positive():
    """Vega is always non-negative."""
    assert bsm_vega(S, K, r, q, T, sigma) > 0


# ── theta ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cp,label", [(1, "call"), (-1, "put")])
def test_theta_finite_diff(cp, label):
    """∂V/∂t = -∂V/∂T; check against -ΔV/ΔT finite diff."""
    dT = 1e-5
    price_fn = _PRICE[label]
    fd = -(price_fn(S, K, r, q, T + dT, sigma) - price_fn(S, K, r, q, T - dT, sigma)) / (2 * dT)
    assert abs(bsm_theta(S, K, r, q, T, sigma, cp) - fd) < _TOL


def test_theta_call_generally_negative():
    """ATM call theta is negative (time value erodes)."""
    assert bsm_theta(S, S, r=0.0, q=0.0, T=T, sigma=sigma, cp=1) < 0


# ── rho ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cp,label", [(1, "call"), (-1, "put")])
def test_rho_finite_diff(cp, label):
    """∂V/∂r scaled by 1/100 matches finite difference (same scaling)."""
    price_fn = _PRICE[label]
    fd = (
        (price_fn(S, K, r + _EPS_R, q, T, sigma) - price_fn(S, K, r - _EPS_R, q, T, sigma))
        / (2 * _EPS_R)
        / 100
    )
    assert abs(bsm_rho(S, K, r, q, T, sigma, cp) - fd) < _TOL


def test_rho_call_positive_put_negative():
    """Call rho > 0 (higher r raises call price); put rho < 0."""
    assert bsm_rho(S, K, r, q, T, sigma, cp=1) > 0
    assert bsm_rho(S, K, r, q, T, sigma, cp=-1) < 0


# ── vanna ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cp,label", [(1, "call"), (-1, "put")])
def test_vanna_finite_diff(cp, label):
    """∂delta/∂σ matches central finite difference."""
    fd = (
        bsm_delta(S, K, r, q, T, sigma + _EPS_V, cp) - bsm_delta(S, K, r, q, T, sigma - _EPS_V, cp)
    ) / (2 * _EPS_V)
    assert abs(bsm_vanna(S, K, r, q, T, sigma, cp) - fd) < _TOL


def test_vanna_call_equals_put():
    """Vanna is identical for calls and puts."""
    va_c = bsm_vanna(S, K, r, q, T, sigma, cp=1)
    va_p = bsm_vanna(S, K, r, q, T, sigma, cp=-1)
    assert abs(va_c - va_p) < 1e-12


# ── volga ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cp,label", [(1, "call"), (-1, "put")])
def test_volga_finite_diff(cp, label):
    """∂vega/∂σ matches central finite difference."""
    fd = (
        bsm_vega(S, K, r, q, T, sigma + _EPS_V, cp) - bsm_vega(S, K, r, q, T, sigma - _EPS_V, cp)
    ) / (2 * _EPS_V)
    assert abs(bsm_volga(S, K, r, q, T, sigma, cp) - fd) < _TOL


def test_volga_call_equals_put():
    """Volga is identical for calls and puts."""
    vo_c = bsm_volga(S, K, r, q, T, sigma, cp=1)
    vo_p = bsm_volga(S, K, r, q, T, sigma, cp=-1)
    assert abs(vo_c - vo_p) < 1e-12


# ── BSM PDE identity ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("cp,label", [(1, "call"), (-1, "put")])
def test_bsm_pde_identity(cp, label):
    """BSM PDE: theta + 0.5*sigma²*S²*gamma + (r-q)*S*delta - r*V = 0."""
    price_fn = _PRICE[label]
    V = price_fn(S, K, r, q, T, sigma)
    θ = bsm_theta(S, K, r, q, T, sigma, cp)
    Δ = bsm_delta(S, K, r, q, T, sigma, cp)
    Γ = bsm_gamma(S, K, r, q, T, sigma, cp)
    pde = θ + 0.5 * sigma**2 * S**2 * Γ + (r - q) * S * Δ - r * V
    assert abs(pde) < 1e-8, f"BSM PDE residual = {pde:.2e}"


# ── bsm_all_greeks ────────────────────────────────────────────────────────────


def test_all_greeks_returns_correct_keys():
    result = bsm_all_greeks(S, K, r, q, T, sigma)
    expected = {"delta", "gamma", "vega", "theta", "rho", "vanna", "volga"}
    assert set(result.keys()) == expected


def test_all_greeks_values_match_individual():
    g = bsm_all_greeks(S, K, r, q, T, sigma, cp=1)
    assert g["delta"] == bsm_delta(S, K, r, q, T, sigma, cp=1)
    assert g["gamma"] == bsm_gamma(S, K, r, q, T, sigma, cp=1)
    assert g["vega"] == bsm_vega(S, K, r, q, T, sigma, cp=1)
    assert g["theta"] == bsm_theta(S, K, r, q, T, sigma, cp=1)
    assert g["rho"] == bsm_rho(S, K, r, q, T, sigma, cp=1)
    assert g["vanna"] == bsm_vanna(S, K, r, q, T, sigma, cp=1)
    assert g["volga"] == bsm_volga(S, K, r, q, T, sigma, cp=1)


# ── strike-grid sanity ────────────────────────────────────────────────────────


@pytest.mark.parametrize("K_val", [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
def test_greeks_are_finite_across_strikes(K_val):
    """All Greeks should be finite for a wide strike range."""
    for cp in (1, -1):
        g = bsm_all_greeks(S, K_val, r, q, T, sigma, cp)
        for name, v in g.items():
            assert np.isfinite(v), f"{name} is not finite for K={K_val}, cp={cp}"
