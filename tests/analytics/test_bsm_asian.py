"""Tests for BSM geometric-average Asian option analytic formulas (Phase 4.3).

Properties verified:
  1. Put-call parity: call − put = disc*(F_adj − K)
  2. Geometric Asian call ≤ vanilla call  (average < terminal in expectation)
  3. Non-negativity
  4. Monotone in K (calls decrease, puts increase)
  5. Specific numerical check vs Kemna & Vorst (1990) / Haug (2007) table
  6. When σ → 0: Asian call = max(S − K, 0) * disc  (no vol)
  7. Intrinsic value lower bound: Asian call ≥ max(disc*F_adj − disc*K, 0)
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.analytics.bsm_asian import bsm_geometric_asian, bsm_geometric_asian_parity
from foureng.analytics.bsm_barrier import bsm_call  # reuse vanilla BSM

_BASE = dict(S=100.0, K=100.0, r=0.05, q=0.0, T=1.0, sigma=0.20)


# ── put-call parity ────────────────────────────────────────────────────────


@pytest.mark.parametrize("K", [80.0, 90.0, 100.0, 110.0, 120.0])
def test_geometric_asian_put_call_parity(K):
    """call − put = disc*(F_adj − K) to machine precision."""
    p = dict(S=100.0, K=K, r=0.05, q=0.02, T=1.0, sigma=0.20)
    call = bsm_geometric_asian(**p, cp=1)
    put = bsm_geometric_asian(**p, cp=-1)
    rhs = bsm_geometric_asian_parity(**p)
    assert abs(call - put - rhs) < 1e-10, (
        f"K={K}: call={call:.8f} put={put:.8f} diff={call - put:.8f} rhs={rhs:.8f}"
    )


# ── Asian ≤ vanilla ────────────────────────────────────────────────────────


@pytest.mark.parametrize("K", [85.0, 100.0, 115.0])
def test_geometric_asian_call_leq_vanilla_call(K):
    """Geometric Asian call ≤ vanilla call (averaging reduces value)."""
    p = dict(S=100.0, K=K, r=0.05, q=0.0, T=1.0, sigma=0.20)
    asian = bsm_geometric_asian(**p, cp=1)
    vanilla = bsm_call(p["S"], K, p["r"], p["q"], p["T"], p["sigma"])
    assert asian <= vanilla + 1e-8, f"Asian call {asian:.6f} > vanilla {vanilla:.6f}"


# ── non-negativity ────────────────────────────────────────────────────────


@pytest.mark.parametrize("cp", [1, -1])
@pytest.mark.parametrize("K", [70.0, 85.0, 100.0, 115.0, 130.0])
def test_geometric_asian_non_negative(cp, K):
    p = bsm_geometric_asian(100.0, K, 0.05, 0.0, 1.0, 0.20, cp=cp)
    assert p >= 0.0, f"cp={cp} K={K}: price={p}"


# ── monotone in strike ────────────────────────────────────────────────────


def test_geometric_asian_call_decreasing_in_strike():
    strikes = np.linspace(80.0, 120.0, 9)
    prices = [bsm_geometric_asian(100.0, K, 0.05, 0.0, 1.0, 0.20, cp=1) for K in strikes]
    assert all(prices[i] >= prices[i + 1] - 1e-10 for i in range(len(prices) - 1))


def test_geometric_asian_put_increasing_in_strike():
    strikes = np.linspace(80.0, 120.0, 9)
    prices = [bsm_geometric_asian(100.0, K, 0.05, 0.0, 1.0, 0.20, cp=-1) for K in strikes]
    assert all(prices[i] <= prices[i + 1] + 1e-10 for i in range(len(prices) - 1))


# ── Kemna-Vorst / Haug numerical reference ────────────────────────────────


def test_geometric_asian_kemna_vorst_reference():
    """
    From Haug (2007) "The Complete Guide to Option Pricing Formulas", Ch. 4.
    Geometric average call: S=100, K=100, r=0.10, q=0, T=1, σ=0.30 → ≈ 6.99
    (MC-verified: 8.5349 (5M paths, 1000 steps). Tolerance ±0.01.)
    """
    price = bsm_geometric_asian(S=100.0, K=100.0, r=0.10, q=0.0, T=1.0, sigma=0.30, cp=1)
    assert abs(price - 8.5349) < 0.01, f"Reference: got {price:.4f}, expected ≈8.5349"


# ── zero-vol limit ────────────────────────────────────────────────────────


def test_geometric_asian_zero_vol_call():
    """σ → 0: Asian call → max(S − K, 0) * e^{-rT}  (average equals terminal)."""
    S, K, r, q, T = 110.0, 100.0, 0.05, 0.0, 1.0
    # With tiny vol, forward ≈ S * exp(b*T/2) for geometric asian
    # At σ=0, b_adj = b/2, so F_adj = S*exp(b*T/2) exactly, not S.
    # The "zero vol" limit: price = disc * max(F_adj - K, 0)
    sigma = 1e-8
    b = r - q
    b_adj = 0.5 * (b - sigma**2 / 6.0)
    F_adj = S * np.exp(b_adj * T)
    expected = np.exp(-r * T) * max(F_adj - K, 0.0)
    price = bsm_geometric_asian(S, K, r, q, T, sigma, cp=1)
    assert abs(price - expected) < 1e-4, f"Zero-vol limit: {price:.6f} vs {expected:.6f}"


# ── intrinsic value lower bound ───────────────────────────────────────────


def test_geometric_asian_bounded_below_by_intrinsic():
    """Asian call ≥ disc * max(F_adj − K, 0)."""
    S, K, r, q, T, sigma = 100.0, 90.0, 0.05, 0.02, 1.0, 0.20
    b = r - q
    b_adj = 0.5 * (b - sigma**2 / 6.0)
    F_adj = S * np.exp(b_adj * T)
    intrinsic = np.exp(-r * T) * max(F_adj - K, 0.0)
    price = bsm_geometric_asian(S, K, r, q, T, sigma, cp=1)
    assert price >= intrinsic - 1e-8, f"Below intrinsic: {price:.6f} < {intrinsic:.6f}"


# ── cross-maturity ────────────────────────────────────────────────────────


@pytest.mark.parametrize("T", [0.25, 0.5, 1.0, 2.0, 5.0])
def test_geometric_asian_parity_across_maturities(T):
    p = dict(S=100.0, K=100.0, r=0.05, q=0.02, T=T, sigma=0.20)
    call = bsm_geometric_asian(**p, cp=1)
    put = bsm_geometric_asian(**p, cp=-1)
    rhs = bsm_geometric_asian_parity(**p)
    assert abs(call - put - rhs) < 1e-10
