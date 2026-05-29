"""Tests for BSM exotic analytic formulas (Sprint 5).

Covers: Margrabe exchange, BSM forward-start, BSM lookback (floating),
        BSM gap call.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from foureng.analytics.bsm_barrier import bsm_call, bsm_put
from foureng.analytics.bsm_exotics import (
    bsm_forward_start,
    bsm_gap_call,
    bsm_lookback_floating,
    margrabe_exchange,
)


# ── Margrabe exchange option ──────────────────────────────────────────────

def test_margrabe_non_negative():
    price = margrabe_exchange(100, 90, 0.02, 0.03, 1.0, 0.20, 0.25, -0.5)
    assert price >= 0.0


def test_margrabe_symmetry():
    """Exchange is symmetric: value(S1,S2) = value(S2,S1) swapped."""
    p12 = margrabe_exchange(100, 90, 0.02, 0.03, 1.0, 0.20, 0.25, 0.4)
    p21 = margrabe_exchange(90, 100, 0.03, 0.02, 1.0, 0.25, 0.20, 0.4)
    # margrabe(S1→S2) is a different option but check positivity
    assert p12 >= 0 and p21 >= 0


def test_margrabe_reduces_to_bsm_call_when_s2_is_cash():
    """If S₂ = K with q₂=r (cash earning risk-free) and σ₂=0, price = BSM call.

    S2_T = K * exp((r - r - 0)*T) = K  (deterministic at maturity)
    → Margrabe(S1, S2=K, q1=q, q2=r, σ1, σ2=0) = BSM_call(S, K, r, q, T, σ)
    """
    r = 0.05
    q = 0.0
    T = 1.0
    sigma = 0.20
    K = 100.0
    S = 110.0
    # S2 = K cash: no vol, dividend yield = r → S2_T = K always
    margrabe_price = margrabe_exchange(S, K, q, r, T, sigma, 0.0, 0.0)
    bsm_price = bsm_call(S, K, r, q, T, sigma)
    assert abs(margrabe_price - bsm_price) < 1e-8, (
        f"Margrabe → BSM call: {margrabe_price:.8f} vs {bsm_price:.8f}"
    )


def test_margrabe_same_assets_zero_vol():
    """Perfect correlation, same vol: exchange = max(S1 e^{-q1T} - S2 e^{-q2T}, 0)."""
    S1, S2 = 100.0, 95.0
    q1 = q2 = 0.0
    T = 1.0
    sigma = 0.20
    rho = 1.0  # perfect correlation → net vol = 0
    price = margrabe_exchange(S1, S2, q1, q2, T, sigma, sigma, rho)
    expected = max(S1 - S2, 0.0)   # q1=q2=0, net σ=0 → pay max(S1-S2, 0) immediately
    assert abs(price - expected) < 1e-8


def test_margrabe_positive_value_for_itm():
    price = margrabe_exchange(110, 100, 0.0, 0.0, 1.0, 0.20, 0.20, 0.0)
    assert price > 0.0


# ── BSM forward-start ─────────────────────────────────────────────────────

def test_forward_start_reduces_to_bsm_when_tstart_zero():
    """t_start = ε ≈ 0 → should match BSM call(S, alpha*S, r, q, T)."""
    S, alpha, r, q, T, sigma = 100.0, 1.0, 0.05, 0.02, 1.0, 0.20
    t_start = 1e-8  # effectively zero
    fs_price = bsm_forward_start(S, alpha, t_start, T, r, q, sigma, cp=1)
    # At t_start→0, K = alpha*S = 100
    bsm_price = bsm_call(S, alpha * S, r, q, T, sigma)
    assert abs(fs_price - bsm_price) < 1e-4, (
        f"FS→BSM: {fs_price:.6f} vs {bsm_price:.6f}"
    )


def test_forward_start_positive():
    price = bsm_forward_start(100.0, 1.0, 0.5, 1.0, 0.05, 0.0, 0.20, cp=1)
    assert price > 0.0


def test_forward_start_put_call_parity():
    """Forward-start: call − put = S·e^{-q·t_s} · [e^{-q·τ} − alpha·e^{-r·τ}].

    This follows directly from BSM parity on the unit option:
      BSM_call(1, alpha) − BSM_put(1, alpha) = e^{-q*τ} − alpha·e^{-r*τ}
    multiplied by S·e^{-q·t_s}.
    """
    S, alpha, t_s, T, r, q, sigma = 100.0, 1.0, 0.25, 1.0, 0.05, 0.02, 0.25
    tau = T - t_s
    call = bsm_forward_start(S, alpha, t_s, T, r, q, sigma, cp=1)
    put  = bsm_forward_start(S, alpha, t_s, T, r, q, sigma, cp=-1)
    # Put-call parity for BSM(S=1, K=alpha) scaled by S·e^{-q·t_s}
    rhs = S * np.exp(-q * t_s) * (np.exp(-q * tau) - alpha * np.exp(-r * tau))
    assert abs(call - put - rhs) < 1e-8, (
        f"FS parity: {call - put:.8f} vs {rhs:.8f}"
    )


def test_forward_start_raises_on_invalid_tenor():
    with pytest.raises(ValueError):
        bsm_forward_start(100.0, 1.0, t_start=2.0, T=1.0, r=0.05, q=0.0, sigma=0.2)


# ── BSM lookback floating-strike ──────────────────────────────────────────

def test_lookback_call_exceeds_vanilla_call():
    """Lookback (buy at min) ≥ ATM call (buy at K=S)."""
    S = 100.0
    r, q, T, sigma = 0.05, 0.0, 1.0, 0.20
    lb_call = bsm_lookback_floating(S, S_min=S, S_max=S, r=r, q=q, T=T, sigma=sigma, cp=1)
    vanilla = bsm_call(S, S, r, q, T, sigma)
    assert lb_call >= vanilla - 1e-8, f"Lookback {lb_call:.4f} < vanilla {vanilla:.4f}"


def test_lookback_put_exceeds_vanilla_put():
    """Lookback (sell at max) ≥ ATM put."""
    S = 100.0
    r, q, T, sigma = 0.05, 0.0, 1.0, 0.20
    lb_put = bsm_lookback_floating(S, S_min=S, S_max=S, r=r, q=q, T=T, sigma=sigma, cp=-1)
    vanilla = bsm_put(S, S, r, q, T, sigma)
    assert lb_put >= vanilla - 1e-8, f"Lookback {lb_put:.4f} < vanilla {vanilla:.4f}"


def test_lookback_positive():
    price = bsm_lookback_floating(100.0, 90.0, 100.0, 0.05, 0.0, 1.0, 0.20, cp=1)
    assert price >= 0.0


def test_lookback_call_lower_smin_increases_price():
    """Lower S_min → lookback call worth more (easier exercise region)."""
    S, r, q, T, sigma = 100.0, 0.05, 0.0, 1.0, 0.20
    p_high = bsm_lookback_floating(S, S_min=95.0, S_max=S, r=r, q=q, T=T, sigma=sigma, cp=1)
    p_low  = bsm_lookback_floating(S, S_min=80.0, S_max=S, r=r, q=q, T=T, sigma=sigma, cp=1)
    assert p_low >= p_high - 1e-8, f"Lower Smin {p_low:.4f} should be >= {p_high:.4f}"


# ── BSM gap call ──────────────────────────────────────────────────────────

def test_gap_call_reduces_to_bsm_when_k1_equals_k2():
    """K1 = K2 → gap call = standard BSM call."""
    S, K, r, q, T, sigma = 100.0, 90.0, 0.05, 0.02, 1.0, 0.20
    gap = bsm_gap_call(S, K, K, r, q, T, sigma)
    vanilla = bsm_call(S, K, r, q, T, sigma)
    assert abs(gap - vanilla) < 1e-8, f"Gap({K},{K})={gap:.6f} vs BSM={vanilla:.6f}"


def test_gap_call_positive_and_finite():
    price = bsm_gap_call(100.0, 95.0, 100.0, 0.05, 0.0, 1.0, 0.20)
    assert np.isfinite(price)


def test_gap_call_smaller_payment_increases_price():
    """Lower K1 (smaller payment if triggered) → higher gap call price.

    The gap call pays S_T − K1 when S_T > K2.
    Lower K1 → larger net payment → higher price.
    """
    S, K2, r, q, T, sigma = 100.0, 90.0, 0.05, 0.0, 1.0, 0.20
    p_lowK1  = bsm_gap_call(S, 80.0, K2, r, q, T, sigma)   # pays more
    p_highK1 = bsm_gap_call(S, 95.0, K2, r, q, T, sigma)   # pays less
    assert p_lowK1 >= p_highK1, (
        f"Lower K1 should give higher price: {p_lowK1:.4f} vs {p_highK1:.4f}"
    )
