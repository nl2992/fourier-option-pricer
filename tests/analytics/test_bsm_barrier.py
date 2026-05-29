"""Tests for BSM analytic single-barrier option formulas (Phase 4.2).

Structural tests that hold for any correct implementation:
  1. In-out parity: knock-out + knock-in = vanilla (calls & puts, all 4 types)
  2. Non-negativity of every barrier price
  3. Barrier-price ≤ vanilla (knock-out discounts the vanilla)
  4. Monotone in barrier level (closer barrier → lower knock-out)
  5. Limit: H→0 (or H→∞) → knock-out → vanilla
  6. Already-breached boundary conditions (instant knockout / activation)
  7. Specific Haug-like numerical spot checks for down-and-out calls
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from foureng.analytics.bsm_barrier import bsm_barrier_price, bsm_call, bsm_put


_PARAMS = dict(r=0.05, q=0.02, T=1.0, sigma=0.25)
_PARAMS_SHORT = dict(r=0.05, q=0.00, T=0.5, sigma=0.30)


# ── in-out parity (the gold standard) ─────────────────────────────────────

@pytest.mark.parametrize("cp", [1, -1])
@pytest.mark.parametrize("case", [
    dict(S=100, K=90,  H=80,  base="down"),   # H < K < S  (down, call OTM, put OTM)
    dict(S=100, K=110, H=80,  base="down"),   # H < S < K
    dict(S=100, K=90,  H=120, base="up"),     # K < S < H
    dict(S=100, K=110, H=120, base="up"),     # S < K < H
])
def test_in_out_parity(cp, case):
    """knock-out + knock-in = vanilla to machine precision."""
    S, K, H = case["S"], case["K"], case["H"]
    p_out = bsm_barrier_price(S, K, H, cp=cp, barrier_type=case["base"]+"_out", **_PARAMS)
    p_in  = bsm_barrier_price(S, K, H, cp=cp, barrier_type=case["base"]+"_in",  **_PARAMS)
    vanilla = bsm_call(S, K, **_PARAMS) if cp == 1 else bsm_put(S, K, **_PARAMS)
    assert abs(p_out + p_in - vanilla) < 1e-8, (
        f"Parity failed: out={p_out:.8f} in={p_in:.8f} sum={p_out+p_in:.8f} vanilla={vanilla:.8f}"
    )


# ── non-negativity ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("bt", ["down_out", "down_in", "up_out", "up_in"])
@pytest.mark.parametrize("cp", [1, -1])
def test_barrier_price_non_negative(bt, cp):
    if bt.startswith("down"):
        H = 80.0
    else:
        H = 120.0
    for K in [90.0, 100.0, 110.0]:
        p = bsm_barrier_price(100.0, K, H, cp=cp, barrier_type=bt, **_PARAMS)
        assert p >= 0.0, f"Negative price: {bt} K={K} cp={cp} p={p}"


# ── knock-out ≤ vanilla ───────────────────────────────────────────────────

@pytest.mark.parametrize("cp", [1, -1])
@pytest.mark.parametrize("H,bt", [(80.0, "down_out"), (120.0, "up_out")])
def test_knockout_bounded_by_vanilla(cp, H, bt):
    for K in [90.0, 100.0, 110.0]:
        S = 100.0
        p_ko = bsm_barrier_price(S, K, H, cp=cp, barrier_type=bt, **_PARAMS)
        vanilla = bsm_call(S, K, **_PARAMS) if cp == 1 else bsm_put(S, K, **_PARAMS)
        assert p_ko <= vanilla + 1e-8, (
            f"{bt} cp={cp} K={K}: knock-out {p_ko:.6f} > vanilla {vanilla:.6f}"
        )


# ── monotone in barrier (knock-out decreases as barrier approaches spot) ──

def test_down_out_call_decreases_as_barrier_rises():
    """Raising H toward S reduces the down-out call (higher chance of knockout)."""
    S, K = 100.0, 90.0
    barriers = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0]
    prices = [bsm_barrier_price(S, K, H, cp=1, barrier_type="down_out", **_PARAMS)
              for H in barriers]
    assert all(prices[i] >= prices[i+1] - 1e-8 for i in range(len(prices)-1)), (
        f"Down-out call should decrease as barrier rises: {prices}"
    )


def test_up_out_put_decreases_as_barrier_falls():
    """Lowering H toward S reduces the up-out put."""
    S, K = 100.0, 110.0
    barriers = [140.0, 135.0, 130.0, 125.0, 120.0, 115.0]
    prices = [bsm_barrier_price(S, K, H, cp=-1, barrier_type="up_out", **_PARAMS)
              for H in barriers]
    assert all(prices[i] >= prices[i+1] - 1e-8 for i in range(len(prices)-1)), (
        f"Up-out put should decrease as barrier falls: {prices}"
    )


# ── limits ────────────────────────────────────────────────────────────────

def test_down_out_call_converges_to_vanilla_at_small_barrier():
    """H → 0: down-out call → vanilla call (barrier never hit)."""
    S, K = 100.0, 90.0
    vanilla = bsm_call(S, K, **_PARAMS)
    p_low_H = bsm_barrier_price(S, K, 1.0, cp=1, barrier_type="down_out", **_PARAMS)
    assert abs(p_low_H - vanilla) < 1e-5, (
        f"H→0: DOC={p_low_H:.6f} should ≈ vanilla={vanilla:.6f}"
    )


def test_up_out_put_converges_to_vanilla_at_large_barrier():
    """H → ∞: up-out put → vanilla put."""
    S, K = 100.0, 110.0
    vanilla = bsm_put(S, K, **_PARAMS)
    p_high_H = bsm_barrier_price(S, K, 1e6, cp=-1, barrier_type="up_out", **_PARAMS)
    assert abs(p_high_H - vanilla) < 1e-5, (
        f"H→∞: UOP={p_high_H:.6f} should ≈ vanilla={vanilla:.6f}"
    )


# ── boundary: spot at barrier → knockout = 0 ─────────────────────────────

def test_down_out_call_zero_when_spot_at_barrier():
    """If S == H, down-and-out call = 0 (immediately knocked out)."""
    p = bsm_barrier_price(80.0, 90.0, 80.0, cp=1, barrier_type="down_out", **_PARAMS)
    assert p == 0.0


def test_up_out_put_zero_when_spot_at_barrier():
    """If S == H, up-and-out put = 0."""
    p = bsm_barrier_price(120.0, 110.0, 120.0, cp=-1, barrier_type="up_out", **_PARAMS)
    assert p == 0.0


# ── boundary: spot already past barrier → knock-in = vanilla ─────────────

def test_down_in_call_equals_vanilla_when_already_knocked_in():
    """If S < H, down-in is already activated: price = vanilla."""
    S, K, H = 70.0, 90.0, 80.0   # S < H
    p_in = bsm_barrier_price(S, K, H, cp=1, barrier_type="down_in", **_PARAMS)
    vanilla = bsm_call(S, K, **_PARAMS)
    assert p_in == approx(vanilla, rel=1e-10)


def test_up_in_put_equals_vanilla_when_already_knocked_in():
    """If S > H, up-in is already activated: price = vanilla."""
    S, K, H = 130.0, 110.0, 120.0   # S > H
    p_in = bsm_barrier_price(S, K, H, cp=-1, barrier_type="up_in", **_PARAMS)
    vanilla = bsm_put(S, K, **_PARAMS)
    assert p_in == approx(vanilla, rel=1e-10)


# ── numerical spot checks (self-consistent round-trip) ────────────────────

def test_down_out_call_specific_value():
    """
    Reference: Haug (2007) Table 2.14 calibration:
    S=100, K=90, H=80, r=0.05, q=0.02, T=1, σ=0.25
    DOC should be significantly less than vanilla call (~16.64).
    Verify value is in (0, vanilla) and consistent with DIC = vanilla - DOC.
    """
    S, K, H = 100.0, 90.0, 80.0
    r, q, T, sigma = 0.05, 0.02, 1.0, 0.25
    vanilla = bsm_call(S, K, r=r, q=q, T=T, sigma=sigma)
    doc = bsm_barrier_price(S, K, H, r=r, q=q, T=T, sigma=sigma,
                             barrier_type="down_out", cp=1)
    dic = bsm_barrier_price(S, K, H, r=r, q=q, T=T, sigma=sigma,
                             barrier_type="down_in", cp=1)
    assert 0 < doc < vanilla, f"DOC={doc:.4f} not in (0, {vanilla:.4f})"
    assert abs(doc + dic - vanilla) < 1e-8


# ── cross-maturity stress ─────────────────────────────────────────────────

@pytest.mark.parametrize("T", [0.1, 0.5, 1.0, 2.0])
def test_in_out_parity_across_maturities(T):
    """In-out parity holds for a range of maturities."""
    S, K, H = 100.0, 95.0, 85.0
    r, q, sigma = 0.04, 0.01, 0.20
    doc = bsm_barrier_price(S, K, H, r=r, q=q, T=T, sigma=sigma,
                             barrier_type="down_out", cp=1)
    dic = bsm_barrier_price(S, K, H, r=r, q=q, T=T, sigma=sigma,
                             barrier_type="down_in", cp=1)
    vanilla = bsm_call(S, K, r=r, q=q, T=T, sigma=sigma)
    assert abs(doc + dic - vanilla) < 1e-8
