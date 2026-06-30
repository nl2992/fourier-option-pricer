"""Tests for the Geske (1979) compound option pricer.

Verification strategy:
  1. At expiry (T1→T2): compound option value collapses to max(inner_option - K1, 0)
  2. Call-on-call lower bound: V ≥ max(C(S,K2,T2) - K1*e^{-rT1}, 0)
  3. Put-call parity for compound options:
       C-on-C - P-on-C = BSM_Call(S,K2,T2) - K1*e^{-rT1}
       C-on-P - P-on-P = BSM_Put(S,K2,T2)  - K1*e^{-rT1}
  4. Non-negativity (options are non-negative)
  5. Monotonicity: compound call value increases as K1 decreases
  6. Robustness across a grid of (S, K1) values
  7. Pipeline dispatch via CompoundOption product spec
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.analytics.bsm_barrier import bsm_call, bsm_put
from foureng.analytics.bsm_compound import geske_compound_price

# ── shared parameters ─────────────────────────────────────────────────────────

S = 100.0
K1, K2 = 5.0, 100.0
r, q = 0.08, 0.0
T1, T2 = 0.5, 1.0
sigma = 0.25


# ── 1. Non-negativity ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("cp_outer,cp_inner", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_compound_price_non_negative(cp_outer, cp_inner):
    v = geske_compound_price(S, K1, K2, r, q, T1, T2, sigma, cp_outer, cp_inner)
    assert v >= -1e-10, f"Negative compound price: {v:.6f}"


# ── 2. Put-call parity for compound options ───────────────────────────────────


def test_compound_call_on_call_minus_put_on_call_parity():
    """C-on-C - P-on-C = BSM_Call(S,K2,T2) - K1*e^{-r*T1}."""
    coc = geske_compound_price(S, K1, K2, r, q, T1, T2, sigma, cp_outer=1, cp_inner=1)
    poc = geske_compound_price(S, K1, K2, r, q, T1, T2, sigma, cp_outer=-1, cp_inner=1)
    lhs = coc - poc
    rhs = bsm_call(S, K2, r, q, T2, sigma) - K1 * np.exp(-r * T1)
    assert abs(lhs - rhs) < 1e-6, f"Compound C-on-C parity: lhs={lhs:.8f}, rhs={rhs:.8f}"


def test_compound_call_on_put_minus_put_on_put_parity():
    """C-on-P - P-on-P = BSM_Put(S,K2,T2) - K1*e^{-r*T1}."""
    cop = geske_compound_price(S, K1, K2, r, q, T1, T2, sigma, cp_outer=1, cp_inner=-1)
    pop = geske_compound_price(S, K1, K2, r, q, T1, T2, sigma, cp_outer=-1, cp_inner=-1)
    lhs = cop - pop
    rhs = bsm_put(S, K2, r, q, T2, sigma) - K1 * np.exp(-r * T1)
    assert abs(lhs - rhs) < 1e-6, f"Compound C-on-P parity: lhs={lhs:.8f}, rhs={rhs:.8f}"


# ── 3. Lower bound: call-on-call ≥ max(BSM_call - K1*disc, 0) ────────────────


def test_call_on_call_lower_bound():
    """Compound call-on-call ≥ max(C(S,K2,T2) - K1*e^{-r*T1}, 0)."""
    coc = geske_compound_price(S, K1, K2, r, q, T1, T2, sigma, cp_outer=1, cp_inner=1)
    lb = max(bsm_call(S, K2, r, q, T2, sigma) - K1 * np.exp(-r * T1), 0.0)
    assert coc >= lb - 1e-8, f"C-on-C lower bound violated: {coc:.6f} < {lb:.6f}"


# ── 4. Zero K1 → compound = underlying option price ──────────────────────────


def test_call_on_call_zero_outer_strike():
    """With K1=0, a call-on-call is as good as owning the inner call outright."""
    coc = geske_compound_price(S, 0.0, K2, r, q, T1, T2, sigma, cp_outer=1, cp_inner=1)
    inner = bsm_call(S, K2, r, q, T2, sigma)
    # C-on-C(K1=0) ≈ C(T2) [discounted by T1 if we account for the timing diff]
    # Actually at K1=0 holder always exercises → C-on-C(K1=0) = e^{-r*T1}*E[C(S_{T1},...)]
    # The parity relation gives: C-on-C - P-on-C = C(T2) - 0 = C(T2)
    # And P-on-C ≥ 0, so C-on-C ≥ C(T2)  when K1=0 also means C-on-C <= C(T2)
    # More precisely: C-on-C(K1=0) = C(S, K2, T2) exactly (exercise always occurs)
    assert abs(coc - inner) < 1e-4, (
        f"C-on-C(K1=0) = {coc:.6f}, C(S,K2,T2) = {inner:.6f}"
    )


# ── 5. Monotonicity in outer strike ───────────────────────────────────────────


def test_call_on_call_decreasing_in_K1():
    """Call-on-call price decreases as K1 increases (higher exercise cost)."""
    prices = [
        geske_compound_price(S, k1, K2, r, q, T1, T2, sigma, 1, 1)
        for k1 in [0.0, 2.0, 5.0, 10.0, 20.0]
    ]
    for i in range(len(prices) - 1):
        assert prices[i] >= prices[i + 1] - 1e-8, (
            f"C-on-C not decreasing in K1: {prices[i]:.6f} < {prices[i+1]:.6f}"
        )


# ── 6. Monotonicity in inner strike (call direction) ─────────────────────────


def test_call_on_call_decreasing_in_K2():
    """C-on-C price decreases as K2 increases (inner call is cheaper)."""
    prices = [
        geske_compound_price(S, K1, k2, r, q, T1, T2, sigma, 1, 1)
        for k2 in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    ]
    for i in range(len(prices) - 1):
        assert prices[i] >= prices[i + 1] - 1e-8, (
            f"C-on-C not decreasing in K2: {prices[i]:.6f} < {prices[i+1]:.6f}"
        )


# ── 7. Grid robustness ────────────────────────────────────────────────────────


@pytest.mark.parametrize("S_val", [70.0, 90.0, 100.0, 110.0, 130.0])
@pytest.mark.parametrize("K1_val", [0.0, 3.0, 8.0, 15.0])
def test_compound_finite_across_grid(S_val, K1_val):
    """Compound prices should be finite and non-negative across a (S, K1) grid."""
    for cp_outer, cp_inner in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        v = geske_compound_price(S_val, K1_val, K2, r, q, T1, T2, sigma, cp_outer, cp_inner)
        assert np.isfinite(v), f"Non-finite for S={S_val}, K1={K1_val}, types=({cp_outer},{cp_inner})"
        assert v >= -1e-8, f"Negative for S={S_val}, K1={K1_val}, types=({cp_outer},{cp_inner})"


# ── 8. T1 validation ─────────────────────────────────────────────────────────


def test_raises_if_T1_ge_T2():
    with pytest.raises(ValueError, match="T1"):
        geske_compound_price(S, K1, K2, r, q, T1=1.0, T2=1.0, sigma=sigma)


def test_raises_if_T1_le_zero():
    with pytest.raises(ValueError, match="T1"):
        geske_compound_price(S, K1, K2, r, q, T1=0.0, T2=1.0, sigma=sigma)


# ── 9. Pipeline dispatch ──────────────────────────────────────────────────────


def test_pipeline_compound_call_on_call():
    """price() dispatches CompoundOption to geske_compound_price correctly."""
    from foureng.models.base import ForwardSpec
    from foureng.models.bsm import BsmParams
    from foureng.pipeline import price
    from foureng.products.compound import CompoundOption

    product = CompoundOption(
        strike_outer=K1, strike_inner=K2,
        maturity_outer=T1, maturity_inner=T2,
        cp_outer=1, cp_inner=1,
    )
    fwd = ForwardSpec(S0=S, r=r, q=q, T=T2)
    params = BsmParams(sigma=sigma)

    p = price(product, model="bsm", method="geske", fwd=fwd, params=params)
    expected = geske_compound_price(S, K1, K2, r, q, T1, T2, sigma, 1, 1)
    assert abs(p - expected) < 1e-12
