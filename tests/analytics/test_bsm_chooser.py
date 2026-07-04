"""Tests for the Rubinstein (1991) simple chooser option pricer.

Verification:
  1. Chooser price ≥ max(BSM_call, BSM_put) (it dominates either option)
  2. As T_choice → 0: chooser → max(call, put)
  3. As T_choice → T_exp: chooser → straddle (call + put)
  4. Lower bound: chooser ≥ BSM_call(T_exp) and ≥ BSM_put(T_exp)
  5. Non-negativity
  6. Monotonicity: increasing T_choice raises chooser price
  7. Pipeline dispatch via ChooserOption product spec
  8. Rubinstein decomposition: chooser = call(K*,T_choice) + put(K,T_exp)
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.analytics.bsm_barrier import bsm_call, bsm_put
from foureng.analytics.bsm_chooser import bsm_chooser_price

# ── shared parameters ─────────────────────────────────────────────────────────

S = 100.0
K = 100.0
r, q = 0.05, 0.02
T_choice = 0.5
T_exp = 1.0
sigma = 0.25


# ── 1. Non-negativity ─────────────────────────────────────────────────────────


def test_chooser_non_negative():
    v = bsm_chooser_price(S, K, r, q, T_choice, T_exp, sigma)
    assert v >= 0.0


# ── 2. Chooser ≥ call and ≥ put ──────────────────────────────────────────────


def test_chooser_dominates_call():
    """Chooser must be worth at least as much as a plain call."""
    ch = bsm_chooser_price(S, K, r, q, T_choice, T_exp, sigma)
    call = bsm_call(S, K, r, q, T_exp, sigma)
    assert ch >= call - 1e-8, f"chooser={ch:.6f} < call={call:.6f}"


def test_chooser_dominates_put():
    """Chooser must be worth at least as much as a plain put."""
    ch = bsm_chooser_price(S, K, r, q, T_choice, T_exp, sigma)
    put = bsm_put(S, K, r, q, T_exp, sigma)
    assert ch >= put - 1e-8, f"chooser={ch:.6f} < put={put:.6f}"


# ── 3. T_choice → T_exp: chooser ≈ straddle ──────────────────────────────────


def test_chooser_approaches_straddle_as_T_choice_to_T_exp():
    """As T_choice → T_exp, chooser → call + put (ATM straddle)."""
    T_choice_near = T_exp * (1 - 1e-6)
    ch = bsm_chooser_price(S, K, r, q, T_choice_near, T_exp, sigma)
    straddle = bsm_call(S, K, r, q, T_exp, sigma) + bsm_put(S, K, r, q, T_exp, sigma)
    assert abs(ch - straddle) < 1e-4, f"Chooser near T_exp: {ch:.6f} vs straddle: {straddle:.6f}"


# ── 4. T_choice very small: chooser ≈ max(call, put) ─────────────────────────


def test_chooser_approaches_max_call_put_as_T_choice_to_zero_zero_dividend():
    """As T_choice → 0 (q=0), the chooser → max(BSM_call, BSM_put)."""
    eps = 1e-5
    ch = bsm_chooser_price(S, K, r, q=0.0, T_choice=eps, T_exp=T_exp, sigma=sigma)
    call = bsm_call(S, K, r, q=0.0, T=T_exp, sigma=sigma)
    put = bsm_put(S, K, r, q=0.0, T=T_exp, sigma=sigma)
    expected = max(call, put)
    assert abs(ch - expected) < 1e-3, f"Chooser near 0 (q=0): {ch:.6f} vs max(C,P): {expected:.6f}"


# ── 5. Rubinstein decomposition ───────────────────────────────────────────────


def test_rubinstein_decomposition():
    """Verify: chooser = call(K*, T_choice) + put(K, T_exp) exactly."""
    K_star = K * np.exp(-(r - q) * (T_exp - T_choice))
    call_part = bsm_call(S, K_star, r, q, T_choice, sigma)
    put_part = bsm_put(S, K, r, q, T_exp, sigma)
    decomp = call_part + put_part
    ch = bsm_chooser_price(S, K, r, q, T_choice, T_exp, sigma)
    assert abs(ch - decomp) < 1e-12, (
        f"Rubinstein decomposition: chooser={ch:.10f}, decomp={decomp:.10f}"
    )


# ── 6. Monotonicity in T_choice ───────────────────────────────────────────────


def test_chooser_increasing_in_T_choice():
    """Longer choice window → higher chooser price (more optionality)."""
    prices = [
        bsm_chooser_price(S, K, r, q, tc, T_exp, sigma) for tc in np.linspace(0.05, T_exp - 0.01, 8)
    ]
    for i in range(len(prices) - 1):
        assert prices[i] <= prices[i + 1] + 1e-6, (
            f"Chooser not increasing in T_choice: {prices[i]:.6f} > {prices[i + 1]:.6f}"
        )


# ── 7. Robustness across (S, K) grid ─────────────────────────────────────────


@pytest.mark.parametrize("S_val", [70.0, 90.0, 100.0, 110.0, 130.0])
@pytest.mark.parametrize("K_val", [80.0, 100.0, 120.0])
def test_chooser_finite_across_grid(S_val, K_val):
    v = bsm_chooser_price(S_val, K_val, r, q, T_choice, T_exp, sigma)
    assert np.isfinite(v)
    assert v >= 0.0


# ── 8. Validation errors ─────────────────────────────────────────────────────


def test_chooser_raises_if_T_choice_ge_T_exp():
    with pytest.raises(ValueError, match="T_choice"):
        bsm_chooser_price(S, K, r, q, T_choice=1.0, T_exp=0.5, sigma=sigma)


def test_chooser_raises_if_T_choice_zero():
    with pytest.raises(ValueError, match="T_choice"):
        bsm_chooser_price(S, K, r, q, T_choice=0.0, T_exp=1.0, sigma=sigma)


# ── 9. Pipeline dispatch ──────────────────────────────────────────────────────


def test_pipeline_chooser():
    """price() dispatches ChooserOption to bsm_chooser_price correctly."""
    from foureng.models.base import ForwardSpec
    from foureng.models.bsm import BsmParams
    from foureng.pipeline import price
    from foureng.products.chooser import ChooserOption

    product = ChooserOption(strike=K, maturity_choice=T_choice, maturity_expiry=T_exp)
    fwd = ForwardSpec(S0=S, r=r, q=q, T=T_exp)
    params = BsmParams(sigma=sigma)

    p = price(product, model="bsm", method="analytic", fwd=fwd, params=params)
    expected = bsm_chooser_price(S, K, r, q, T_choice, T_exp, sigma)
    assert abs(p - expected) < 1e-12
