"""Tests for BSM double barrier pricing (eigenfunction expansion).

Key invariants:
  1. In-out parity: double_out + double_in = vanilla.
  2. Already-breached spot → KO=0, KI=vanilla.
  3. Wider barriers → higher KO price (monotonicity).
  4. Double-KO ≤ single-barrier KO ≤ vanilla.
  5. Barrier symmetry: call/put in-out parity.
  6. Convergence: n_terms insensitivity (10 vs 100 terms).
  7. Pipeline dispatch: price() with method='double_barrier_bsm'.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.analytics.bsm_barrier import bsm_call, bsm_double_barrier_price, bsm_put


# ── shared fixture ─────────────────────────────────────────────────────────

S, K, L, U = 100.0, 100.0, 80.0, 130.0
r, q, sigma, T = 0.05, 0.00, 0.20, 1.0


# ── 1. In-out parity ───────────────────────────────────────────────────────


def test_double_barrier_inout_parity_call():
    ko = bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=1, barrier_type="double_out")
    ki = bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=1, barrier_type="double_in")
    vanilla = bsm_call(S, K, r, q, T, sigma)
    assert abs(ko + ki - vanilla) < 5e-6, f"parity error: ko={ko:.6f}, ki={ki:.6f}, vanilla={vanilla:.6f}"


def test_double_barrier_inout_parity_put():
    ko = bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=-1, barrier_type="double_out")
    ki = bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=-1, barrier_type="double_in")
    vanilla = bsm_put(S, K, r, q, T, sigma)
    assert abs(ko + ki - vanilla) < 5e-6, f"put parity error: ko={ko:.6f}, ki={ki:.6f}, vanilla={vanilla:.6f}"


# ── 2. Already-breached spot ───────────────────────────────────────────────


def test_spot_at_lower_barrier_ko_zero():
    price = bsm_double_barrier_price(L, K, L, U, r, q, T, sigma, cp=1, barrier_type="double_out")
    assert price == 0.0


def test_spot_at_upper_barrier_ko_zero():
    price = bsm_double_barrier_price(U, K, L, U, r, q, T, sigma, cp=1, barrier_type="double_out")
    assert price == 0.0


def test_spot_below_lower_barrier_ki_vanilla():
    spot_below = L - 5.0
    ki = bsm_double_barrier_price(spot_below, K, L, U, r, q, T, sigma, cp=1, barrier_type="double_in")
    vanilla = bsm_call(spot_below, K, r, q, T, sigma)
    assert abs(ki - vanilla) < 1e-10


# ── 3. Monotonicity in barrier width ──────────────────────────────────────


def test_double_ko_monotone_in_width():
    """Wider corridors → lower knockout probability → higher KO call price."""
    narrow = bsm_double_barrier_price(S, K, 90.0, 115.0, r, q, T, sigma, cp=1)
    medium = bsm_double_barrier_price(S, K, 80.0, 130.0, r, q, T, sigma, cp=1)
    wide   = bsm_double_barrier_price(S, K, 60.0, 160.0, r, q, T, sigma, cp=1)
    assert narrow <= medium <= wide, f"Not monotone: narrow={narrow:.4f}, medium={medium:.4f}, wide={wide:.4f}"


# ── 4. Double KO ≤ single barrier KO ≤ vanilla ────────────────────────────


def test_double_ko_le_single_barrier_ko():
    from foureng.analytics.bsm_barrier import bsm_barrier_price

    # Same lower barrier on both; upper barrier set very high so double ≈ single-down
    double_ko = bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=1)
    # Single down-and-out call (reference)
    single_do = bsm_barrier_price(S, K, L, r, q, T, sigma, "down_out", cp=1)
    vanilla = bsm_call(S, K, r, q, T, sigma)

    # Double KO is harder to survive than single KO, so price ≤ single
    assert double_ko <= single_do + 1e-6, f"double_ko={double_ko:.4f} > single_do={single_do:.4f}"
    assert single_do <= vanilla + 1e-6


# ── 5. Non-negativity across strikes ──────────────────────────────────────


def test_non_negative_across_strikes():
    for K_val in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]:
        for bt in ("double_out", "double_in"):
            for cp_val in (1, -1):
                p = bsm_double_barrier_price(S, K_val, L, U, r, q, T, sigma, cp=cp_val, barrier_type=bt)
                assert p >= -1e-8, f"Negative price {p:.6f} for K={K_val}, bt={bt}, cp={cp_val}"


# ── 6. n_terms convergence ────────────────────────────────────────────────


def test_convergence_in_n_terms():
    p30 = bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=1, n_terms=30)
    p100 = bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=1, n_terms=100)
    assert abs(p30 - p100) < 1e-8, f"n_terms=30 and n_terms=100 differ: {p30:.8f} vs {p100:.8f}"


# ── 7. Input validation ────────────────────────────────────────────────────


def test_invalid_barriers():
    with pytest.raises(ValueError, match="0 < L < U"):
        bsm_double_barrier_price(S, K, 120.0, 80.0, r, q, T, sigma)  # L > U


def test_invalid_cp():
    with pytest.raises(ValueError, match="cp must be"):
        bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=0)


# ── 8. Pipeline dispatch ────────────────────────────────────────────────────


def test_pipeline_double_barrier_bsm():
    from foureng.models.base import ForwardSpec
    from foureng.models.bsm import BsmParams
    from foureng.pipeline import price
    from foureng.products.barrier import DoubleBarrierOption

    product = DoubleBarrierOption(
        strike=K, lower_barrier=L, upper_barrier=U, maturity=T, cp=1
    )
    fwd = ForwardSpec(S0=S, r=r, q=q, T=T)
    params = BsmParams(sigma=sigma)

    p_pipeline = price(product, "bsm", "double_barrier_bsm", fwd, params)
    p_direct = bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=1)

    assert abs(p_pipeline - p_direct) < 1e-12, (
        f"Pipeline vs direct mismatch: {p_pipeline:.8f} vs {p_direct:.8f}"
    )
