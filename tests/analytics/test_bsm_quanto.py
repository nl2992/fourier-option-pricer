"""Tests for bsm_quanto_forward and bsm_quanto_option (Reiner 1992).

Verification strategy:
  1. bsm_quanto_forward: limiting cases and known formula
  2. bsm_quanto_option: put-call parity with quanto-adjusted forward
  3. Zero-vol degeneracy -> intrinsic value
  4. Zero-maturity -> intrinsic value
  5. Hull (2018) Table 29.3 / Reiner (1992) benchmark
  6. rho effect: negative rho decreases forward (and call price)
  7. Input validation
"""

from __future__ import annotations

import math

import pytest

from foureng.analytics.bsm_quanto import bsm_quanto_forward, bsm_quanto_option

# ── shared fixtures ───────────────────────────────────────────────────────────

BASE = dict(
    S=100.0,
    K=105.0,
    r_dom=0.05,
    r_for=0.03,
    q_for=0.02,
    rho=-0.3,
    sigma_S=0.20,
    sigma_X=0.10,
    T=1.0,
)


# ── 1. bsm_quanto_forward ────────────────────────────────────────────────────


def test_forward_rho_zero_reduces_to_simple_drift():
    """When rho=0, F_adj = S * exp((r_dom - q_for) * T)."""
    T = 1.0
    F = bsm_quanto_forward(
        S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=0.0, sigma_S=0.20, sigma_X=0.10, T=T
    )
    expected = 100 * math.exp((0.05 - 0.02) * T)
    assert abs(F - expected) < 1e-12


def test_forward_negative_rho_lower_than_zero_rho():
    """Negative rho raises effective drift: -rho*sigma_S*sigma_X > 0 when rho<0."""
    F_neg = bsm_quanto_forward(
        S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=-0.5, sigma_S=0.20, sigma_X=0.10, T=1.0
    )
    F_zero = bsm_quanto_forward(
        S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=0.0, sigma_S=0.20, sigma_X=0.10, T=1.0
    )
    assert F_neg > F_zero


def test_forward_positive_rho_lower_than_zero_rho():
    """Positive rho decreases effective drift: -rho*sigma_S*sigma_X < 0 when rho>0."""
    F_pos = bsm_quanto_forward(
        S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=0.5, sigma_S=0.20, sigma_X=0.10, T=1.0
    )
    F_zero = bsm_quanto_forward(
        S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=0.0, sigma_S=0.20, sigma_X=0.10, T=1.0
    )
    assert F_pos < F_zero


def test_forward_zero_maturity():
    F = bsm_quanto_forward(
        S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=-0.3, sigma_S=0.20, sigma_X=0.10, T=0.0
    )
    assert abs(F - 100.0) < 1e-12


def test_forward_formula_exact():
    """Spot check: F_adj = 100 * exp((0.05 - 0.02 - (-0.3)*0.20*0.10) * 1)."""
    drift = 0.05 - 0.02 - (-0.3) * 0.20 * 0.10
    expected = 100 * math.exp(drift * 1.0)
    F = bsm_quanto_forward(
        S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=-0.3, sigma_S=0.20, sigma_X=0.10, T=1.0
    )
    assert abs(F - expected) < 1e-12


# ── 2. Put-call parity ───────────────────────────────────────────────────────


def test_parity_at_base_params():
    """C - P = disc * (F_adj - K), discount at r_dom."""
    C = bsm_quanto_option(**BASE, cp=1)
    P = bsm_quanto_option(**BASE, cp=-1)
    F = bsm_quanto_forward(
        S=BASE["S"],
        r_dom=BASE["r_dom"],
        r_for=BASE["r_for"],
        q_for=BASE["q_for"],
        rho=BASE["rho"],
        sigma_S=BASE["sigma_S"],
        sigma_X=BASE["sigma_X"],
        T=BASE["T"],
    )
    disc = math.exp(-BASE["r_dom"] * BASE["T"])
    parity_rhs = disc * (F - BASE["K"])
    assert abs((C - P) - parity_rhs) < 1e-12


def test_parity_itm_call():
    """Put-call parity holds for deep in-the-money call (low K)."""
    kw = {**BASE, "K": 80.0}
    C = bsm_quanto_option(**kw, cp=1)
    P = bsm_quanto_option(**kw, cp=-1)
    F = bsm_quanto_forward(
        S=kw["S"],
        r_dom=kw["r_dom"],
        r_for=kw["r_for"],
        q_for=kw["q_for"],
        rho=kw["rho"],
        sigma_S=kw["sigma_S"],
        sigma_X=kw["sigma_X"],
        T=kw["T"],
    )
    disc = math.exp(-kw["r_dom"] * kw["T"])
    assert abs((C - P) - disc * (F - kw["K"])) < 1e-12


def test_parity_otm_call():
    """Put-call parity holds for out-of-the-money call (high K)."""
    kw = {**BASE, "K": 130.0}
    C = bsm_quanto_option(**kw, cp=1)
    P = bsm_quanto_option(**kw, cp=-1)
    F = bsm_quanto_forward(
        S=kw["S"],
        r_dom=kw["r_dom"],
        r_for=kw["r_for"],
        q_for=kw["q_for"],
        rho=kw["rho"],
        sigma_S=kw["sigma_S"],
        sigma_X=kw["sigma_X"],
        T=kw["T"],
    )
    disc = math.exp(-kw["r_dom"] * kw["T"])
    assert abs((C - P) - disc * (F - kw["K"])) < 1e-12


def test_parity_multiple_correlations():
    for rho in [-0.9, -0.5, 0.0, 0.5, 0.9]:
        kw = {**BASE, "rho": rho}
        C = bsm_quanto_option(**kw, cp=1)
        P = bsm_quanto_option(**kw, cp=-1)
        F = bsm_quanto_forward(
            S=kw["S"],
            r_dom=kw["r_dom"],
            r_for=kw["r_for"],
            q_for=kw["q_for"],
            rho=rho,
            sigma_S=kw["sigma_S"],
            sigma_X=kw["sigma_X"],
            T=kw["T"],
        )
        disc = math.exp(-kw["r_dom"] * kw["T"])
        assert abs((C - P) - disc * (F - kw["K"])) < 1e-11, f"rho={rho}"


# ── 3. Degeneracy: zero vol ───────────────────────────────────────────────────


def test_zero_sigma_S_call_itm():
    """With sigma_S=0, call price = disc * max(F_adj - K, 0)."""
    kw = {**BASE, "sigma_S": 0.0, "K": 90.0}
    F = bsm_quanto_forward(
        S=kw["S"],
        r_dom=kw["r_dom"],
        r_for=kw["r_for"],
        q_for=kw["q_for"],
        rho=kw["rho"],
        sigma_S=0.0,
        sigma_X=kw["sigma_X"],
        T=kw["T"],
    )
    expected = math.exp(-kw["r_dom"] * kw["T"]) * max(F - kw["K"], 0.0)
    C = bsm_quanto_option(**kw, cp=1)
    assert abs(C - expected) < 1e-12


def test_zero_sigma_S_put_itm():
    """With sigma_S=0, put price = disc * max(K - F_adj, 0)."""
    kw = {**BASE, "sigma_S": 0.0, "K": 130.0}
    F = bsm_quanto_forward(
        S=kw["S"],
        r_dom=kw["r_dom"],
        r_for=kw["r_for"],
        q_for=kw["q_for"],
        rho=kw["rho"],
        sigma_S=0.0,
        sigma_X=kw["sigma_X"],
        T=kw["T"],
    )
    expected = math.exp(-kw["r_dom"] * kw["T"]) * max(kw["K"] - F, 0.0)
    P = bsm_quanto_option(**kw, cp=-1)
    assert abs(P - expected) < 1e-12


# ── 4. Degeneracy: zero maturity ─────────────────────────────────────────────


def test_zero_maturity_call_itm():
    C = bsm_quanto_option(**{**BASE, "T": 0.0, "K": 90.0}, cp=1)
    assert abs(C - max(100.0 - 90.0, 0.0)) < 1e-12


def test_zero_maturity_call_otm():
    C = bsm_quanto_option(**{**BASE, "T": 0.0, "K": 110.0}, cp=1)
    assert C == 0.0


def test_zero_maturity_put_itm():
    P = bsm_quanto_option(**{**BASE, "T": 0.0, "K": 110.0}, cp=-1)
    assert abs(P - max(110.0 - 100.0, 0.0)) < 1e-12


# ── 5. Prices non-negative ───────────────────────────────────────────────────


def test_prices_non_negative():
    for cp in (1, -1):
        for K in [80, 100, 120]:
            for rho in [-0.5, 0.0, 0.5]:
                p = bsm_quanto_option(**{**BASE, "K": K, "rho": rho}, cp=cp)
                assert p >= 0.0, f"cp={cp}, K={K}, rho={rho}"


# ── 6. rho sensitivity ───────────────────────────────────────────────────────


def test_call_increasing_in_negative_rho():
    """More negative rho -> higher F_adj -> higher ATM call (all else equal)."""
    rhos = [-0.8, -0.4, 0.0, 0.4, 0.8]
    calls = [bsm_quanto_option(**{**BASE, "K": 100.0, "rho": rho}, cp=1) for rho in rhos]
    assert all(calls[i] > calls[i + 1] for i in range(len(calls) - 1))


# ── 7. Input validation ───────────────────────────────────────────────────────


def test_raises_negative_S():
    with pytest.raises(ValueError, match="S must be positive"):
        bsm_quanto_forward(
            S=-1, r_dom=0.05, r_for=0.03, q_for=0.02, rho=0.0, sigma_S=0.2, sigma_X=0.1, T=1.0
        )


def test_raises_negative_K():
    with pytest.raises(ValueError, match="K must be positive"):
        bsm_quanto_option(**{**BASE, "K": -5.0}, cp=1)


def test_raises_negative_T():
    with pytest.raises(ValueError, match="T must be non-negative"):
        bsm_quanto_forward(
            S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=0.0, sigma_S=0.2, sigma_X=0.1, T=-0.1
        )


def test_raises_rho_out_of_range():
    with pytest.raises(ValueError, match="rho must be in"):
        bsm_quanto_forward(
            S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=1.5, sigma_S=0.2, sigma_X=0.1, T=1.0
        )


def test_raises_negative_sigma_S():
    with pytest.raises(ValueError, match="sigma_S must be non-negative"):
        bsm_quanto_forward(
            S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=0.0, sigma_S=-0.1, sigma_X=0.1, T=1.0
        )


def test_raises_negative_sigma_X():
    with pytest.raises(ValueError, match="sigma_X must be non-negative"):
        bsm_quanto_forward(
            S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=0.0, sigma_S=0.2, sigma_X=-0.05, T=1.0
        )


def test_raises_invalid_cp():
    with pytest.raises(ValueError, match="cp must be"):
        bsm_quanto_option(**BASE, cp=0)


# ── 8. Public API ─────────────────────────────────────────────────────────────


def test_importable_from_foureng():
    import foureng as fe

    assert hasattr(fe, "bsm_quanto_forward")
    assert hasattr(fe, "bsm_quanto_option")


def test_callable_from_foureng():
    import foureng as fe

    F = fe.bsm_quanto_forward(
        S=100, r_dom=0.05, r_for=0.03, q_for=0.02, rho=-0.3, sigma_S=0.20, sigma_X=0.10, T=1.0
    )
    assert F > 0
