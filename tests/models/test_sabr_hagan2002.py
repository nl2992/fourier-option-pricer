"""Tests for the SABR Hagan (2002) lognormal implied volatility model.

The SABR model:
    dF = sigma_t F^beta dW1
    d sigma_t = nu sigma_t dW2
    <dW1, dW2> = rho dt

Hagan (2002) provides an analytical approximation for the lognormal (Black-Scholes)
implied volatility sigma_BS(F, K, T; alpha, beta, rho, nu).

Tests
-----
1. ATM self-consistency: sabr_hagan_implied_vol(F, F, ...) matches ATM limiting formula
2. BSM limit: beta=1, nu→0 → sigma_BS = alpha (the lognormal vol)
3. Put-call parity for sabr_call_price / sabr_put_price
4. Non-negativity of prices across a grid of strikes
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.models.sabr import (
    SabrParams,
    sabr_call_price,
    sabr_hagan_implied_vol,
    sabr_put_price,
)

# ── 1. ATM self-consistency ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "alpha,beta,rho,nu",
    [
        (0.3, 0.5, -0.3, 0.4),
        (0.2, 1.0, 0.0, 0.5),
        (0.5, 0.0, 0.1, 0.3),
        (0.4, 0.7, -0.5, 0.6),
    ],
)
def test_atm_self_consistency(alpha, beta, rho, nu):
    """sabr_hagan_implied_vol(F, F, T, ...) matches the ATM limiting formula exactly."""
    F, T = 100.0, 1.0
    FK_mid = F  # ATM
    correction_2 = 1.0 + (
        (1.0 - beta) ** 2 / 24.0 * alpha ** 2 / FK_mid ** (2.0 - 2.0 * beta)
        + rho * beta * nu * alpha / (4.0 * FK_mid ** (1.0 - beta))
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    ) * T
    atm_formula = alpha / F ** (1.0 - beta) * correction_2
    iv = sabr_hagan_implied_vol(F, F, T, alpha, beta, rho, nu)
    assert abs(iv - atm_formula) < 1e-10, (
        f"ATM mismatch: formula={atm_formula:.8f}, function={iv:.8f}"
    )


# ── 2. BSM limit: beta=1, nu→0 → sigma_BS = alpha ───────────────────────


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.3, 0.5])
def test_bsm_limit_beta1_nu0(alpha):
    """With beta=1, nu=0, SABR reduces to BSM with vol=alpha (all strikes)."""
    F, T = 100.0, 1.0
    beta, rho, nu = 1.0, 0.0, 1e-10  # near-zero nu
    for K in [80.0, 90.0, 100.0, 110.0, 120.0]:
        iv = sabr_hagan_implied_vol(F, K, T, alpha, beta, rho, nu)
        assert abs(iv - alpha) < 1e-6, (
            f"beta=1 nu≈0: expected IV={alpha}, got {iv:.8f} for K={K}"
        )


# ── 3. Put-call parity ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "K",
    [80.0, 90.0, 100.0, 110.0, 120.0],
)
def test_put_call_parity(K):
    """C - P = (F - K) * disc for SABR call and put prices."""
    S, T = 100.0, 1.0
    r, q = 0.05, 0.01
    alpha, beta, rho, nu = 2.0, 0.5, -0.3, 0.4  # alpha=2 gives ~20% ATM vol with beta=0.5, S=100

    F = S * np.exp((r - q) * T)
    disc = np.exp(-r * T)

    c = sabr_call_price(S, K, T, r, q, alpha, beta, rho, nu)
    p = sabr_put_price(S, K, T, r, q, alpha, beta, rho, nu)

    pcp_lhs = c - p
    pcp_rhs = (F - K) * disc
    assert abs(pcp_lhs - pcp_rhs) < 1e-8, (
        f"Put-call parity failed: C-P={pcp_lhs:.8f}, (F-K)*disc={pcp_rhs:.8f} for K={K}"
    )


# ── 4. Non-negativity of prices ──────────────────────────────────────────


@pytest.mark.parametrize("cp,fn", [(1, sabr_call_price), (-1, sabr_put_price)])
@pytest.mark.parametrize("K", [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
def test_price_non_negative(cp, fn, K):
    """SABR call and put prices are non-negative across a grid of strikes."""
    S, T = 100.0, 1.0
    r, q = 0.05, 0.0
    alpha, beta, rho, nu = 2.0, 0.5, -0.3, 0.4
    price = fn(S, K, T, r, q, alpha, beta, rho, nu)
    assert price >= 0.0, f"Negative price: {price:.8f} for K={K}"


# ── 5. SabrParams validation ─────────────────────────────────────────────


def test_sabr_params_valid():
    """SabrParams accepts valid parameters."""
    p = SabrParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
    assert p.alpha == 0.3
    assert p.name == "sabr"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"alpha": -0.1, "beta": 0.5, "rho": 0.0, "nu": 0.4}, "alpha"),
        ({"alpha": 0.3, "beta": 1.5, "rho": 0.0, "nu": 0.4}, "beta"),
        ({"alpha": 0.3, "beta": 0.5, "rho": -1.0, "nu": 0.4}, "rho"),
        ({"alpha": 0.3, "beta": 0.5, "rho": 0.0, "nu": -0.1}, "nu"),
    ],
)
def test_sabr_params_validation(kwargs, match):
    """SabrParams raises ValueError for invalid parameters."""
    with pytest.raises(ValueError, match=match):
        SabrParams(**kwargs)


# ── 6. Smile monotonicity (informational) ────────────────────────────────


def test_smile_has_positive_skew_for_negative_rho():
    """With rho < 0, the SABR smile has a negative skew (lower IV for higher strikes)."""
    F, T = 100.0, 1.0
    alpha, beta, rho, nu = 2.0, 0.5, -0.5, 0.4
    K_low, K_high = 80.0, 120.0
    iv_low = sabr_hagan_implied_vol(F, K_low, T, alpha, beta, rho, nu)
    iv_high = sabr_hagan_implied_vol(F, K_high, T, alpha, beta, rho, nu)
    assert iv_low > iv_high, (
        f"Negative rho → higher IV for low strikes: iv_low={iv_low:.4f}, iv_high={iv_high:.4f}"
    )
