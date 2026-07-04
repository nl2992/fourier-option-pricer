"""Tests for the BSM double-barrier analytic pricer (eigenfunction expansion).

The DKO/DKI prices are computed via the sine-series eigenfunction expansion of the
GBM transition density absorbed at two barriers L and U (Kunitomo-Ikeda 1992,
Haug 2007 Ch. 2.17).

Tests
-----
1. DKO + DKI = vanilla (in-out parity) — calls and puts
2. DKO call → 0 as L → S (lower barrier tightens to spot)
3. DKO call → 0 as U → S (upper barrier tightens to spot)
4. DKO call < vanilla call (always cheaper)
5. As L → 0 and U → ∞, DKO call approaches vanilla call
6. Numerical convergence: n_max=50 vs n_max=150 agree to 1e-6
7. MC reference: DKO call within 3× MC std-err for reference parameters
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.analytics.bsm_barrier import bsm_call, bsm_double_barrier_price, bsm_put

# ── reference parameters (used in several tests) ──────────────────────────
# S=100, K=100, L=80, U=120, r=0.05, q=0, σ=0.25, T=0.5
_REF = dict(S=100.0, K=100.0, L=80.0, U=120.0, r=0.05, q=0.0, T=0.5, sigma=0.25)

# ── 1. in-out parity ──────────────────────────────────────────────────────


@pytest.mark.parametrize("cp", [1, -1])
@pytest.mark.parametrize(
    "params",
    [
        dict(S=100.0, K=100.0, L=80.0, U=120.0, r=0.05, q=0.0, T=0.5, sigma=0.25),
        dict(S=100.0, K=90.0, L=85.0, U=115.0, r=0.03, q=0.01, T=1.0, sigma=0.20),
        dict(S=100.0, K=110.0, L=70.0, U=130.0, r=0.08, q=0.02, T=0.25, sigma=0.30),
    ],
)
def test_in_out_parity(cp, params):
    """DKO + DKI = vanilla to machine precision."""
    dko = bsm_double_barrier_price(**params, cp=cp, knockout=True)
    dki = bsm_double_barrier_price(**params, cp=cp, knockout=False)
    S, K = params["S"], params["K"]
    r, q, T, sigma = params["r"], params["q"], params["T"], params["sigma"]
    vanilla = bsm_call(S, K, r, q, T, sigma) if cp == 1 else bsm_put(S, K, r, q, T, sigma)
    assert abs(dko + dki - vanilla) < 1e-8, (
        f"Parity failed: DKO={dko:.8f} DKI={dki:.8f} sum={dko + dki:.8f} vanilla={vanilla:.8f}"
    )


# ── 2. DKO → 0 as L → S ──────────────────────────────────────────────────


def test_dko_call_vanishes_as_lower_barrier_tightens():
    """DKO call → 0 as L → S from below."""
    S, K, U, r, q, T, sigma = 100.0, 100.0, 120.0, 0.05, 0.0, 0.5, 0.25
    prices = [
        bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=1) for L in [99.0, 99.5, 99.9]
    ]
    assert prices[0] > prices[1] > prices[2], "DKO should decrease as L approaches S"
    assert prices[-1] < 0.01, f"DKO should be near 0 when L is close to S: {prices[-1]:.6f}"


# ── 3. DKO → 0 as U → S ──────────────────────────────────────────────────


def test_dko_call_vanishes_as_upper_barrier_tightens():
    """DKO call → 0 as U → S from above."""
    S, K, L, r, q, T, sigma = 100.0, 100.0, 80.0, 0.05, 0.0, 0.5, 0.25
    prices = [
        bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=1) for U in [101.0, 100.5, 100.1]
    ]
    assert prices[0] > prices[1] > prices[2], "DKO should decrease as U approaches S"
    assert prices[-1] < 0.01, f"DKO should be near 0 when U is close to S: {prices[-1]:.6f}"


# ── 4. DKO ≤ vanilla ─────────────────────────────────────────────────────


@pytest.mark.parametrize("cp", [1, -1])
@pytest.mark.parametrize("K", [80.0, 90.0, 100.0, 110.0, 120.0])
def test_dko_bounded_by_vanilla(cp, K):
    """DKO ≤ vanilla for all strikes."""
    S, L, U = 100.0, 70.0, 130.0
    r, q, T, sigma = 0.05, 0.0, 0.5, 0.25
    # Skip degenerate cases
    if cp == 1 and K >= U:
        return
    if cp == -1 and K <= L:
        return
    dko = bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=cp)
    vanilla = bsm_call(S, K, r, q, T, sigma) if cp == 1 else bsm_put(S, K, r, q, T, sigma)
    assert dko <= vanilla + 1e-8, f"DKO={dko:.6f} > vanilla={vanilla:.6f} for K={K}, cp={cp}"


# ── 5. As L → 0 and U → ∞, DKO → vanilla ────────────────────────────────


def test_dko_call_approaches_vanilla_for_wide_barriers():
    """DKO call → vanilla when barriers are far from spot."""
    S, K, r, q, T, sigma = 100.0, 100.0, 0.05, 0.0, 0.5, 0.25
    L_tight, U_tight = 80.0, 120.0
    L_wide, U_wide = 1.0, 10000.0

    dko_tight = bsm_double_barrier_price(S, K, L_tight, U_tight, r, q, T, sigma)
    dko_wide = bsm_double_barrier_price(S, K, L_wide, U_wide, r, q, T, sigma)
    vanilla = bsm_call(S, K, r, q, T, sigma)

    assert dko_wide > dko_tight, "Wider barriers should give higher DKO"
    assert abs(dko_wide - vanilla) < 0.01, (
        f"Wide-barrier DKO={dko_wide:.4f} should be close to vanilla={vanilla:.4f}"
    )


# ── 6. Numerical convergence: n_max=50 vs n_max=150 ─────────────────────


@pytest.mark.parametrize("cp", [1, -1])
def test_series_convergence(cp):
    """n_max=50 and n_max=150 agree to 1e-6."""
    p50 = bsm_double_barrier_price(**_REF, cp=cp, n_max=50)
    p150 = bsm_double_barrier_price(**_REF, cp=cp, n_max=150)
    assert abs(p50 - p150) < 1e-6, f"Convergence failed: p50={p50:.8f} p150={p150:.8f}"


# ── 7. MC reference (3× stderr tolerance) ────────────────────────────────


def _mc_dko_call_bb(
    S: float,
    K: float,
    L: float,
    U: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    n_paths: int = 10000,
    n_steps: int = 500,
    seed: int = 42,
) -> tuple:
    """GBM Monte Carlo for DKO call with Brownian-bridge barrier correction.

    Uses the Brownian bridge probability of hitting a barrier between grid
    points (Beaglehole-Dybvig-Zhou / Andersen-Brotherton-Ratcliffe):

        P(min X ≤ log(L/S) between t and t+dt | S_t, S_{t+dt})
            = exp(-2 * log(S_t/L) * log(S_{t+dt}/L) / (σ² dt))

    This corrects the positive bias of naive discrete monitoring, allowing
    the MC estimate to converge to the continuous-barrier analytic price
    with a moderate number of steps.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol_sq_dt = sigma**2 * dt

    Z = rng.standard_normal((n_steps, n_paths))
    dX = drift + sigma * np.sqrt(dt) * Z
    X = np.cumsum(dX, axis=0)
    S_paths = S * np.exp(X)

    # Build full path including t=0
    S_full = np.vstack([np.full((1, n_paths), S), S_paths])  # (n_steps+1, n_paths)

    knocked_out = np.zeros(n_paths, dtype=bool)

    for i in range(n_steps):
        S_prev = S_full[i]
        S_curr = S_full[i + 1]

        # Discrete knock-out at grid points
        discrete_ko = (S_curr <= L) | (S_curr >= U)

        # BB probability of touching L from above
        above_L = (S_prev > L) & (S_curr > L)
        prob_L = np.where(
            above_L,
            np.exp(-2.0 * np.log(S_prev / L) * np.log(S_curr / L) / vol_sq_dt),
            0.0,
        )
        # BB probability of touching U from below
        below_U = (S_prev < U) & (S_curr < U)
        prob_U = np.where(
            below_U,
            np.exp(-2.0 * np.log(U / S_prev) * np.log(U / S_curr) / vol_sq_dt),
            0.0,
        )

        u1 = rng.uniform(size=n_paths)
        u2 = rng.uniform(size=n_paths)
        bb_ko = (u1 < prob_L) | (u2 < prob_U)

        knocked_out = knocked_out | discrete_ko | bb_ko

    S_T = S_paths[-1]
    payoff = np.where(knocked_out, 0.0, np.maximum(S_T - K, 0.0))
    disc = np.exp(-r * T)
    price = disc * np.mean(payoff)
    stderr = disc * np.std(payoff) / np.sqrt(n_paths)
    return price, stderr


def test_dko_call_vs_mc_reference():
    """DKO call price within 3× MC std-err (with Brownian-bridge correction).

    Reference parameters: S=100, K=100, L=80, U=120, r=0.05, q=0, σ=0.25, T=0.5
    MC: 10000 paths, 500 steps, Brownian-bridge barrier correction.

    The BB correction makes the discrete MC consistent with the continuous-time
    analytic formula, allowing agreement within statistical noise.
    """
    S, K, L, U, r, q, T, sigma = 100.0, 100.0, 80.0, 120.0, 0.05, 0.0, 0.5, 0.25

    analytic = bsm_double_barrier_price(S, K, L, U, r, q, T, sigma, cp=1, knockout=True)
    mc_price, mc_stderr = _mc_dko_call_bb(S, K, L, U, r, q, T, sigma, n_paths=10000, n_steps=500)

    tol = 3.0 * mc_stderr
    assert abs(analytic - mc_price) < tol, (
        f"DKO call: analytic={analytic:.4f}, MC+BB={mc_price:.4f}±{mc_stderr:.4f} "
        f"(3σ tol={tol:.4f})"
    )
