"""L5 evidence: exact closed form for discrete geometric Asian options.

Tests foureng.pricers.analytic_bsm.bsm_geometric_asian against known
mathematical properties and a frozen reference value.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.pricers.analytic_bsm import bsm_european, bsm_geometric_asian

# Standard parameters
S = 100.0
K = 100.0
r = 0.05
q = 0.0
sigma = 0.2
T = 1.0


class TestGeometricAsianBSM:
    def test_n1_equals_european(self):
        """N=1 monitoring date → European option (geometry of single observation is spot)."""
        asian = bsm_geometric_asian(S, K, r, q, sigma, T, N=1, cp=1)
        euro = bsm_european(S, K, r, q, sigma, T, cp=1)
        assert asian == pytest.approx(euro, rel=1e-10)

    def test_n1_put_equals_european_put(self):
        """N=1 put → European put."""
        asian = bsm_geometric_asian(S, K, r, q, sigma, T, N=1, cp=-1)
        euro = bsm_european(S, K, r, q, sigma, T, cp=-1)
        assert asian == pytest.approx(euro, rel=1e-10)

    def test_geometric_less_than_arithmetic_mc(self):
        """Geometric Asian ≤ arithmetic Asian by Jensen's inequality (MC verification)."""
        rng = np.random.default_rng(42)
        n_paths = 50_000
        N_mon = 12
        dt = T / N_mon

        # Simulate log-normal paths
        log_returns = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(
            (n_paths, N_mon)
        )
        log_paths = np.cumsum(log_returns, axis=1)
        paths = S * np.exp(log_paths)

        geo_avg = np.exp(np.mean(np.log(paths), axis=1))
        arith_avg = np.mean(paths, axis=1)

        disc = np.exp(-r * T)
        mc_geo_call = disc * np.mean(np.maximum(geo_avg - K, 0.0))
        mc_arith_call = disc * np.mean(np.maximum(arith_avg - K, 0.0))
        closed_geo = bsm_geometric_asian(S, K, r, q, sigma, T, N=N_mon, cp=1)

        # Geometric < arithmetic by Jensen's
        assert mc_geo_call < mc_arith_call

        # Closed form should be consistent with MC within a reasonable tolerance
        # (MC has noise; use a wider tolerance)
        assert abs(closed_geo - mc_geo_call) / closed_geo < 0.03  # 3% MC noise tolerance

    def test_put_call_parity(self):
        """Geometric Asian put-call parity: C - P = disc*(F_G - K)."""
        N = 12
        call = bsm_geometric_asian(S, K, r, q, sigma, T, N=N, cp=1)
        put = bsm_geometric_asian(S, K, r, q, sigma, T, N=N, cp=-1)

        # Compute F_G (geometric-average forward) internally
        mu_G = np.log(S) + (r - q - 0.5 * sigma**2) * T * (N + 1) / (2 * N)
        var_G = sigma**2 * T * (N + 1) * (2 * N + 1) / (6 * N**2)
        F_G = np.exp(mu_G + 0.5 * var_G)
        disc = np.exp(-r * T)

        parity_rhs = disc * (F_G - K)
        assert call - put == pytest.approx(parity_rhs, rel=1e-8)

    def test_convergence_as_n_increases(self):
        """Price should converge as N increases (continuous approximation)."""
        prices = [bsm_geometric_asian(S, K, r, q, sigma, T, N=n, cp=1) for n in [1, 4, 12, 52, 252]]
        # Prices should be decreasing as N increases (more monitoring = closer to continuous)
        # or at least converging. Check monotone and bounded.
        for i in range(len(prices) - 1):
            # Prices decrease as N increases for ATM call (geometric avg < spot on average)
            assert prices[i] >= prices[i + 1] - 1e-4  # allow small non-monotonicity

    def test_zero_vol_limit(self):
        """Zero volatility: deterministic path, payoff is discounted deterministic value."""
        N = 12
        sigma_tiny = 1e-10
        asian = bsm_geometric_asian(S, K, r, q, sigma_tiny, T, N=N, cp=1)
        # Deterministic path: S_t = S * exp((r-q)*t_i) for t_i = i*T/N
        # Geometric average = S * exp((r-q)*T*(N+1)/(2*N))
        # (same as F_G when sigma≈0)
        F_det = S * np.exp((r - q) * T * (N + 1) / (2 * N))
        disc = np.exp(-r * T)
        expected = disc * max(F_det - K, 0.0)
        assert asian == pytest.approx(expected, rel=1e-4)

    def test_frozen_reference_n12(self):
        """Frozen reference: S=100, K=100, r=0.05, q=0, sigma=0.2, T=1, N=12."""
        call = bsm_geometric_asian(100, 100, 0.05, 0, 0.2, 1.0, 12, cp=1)
        assert call == pytest.approx(5.940200, rel=1e-4)

    def test_frozen_reference_n12_put(self):
        """Frozen put reference: S=100, K=100, r=0.05, q=0, sigma=0.2, T=1, N=12."""
        put = bsm_geometric_asian(100, 100, 0.05, 0, 0.2, 1.0, 12, cp=-1)
        assert put == pytest.approx(3.651734, rel=1e-4)

    def test_geometric_below_european_for_atm_call(self):
        """Geometric Asian ATM call ≤ European ATM call (geometric avg ≤ arithmetic avg)."""
        euro = bsm_european(S, K, r, q, sigma, T, cp=1)
        geo = bsm_geometric_asian(S, K, r, q, sigma, T, N=12, cp=1)
        assert geo <= euro

    def test_geometric_below_european_for_various_n(self):
        """Geometric Asian call ≤ European call for various N (Jensen's)."""
        euro = bsm_european(S, K, r, q, sigma, T, cp=1)
        for N in [2, 4, 12, 52]:
            geo = bsm_geometric_asian(S, K, r, q, sigma, T, N=N, cp=1)
            assert geo <= euro + 1e-10  # small tolerance for N=1 case

    def test_non_negative_prices(self):
        """Geometric Asian option prices are non-negative."""
        for K_test in [80.0, 100.0, 120.0]:
            for cp in [1, -1]:
                price = bsm_geometric_asian(S, K_test, r, q, sigma, T, N=12, cp=cp)
                assert price >= 0.0
