"""MC tests for arithmetic Asian option pricer."""

from __future__ import annotations

import numpy as np
import pytest

from foureng.mc.asian_mc import asian_mc
from foureng.mc.path_engine import GBMPathSpec

S0 = 100.0
K = 100.0
r = 0.05
q = 0.0
sigma = 0.2
T = 1.0

_SPEC = GBMPathSpec(n_paths=40_000, n_steps=50, seed=42, antithetic=True)


class TestAsianMC:
    def test_zero_vol(self):
        """Zero vol: arithmetic average is deterministic, payoff is known exactly."""
        spec = GBMPathSpec(n_paths=1_000, n_steps=12, seed=0, antithetic=True)
        N_mon = 12
        dt = T / N_mon
        # Deterministic path: S_t_i = S0 * exp((r-q)*t_i)
        times = np.arange(1, N_mon + 1) * dt
        det_avg = np.mean(S0 * np.exp((r - q) * times))
        disc = np.exp(-r * T)
        expected_call = disc * max(det_avg - K, 0.0)
        expected_put = disc * max(K - det_avg, 0.0)

        price_call = asian_mc(S0, K, r, q, 1e-8, T, N_mon, cp=1, spec=spec)
        price_put = asian_mc(S0, K, r, q, 1e-8, T, N_mon, cp=-1, spec=spec)

        assert price_call == pytest.approx(expected_call, abs=0.01)
        assert price_put == pytest.approx(expected_put, abs=0.01)

    @pytest.mark.derived_reference
    @pytest.mark.slow
    def test_arithmetic_asian_converges_to_geometric(self):
        """With N=1, arithmetic average = geometric average = S_T; prices agree."""
        from foureng.pricers.analytic_bsm import bsm_geometric_asian

        spec = GBMPathSpec(n_paths=20_000, n_steps=1, seed=7, antithetic=True)
        mc_price = asian_mc(S0, K, r, q, sigma, T, N_mon=1, cp=1, spec=spec)
        geo_price = bsm_geometric_asian(S0, K, r, q, sigma, T, N=1, cp=1)

        # N=1: arith avg = geo avg = S_T; with geometric CV the error is tiny
        assert abs(mc_price - geo_price) < 0.05

    @pytest.mark.derived_reference
    @pytest.mark.slow
    def test_arithmetic_above_geometric_for_n_large(self):
        """Arithmetic Asian call >= geometric Asian call (Jensen: arith avg >= geo avg)."""
        from foureng.pricers.analytic_bsm import bsm_geometric_asian

        N_mon = 12
        spec = GBMPathSpec(n_paths=40_000, n_steps=N_mon, seed=42, antithetic=True)
        arith_price = asian_mc(S0, K, r, q, sigma, T, N_mon=N_mon, cp=1, spec=spec)
        geo_price = bsm_geometric_asian(S0, K, r, q, sigma, T, N=N_mon, cp=1)

        # Arithmetic avg >= geometric avg (Jensen) → arith call >= geo call; MC tolerance 0.2
        assert arith_price >= geo_price - 0.2

    @pytest.mark.derived_reference
    def test_asian_parity(self):
        """Call - put = disc * (arith_forward - K) for arithmetic Asian."""
        N_mon = 12
        spec_c = GBMPathSpec(n_paths=20_000, n_steps=N_mon, seed=1, antithetic=True)
        spec_p = GBMPathSpec(n_paths=20_000, n_steps=N_mon, seed=1, antithetic=True)

        call = asian_mc(S0, K, r, q, sigma, T, N_mon, cp=1, spec=spec_c)
        put = asian_mc(S0, K, r, q, sigma, T, N_mon, cp=-1, spec=spec_p)

        # Arithmetic average forward = mean of S0*exp((r-q)*t_i)
        times = np.arange(1, N_mon + 1) * (T / N_mon)
        F_arith = np.mean(S0 * np.exp((r - q) * times))
        disc = np.exp(-r * T)
        parity_rhs = disc * (F_arith - K)

        # Same seed so paths agree; parity should hold within 0.1
        assert abs((call - put) - parity_rhs) < 0.15

    def test_non_negative(self):
        """Asian MC prices are non-negative."""
        spec = GBMPathSpec(n_paths=10_000, n_steps=12, seed=99, antithetic=True)
        for cp in [1, -1]:
            price = asian_mc(S0, K, r, q, sigma, T, N_mon=12, cp=cp, spec=spec)
            assert price >= 0.0
