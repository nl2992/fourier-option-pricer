"""MC tests for single-barrier option pricer with BGK continuity correction."""

from __future__ import annotations

import pytest

from foureng.mc.barrier_mc import barrier_mc
from foureng.mc.path_engine import GBMPathSpec
from foureng.pricers.analytic_bsm import bsm_barrier, bsm_european

S0 = 100.0
K = 100.0
r = 0.05
q = 0.0
sigma = 0.2
T = 1.0

_SPEC = GBMPathSpec(n_paths=30_000, n_steps=500, seed=42, antithetic=True)


class TestBarrierMC:
    def test_barrier_far_below_spot(self):
        """Down-out call with H << S0 is virtually never knocked; price ≈ vanilla BSM."""
        H_low = 10.0
        mc_price = barrier_mc(S0, K, H_low, r, q, sigma, T, 500, "down_out", 0.0, 1, _SPEC)
        vanilla = bsm_european(S0, K, r, q, sigma, T, cp=1)
        assert abs(mc_price - vanilla) < 0.10

    def test_in_out_parity(self):
        """Knock-in + knock-out = vanilla (in-out parity), within MC tolerance."""
        H = 90.0
        spec = GBMPathSpec(n_paths=20_000, n_steps=500, seed=5, antithetic=True)
        ko = barrier_mc(S0, K, H, r, q, sigma, T, 500, "down_out", 0.0, 1, spec)
        ki = barrier_mc(S0, K, H, r, q, sigma, T, 500, "down_in", 0.0, 1, spec)
        vanilla = bsm_european(S0, K, r, q, sigma, T, cp=1)
        # ki uses parity internally so sum should be exact
        assert abs(ko + ki - vanilla) < 1e-6

    @pytest.mark.derived_reference
    @pytest.mark.slow
    def test_down_out_call_vs_analytic(self):
        """Down-out call MC (BGK, n_steps=1000) within 0.5 of BSM analytic."""
        H = 90.0
        spec = GBMPathSpec(n_paths=30_000, n_steps=1000, seed=42, antithetic=True)
        mc_price = barrier_mc(S0, K, H, r, q, sigma, T, 1000, "down_out", 0.0, 1, spec)
        analytic = bsm_barrier(S0, K, H, r, q, sigma, T, "down_out", rebate=0.0, cp=1)
        assert abs(mc_price - analytic) < 0.50

    def test_up_out_call_far_barrier(self):
        """Up-out call with H >> S0: price ≈ vanilla (barrier is never hit)."""
        H_high = 200.0
        mc_price = barrier_mc(S0, K, H_high, r, q, sigma, T, 500, "up_out", 0.0, 1, _SPEC)
        vanilla = bsm_european(S0, K, r, q, sigma, T, cp=1)
        assert abs(mc_price - vanilla) < 0.15

    def test_non_negative(self):
        """All barrier option prices are non-negative."""
        for btype in ["down_out", "down_in", "up_out", "up_in"]:
            H = 90.0 if btype.startswith("down") else 120.0
            spec = GBMPathSpec(n_paths=5_000, n_steps=200, seed=0, antithetic=True)
            price = barrier_mc(S0, K, H, r, q, sigma, T, 200, btype, 0.0, 1, spec)
            assert price >= 0.0, f"negative price for {btype}: {price}"
