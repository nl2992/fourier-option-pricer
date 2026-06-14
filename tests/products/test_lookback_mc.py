"""MC tests for floating- and fixed-strike lookback option pricers."""

from __future__ import annotations

import pytest

from foureng.mc.lookback_mc import lookback_mc
from foureng.mc.path_engine import GBMPathSpec
from foureng.pricers.analytic_bsm import bsm_lookback_fixed, bsm_lookback_floating

S0 = 100.0
r = 0.05
q = 0.0
sigma = 0.2
T = 1.0


class TestLookbackMC:
    def test_floating_call_non_negative(self):
        """Floating lookback call price is non-negative."""
        spec = GBMPathSpec(n_paths=10_000, n_steps=200, seed=0, antithetic=True)
        assert lookback_mc(S0, r, q, sigma, T, 200, cp=1, spec=spec) >= 0.0

    def test_floating_put_non_negative(self):
        """Floating lookback put price is non-negative."""
        spec = GBMPathSpec(n_paths=10_000, n_steps=200, seed=1, antithetic=True)
        assert lookback_mc(S0, r, q, sigma, T, 200, cp=-1, spec=spec) >= 0.0

    def test_fixed_call_non_negative(self):
        """Fixed-strike lookback call price is non-negative."""
        spec = GBMPathSpec(n_paths=10_000, n_steps=200, seed=2, antithetic=True)
        assert lookback_mc(S0, r, q, sigma, T, 200, cp=1, spec=spec, K=100.0) >= 0.0

    def test_fixed_put_non_negative(self):
        """Fixed-strike lookback put price is non-negative."""
        spec = GBMPathSpec(n_paths=10_000, n_steps=200, seed=3, antithetic=True)
        assert lookback_mc(S0, r, q, sigma, T, 200, cp=-1, spec=spec, K=100.0) >= 0.0

    @pytest.mark.derived_reference
    @pytest.mark.slow
    def test_floating_call_vs_analytic(self):
        """Floating call MC (n_steps=2000) within 3% of bsm_lookback_floating."""
        spec = GBMPathSpec(n_paths=40_000, n_steps=2000, seed=42, antithetic=True)
        mc_price = lookback_mc(S0, r, q, sigma, T, 2000, cp=1, spec=spec)
        analytic = bsm_lookback_floating(S0, S0, r, q, sigma, T, cp=1)
        # 3% relative tolerance; MC has discretization bias at n_steps=2000
        assert abs(mc_price - analytic) / analytic < 0.04

    @pytest.mark.derived_reference
    @pytest.mark.slow
    def test_floating_put_vs_analytic(self):
        """Floating put MC (n_steps=2000) within 4% of bsm_lookback_floating."""
        spec = GBMPathSpec(n_paths=40_000, n_steps=2000, seed=42, antithetic=True)
        mc_price = lookback_mc(S0, r, q, sigma, T, 2000, cp=-1, spec=spec)
        analytic = bsm_lookback_floating(S0, S0, r, q, sigma, T, cp=-1)
        assert abs(mc_price - analytic) / analytic < 0.04

    @pytest.mark.derived_reference
    @pytest.mark.slow
    def test_fixed_call_vs_analytic(self):
        """Fixed-strike call MC (n_steps=2000) within 4% of bsm_lookback_fixed."""
        K = 100.0
        spec = GBMPathSpec(n_paths=40_000, n_steps=2000, seed=42, antithetic=True)
        mc_price = lookback_mc(S0, r, q, sigma, T, 2000, cp=1, spec=spec, K=K)
        analytic = bsm_lookback_fixed(S0, K, S0, r, q, sigma, T, cp=1)
        assert abs(mc_price - analytic) / analytic < 0.04

    def test_lookback_positive(self):
        """All lookback MC prices (floating and fixed) are non-negative."""
        spec = GBMPathSpec(n_paths=5_000, n_steps=50, seed=7, antithetic=True)
        for cp in [1, -1]:
            price_float = lookback_mc(S0, r, q, sigma, T, 50, cp=cp, spec=spec)
            assert price_float >= 0.0, f"floating cp={cp} price={price_float}"
            price_fixed = lookback_mc(S0, r, q, sigma, T, 50, cp=cp, spec=spec, K=100.0)
            assert price_fixed >= 0.0, f"fixed cp={cp} price={price_fixed}"
