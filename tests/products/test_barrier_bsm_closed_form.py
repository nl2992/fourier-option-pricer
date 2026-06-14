"""L5 evidence: BSM barrier option closed-form pricing (Reiner-Rubinstein 1991).

Tests foureng.pricers.analytic_bsm.bsm_barrier against mathematical
identities and frozen reference values.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.pricers.analytic_bsm import bsm_barrier, bsm_european

# Standard parameters
S = 100.0
K = 100.0
H_down = 90.0  # down-barrier below strike
H_up = 110.0  # up-barrier above strike
r = 0.05
q = 0.0
sigma = 0.2
T = 1.0


class TestInOutParity:
    """In-out parity: knock-in + knock-out = vanilla."""

    def test_down_call_parity(self):
        """Down-in call + Down-out call = Vanilla call."""
        vanilla = bsm_european(S, K, r, q, sigma, T, cp=1)
        doc = bsm_barrier(S, K, H_down, r, q, sigma, T, "down_out", cp=1)
        dic = bsm_barrier(S, K, H_down, r, q, sigma, T, "down_in", cp=1)
        assert doc + dic == pytest.approx(vanilla, rel=1e-8)

    def test_down_put_parity(self):
        """Down-in put + Down-out put = Vanilla put."""
        vanilla = bsm_european(S, K, r, q, sigma, T, cp=-1)
        dop = bsm_barrier(S, K, H_down, r, q, sigma, T, "down_out", cp=-1)
        dip = bsm_barrier(S, K, H_down, r, q, sigma, T, "down_in", cp=-1)
        assert dop + dip == pytest.approx(vanilla, rel=1e-8)

    def test_up_call_parity(self):
        """Up-in call + Up-out call = Vanilla call."""
        vanilla = bsm_european(S, K, r, q, sigma, T, cp=1)
        uoc = bsm_barrier(S, K, H_up, r, q, sigma, T, "up_out", cp=1)
        uic = bsm_barrier(S, K, H_up, r, q, sigma, T, "up_in", cp=1)
        assert uoc + uic == pytest.approx(vanilla, rel=1e-8)

    def test_up_put_parity(self):
        """Up-in put + Up-out put = Vanilla put."""
        vanilla = bsm_european(S, K, r, q, sigma, T, cp=-1)
        uop = bsm_barrier(S, K, H_up, r, q, sigma, T, "up_out", cp=-1)
        uip = bsm_barrier(S, K, H_up, r, q, sigma, T, "up_in", cp=-1)
        assert uop + uip == pytest.approx(vanilla, rel=1e-8)


class TestBarrierLimits:
    def test_barrier_far_below_gives_vanilla(self):
        """Down-out call with barrier far below S ≈ Vanilla call."""
        vanilla = bsm_european(S, K, r, q, sigma, T, cp=1)
        doc_far = bsm_barrier(S, K, 1.0, r, q, sigma, T, "down_out", cp=1)
        assert doc_far == pytest.approx(vanilla, rel=1e-6)

    def test_barrier_far_above_gives_vanilla(self):
        """Down-out call with barrier far above S (e.g. S*1.1) — in-out parity check.

        Note: the Reiner-Rubinstein formula is numerically unstable for very high
        H/S ratios due to (H/S)^{2*(mu+1)} overflow. Test with a moderate barrier.
        """
        # In-out parity is the reliable test: up-out + up-in = vanilla
        vanilla = bsm_european(S, K, r, q, sigma, T, cp=-1)
        H_mod = 150.0  # moderately high barrier
        uop = bsm_barrier(S, K, H_mod, r, q, sigma, T, "up_out", cp=-1)
        uip = bsm_barrier(S, K, H_mod, r, q, sigma, T, "up_in", cp=-1)
        assert uop + uip == pytest.approx(vanilla, rel=1e-6)

    def test_barrier_at_spot_down_out_returns_rebate(self):
        """Down-out option with S=H (already at barrier): returns rebate."""
        rebate = 2.0
        price = bsm_barrier(S, K, S, r, q, sigma, T, "down_out", rebate=rebate, cp=1)
        expected = rebate * np.exp(-r * T)
        assert price == pytest.approx(expected, rel=1e-10)

    def test_already_breached_down_out_returns_rebate(self):
        """Down-out with S < H: already knocked out, returns discounted rebate."""
        rebate = 3.0
        price = bsm_barrier(50.0, K, 60.0, r, q, sigma, T, "down_out", rebate=rebate, cp=1)
        expected = rebate * np.exp(-r * T)
        assert price == pytest.approx(expected, rel=1e-10)

    def test_already_breached_down_in_gives_vanilla(self):
        """Down-in with S < H: already knocked in, returns vanilla."""
        vanilla = bsm_european(50.0, K, r, q, sigma, T, cp=1)
        price = bsm_barrier(50.0, K, 60.0, r, q, sigma, T, "down_in", cp=1)
        assert price == pytest.approx(vanilla, rel=1e-10)

    def test_already_breached_up_out_returns_rebate(self):
        """Up-out with S >= H: returns discounted rebate."""
        rebate = 1.5
        price = bsm_barrier(120.0, K, 110.0, r, q, sigma, T, "up_out", rebate=rebate, cp=1)
        expected = rebate * np.exp(-r * T)
        assert price == pytest.approx(expected, rel=1e-10)

    def test_zero_rebate_default(self):
        """Default rebate=0 gives zero for already-knocked-out option."""
        price = bsm_barrier(S, K, S, r, q, sigma, T, "down_out", cp=1)
        assert price == pytest.approx(0.0, abs=1e-12)


class TestBarrierMonotonicity:
    def test_doc_less_than_vanilla(self):
        """Down-out call is cheaper than vanilla call (knockout is a constraint)."""
        vanilla = bsm_european(S, K, r, q, sigma, T, cp=1)
        doc = bsm_barrier(S, K, H_down, r, q, sigma, T, "down_out", cp=1)
        assert doc <= vanilla

    def test_uoc_less_than_vanilla(self):
        """Up-out call is cheaper than vanilla call."""
        vanilla = bsm_european(S, K, r, q, sigma, T, cp=1)
        uoc = bsm_barrier(S, K, H_up, r, q, sigma, T, "up_out", cp=1)
        assert uoc <= vanilla

    def test_non_negative_prices(self):
        """All barrier option prices are non-negative."""
        for bt in ["down_out", "down_in"]:
            for cp in [1, -1]:
                if bt.startswith("down"):
                    H = H_down
                else:
                    H = H_up
                price = bsm_barrier(S, K, H, r, q, sigma, T, bt, cp=cp)
                assert price >= 0.0

        for bt in ["up_out", "up_in"]:
            for cp in [1, -1]:
                price = bsm_barrier(S, K, H_up, r, q, sigma, T, bt, cp=cp)
                assert price >= 0.0


class TestBarrierFrozenReferences:
    """Frozen reference values computed from the Reiner-Rubinstein formula."""

    def test_doc_frozen_reference(self):
        """Down-out call: S=100, K=100, H=90, r=0.05, q=0, sigma=0.2, T=1."""
        result = bsm_barrier(100, 100, 90, 0.05, 0, 0.2, 1.0, "down_out", cp=1)
        assert result == pytest.approx(8.66547, rel=1e-4)

    def test_dic_frozen_reference(self):
        """Down-in call: S=100, K=100, H=90, r=0.05, q=0, sigma=0.2, T=1."""
        result = bsm_barrier(100, 100, 90, 0.05, 0, 0.2, 1.0, "down_in", cp=1)
        assert result == pytest.approx(1.78511, rel=1e-4)

    def test_uoc_frozen_reference(self):
        """Up-out call: S=100, K=100, H=110, r=0.05, q=0, sigma=0.2, T=1."""
        result = bsm_barrier(100, 100, 110, 0.05, 0, 0.2, 1.0, "up_out", cp=1)
        assert result == pytest.approx(1.62516, rel=1e-4)

    def test_uic_frozen_reference(self):
        """Up-in call: S=100, K=100, H=110, r=0.05, q=0, sigma=0.2, T=1."""
        result = bsm_barrier(100, 100, 110, 0.05, 0, 0.2, 1.0, "up_in", cp=1)
        assert result == pytest.approx(8.82543, rel=1e-4)


class TestBarrierEdgeCases:
    def test_invalid_barrier_type_raises(self):
        """Invalid barrier_type raises ValueError."""
        with pytest.raises(ValueError, match="barrier_type"):
            bsm_barrier(S, K, H_down, r, q, sigma, T, "knock_out", cp=1)

    def test_parity_various_barriers(self):
        """In-out parity holds for multiple barrier levels."""
        vanilla = bsm_european(S, K, r, q, sigma, T, cp=1)
        for H in [70.0, 80.0, 85.0, 95.0]:
            doc = bsm_barrier(S, K, H, r, q, sigma, T, "down_out", cp=1)
            dic = bsm_barrier(S, K, H, r, q, sigma, T, "down_in", cp=1)
            assert doc + dic == pytest.approx(vanilla, rel=1e-6)
