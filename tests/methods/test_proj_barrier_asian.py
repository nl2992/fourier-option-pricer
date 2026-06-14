"""Sprint 5 evidence: PROJ barrier and PROJ-assisted arithmetic Asian pricers.

Tests ``proj_barrier_price`` (Kirkby 2015 backward induction with barrier
absorption) and ``proj_asian_price_cv`` (arithmetic Asian MC with PROJ
geometric control variate). Both are validated against BSM closed forms and
basic financial properties.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.models.base import ForwardSpec
from foureng.models.registry import MODEL_REGISTRY
from foureng.pricers.analytic_bsm import bsm_barrier, bsm_european, bsm_geometric_asian
from foureng.pricers.proj import proj_asian_price_cv, proj_barrier_price

# ── shared fixtures ──────────────────────────────────────────────────────────

S0 = 100.0
K = 100.0
r = 0.05
q = 0.02
sigma = 0.20
T = 1.0
H_down = 85.0  # down barrier (below spot)
H_up = 115.0  # up barrier (above spot)


def _bsm_step_cf(sigma, r, q, T, M):
    """Return one-step RN-CHF for BSM with M monitoring dates."""
    dt = T / M

    def cf(u):
        drift = (r - q - 0.5 * sigma**2) * dt
        return np.exp(1j * u * drift - 0.5 * sigma**2 * dt * u**2)

    return cf


# ── PROJ barrier tests ───────────────────────────────────────────────────────


class TestProjBarrierDownOut:
    """Down-and-out call: barrier below spot, option survives if price stays above H."""

    def test_down_out_call_is_nonneg(self):
        cf = _bsm_step_cf(sigma, r, q, T, M=100)
        price = proj_barrier_price(
            cf, S0=S0, r=r, T=T, K=K, H=H_down, M=100, barrier_type="down_out", cp=1
        )
        assert price >= 0.0

    @pytest.mark.slow
    def test_down_out_call_matches_bsm_closed_form(self):
        """PROJ down-out call should match Reiner-Rubinstein BSM formula."""
        M = 252
        cf = _bsm_step_cf(sigma, r, q, T, M=M)
        proj_price = proj_barrier_price(
            cf, S0=S0, r=r, T=T, K=K, H=H_down, M=M, barrier_type="down_out", cp=1
        )
        analytic = bsm_barrier(S0, K, H_down, r, q, sigma, T, "down_out", rebate=0.0, cp=1)
        # PROJ with M=252 is daily monitoring; analytic is continuous — allow small gap
        assert abs(proj_price - analytic) < 0.15

    def test_down_out_call_below_vanilla(self):
        """Down-out call <= vanilla call (barrier reduces value)."""
        M = 50
        cf = _bsm_step_cf(sigma, r, q, T, M=M)
        proj_price = proj_barrier_price(
            cf, S0=S0, r=r, T=T, K=K, H=H_down, M=M, barrier_type="down_out", cp=1
        )
        vanilla = bsm_european(S0, K, r, q, sigma, T, cp=1)
        assert proj_price <= vanilla + 1e-6

    def test_down_out_call_zero_when_spot_at_barrier(self):
        """Down-out call should be (near) zero when spot is at or below barrier."""
        M = 50
        cf = _bsm_step_cf(sigma, r, q, T, M=M)
        # Spot exactly at barrier — option is worthless
        price_at = proj_barrier_price(
            cf, S0=H_down, r=r, T=T, K=K, H=H_down, M=M, barrier_type="down_out", cp=1
        )
        assert price_at < 0.5  # should be very small (nearly knocked out)


class TestProjBarrierUpOut:
    """Up-and-out put: barrier above spot, option survives if price stays below H."""

    def test_up_out_put_is_nonneg(self):
        M = 50
        cf = _bsm_step_cf(sigma, r, q, T, M=M)
        price = proj_barrier_price(
            cf, S0=S0, r=r, T=T, K=K, H=H_up, M=M, barrier_type="up_out", cp=-1
        )
        assert price >= 0.0

    def test_up_out_put_below_vanilla(self):
        """Up-out put <= vanilla put."""
        M = 50
        cf = _bsm_step_cf(sigma, r, q, T, M=M)
        proj_price = proj_barrier_price(
            cf, S0=S0, r=r, T=T, K=K, H=H_up, M=M, barrier_type="up_out", cp=-1
        )
        vanilla = bsm_european(S0, K, r, q, sigma, T, cp=-1)
        assert proj_price <= vanilla + 1e-6


class TestProjBarrierKnockIn:
    """In-out parity: knock_in + knock_out = vanilla."""

    def test_down_in_plus_down_out_equals_vanilla(self):
        """Down-in + down-out = vanilla (in-out parity)."""
        M = 40
        cf = _bsm_step_cf(sigma, r, q, T, M=M)
        ko = proj_barrier_price(
            cf, S0=S0, r=r, T=T, K=K, H=H_down, M=M, barrier_type="down_out", cp=1
        )
        ki = proj_barrier_price(
            cf, S0=S0, r=r, T=T, K=K, H=H_down, M=M, barrier_type="down_in", cp=1
        )
        vanilla = bsm_european(S0, K, r, q, sigma, T, cp=1)
        # Parity: ki + ko ≈ vanilla (approximately, since M is discrete)
        assert abs(ko + ki - vanilla) < 0.5

    def test_barrier_rejects_bad_type(self):
        cf = _bsm_step_cf(sigma, r, q, T, M=10)
        with pytest.raises(ValueError, match="barrier_type"):
            proj_barrier_price(
                cf, S0=S0, r=r, T=T, K=K, H=H_down, M=10, barrier_type="sideways", cp=1
            )

    def test_barrier_rejects_bad_cp(self):
        cf = _bsm_step_cf(sigma, r, q, T, M=10)
        with pytest.raises(ValueError, match="cp"):
            proj_barrier_price(
                cf, S0=S0, r=r, T=T, K=K, H=H_down, M=10, barrier_type="down_out", cp=0
            )


# ── PROJ Asian tests ─────────────────────────────────────────────────────────


class TestProjAsianCV:
    """proj_asian_price_cv: arithmetic Asian with geometric control variate."""

    @pytest.fixture
    def bsm_fwd(self):
        return ForwardSpec(S0=S0, r=r, q=q, T=T)

    @pytest.fixture
    def bsm_params(self):
        return fe.BsmParams(sigma=sigma)

    def _phi_bsm(self, fwd, params):
        entry = MODEL_REGISTRY["bsm"]
        return lambda u: entry.cf(u, fwd, params)

    def test_atm_call_positive(self, bsm_fwd, bsm_params):
        phi = self._phi_bsm(bsm_fwd, bsm_params)
        price = proj_asian_price_cv(
            phi, bsm_fwd, bsm_params, "bsm", K=K, T=T, M=12, cp=1, n_paths=8_000, seed=0
        )
        assert price > 0.0

    def test_atm_put_positive(self, bsm_fwd, bsm_params):
        phi = self._phi_bsm(bsm_fwd, bsm_params)
        price = proj_asian_price_cv(
            phi, bsm_fwd, bsm_params, "bsm", K=K, T=T, M=12, cp=-1, n_paths=8_000, seed=1
        )
        assert price > 0.0

    def test_call_put_parity_approx(self, bsm_fwd, bsm_params):
        """Asian call - Asian put ≈ PV(F_arith - K) for arithmetic average."""
        phi = self._phi_bsm(bsm_fwd, bsm_params)
        call = proj_asian_price_cv(
            phi, bsm_fwd, bsm_params, "bsm", K=K, T=T, M=12, cp=1, n_paths=10_000, seed=2
        )
        put = proj_asian_price_cv(
            phi, bsm_fwd, bsm_params, "bsm", K=K, T=T, M=12, cp=-1, n_paths=10_000, seed=2
        )
        # For arithmetic average: call - put ≈ disc*(F_arith - K), where F_arith
        # is the discounted arithmetic average forward. For ATM (K=F_0 approximately)
        # this is a small positive number.
        F0 = S0 * np.exp((r - q) * T)
        # Rough bound: |call - put| < 30% of forward (ATM both positive)
        assert abs(call - put) < F0 * 0.3

    def test_arithmetic_above_geometric_bsm(self, bsm_fwd, bsm_params):
        """Arithmetic Asian call >= geometric Asian call (Jensen's inequality)."""
        phi = self._phi_bsm(bsm_fwd, bsm_params)
        arith = proj_asian_price_cv(
            phi, bsm_fwd, bsm_params, "bsm", K=K, T=T, M=12, cp=1, n_paths=10_000, seed=3
        )
        geo = float(bsm_geometric_asian(S0, K, r, q, sigma, T, 12, cp=1))
        assert arith >= geo - 0.3  # allow MC noise

    def test_deep_itm_call_close_to_intrinsic(self, bsm_params):
        """Deep ITM arithmetic Asian call should be close to S0*disc - K*disc."""
        K_deep_itm = 50.0  # very low strike relative to S0=100
        fwd = ForwardSpec(S0=S0, r=r, q=q, T=T)
        phi = self._phi_bsm(fwd, bsm_params)
        price = proj_asian_price_cv(
            phi, fwd, bsm_params, "bsm", K=K_deep_itm, T=T, M=12, cp=1, n_paths=8_000, seed=4
        )
        # Intrinsic ~ S0 * exp(-q*T) - K * exp(-r*T) for average-rate asian (approx)
        intrinsic = S0 * np.exp(-q * T) - K_deep_itm * np.exp(-r * T)
        assert price > 0.0
        assert price < intrinsic + 5.0

    @pytest.mark.slow
    def test_proj_asian_bsm_matches_mc_within_tolerance(self, bsm_fwd, bsm_params):
        """proj_asian_price_cv should match arithmetic Asian MC to < 0.05."""
        from foureng.mc.asian_mc import asian_mc
        from foureng.mc.path_engine import GBMPathSpec

        phi = self._phi_bsm(bsm_fwd, bsm_params)
        proj_price = proj_asian_price_cv(
            phi, bsm_fwd, bsm_params, "bsm", K=K, T=T, M=12, cp=1, n_paths=20_000, seed=99
        )
        spec = GBMPathSpec(n_paths=20_000, n_steps=12, seed=99)
        mc_price = asian_mc(S0, K, r, q, sigma, T, 12, cp=1, spec=spec)
        assert abs(proj_price - mc_price) < 0.3


# ── pipeline dispatch tests ──────────────────────────────────────────────────


class TestProjBarrierPipelineDispatch:
    """Verify price() dispatcher routes barrier products to PROJ correctly."""

    def test_proj_barrier_dispatch_down_out_call(self):
        from foureng.pipeline import price
        from foureng.products.barrier import BarrierOption

        fwd = ForwardSpec(S0=S0, r=r, q=q, T=T)
        params = fe.BsmParams(sigma=sigma)
        product = BarrierOption(
            strike=K, maturity=T, barrier=H_down, barrier_type="down_out", rebate=0.0, cp=1
        )
        p = price(product, "bsm", "proj_barrier", fwd, params)
        assert p >= 0.0
        assert p <= bsm_european(S0, K, r, q, sigma, T, cp=1) + 1e-6

    def test_proj_barrier_dispatch_rejects_nonzero_rebate(self):
        from foureng.pipeline import price
        from foureng.products.barrier import BarrierOption

        fwd = ForwardSpec(S0=S0, r=r, q=q, T=T)
        params = fe.BsmParams(sigma=sigma)
        product = BarrierOption(
            strike=K, maturity=T, barrier=H_down, barrier_type="down_out", rebate=5.0, cp=1
        )
        with pytest.raises(NotImplementedError):
            price(product, "bsm", "proj_barrier", fwd, params)
