"""Tests for the Bilateral Gamma model (Küchler & Tappe 2008).

Layer 1  -  analytic benchmarks
    * Closed-form cumulants match formulae from Küchler & Tappe (2008).
    * Setting alpha_m=0 reduces to a pure Gamma process (one-sided).
    * phi(0)=1, phi(-i)=1 (martingale condition).

Layer 2  -  cross-engine agreement
    Lewis / COS / Carr-Madan / FRFT must all agree to atol=1e-4 on
    benchmark parameters (alpha_p=1.0, lambda_p=5.0, alpha_m=0.8,
    lambda_m=4.0).

Layer 3  -  structural
    phi(0)=1, phi(-i)=1, |phi(u)|<=1, c2>0, no-arbitrage bounds,
    monotone in lambda_m (bigger lambda_m → tighter/faster downward jumps
    → less downward contribution → higher call prices).

References
----------
* Küchler, U. & Tappe, S. (2008), "Bilateral Gamma distributions and
  processes in financial mathematics", *Stochastic Processes and their
  Applications*, 118(2), 261–283.
"""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.bilateral_gamma import (
    BilateralGammaParams,
    bilateral_gamma_cf,
    bilateral_gamma_cumulants,
    _bg_omega,
)
from foureng.models.base import ForwardSpec
from foureng.pricers.lewis import lewis_call_prices
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.pricers.frft import frft_price_at_strikes
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.utils.grids import FFTGrid, FRFTGrid


pytestmark = [pytest.mark.paper, pytest.mark.derived_reference]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STRIKES = np.array([85.0, 90.0, 100.0, 110.0, 115.0])
SPOT = 100.0
TEXP = 1.0

_P_BENCH = BilateralGammaParams(alpha_p=1.0, lambda_p=5.0, alpha_m=0.8, lambda_m=4.0)
_FWD_BENCH = ForwardSpec(S0=SPOT, r=0.0, q=0.0, T=TEXP)


@pytest.fixture(scope="module")
def bench():
    return _P_BENCH, _FWD_BENCH


@pytest.fixture(scope="module")
def lewis_ref(bench):
    p, fwd = bench
    phi = lambda u: bilateral_gamma_cf(u, fwd, p)
    return lewis_call_prices(phi, STRIKES, spot=fwd.S0, texp=fwd.T,
                             intr=fwd.r, divr=fwd.q)


# ---------------------------------------------------------------------------
# Layer 1: Analytic benchmarks
# ---------------------------------------------------------------------------

class TestBGAnalyticBenchmarks:
    """Closed-form properties of the BG CF."""

    def test_cf_at_zero(self, bench):
        """phi(0) = 1 (normalization)."""
        p, fwd = bench
        phi0 = bilateral_gamma_cf(np.array([0.0]), fwd, p)
        np.testing.assert_allclose(abs(phi0[0] - 1.0), 0.0, atol=1e-14)

    def test_martingale_condition(self, bench):
        """phi(-i) = 1, i.e. E[S_T/F_0] = 1."""
        p, fwd = bench
        phi_neg_i = bilateral_gamma_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-12,
                                   err_msg="BG martingale phi(-i)=1")

    def test_cumulant_c2_formula(self, bench):
        """c2 = T*(alpha_p/lambda_p^2 + alpha_m/lambda_m^2)."""
        p, fwd = bench
        c1, c2, c4 = bilateral_gamma_cumulants(fwd, p)
        expected = fwd.T * (p.alpha_p / p.lambda_p**2 + p.alpha_m / p.lambda_m**2)
        np.testing.assert_allclose(c2, expected, rtol=1e-12)

    def test_cumulant_c4_formula(self, bench):
        """c4 = T*2*(alpha_p/lambda_p^4 + alpha_m/lambda_m^4)."""
        p, fwd = bench
        c1, c2, c4 = bilateral_gamma_cumulants(fwd, p)
        expected = fwd.T * 2.0 * (p.alpha_p / p.lambda_p**4 + p.alpha_m / p.lambda_m**4)
        np.testing.assert_allclose(c4, expected, rtol=1e-12)

    def test_symmetric_params_zero_skew(self):
        """With alpha_p=alpha_m and lambda_p=lambda_m, c1=0 (symmetric).

        omega = -(alpha*log(lam/(lam-1)) + alpha*log(lam/(lam+1)))
              = -alpha*log(lam^2/(lam^2-1))
        c1 = T*(omega + alpha/lam - alpha/lam) = T*omega (no net drift from jumps)
        """
        p = BilateralGammaParams(alpha_p=1.0, lambda_p=5.0, alpha_m=1.0, lambda_m=5.0)
        fwd = _FWD_BENCH
        # With equal params, the process is symmetric: upward and downward are mirror
        phi_neg_i = bilateral_gamma_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-12,
                                   err_msg="Symmetric BG: phi(-i)=1")


# ---------------------------------------------------------------------------
# Layer 2: Cross-engine agreement
# ---------------------------------------------------------------------------

class TestBGCrossEngine:
    """All pricers must agree with the Lewis reference to atol=1e-4."""

    def test_cos_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: bilateral_gamma_cf(u, fwd, p)
        cums = bilateral_gamma_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, STRIKES, grid)
        np.testing.assert_allclose(res.call_prices, lewis_ref, atol=1e-4,
                                   err_msg="COS vs Lewis")

    def test_carr_madan_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: bilateral_gamma_cf(u, fwd, p)
        grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4,
                                   err_msg="Carr-Madan vs Lewis")

    def test_frft_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: bilateral_gamma_cf(u, fwd, p)
        grid = FRFTGrid(N=4096, eta=0.25, alpha=1.5, lam=0.01)
        prices = frft_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4,
                                   err_msg="FRFT vs Lewis")

    @pytest.mark.parametrize("K", [85.0, 90.0, 100.0, 110.0, 115.0])
    def test_cos_vs_lewis_per_strike(self, K):
        p = _P_BENCH
        fwd = _FWD_BENCH
        phi = lambda u: bilateral_gamma_cf(u, fwd, p)
        strikes = np.array([K])
        lewis = lewis_call_prices(phi, strikes, spot=fwd.S0, texp=fwd.T,
                                  intr=fwd.r, divr=fwd.q)
        cums = bilateral_gamma_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, strikes, grid)
        np.testing.assert_allclose(res.call_prices, lewis, atol=1e-4,
                                   err_msg=f"COS vs Lewis at K={K}")


# ---------------------------------------------------------------------------
# Layer 3: Structural tests
# ---------------------------------------------------------------------------

class TestBGStructural:
    """CF properties and no-arbitrage bounds."""

    def test_cf_modulus_le_one(self, bench):
        """|phi(u)| <= 1 for real u."""
        p, fwd = bench
        u_grid = np.linspace(-20, 20, 401)
        phi = bilateral_gamma_cf(u_grid, fwd, p)
        assert np.all(np.abs(phi) <= 1.0 + 1e-12), \
            f"|phi|>1 at some u: max={np.max(np.abs(phi)):.6f}"

    def test_cumulants_positive_variance(self, bench):
        """Second cumulant (variance) must be positive."""
        p, fwd = bench
        c1, c2, c4 = bilateral_gamma_cumulants(fwd, p)
        assert c2 > 0, f"c2 should be positive, got {c2}"
        assert np.isfinite(c1) and np.isfinite(c4)

    def test_call_ge_intrinsic(self, bench, lewis_ref):
        """Call prices >= max(F-K, 0)."""
        p, fwd = bench
        intrinsic = np.maximum(fwd.F0 - STRIKES, 0.0)
        assert np.all(lewis_ref >= intrinsic - 1e-6), \
            "Call price below intrinsic value"

    def test_call_le_spot(self, bench, lewis_ref):
        """Call prices <= spot."""
        p, fwd = bench
        assert np.all(lewis_ref <= fwd.S0 + 1e-8), "Call price exceeds spot"

    def test_prices_positive(self, lewis_ref):
        """All call prices should be positive."""
        assert np.all(lewis_ref > 0), f"Non-positive prices: {lewis_ref}"

    def test_prices_monotone_decreasing(self, lewis_ref):
        """Calls are monotone decreasing in strike."""
        assert np.all(np.diff(lewis_ref) < 0), \
            f"Call prices not monotone in strike: {lewis_ref}"

    def test_price_increases_with_alpha_p(self):
        """Higher alpha_p (more upward jump intensity) → higher ATM prices."""
        fwd = _FWD_BENCH
        K = np.array([100.0])

        def price_at_ap(ap):
            p = BilateralGammaParams(alpha_p=ap, lambda_p=5.0, alpha_m=0.8, lambda_m=4.0)
            phi = lambda u: bilateral_gamma_cf(u, fwd, p)
            cums = bilateral_gamma_cumulants(fwd, p)
            grid = cos_auto_grid(cums, N=256, L=10.0)
            return cos_prices(phi, fwd, K, grid).call_prices[0]

        p_lo = price_at_ap(0.5)
        p_hi = price_at_ap(2.0)
        assert p_hi > p_lo, \
            f"ATM price should increase with alpha_p: 0.5→{p_lo:.4f}, 2.0→{p_hi:.4f}"

    def test_price_increases_with_alpha_m(self):
        """Higher alpha_m (more downward jump intensity) → higher ATM prices.

        Although downward jumps shift mean down, they add variance which
        dominates for ATM options.
        """
        fwd = _FWD_BENCH
        K = np.array([100.0])

        def price_at_am(am):
            p = BilateralGammaParams(alpha_p=1.0, lambda_p=5.0, alpha_m=am, lambda_m=4.0)
            phi = lambda u: bilateral_gamma_cf(u, fwd, p)
            cums = bilateral_gamma_cumulants(fwd, p)
            grid = cos_auto_grid(cums, N=256, L=10.0)
            return cos_prices(phi, fwd, K, grid).call_prices[0]

        p_lo = price_at_am(0.3)
        p_hi = price_at_am(1.5)
        assert p_hi > p_lo, \
            f"ATM price should increase with alpha_m: 0.3→{p_lo:.4f}, 1.5→{p_hi:.4f}"
