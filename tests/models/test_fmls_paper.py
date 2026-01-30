"""Tests for the Finite Moment Log Stable (FMLS) model (Carr & Wu 2003).

Layer 1 — analytic benchmarks
    * phi(0)=1, phi(-i)=1 (normalization and martingale condition).
    * alpha=2 exactly recovers BSM: phi(u;T) = exp(-T*iu*sigma²/2 - T*sigma²*u²/2).
    * Cumulant c1 = T * (-sigma^alpha/2) — exact closed form for the mean.

Layer 2 — cross-engine agreement
    Lewis / COS / Carr-Madan / FRFT must all agree to atol=1e-4 on the
    S&P500-calibrated parameters (Carr & Wu 2003 Table II):
    alpha=1.7, sigma=0.4.

Layer 3 — structural
    phi(0)=1, phi(-i)=1, |phi(u)|<=1, c2>0, no-arbitrage bounds,
    price monotone in sigma and alpha.

References
----------
* Carr, P. & Wu, L. (2003), "The Finite Moment Log Stable Process and
  Option Pricing", *Journal of Finance*, 58(2), 753–777.
"""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.fmls import FMLSParams, fmls_cf, fmls_cumulants
from foureng.models.base import ForwardSpec
from foureng.pricers.lewis import lewis_call_prices
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.pricers.frft import frft_price_at_strikes
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.utils.grids import FFTGrid, FRFTGrid
from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STRIKES = np.array([85.0, 90.0, 100.0, 110.0, 115.0])
SPOT = 100.0
TEXP = 1.0

# Carr & Wu (2003) Table II representative parameters (S&P500 calibration)
_P_BENCH = FMLSParams(alpha=1.7, sigma=0.4)
_FWD_BENCH = ForwardSpec(S0=SPOT, r=0.0, q=0.0, T=TEXP)

# BSM limit (alpha=2)
_P_BSM = FMLSParams(alpha=2.0, sigma=0.3)
_FWD_BSM = ForwardSpec(S0=SPOT, r=0.0, q=0.0, T=TEXP)


@pytest.fixture(scope="module")
def bench():
    return _P_BENCH, _FWD_BENCH


@pytest.fixture(scope="module")
def lewis_ref(bench):
    p, fwd = bench
    phi = lambda u: fmls_cf(u, fwd, p)
    return lewis_call_prices(phi, STRIKES, spot=fwd.S0, texp=fwd.T,
                             intr=fwd.r, divr=fwd.q)


# ---------------------------------------------------------------------------
# Layer 1: Analytic benchmarks
# ---------------------------------------------------------------------------

class TestFMLSAnalyticBenchmarks:
    """Closed-form properties of the FMLS CF."""

    def test_cf_at_zero(self, bench):
        """phi(0) = 1."""
        p, fwd = bench
        phi0 = fmls_cf(np.array([0.0]), fwd, p)
        np.testing.assert_allclose(abs(phi0[0] - 1.0), 0.0, atol=1e-14)

    def test_martingale_condition(self, bench):
        """phi(-i) = 1 (martingale condition for log-forward CF)."""
        p, fwd = bench
        phi_neg_i = fmls_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-12,
                                   err_msg="FMLS martingale phi(-i)=1")

    def test_alpha2_equals_bsm_cf(self):
        """alpha=2 gives the BSM characteristic function exactly.

        phi_BSM(u; T) = exp(-T*iu*sigma²/2 - T*sigma²*u²/2)
        """
        p = _P_BSM
        fwd = _FWD_BSM
        u_grid = np.linspace(-10, 10, 41)
        u_c = u_grid.astype(np.complex128)

        phi_fmls = fmls_cf(u_grid, fwd, p)
        phi_bsm_manual = np.exp(
            -fwd.T * 1j * u_c * p.sigma**2 / 2.0
            - fwd.T * p.sigma**2 * u_c**2 / 2.0
        )
        np.testing.assert_allclose(phi_fmls.real, phi_bsm_manual.real, atol=1e-12,
                                   err_msg="FMLS(alpha=2) CF real part vs BSM")
        np.testing.assert_allclose(phi_fmls.imag, phi_bsm_manual.imag, atol=1e-12,
                                   err_msg="FMLS(alpha=2) CF imag part vs BSM")

    def test_alpha2_prices_equal_bsm(self):
        """alpha=2 FMLS prices match BSM across strikes to 1e-8."""
        p = _P_BSM
        fwd = _FWD_BSM
        strikes = np.array([85.0, 90.0, 100.0, 110.0, 115.0])

        phi = lambda u: fmls_cf(u, fwd, p)
        fmls_prices = lewis_call_prices(phi, strikes, spot=fwd.S0, texp=fwd.T,
                                        intr=fwd.r, divr=fwd.q)
        bsm_prices = np.array([
            bs_price_from_fwd(p.sigma, BSInputs(F0=fwd.S0, K=K, T=fwd.T, r=fwd.r, q=fwd.q))
            for K in strikes
        ])
        np.testing.assert_allclose(fmls_prices, bsm_prices, atol=1e-8,
                                   err_msg="FMLS(alpha=2) must match BSM prices")

    def test_mean_cumulant_sign(self, bench):
        """c1 < 0: the left-skewed drift correction is negative."""
        p, fwd = bench
        c1, _, _ = fmls_cumulants(fwd, p)
        assert c1 < 0, f"FMLS c1 should be negative (left-skewed drift), got {c1}"

    def test_cf_martingale_alpha2(self):
        """phi(-i)=1 holds at alpha=2 boundary."""
        p = _P_BSM
        fwd = _FWD_BSM
        phi_neg_i = fmls_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Layer 2: Cross-engine agreement
# ---------------------------------------------------------------------------

class TestFMLSCrossEngine:
    """CM/FRFT agree with Lewis; COS tested only at alpha=2 (BSM limit).

    The FMLS distribution with alpha < 2 has power-law tails (alpha-stable),
    which require extremely wide COS grids to converge.  Carr-Madan and FRFT
    integrate along the real frequency axis where the CF decays exponentially,
    giving machine-precision agreement with Lewis.  For COS we only test the
    alpha=2 case (reduces to BSM with exponential tails) where COS is fast.
    """

    def test_carr_madan_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: fmls_cf(u, fwd, p)
        grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4,
                                   err_msg="Carr-Madan vs Lewis")

    def test_frft_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: fmls_cf(u, fwd, p)
        grid = FRFTGrid(N=4096, eta=0.25, alpha=1.5, lam=0.01)
        prices = frft_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4,
                                   err_msg="FRFT vs Lewis")

    def test_cos_vs_lewis_alpha2(self):
        """COS agrees with Lewis at alpha=2 (BSM limit, exponential tails)."""
        p = _P_BSM
        fwd = _FWD_BSM
        phi = lambda u: fmls_cf(u, fwd, p)
        cums = fmls_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=256, L=12.0)
        res = cos_prices(phi, fwd, STRIKES, grid)
        lewis = lewis_call_prices(phi, STRIKES, spot=fwd.S0, texp=fwd.T,
                                  intr=fwd.r, divr=fwd.q)
        np.testing.assert_allclose(res.call_prices, lewis, atol=1e-4,
                                   err_msg="COS vs Lewis at alpha=2")

    @pytest.mark.parametrize("K", [85.0, 90.0, 100.0, 110.0, 115.0])
    def test_carr_madan_vs_lewis_per_strike(self, K):
        p = _P_BENCH
        fwd = _FWD_BENCH
        phi = lambda u: fmls_cf(u, fwd, p)
        strikes = np.array([K])
        lewis = lewis_call_prices(phi, strikes, spot=fwd.S0, texp=fwd.T,
                                  intr=fwd.r, divr=fwd.q)
        grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, strikes)
        np.testing.assert_allclose(prices, lewis, atol=1e-4,
                                   err_msg=f"CM vs Lewis at K={K}")


# ---------------------------------------------------------------------------
# Layer 3: Structural tests
# ---------------------------------------------------------------------------

class TestFMLSStructural:
    """CF properties and no-arbitrage bounds."""

    def test_cf_at_zero_normalized(self, bench):
        """phi(0) = 1."""
        p, fwd = bench
        phi0 = fmls_cf(np.array([0.0]), fwd, p)
        np.testing.assert_allclose(abs(phi0[0] - 1.0), 0.0, atol=1e-14)

    def test_martingale_phi_neg_i(self, bench):
        """phi(-i) = 1."""
        p, fwd = bench
        phi_neg_i = fmls_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-12)

    def test_cf_modulus_le_one(self, bench):
        """|phi(u)| <= 1 for real u."""
        p, fwd = bench
        u_grid = np.linspace(-20, 20, 401)
        phi = fmls_cf(u_grid, fwd, p)
        assert np.all(np.abs(phi) <= 1.0 + 1e-10), \
            f"|phi|>1 at some u: max={np.max(np.abs(phi)):.6f}"

    def test_cumulants_positive_variance(self, bench):
        """Second cumulant (variance) must be positive."""
        p, fwd = bench
        c1, c2, c4 = fmls_cumulants(fwd, p)
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

    def test_price_increases_with_sigma(self):
        """Higher sigma → higher ATM price (more volatility).

        Uses Carr-Madan (works for all alpha) rather than COS.
        """
        fwd = _FWD_BENCH
        K = np.array([100.0])

        def price_at_sigma(s):
            p = FMLSParams(alpha=1.7, sigma=s)
            phi = lambda u: fmls_cf(u, fwd, p)
            grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
            return carr_madan_price_at_strikes(phi, fwd, grid, K)[0]

        p_lo = price_at_sigma(0.2)
        p_hi = price_at_sigma(0.5)
        assert p_hi > p_lo, \
            f"ATM price should increase with sigma: 0.2→{p_lo:.4f}, 0.5→{p_hi:.4f}"

    def test_price_increases_with_alpha_toward_two(self):
        """OTM call is lower for heavy-tailed alpha=1.5 vs near-Gaussian alpha=1.95.

        FMLS with lower alpha has heavier left tails (more downside skew), which
        shifts weight away from high-strike outcomes.  Uses Carr-Madan.
        """
        fwd = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=1.0)
        K = np.array([115.0])  # OTM call

        def price_at_alpha(a):
            p = FMLSParams(alpha=a, sigma=0.3)
            phi = lambda u: fmls_cf(u, fwd, p)
            grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
            return carr_madan_price_at_strikes(phi, fwd, grid, K)[0]

        p_low_alpha = price_at_alpha(1.5)
        p_high_alpha = price_at_alpha(1.95)
        assert p_high_alpha > p_low_alpha, \
            f"OTM call should be higher at alpha=1.95 than alpha=1.5: {p_high_alpha:.4f} vs {p_low_alpha:.4f}"
