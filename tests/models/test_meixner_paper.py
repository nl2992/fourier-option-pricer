"""Tests for the Meixner process (Schoutens & Teugels 1998).

Layer 1 — analytic benchmarks
    * b=0 gives a symmetric distribution (phi is real-valued on real axis).
    * Drift correction ω ensures E[exp(X_T)] = 1 (martingale condition).
    * phi(0) = 1 (normalization).

Layer 2 — cross-engine agreement
    Lewis / COS / Carr-Madan / FRFT must all agree to atol=1e-4 on a
    representative set (a=0.3978, b=-1.1326, delta=0.3462) from Schoutens
    (2002) S&P500 calibration.

Layer 3 — structural
    phi(0)=1, phi(-i)=1, |phi(u)|<=1, c2>0, no-arbitrage bounds,
    monotonicity in vol parameter a and intensity delta.

References
----------
* Schoutens, W. (2002), "The Meixner Process: Theory and Applications in
  Finance", EURANDOM, Eindhoven.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press, Chapter 4.
"""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.meixner import (
    MeixnerParams,
    meixner_cf,
    meixner_cumulants,
    _meixner_omega,
)
from foureng.models.base import ForwardSpec
from foureng.pricers.lewis import lewis_call_prices
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.pricers.frft import frft_price_at_strikes
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.utils.grids import FFTGrid, FRFTGrid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STRIKES = np.array([85.0, 90.0, 100.0, 110.0, 115.0])
SPOT = 100.0
TEXP = 1.0

# Schoutens (2002) S&P500 calibration parameters
_P_BENCH = MeixnerParams(a=0.3978, b=-1.1326, delta=0.3462)
_FWD_BENCH = ForwardSpec(S0=SPOT, r=0.0, q=0.0, T=TEXP)

# Symmetric case (b=0)
_P_SYM = MeixnerParams(a=0.3, b=0.0, delta=0.5)


@pytest.fixture(scope="module")
def bench():
    return _P_BENCH, _FWD_BENCH


@pytest.fixture(scope="module")
def lewis_ref(bench):
    p, fwd = bench
    phi = lambda u: meixner_cf(u, fwd, p)
    return lewis_call_prices(phi, STRIKES, spot=fwd.S0, texp=fwd.T,
                             intr=fwd.r, divr=fwd.q)


# ---------------------------------------------------------------------------
# Layer 1: Analytic benchmarks
# ---------------------------------------------------------------------------

class TestMeixnerAnalyticBenchmarks:
    """Closed-form properties of the Meixner CF."""

    def test_cf_at_zero(self, bench):
        """phi(0) = 1 (normalization)."""
        p, fwd = bench
        phi0 = meixner_cf(np.array([0.0]), fwd, p)
        np.testing.assert_allclose(abs(phi0[0] - 1.0), 0.0, atol=1e-14)

    def test_martingale_condition(self, bench):
        """phi(-i) = 1, i.e. E[S_T/F_0] = 1."""
        p, fwd = bench
        phi_neg_i = meixner_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-12,
                                   err_msg="Meixner martingale phi(-i)=1")

    def test_symmetric_modulus_even(self):
        """With b=0, |phi(u)| is even in u (symmetric distribution modulus)."""
        p = _P_SYM
        fwd = _FWD_BENCH
        u_pos = np.linspace(0.5, 5.0, 20)
        phi_pos = np.abs(meixner_cf(u_pos, fwd, p))
        phi_neg = np.abs(meixner_cf(-u_pos, fwd, p))
        np.testing.assert_allclose(phi_pos, phi_neg, atol=1e-14,
                                   err_msg="Meixner(b=0): |phi(u)| should equal |phi(-u)|")

    def test_symmetric_martingale(self):
        """Symmetric case also satisfies the martingale condition."""
        p = _P_SYM
        fwd = _FWD_BENCH
        phi_neg_i = meixner_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-12)

    def test_omega_positive_negative_b(self):
        """With b < 0 (left skew), omega is positive (upward drift correction).

        Negative b shifts probability mass to the left, so the forward
        price without correction would be below F_0; the drift correction
        omega must be positive to restore the martingale condition.
        """
        p = _P_BENCH  # b = -1.1326 < 0
        omega = _meixner_omega(p)
        assert omega > 0, f"Expected omega > 0 for b < 0, got {omega:.6f}"

    def test_omega_negative_for_b_zero(self):
        """With b=0 (symmetric distribution), omega < 0 (Jensen's inequality).

        Even a symmetric log-return distribution needs a downward drift
        correction because E[exp(X)] = exp(mean + variance/2) > 1 when
        the variance term is positive.
        """
        p = _P_SYM  # b = 0, but omega still non-zero
        omega = _meixner_omega(p)
        assert omega < 0, f"Expected omega < 0 for b=0 (Jensen effect), got {omega:.6f}"


# ---------------------------------------------------------------------------
# Layer 2: Cross-engine agreement
# ---------------------------------------------------------------------------

class TestMeixnerCrossEngine:
    """All pricers must agree with the Lewis reference to atol=1e-4."""

    def test_cos_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: meixner_cf(u, fwd, p)
        cums = meixner_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, STRIKES, grid)
        np.testing.assert_allclose(res.call_prices, lewis_ref, atol=1e-4,
                                   err_msg="COS vs Lewis")

    def test_carr_madan_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: meixner_cf(u, fwd, p)
        grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4,
                                   err_msg="Carr-Madan vs Lewis")

    def test_frft_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: meixner_cf(u, fwd, p)
        grid = FRFTGrid(N=4096, eta=0.25, alpha=1.5, lam=0.01)
        prices = frft_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4,
                                   err_msg="FRFT vs Lewis")

    @pytest.mark.parametrize("K", [85.0, 90.0, 100.0, 110.0, 115.0])
    def test_cos_vs_lewis_per_strike(self, K):
        p = _P_BENCH
        fwd = _FWD_BENCH
        phi = lambda u: meixner_cf(u, fwd, p)
        strikes = np.array([K])
        lewis = lewis_call_prices(phi, strikes, spot=fwd.S0, texp=fwd.T,
                                  intr=fwd.r, divr=fwd.q)
        cums = meixner_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, strikes, grid)
        np.testing.assert_allclose(res.call_prices, lewis, atol=1e-4,
                                   err_msg=f"COS vs Lewis at K={K}")


# ---------------------------------------------------------------------------
# Layer 3: Structural tests
# ---------------------------------------------------------------------------

class TestMeixnerStructural:
    """CF properties and no-arbitrage bounds."""

    def test_cf_modulus_le_one(self, bench):
        """|phi(u)| <= 1 for real u."""
        p, fwd = bench
        u_grid = np.linspace(-20, 20, 401)
        phi = meixner_cf(u_grid, fwd, p)
        assert np.all(np.abs(phi) <= 1.0 + 1e-12), \
            f"|phi|>1 at some u: max={np.max(np.abs(phi)):.6f}"

    def test_cumulants_positive_variance(self, bench):
        """Second cumulant (variance) must be positive."""
        p, fwd = bench
        c1, c2, c4 = meixner_cumulants(fwd, p)
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

    def test_price_increases_with_delta(self):
        """Higher intensity delta → higher ATM prices (more variance)."""
        fwd = _FWD_BENCH
        K = np.array([100.0])

        def price_at_delta(d):
            p = MeixnerParams(a=0.3978, b=-1.1326, delta=d)
            phi = lambda u: meixner_cf(u, fwd, p)
            cums = meixner_cumulants(fwd, p)
            grid = cos_auto_grid(cums, N=256, L=10.0)
            return cos_prices(phi, fwd, K, grid).call_prices[0]

        p_lo = price_at_delta(0.2)
        p_hi = price_at_delta(0.8)
        assert p_hi > p_lo, \
            f"ATM price should increase with delta: delta=0.2→{p_lo:.4f}, 0.8→{p_hi:.4f}"

    def test_price_increases_with_a(self):
        """Higher scale a → higher ATM prices (heavier tails / more variance)."""
        fwd = _FWD_BENCH
        K = np.array([100.0])

        def price_at_a(av):
            p = MeixnerParams(a=av, b=-0.5, delta=0.5)
            phi = lambda u: meixner_cf(u, fwd, p)
            cums = meixner_cumulants(fwd, p)
            grid = cos_auto_grid(cums, N=256, L=10.0)
            return cos_prices(phi, fwd, K, grid).call_prices[0]

        p_lo = price_at_a(0.1)
        p_hi = price_at_a(0.4)
        assert p_hi > p_lo, \
            f"ATM price should increase with a: a=0.1→{p_lo:.4f}, 0.4→{p_hi:.4f}"
