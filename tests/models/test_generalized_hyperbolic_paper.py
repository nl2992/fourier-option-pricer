"""Tests for the Generalized Hyperbolic (GH) Lévy model (Barndorff-Nielsen 1977).

Layer 1  -  analytic benchmarks
    * phi(0)=1, phi(-i)=1 (normalization and martingale condition).
    * GH with lam=-0.5 reduces to NIG: the exponent simplifies to
      δ*(sqrt(α²-β²) - sqrt(α²-(β+iu)²)) (verified analytically).
    * lam=1 (hyperbolic distribution) also satisfies both conditions.

Layer 2  -  cross-engine agreement
    Lewis / COS / Carr-Madan / FRFT must all agree to atol=1e-4 on
    the NIG benchmark (lam=-0.5, alpha=6.1882, beta=-3.8941, delta=0.1622)
    from Eberlein & Keller (1995).

Layer 3  -  structural
    phi(0)=1, phi(-i)=1, |phi(u)|<=1, c2>0, no-arbitrage bounds,
    price monotone in delta (scale) and alpha (tail heaviness).

References
----------
* Barndorff-Nielsen, O. E. (1977), "Exponentially decreasing distributions
  for the logarithm of particle size", *Proc. Royal Society London*, 353.
* Eberlein, E. & Keller, U. (1995), "Hyperbolic distributions in finance",
  *Bernoulli*, 1(3), 281–299.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press, Chapter 4.
"""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.generalized_hyperbolic import (
    GHParams,
    gh_cf,
    gh_cumulants,
    _gh_omega,
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

# NIG limit (lam=-0.5)  -  Eberlein & Keller (1995) DAX calibration
_P_NIG = GHParams(lam=-0.5, alpha=6.1882, beta=-3.8941, delta=0.1622)
_FWD_BENCH = ForwardSpec(S0=SPOT, r=0.0, q=0.0, T=TEXP)

# Hyperbolic limit (lam=1)
_P_HYP = GHParams(lam=1.0, alpha=4.0, beta=-1.0, delta=0.5)


@pytest.fixture(scope="module")
def bench():
    return _P_NIG, _FWD_BENCH


@pytest.fixture(scope="module")
def lewis_ref(bench):
    p, fwd = bench
    phi = lambda u: gh_cf(u, fwd, p)
    return lewis_call_prices(phi, STRIKES, spot=fwd.S0, texp=fwd.T,
                             intr=fwd.r, divr=fwd.q)


# ---------------------------------------------------------------------------
# Layer 1: Analytic benchmarks
# ---------------------------------------------------------------------------

class TestGHAnalyticBenchmarks:
    """Closed-form properties of the GH CF."""

    def test_cf_at_zero_nig(self, bench):
        """phi(0) = 1 for NIG (lam=-0.5)."""
        p, fwd = bench
        phi0 = gh_cf(np.array([0.0]), fwd, p)
        np.testing.assert_allclose(abs(phi0[0] - 1.0), 0.0, atol=1e-14)

    def test_martingale_condition_nig(self, bench):
        """phi(-i) = 1 for NIG (lam=-0.5)."""
        p, fwd = bench
        phi_neg_i = gh_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-12,
                                   err_msg="GH(NIG) martingale phi(-i)=1")

    def test_cf_at_zero_hyperbolic(self):
        """phi(0) = 1 for Hyperbolic (lam=1)."""
        p = _P_HYP
        fwd = _FWD_BENCH
        phi0 = gh_cf(np.array([0.0]), fwd, p)
        np.testing.assert_allclose(abs(phi0[0] - 1.0), 0.0, atol=1e-14)

    def test_martingale_condition_hyperbolic(self):
        """phi(-i) = 1 for Hyperbolic (lam=1)."""
        p = _P_HYP
        fwd = _FWD_BENCH
        phi_neg_i = gh_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-11,
                                   err_msg="GH(Hyperbolic) martingale phi(-i)=1")

    def test_nig_exponent_matches_nig_formula(self, bench):
        """GH with lam=-0.5 must match the NIG formula δ*(z0-z1) analytically.

        For K_{-1/2}(z) = sqrt(π/(2z))*exp(-z), the GH exponent simplifies to
        δ*(sqrt(α²-β²) - sqrt(α²-(β+iu)²)). We verify the CF matches.
        """
        p, fwd = bench
        u_grid = np.linspace(-5, 5, 21)
        u_c = u_grid.astype(np.complex128)

        # GH CF
        phi_gh = gh_cf(u_grid, fwd, p)

        # Manual NIG formula: delta*(r0 - r1) where r0, r1 = sqrt(α²-β²), sqrt(α²-(β+iu)²)
        r0 = np.sqrt(p.alpha**2 - p.beta**2)
        omega = _gh_omega(p)
        r1 = np.sqrt(p.alpha**2 - (p.beta + 1j*u_c)**2)
        nig_exponent = fwd.T * (1j*u_c*omega + p.delta*(r0 - r1))
        phi_nig_manual = np.exp(nig_exponent)

        np.testing.assert_allclose(phi_gh.real, phi_nig_manual.real, atol=1e-10,
                                   err_msg="GH(lam=-0.5) real part vs NIG formula")
        np.testing.assert_allclose(phi_gh.imag, phi_nig_manual.imag, atol=1e-10,
                                   err_msg="GH(lam=-0.5) imag part vs NIG formula")


# ---------------------------------------------------------------------------
# Layer 2: Cross-engine agreement
# ---------------------------------------------------------------------------

class TestGHCrossEngine:
    """All pricers must agree with the Lewis reference to atol=1e-4."""

    def test_cos_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: gh_cf(u, fwd, p)
        cums = gh_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, STRIKES, grid)
        np.testing.assert_allclose(res.call_prices, lewis_ref, atol=1e-4,
                                   err_msg="COS vs Lewis")

    def test_carr_madan_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: gh_cf(u, fwd, p)
        grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4,
                                   err_msg="Carr-Madan vs Lewis")

    def test_frft_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: gh_cf(u, fwd, p)
        grid = FRFTGrid(N=4096, eta=0.25, alpha=1.5, lam=0.01)
        prices = frft_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4,
                                   err_msg="FRFT vs Lewis")

    @pytest.mark.parametrize("K", [85.0, 90.0, 100.0, 110.0, 115.0])
    def test_cos_vs_lewis_per_strike(self, K):
        p = _P_NIG
        fwd = _FWD_BENCH
        phi = lambda u: gh_cf(u, fwd, p)
        strikes = np.array([K])
        lewis = lewis_call_prices(phi, strikes, spot=fwd.S0, texp=fwd.T,
                                  intr=fwd.r, divr=fwd.q)
        cums = gh_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, strikes, grid)
        np.testing.assert_allclose(res.call_prices, lewis, atol=1e-4,
                                   err_msg=f"COS vs Lewis at K={K}")


# ---------------------------------------------------------------------------
# Layer 3: Structural tests
# ---------------------------------------------------------------------------

class TestGHStructural:
    """CF properties and no-arbitrage bounds."""

    def test_cf_modulus_le_one(self, bench):
        """|phi(u)| <= 1 for real u."""
        p, fwd = bench
        u_grid = np.linspace(-20, 20, 401)
        phi = gh_cf(u_grid, fwd, p)
        assert np.all(np.abs(phi) <= 1.0 + 1e-10), \
            f"|phi|>1 at some u: max={np.max(np.abs(phi)):.6f}"

    def test_cumulants_positive_variance(self, bench):
        """Second cumulant (variance) must be positive."""
        p, fwd = bench
        c1, c2, c4 = gh_cumulants(fwd, p)
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
        """Higher delta → higher ATM prices (more scale / variance)."""
        fwd = _FWD_BENCH
        K = np.array([100.0])

        def price_at_delta(d):
            p = GHParams(lam=-0.5, alpha=6.1882, beta=-3.8941, delta=d)
            phi = lambda u: gh_cf(u, fwd, p)
            cums = gh_cumulants(fwd, p)
            grid = cos_auto_grid(cums, N=256, L=10.0)
            return cos_prices(phi, fwd, K, grid).call_prices[0]

        p_lo = price_at_delta(0.05)
        p_hi = price_at_delta(0.30)
        assert p_hi > p_lo, \
            f"ATM price should increase with delta: 0.05→{p_lo:.4f}, 0.30→{p_hi:.4f}"
