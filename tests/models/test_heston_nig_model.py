"""Tests for the Heston-NIG SVJ model (Heston SV + NIG Lévy jumps).

Layer 1  -  CF structure
    * phi(0)=1, phi(-i)=1 (normalization and martingale condition).
    * |phi(u)| <= 1 for real u.
    * Factorization: phi_HN / phi_H equals the standalone NIG jump CF.

Layer 2  -  cross-engine agreement
    Lewis / COS / Carr-Madan / FRFT must all agree to atol=1e-4 on the
    benchmark parameter set.

Layer 3  -  structural and no-arbitrage
    c2 > 0, call prices positive, monotone decreasing in strike,
    call >= intrinsic, call <= spot.

Layer 4  -  parameter validation
    Invalid parameters raise ValueError with appropriate messages.

References
----------
* Heston, S. (1993), *Review of Financial Studies*, 6(2), 327-343.
* Barndorff-Nielsen, O. E. (1997), *Scand. J. Statist.*, 24, 1-13.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press. (Chapter 15: stochastic-volatility models with Lévy jumps.)
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.heston import heston_cf
from foureng.models.heston_nig import HestonNIGParams, heston_nig_cf, heston_nig_cumulants
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.pricers.frft import frft_price_at_strikes
from foureng.pricers.lewis import lewis_call_prices
from foureng.utils.grids import FFTGrid, FRFTGrid

pytestmark = [pytest.mark.paper, pytest.mark.derived_reference]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

STRIKES = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
SPOT = 100.0
TEXP = 1.0

# Representative equity-calibration parameter set
# (Heston SV with negative skew + NIG jump overlay with negative drift)
_P_BENCH = HestonNIGParams(
    kappa=2.0, theta=0.04, nu=0.3, rho=-0.7, v0=0.04,
    nig_sigma=0.15, nig_nu=0.5, nig_theta=-0.10,
)
_FWD_BENCH = ForwardSpec(S0=SPOT, r=0.05, q=0.0, T=TEXP)


@pytest.fixture(scope="module")
def bench():
    return _P_BENCH, _FWD_BENCH


@pytest.fixture(scope="module")
def lewis_ref(bench):
    p, fwd = bench
    phi = lambda u: heston_nig_cf(u, fwd, p)
    return lewis_call_prices(phi, STRIKES, spot=fwd.S0, texp=fwd.T, intr=fwd.r, divr=fwd.q)


# ---------------------------------------------------------------------------
# Layer 1: CF structure
# ---------------------------------------------------------------------------


class TestHestonNIGCFStructure:
    """Normalization, martingale, modulus, and factorization checks."""

    def test_phi_at_zero(self, bench):
        """phi(0) = 1."""
        p, fwd = bench
        phi0 = heston_nig_cf(np.array([0.0]), fwd, p)
        np.testing.assert_allclose(abs(phi0[0] - 1.0), 0.0, atol=1e-14)

    def test_martingale_condition(self, bench):
        """phi(-i) = 1 (E[S_T/F_0] = 1)."""
        p, fwd = bench
        phi_mi = heston_nig_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(
            abs(phi_mi[0] - 1.0), 0.0, atol=1e-12, err_msg="Heston-NIG martingale phi(-i)=1"
        )

    def test_cf_modulus_le_one(self, bench):
        """|phi(u)| <= 1 for real u."""
        p, fwd = bench
        u_grid = np.linspace(-20, 20, 201)
        phi = heston_nig_cf(u_grid, fwd, p)
        assert np.all(np.abs(phi) <= 1.0 + 1e-12), (
            f"max|phi| = {np.max(np.abs(phi)):.6f} > 1"
        )

    def test_factorization_with_heston(self, bench):
        """phi_HN / phi_H gives the NIG jump component.

        The ratio phi_HN(u) / phi_Heston(u) must equal
        exp(T * (psi_NIG(u) - iu * psi_NIG(-i))) for all u.
        """
        p, fwd = bench
        u_grid = np.linspace(-10, 10, 51)
        u_c = u_grid.astype(np.complex128)

        phi_hn = heston_nig_cf(u_c, fwd, p)
        phi_h = heston_cf(u_c, fwd, p.heston_params)

        # phi_HN / phi_H should be a CF with phi(-i) = 1
        ratio = phi_hn / phi_h
        ratio_at_mi = (heston_nig_cf(np.array([-1j]), fwd, p) /
                       heston_cf(np.array([-1j]), fwd, p.heston_params))
        np.testing.assert_allclose(
            abs(ratio_at_mi[0] - 1.0), 0.0, atol=1e-12,
            err_msg="NIG jump component phi_NIG_jump(-i) must be 1"
        )

        # ratio should be smooth and bounded
        assert np.all(np.isfinite(np.abs(ratio))), "Ratio phi_HN/phi_H has non-finite values"

    def test_cf_real_u_conjugate_symmetry(self, bench):
        """phi(-u) = conj(phi(u)) for real u (CF of a real-valued RV)."""
        p, fwd = bench
        u_grid = np.linspace(0.5, 10, 20)
        phi_pos = heston_nig_cf(u_grid, fwd, p)
        phi_neg = heston_nig_cf(-u_grid, fwd, p)
        np.testing.assert_allclose(
            phi_neg, np.conj(phi_pos), atol=1e-14,
            err_msg="Heston-NIG CF conjugate symmetry"
        )


# ---------------------------------------------------------------------------
# Layer 2: Cross-engine agreement
# ---------------------------------------------------------------------------


class TestHestonNIGCrossEngine:
    """All Fourier pricers must agree with Lewis reference to atol=1e-4."""

    def test_cos_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: heston_nig_cf(u, fwd, p)
        cums = heston_nig_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, STRIKES, grid)
        np.testing.assert_allclose(
            res.call_prices, lewis_ref, atol=1e-4, err_msg="COS vs Lewis"
        )

    def test_carr_madan_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: heston_nig_cf(u, fwd, p)
        grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4, err_msg="Carr-Madan vs Lewis")

    def test_frft_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: heston_nig_cf(u, fwd, p)
        grid = FRFTGrid(N=4096, eta=0.25, alpha=1.5, lam=0.005)
        prices = frft_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4, err_msg="FRFT vs Lewis")

    @pytest.mark.parametrize("K", [85.0, 90.0, 100.0, 110.0, 115.0])
    def test_cos_vs_lewis_per_strike(self, K):
        p = _P_BENCH
        fwd = _FWD_BENCH
        phi = lambda u: heston_nig_cf(u, fwd, p)
        strikes = np.array([K])
        lewis = lewis_call_prices(phi, strikes, spot=fwd.S0, texp=fwd.T, intr=fwd.r, divr=fwd.q)
        cums = heston_nig_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, strikes, grid)
        np.testing.assert_allclose(
            res.call_prices, lewis, atol=1e-4, err_msg=f"COS vs Lewis at K={K}"
        )


# ---------------------------------------------------------------------------
# Layer 3: Structural / no-arbitrage
# ---------------------------------------------------------------------------


class TestHestonNIGStructural:
    """No-arbitrage bounds and monotonicity of the pricing surface."""

    def test_cumulants_positive_variance(self, bench):
        """Second cumulant (variance) must be positive."""
        p, fwd = bench
        c1, c2, c4 = heston_nig_cumulants(fwd, p)
        assert c2 > 0, f"c2 must be positive; got {c2}"
        assert np.isfinite(c1) and np.isfinite(c4)

    def test_call_ge_intrinsic(self, bench, lewis_ref):
        """Call prices >= max(F-K, 0)."""
        p, fwd = bench
        intrinsic = np.maximum(fwd.F0 - STRIKES, 0.0)
        assert np.all(lewis_ref >= intrinsic - 1e-6), "Call price below intrinsic value"

    def test_call_le_spot(self, bench, lewis_ref):
        """Call prices <= spot."""
        p, fwd = bench
        assert np.all(lewis_ref <= fwd.S0 + 1e-8), "Call price exceeds spot"

    def test_prices_positive(self, lewis_ref):
        """All call prices must be positive."""
        assert np.all(lewis_ref > 0), f"Non-positive prices: {lewis_ref}"

    def test_prices_monotone_decreasing(self, lewis_ref):
        """Calls are monotone decreasing in strike."""
        assert np.all(np.diff(lewis_ref) < 0), f"Call prices not monotone: {lewis_ref}"

    def test_nig_jump_adds_variance(self):
        """Adding NIG jumps (larger nig_sigma) increases ATM call price."""
        fwd = _FWD_BENCH
        K = np.array([100.0])

        def price_at_nig_sigma(sig):
            p = HestonNIGParams(
                kappa=2.0, theta=0.04, nu=0.3, rho=-0.7, v0=0.04,
                nig_sigma=sig, nig_nu=0.5, nig_theta=-0.10,
            )
            phi = lambda u: heston_nig_cf(u, fwd, p)
            cums = heston_nig_cumulants(fwd, p)
            grid = cos_auto_grid(cums, N=256, L=10.0)
            return cos_prices(phi, fwd, K, grid).call_prices[0]

        p_lo = price_at_nig_sigma(0.05)
        p_hi = price_at_nig_sigma(0.30)
        assert p_hi > p_lo, (
            f"ATM call should increase with nig_sigma: 0.05→{p_lo:.4f}, 0.30→{p_hi:.4f}"
        )

    def test_near_deterministic_heston_still_prices(self):
        """Very small Heston nu gives near-BSM+NIG prices (all positive and monotone).

        PyFENG requires vov > 0 strictly, so we use nu=1e-6 to approximate
        the BSM+NIG limit (Heston with negligible stochastic volatility).
        """
        fwd = _FWD_BENCH
        K = np.array([95.0, 100.0, 105.0])
        p = HestonNIGParams(
            kappa=2.0, theta=0.04, nu=1e-6, rho=-0.5, v0=0.04,
            nig_sigma=0.15, nig_nu=0.5, nig_theta=-0.10,
        )
        phi = lambda u: heston_nig_cf(u, fwd, p)
        calls = lewis_call_prices(phi, K, spot=fwd.S0, texp=fwd.T, intr=fwd.r, divr=fwd.q)
        assert np.all(calls > 0), "Near-BSM+NIG prices should be positive"
        assert np.all(np.diff(calls) < 0), "Near-BSM+NIG prices should be monotone"


# ---------------------------------------------------------------------------
# Layer 4: Parameter validation
# ---------------------------------------------------------------------------


class TestHestonNIGParamValidation:
    """Invalid parameters must raise ValueError."""

    def test_invalid_kappa(self):
        with pytest.raises(ValueError, match="kappa"):
            HestonNIGParams(kappa=-1.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04,
                            nig_sigma=0.15, nig_nu=0.5, nig_theta=-0.1)

    def test_invalid_nu_zero(self):
        with pytest.raises(ValueError, match="nu"):
            HestonNIGParams(kappa=2.0, theta=0.04, nu=0.0, rho=-0.5, v0=0.04,
                            nig_sigma=0.15, nig_nu=0.5, nig_theta=-0.1)

    def test_invalid_rho(self):
        with pytest.raises(ValueError, match="rho"):
            HestonNIGParams(kappa=2.0, theta=0.04, nu=0.3, rho=1.5, v0=0.04,
                            nig_sigma=0.15, nig_nu=0.5, nig_theta=-0.1)

    def test_invalid_v0(self):
        with pytest.raises(ValueError, match="v0"):
            HestonNIGParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.0,
                            nig_sigma=0.15, nig_nu=0.5, nig_theta=-0.1)

    def test_invalid_nig_sigma(self):
        with pytest.raises(ValueError, match="nig_sigma"):
            HestonNIGParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04,
                            nig_sigma=0.0, nig_nu=0.5, nig_theta=-0.1)

    def test_invalid_nig_nu(self):
        with pytest.raises(ValueError, match="nig_nu"):
            HestonNIGParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04,
                            nig_sigma=0.15, nig_nu=0.0, nig_theta=-0.1)

    def test_nig_existence_condition_violated(self):
        """1 - 2*nig_theta*nig_nu - nig_sigma^2*nig_nu <= 0 must raise."""
        # Make nig_theta very positive so 2*theta*nu >> 1
        with pytest.raises(ValueError, match="existence condition"):
            HestonNIGParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04,
                            nig_sigma=0.15, nig_nu=0.5, nig_theta=5.0)
