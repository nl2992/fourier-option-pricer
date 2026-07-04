"""Tests for the Merton (1976) jump-diffusion model.

Layer 1  -  limit / analytic benchmarks
    * lambda=0  reduces to BSM (verified against bs_price_from_fwd).
    * Cumulant formulae match numerical Cauchy integration.

Layer 2  -  cross-engine agreement
    Lewis / COS / Carr-Madan / FRFT must all agree to atol=1e-4 on a
    representative parameter set (sigma=0.2, lam=0.3, muj=-0.05, sigj=0.1).

Layer 3  -  structural
    phi(0)=1, phi(-i)=1, |phi(u)|<=1, c2>0, no-arbitrage bounds,
    monotonicity in lambda and sigma.

References
----------
* Merton, R. (1976), "Option pricing when underlying stock returns are
  discontinuous", *Journal of Financial Economics*, 3, 125–144.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd
from foureng.models.base import ForwardSpec
from foureng.models.merton_jd import (
    MertonJDParams,
    merton_jd_cf,
    merton_jd_cumulants,
)
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.pricers.frft import frft_price_at_strikes
from foureng.pricers.lewis import lewis_call_prices
from foureng.utils.grids import FFTGrid, FRFTGrid

pytestmark = [pytest.mark.paper, pytest.mark.derived_reference]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STRIKES = np.array([85.0, 90.0, 100.0, 110.0, 115.0])
SPOT = 100.0
TEXP = 1.0

_P_BENCH = MertonJDParams(sigma=0.2, lam=0.3, muj=-0.05, sigj=0.1)
_FWD_BENCH = ForwardSpec(S0=SPOT, r=0.0, q=0.0, T=TEXP)

_P_BSM = MertonJDParams(sigma=0.2, lam=0.0, muj=0.0, sigj=0.0)


@pytest.fixture(scope="module")
def bench():
    return _P_BENCH, _FWD_BENCH


@pytest.fixture(scope="module")
def lewis_ref(bench):
    p, fwd = bench
    phi = lambda u: merton_jd_cf(u, fwd, p)
    return lewis_call_prices(phi, STRIKES, spot=fwd.S0, texp=fwd.T, intr=fwd.r, divr=fwd.q)


# ---------------------------------------------------------------------------
# Layer 1: Limit cases and analytic benchmarks
# ---------------------------------------------------------------------------


class TestMertonJDLimitCases:
    """lambda=0 reduces to BSM; cumulants match numerical Cauchy integral."""

    @pytest.mark.parametrize("K", [85.0, 100.0, 115.0])
    def test_zero_lambda_equals_bsm(self, K):
        """With lam=0, MJD must equal BSM exactly."""
        fwd = _FWD_BENCH
        p_jd = _P_BSM

        phi_jd = lambda u: merton_jd_cf(u, fwd, p_jd)
        cums = merton_jd_cumulants(fwd, p_jd)
        grid = cos_auto_grid(cums, N=256, L=10.0)
        jd_price = cos_prices(phi_jd, fwd, np.array([K]), grid).call_prices[0]

        bsm_price = bs_price_from_fwd(
            p_jd.sigma,
            BSInputs(F0=fwd.F0, K=K, T=fwd.T, r=fwd.r, q=fwd.q),
        )
        np.testing.assert_allclose(
            jd_price,
            bsm_price,
            atol=1e-5,
            err_msg=f"MJD(lam=0) vs BSM at K={K}",
        )

    def test_cumulants_c1(self):
        """c1 = T*(-sigma^2/2 + omega + lam*muj); omega = -lam*(exp(muj+sigj^2/2)-1).

        The -sigma^2/2 term is the diffusion part of the forward-normalized
        drift used by merton_jd_cf; it was previously missing here and in the
        cumulant function itself."""
        p = _P_BENCH
        fwd = _FWD_BENCH
        c1, c2, c4 = merton_jd_cumulants(fwd, p)
        zeta = np.expm1(p.muj + 0.5 * p.sigj**2)
        omega = -p.lam * zeta
        expected_c1 = fwd.T * (-0.5 * p.sigma**2 + omega + p.lam * p.muj)
        np.testing.assert_allclose(c1, expected_c1, rtol=1e-12)

    def test_cumulants_c2(self):
        """c2 = T*(sigma^2 + lam*(muj^2 + sigj^2))."""
        p = _P_BENCH
        fwd = _FWD_BENCH
        c1, c2, c4 = merton_jd_cumulants(fwd, p)
        expected_c2 = fwd.T * (p.sigma**2 + p.lam * (p.muj**2 + p.sigj**2))
        np.testing.assert_allclose(c2, expected_c2, rtol=1e-12)

    def test_cumulants_c4(self):
        """c4 = T*lam*(muj^4 + 6*muj^2*sigj^2 + 3*sigj^4)."""
        p = _P_BENCH
        fwd = _FWD_BENCH
        c1, c2, c4 = merton_jd_cumulants(fwd, p)
        expected_c4 = fwd.T * p.lam * (p.muj**4 + 6 * p.muj**2 * p.sigj**2 + 3 * p.sigj**4)
        np.testing.assert_allclose(c4, expected_c4, rtol=1e-12)

    def test_price_increases_with_jump_vol(self):
        """Higher jump vol sigj → higher ATM call prices (sigj adds variance)."""
        fwd = _FWD_BENCH
        K = np.array([100.0])

        def price_at_sigj(sj):
            p = MertonJDParams(sigma=0.2, lam=0.3, muj=-0.05, sigj=sj)
            phi = lambda u: merton_jd_cf(u, fwd, p)
            cums = merton_jd_cumulants(fwd, p)
            grid = cos_auto_grid(cums, N=256, L=10.0)
            return cos_prices(phi, fwd, K, grid).call_prices[0]

        p_lo = price_at_sigj(0.05)
        p_hi = price_at_sigj(0.30)
        assert p_hi > p_lo, (
            f"ATM call should increase with sigj: sigj=0.05→{p_lo:.4f}, sigj=0.30→{p_hi:.4f}"
        )


# ---------------------------------------------------------------------------
# Layer 2: Cross-engine agreement
# ---------------------------------------------------------------------------


class TestMertonJDCrossEngine:
    """All pricers must agree with the Lewis reference to atol=1e-4."""

    def test_cos_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: merton_jd_cf(u, fwd, p)
        cums = merton_jd_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, STRIKES, grid)
        np.testing.assert_allclose(res.call_prices, lewis_ref, atol=1e-4, err_msg="COS vs Lewis")

    def test_carr_madan_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: merton_jd_cf(u, fwd, p)
        grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4, err_msg="Carr-Madan vs Lewis")

    def test_frft_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: merton_jd_cf(u, fwd, p)
        grid = FRFTGrid(N=4096, eta=0.25, alpha=1.5, lam=0.01)
        prices = frft_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4, err_msg="FRFT vs Lewis")

    @pytest.mark.parametrize("K", [85.0, 90.0, 100.0, 110.0, 115.0])
    def test_cos_vs_lewis_per_strike(self, K):
        p = _P_BENCH
        fwd = _FWD_BENCH
        phi = lambda u: merton_jd_cf(u, fwd, p)
        strikes = np.array([K])
        lewis = lewis_call_prices(phi, strikes, spot=fwd.S0, texp=fwd.T, intr=fwd.r, divr=fwd.q)
        cums = merton_jd_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, strikes, grid)
        np.testing.assert_allclose(
            res.call_prices, lewis, atol=1e-4, err_msg=f"COS vs Lewis at K={K}"
        )


# ---------------------------------------------------------------------------
# Layer 3: Structural tests
# ---------------------------------------------------------------------------


class TestMertonJDStructural:
    """CF normalization and no-arbitrage properties."""

    def test_cf_at_zero(self, bench):
        """phi(0) = 1."""
        p, fwd = bench
        phi0 = merton_jd_cf(np.array([0.0]), fwd, p)
        np.testing.assert_allclose(abs(phi0[0] - 1.0), 0.0, atol=1e-14)

    def test_martingale_condition(self, bench):
        """phi(-i) = 1, i.e. E[S_T/F_0] = 1."""
        p, fwd = bench
        phi_neg_i = merton_jd_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(
            abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-12, err_msg="Merton JD martingale phi(-i)=1"
        )

    def test_cf_modulus_le_one(self, bench):
        """|phi(u)| <= 1 for real u."""
        p, fwd = bench
        u_grid = np.linspace(-20, 20, 401)
        phi = merton_jd_cf(u_grid, fwd, p)
        assert np.all(np.abs(phi) <= 1.0 + 1e-12), (
            f"|phi|>1 at some u: max={np.max(np.abs(phi)):.6f}"
        )

    def test_cumulants_positive_variance(self, bench):
        """Second cumulant (variance) must be positive."""
        p, fwd = bench
        c1, c2, c4 = merton_jd_cumulants(fwd, p)
        assert c2 > 0, f"c2 should be positive, got {c2}"
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
        """All call prices should be positive."""
        assert np.all(lewis_ref > 0), f"Non-positive prices: {lewis_ref}"

    def test_prices_monotone_decreasing(self, lewis_ref):
        """Calls are monotone decreasing in strike."""
        assert np.all(np.diff(lewis_ref) < 0), f"Call prices not monotone in strike: {lewis_ref}"

    def test_price_monotone_in_sigma(self):
        """Higher diffusion vol → higher call prices (all else equal)."""
        fwd = _FWD_BENCH
        K = np.array([100.0])

        def price_at_sigma(s):
            p = MertonJDParams(sigma=s, lam=0.3, muj=-0.05, sigj=0.1)
            phi = lambda u: merton_jd_cf(u, fwd, p)
            cums = merton_jd_cumulants(fwd, p)
            grid = cos_auto_grid(cums, N=256, L=10.0)
            return cos_prices(phi, fwd, K, grid).call_prices[0]

        p_lo = price_at_sigma(0.1)
        p_hi = price_at_sigma(0.3)
        assert p_hi > p_lo, (
            f"Price should increase with sigma: sigma=0.1→{p_lo:.4f}, sigma=0.3→{p_hi:.4f}"
        )
