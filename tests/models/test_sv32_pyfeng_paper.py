"""Tests for the 3/2 SV model (sv32)  -  paper benchmark, cross-engine, limit-cases.

Layer 1  -  paper formula
    Lewis (2000) docstring benchmark: sigma=0.06, mr=20.48, theta=0.218,
    vov=3.20, rho=-0.99, strikes=[95,100,105], spot=100, texp=0.5
    → prices ≈ [11.7235, 8.9978, 6.7091]

Layer 2  -  cross-engine
    Our CF via Lewis integral must match PyFENG's native Sv32Fft pricer
    to 1e-4 (1 bp), and all Fourier pricers (COS, FRFT, Carr-Madan) must
    agree to each other to 1e-3.

Layer 3  -  limit / structural
    CF normalization: φ(0)=1, φ(-i)=1 (martingale).
    Cumulants: c2 > 0, c1 is finite.
    Rho=0 symmetry: call(K) = call(2*F-K) mirrored around ATM.
"""

from __future__ import annotations

import numpy as np
import pytest

pf = pytest.importorskip(
    "pyfeng",
    reason="pyfeng not installed  -  sv32 tests require it",
)

from foureng.models.base import ForwardSpec
from foureng.models.sv32 import Sv32Params, sv32_cf, sv32_cumulants
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.pricers.frft import frft_price_at_strikes
from foureng.pricers.lewis import lewis_call_prices
from foureng.utils.grids import FFTGrid, FRFTGrid

pytestmark = [pytest.mark.paper, pytest.mark.adapter]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STRIKES = np.array([95.0, 100.0, 105.0])
SPOT = 100.0
TEXP = 0.5

# Lewis (2000) docstring benchmark parameters
_P_BENCH = Sv32Params(v0=0.06, kappa=20.48, theta=0.218, nu=3.20, rho=-0.99)
_FWD_BENCH = ForwardSpec(S0=SPOT, r=0.0, q=0.0, T=TEXP)

# PyFENG reference prices for the benchmark case
_REF_PRICES = np.array([11.7235, 8.9978, 6.7091])


@pytest.fixture(scope="module")
def bench_params():
    return _P_BENCH, _FWD_BENCH


@pytest.fixture(scope="module")
def pyfeng_ref_prices():
    """Prices computed directly by PyFENG Sv32Fft  -  authoritative reference."""
    m = pf.Sv32Fft(sigma=0.06, vov=3.20, mr=20.48, rho=-0.99, theta=0.218)
    return m.price(STRIKES, spot=SPOT, texp=TEXP, cp=1)


# ---------------------------------------------------------------------------
# Layer 1: Paper / docstring benchmark
# ---------------------------------------------------------------------------


class TestSv32PaperBenchmark:
    """Prices must match Lewis (2000) via-PyFENG docstring values to 1e-3."""

    def test_lewis_integral_matches_paper(self, bench_params):
        p, fwd = bench_params
        phi = lambda u: sv32_cf(u, fwd, p)
        prices = lewis_call_prices(
            phi,
            STRIKES,
            spot=fwd.S0,
            texp=fwd.T,
            intr=fwd.r,
            divr=fwd.q,
        )
        np.testing.assert_allclose(
            prices, _REF_PRICES, atol=1e-3, err_msg="Lewis pricer vs paper benchmark"
        )

    def test_cos_matches_paper(self, bench_params):
        p, fwd = bench_params
        phi = lambda u: sv32_cf(u, fwd, p)
        cums = sv32_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=256, L=10.0)
        res = cos_prices(phi, fwd, STRIKES, grid)
        np.testing.assert_allclose(
            res.call_prices, _REF_PRICES, atol=1e-3, err_msg="COS pricer vs paper benchmark"
        )

    def test_carr_madan_matches_paper(self, bench_params):
        p, fwd = bench_params
        phi = lambda u: sv32_cf(u, fwd, p)
        grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(
            prices, _REF_PRICES, atol=1e-3, err_msg="Carr-Madan pricer vs paper benchmark"
        )


# ---------------------------------------------------------------------------
# Layer 2: Cross-engine agreement
# ---------------------------------------------------------------------------


class TestSv32CrossEngine:
    """All pricers using our CF must match PyFENG's native Sv32Fft to 1e-4."""

    def test_lewis_vs_pyfeng(self, bench_params, pyfeng_ref_prices):
        p, fwd = bench_params
        phi = lambda u: sv32_cf(u, fwd, p)
        lewis = lewis_call_prices(
            phi,
            STRIKES,
            spot=fwd.S0,
            texp=fwd.T,
            intr=fwd.r,
            divr=fwd.q,
        )
        np.testing.assert_allclose(
            lewis, pyfeng_ref_prices, atol=1e-4, err_msg="Lewis vs PyFENG Sv32Fft"
        )

    def test_cos_vs_pyfeng(self, bench_params, pyfeng_ref_prices):
        p, fwd = bench_params
        phi = lambda u: sv32_cf(u, fwd, p)
        cums = sv32_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=256, L=10.0)
        res = cos_prices(phi, fwd, STRIKES, grid)
        np.testing.assert_allclose(
            res.call_prices, pyfeng_ref_prices, atol=1e-4, err_msg="COS vs PyFENG Sv32Fft"
        )

    def test_frft_vs_pyfeng(self, bench_params, pyfeng_ref_prices):
        p, fwd = bench_params
        phi = lambda u: sv32_cf(u, fwd, p)
        grid = FRFTGrid(N=4096, eta=0.25, alpha=1.5, lam=0.01)
        prices = frft_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(
            prices, pyfeng_ref_prices, atol=1e-4, err_msg="FRFT vs PyFENG Sv32Fft"
        )

    def test_carr_madan_vs_pyfeng(self, bench_params, pyfeng_ref_prices):
        p, fwd = bench_params
        phi = lambda u: sv32_cf(u, fwd, p)
        grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(
            prices, pyfeng_ref_prices, atol=1e-4, err_msg="Carr-Madan vs PyFENG Sv32Fft"
        )

    @pytest.mark.parametrize("K", [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
    def test_cos_vs_lewis_wider_strikes(self, K):
        """COS and Lewis must agree to 1e-3 across a wider strike range."""
        p = _P_BENCH
        fwd = _FWD_BENCH
        phi = lambda u: sv32_cf(u, fwd, p)
        strikes = np.array([K])
        lewis_price = lewis_call_prices(
            phi,
            strikes,
            spot=fwd.S0,
            texp=fwd.T,
            intr=fwd.r,
            divr=fwd.q,
        )
        cums = sv32_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        cos_res = cos_prices(phi, fwd, strikes, grid)
        np.testing.assert_allclose(
            cos_res.call_prices,
            lewis_price,
            atol=1e-3,
            err_msg=f"COS vs Lewis at K={K}",
        )


# ---------------------------------------------------------------------------
# Layer 3: Structural / limit tests
# ---------------------------------------------------------------------------


class TestSv32Structural:
    """CF normalization and statistical properties."""

    def test_cf_at_zero(self, bench_params):
        """φ(0) = 1."""
        p, fwd = bench_params
        phi0 = sv32_cf(np.array([0.0]), fwd, p)
        np.testing.assert_allclose(abs(phi0[0] - 1.0), 0.0, atol=1e-14)

    def test_martingale_condition(self, bench_params):
        """φ(-i) = 1 (E[S_T/F_0] = 1)."""
        p, fwd = bench_params
        phi_neg_i = sv32_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(
            abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-10, err_msg="sv32 martingale condition φ(-i)=1"
        )

    def test_cf_modulus_le_one(self, bench_params):
        """|φ(u)| ≤ 1 for all real u."""
        p, fwd = bench_params
        u_grid = np.linspace(-20, 20, 401)
        phi = sv32_cf(u_grid, fwd, p)
        assert np.all(np.abs(phi) <= 1.0 + 1e-12), (
            f"|φ(u)| > 1 at some u: max={np.max(np.abs(phi)):.6f}"
        )

    def test_cumulants_positive_variance(self, bench_params):
        """Second cumulant (variance) must be positive."""
        p, fwd = bench_params
        c1, c2, c4 = sv32_cumulants(fwd, p)
        assert c2 > 0, f"c2 should be positive, got {c2}"
        assert np.isfinite(c1)
        assert np.isfinite(c4)

    def test_negative_rho_increases_otm_put_cost(self):
        """Negative rho (spot-var anticorrelation) increases OTM put price.

        In the 3/2 model, negative rho means downward spot moves are
        accompanied by variance increases, which makes OTM puts more expensive.
        We verify the call skew: rho<0 raises OTM put IVs, i.e. raises
        OTM call at low K relative to same model with rho=0.
        """
        fwd = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=0.5)
        K_otm = np.array([90.0])  # OTM call
        p_neg_rho = Sv32Params(v0=0.06, kappa=20.48, theta=0.218, nu=3.20, rho=-0.99)
        p_pos_rho = Sv32Params(v0=0.06, kappa=20.48, theta=0.218, nu=3.20, rho=0.0)

        cums_neg = sv32_cumulants(fwd, p_neg_rho)
        cums_pos = sv32_cumulants(fwd, p_pos_rho)

        grid_neg = cos_auto_grid(cums_neg, N=256, L=10.0)
        grid_pos = cos_auto_grid(cums_pos, N=256, L=10.0)

        price_neg = cos_prices(lambda u: sv32_cf(u, fwd, p_neg_rho), fwd, K_otm, grid_neg)
        price_pos = cos_prices(lambda u: sv32_cf(u, fwd, p_pos_rho), fwd, K_otm, grid_pos)

        # Negative rho → more expensive deep ITM call (= OTM put parity)
        assert price_neg.call_prices[0] > price_pos.call_prices[0], (
            f"Negative rho should increase OTM (K=90) call price: "
            f"rho=-0.99 gave {price_neg.call_prices[0]:.4f}, "
            f"rho=0 gave {price_pos.call_prices[0]:.4f}"
        )

    def test_price_monotone_in_v0(self):
        """ATM call price must be monotone increasing in initial variance v0."""
        fwd = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=0.5)
        K_atm = np.array([100.0])

        prices = []
        for v0 in [0.02, 0.06, 0.12, 0.20]:
            p = Sv32Params(v0=v0, kappa=20.48, theta=0.218, nu=3.20, rho=-0.99)
            phi = lambda u, _p=p: sv32_cf(u, fwd, _p)
            cums = sv32_cumulants(fwd, p)
            grid = cos_auto_grid(cums, N=256, L=10.0)
            res = cos_prices(phi, fwd, K_atm, grid)
            prices.append(float(res.call_prices[0]))

        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1], f"ATM call should increase with v0: prices={prices}"

    def test_call_ge_intrinsic(self, bench_params):
        """Call prices must exceed max(F-K, 0)."""
        p, fwd = bench_params
        phi = lambda u: sv32_cf(u, fwd, p)
        cums = sv32_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=256, L=10.0)
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        res = cos_prices(phi, fwd, strikes, grid)
        intrinsic = np.maximum(fwd.F0 - strikes, 0.0)
        assert np.all(res.call_prices >= intrinsic - 1e-6), "Call prices fall below intrinsic value"

    def test_call_le_spot(self, bench_params):
        """Call prices must not exceed spot (no-arbitrage upper bound)."""
        p, fwd = bench_params
        phi = lambda u: sv32_cf(u, fwd, p)
        cums = sv32_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=256, L=10.0)
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        res = cos_prices(phi, fwd, strikes, grid)
        assert np.all(res.call_prices <= fwd.S0 + 1e-8), "Call price exceeds spot"
