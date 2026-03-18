"""Tests for the Rough Heston model (El Euch & Rosenbaum 2019).

Layer 1  -  paper / PyFENG benchmark
    Prices from our CF via Lewis integral must match PyFENG's native
    RoughHestonFft.price() to 1e-4.  The docstring benchmark parameters
    (sigma=0.0392, vov=0.1, mr=0.3156, rho=-0.681, theta=0.3156, alpha=0.62,
    spot=100, T=1) produce prices ≈ [40.407, 31.356, 11.473, 1.888] for
    strikes [60, 70, 100, 140].

Layer 2  -  cross-engine
    COS, FRFT, Carr-Madan all agree with the Lewis reference to 1e-3.
    Note: Carr-Madan and FRFT use eta=0.1 (not the default 0.25) to keep
    the max frequency within the rough Heston CF's stability range.

Layer 3  -  structural (fast)
    CF normalization: phi(0)=1, martingale: phi(-i)=1, |phi(u)| <= 1.
    Positive variance cumulant. No-arbitrage bounds.

Performance note
----------------
The rough Heston CF solves a fractional Riccati ODE numerically (Adam's
method), making each evaluation ~O(N_steps) expensive. All pricing tests
are marked ``@pytest.mark.slow`` and excluded from the default fast suite
(``-m "not slow"``). Only CF structural tests (which call the CF at a
handful of points) run in the fast suite.

Import note
-----------
We import RoughHestonFft directly from ``pyfeng.sv_fft`` to avoid the broken
``pyfeng.ex`` path (``scipy.misc.derivative`` removed in newer SciPy).
"""

from __future__ import annotations

import numpy as np
import pytest

pf_sv_fft = pytest.importorskip(
    "pyfeng.sv_fft",
    reason="pyfeng.sv_fft not available  -  rough Heston tests require it",
)
RoughHestonFft = pf_sv_fft.RoughHestonFft

from foureng.models.base import ForwardSpec
from foureng.models.rough_heston import (
    RoughHestonParams,
    rough_heston_cf,
    rough_heston_cumulants,
)
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.pricers.frft import frft_price_at_strikes
from foureng.pricers.lewis import lewis_call_prices
from foureng.utils.grids import FFTGrid, FRFTGrid

pytestmark = [pytest.mark.paper, pytest.mark.adapter]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STRIKES = np.array([60.0, 70.0, 100.0, 140.0])
SPOT = 100.0
TEXP = 1.0

# El Euch-Rosenbaum benchmark parameters (PyFENG docstring, example 1)
_P_BENCH = RoughHestonParams(
    sigma=0.0392,
    vov=0.1,
    mr=0.3156,
    rho=-0.681,
    theta=0.3156,
    alpha=0.62,
)
_FWD_BENCH = ForwardSpec(S0=SPOT, r=0.0, q=0.0, T=TEXP)


@pytest.fixture(scope="module")
def bench():
    return _P_BENCH, _FWD_BENCH


@pytest.fixture(scope="module")
def pyfeng_ref_prices():
    """PyFENG native prices  -  authoritative reference."""
    m = RoughHestonFft(sigma=0.0392, vov=0.1, mr=0.3156, rho=-0.681, theta=0.3156, alpha=0.62)
    return m.price(STRIKES, spot=SPOT, texp=TEXP, cp=1)


@pytest.fixture(scope="module")
def lewis_ref(bench):
    p, fwd = bench
    phi = lambda u: rough_heston_cf(u, fwd, p)
    return lewis_call_prices(phi, STRIKES, spot=fwd.S0, texp=fwd.T, intr=fwd.r, divr=fwd.q)


# ---------------------------------------------------------------------------
# Layer 1: Paper / PyFENG benchmark
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRoughHestonPaperBenchmark:
    """Lewis integral must match PyFENG's RoughHestonFft.price() to 1e-4."""

    def test_lewis_matches_pyfeng(self, lewis_ref, pyfeng_ref_prices):
        np.testing.assert_allclose(
            lewis_ref,
            pyfeng_ref_prices,
            atol=1e-4,
            err_msg="Lewis vs PyFENG RoughHestonFft direct prices",
        )

    def test_prices_positive(self, lewis_ref):
        assert np.all(lewis_ref > 0), "All prices should be positive"

    def test_prices_monotone_decreasing(self, lewis_ref):
        """Calls are monotone decreasing in strike."""
        assert np.all(np.diff(lewis_ref) < 0), f"Call prices not monotone in strike: {lewis_ref}"


# ---------------------------------------------------------------------------
# Layer 2: Cross-engine agreement
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRoughHestonCrossEngine:
    """All pricers using our CF must agree with PyFENG to 1e-3."""

    def test_cos_vs_pyfeng(self, bench, pyfeng_ref_prices):
        p, fwd = bench
        phi = lambda u: rough_heston_cf(u, fwd, p)
        cums = rough_heston_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, STRIKES, grid)
        np.testing.assert_allclose(
            res.call_prices, pyfeng_ref_prices, atol=1e-3, err_msg="COS vs PyFENG"
        )

    def test_carr_madan_vs_pyfeng(self, bench, pyfeng_ref_prices):
        # The rough Heston CF is evaluated at u - i*(alpha+1) in Carr-Madan.
        # At large u with the standard eta=0.25 grid, the CF overflows.
        # Use eta=0.1 to keep u_max ≈ 200, where the CF is stable.
        p, fwd = bench
        phi = lambda u: rough_heston_cf(u, fwd, p)
        grid = FFTGrid(N=4096, eta=0.1, alpha=1.5)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prices = carr_madan_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(
            prices, pyfeng_ref_prices, atol=1e-3, err_msg="Carr-Madan (eta=0.1) vs PyFENG"
        )

    def test_frft_vs_pyfeng(self, bench, pyfeng_ref_prices):
        # FRFT also evaluates CF at large u, causing overflow at standard
        # grid settings. Use eta=0.1 to keep u_max within CF stability range.
        p, fwd = bench
        phi = lambda u: rough_heston_cf(u, fwd, p)
        grid = FRFTGrid(N=4096, eta=0.1, alpha=1.5, lam=0.01)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prices = frft_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(
            prices, pyfeng_ref_prices, atol=1e-3, err_msg="FRFT (eta=0.1) vs PyFENG"
        )

    @pytest.mark.parametrize("K", [85.0, 100.0, 115.0])
    def test_cos_vs_lewis_atm_region(self, K):
        """COS and Lewis agree to 1e-3 near the money.

        Restricted to 3 strikes (cf. 5 for other models) because each CF
        evaluation involves solving the fractional Riccati ODE  -  slow.
        """
        p = _P_BENCH
        fwd = _FWD_BENCH
        phi = lambda u: rough_heston_cf(u, fwd, p)
        strikes = np.array([K])
        lewis = lewis_call_prices(phi, strikes, spot=fwd.S0, texp=fwd.T, intr=fwd.r, divr=fwd.q)
        cums = rough_heston_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        res = cos_prices(phi, fwd, strikes, grid)
        np.testing.assert_allclose(
            res.call_prices, lewis, atol=1e-3, err_msg=f"COS vs Lewis at K={K}"
        )


# ---------------------------------------------------------------------------
# Layer 3: Structural tests
# ---------------------------------------------------------------------------


class TestRoughHestonStructural:
    """CF normalization and no-arbitrage properties."""

    def test_cf_at_zero(self, bench):
        """phi(0) = 1."""
        p, fwd = bench
        phi0 = rough_heston_cf(np.array([0.0]), fwd, p)
        np.testing.assert_allclose(abs(phi0[0] - 1.0), 0.0, atol=1e-14)

    def test_martingale_condition(self, bench):
        """phi(-i) = 1 (E[S_T / F_0] = 1)."""
        p, fwd = bench
        phi_neg_i = rough_heston_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(
            abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-8, err_msg="rough Heston martingale phi(-i)=1"
        )

    def test_cf_modulus_le_one(self, bench):
        """|phi(u)| <= 1 for real u."""
        p, fwd = bench
        u_grid = np.linspace(-10, 10, 201)
        phi = rough_heston_cf(u_grid, fwd, p)
        assert np.all(np.abs(phi) <= 1.0 + 1e-10), (
            f"|phi|>1 at some u: max={np.max(np.abs(phi)):.6f}"
        )

    def test_cumulants_positive_variance(self, bench):
        """Second cumulant (variance) must be positive."""
        p, fwd = bench
        c1, c2, c4 = rough_heston_cumulants(fwd, p)
        assert c2 > 0, f"c2 should be positive, got {c2}"
        assert np.isfinite(c1) and np.isfinite(c4)

    @pytest.mark.slow
    def test_call_ge_intrinsic(self, bench, lewis_ref):
        """Call prices >= max(F-K, 0)."""
        p, fwd = bench
        intrinsic = np.maximum(fwd.F0 - STRIKES, 0.0)
        assert np.all(lewis_ref >= intrinsic - 1e-6), "Call price below intrinsic value"

    @pytest.mark.slow
    def test_call_le_spot(self, bench, lewis_ref):
        """Call prices <= spot."""
        p, fwd = bench
        assert np.all(lewis_ref <= fwd.S0 + 1e-8), "Call price exceeds spot"

    @pytest.mark.slow
    def test_rough_heston_vs_standard_heston_short_mat(self):
        """At short maturities, rough Heston ATM skew differs from Heston.

        Rough Heston (H < 0.5) produces a steeper ATM skew than standard
        Heston at short maturities  -  the key empirical motivation. We verify
        that the price surfaces differ (not a limit test, just a smoke test).
        """
        pytest.importorskip("pyfeng", reason="pyfeng not installed")

        T = 0.25
        K_atm = np.array([100.0])
        K_otm = np.array([105.0])

        # Standard Heston (using Heston CF)
        from foureng.models.heston import HestonParams, heston_cf

        p_heston = HestonParams(kappa=0.3156, theta=0.3156, nu=0.1, rho=-0.681, v0=0.0392)
        fwd_h = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=T)
        from foureng.models.heston import heston_cumulants
        from foureng.pricers.cos import cos_auto_grid, cos_prices

        phi_h = lambda u: heston_cf(u, fwd_h, p_heston)
        cums_h = heston_cumulants(fwd_h, p_heston)
        grid_h = cos_auto_grid(cums_h, N=256, L=10.0)
        p_atm_h = cos_prices(phi_h, fwd_h, K_atm, grid_h).call_prices[0]
        p_otm_h = cos_prices(phi_h, fwd_h, K_otm, grid_h).call_prices[0]
        skew_heston = p_atm_h - p_otm_h  # rough proxy for skew

        # Rough Heston
        p_rh = RoughHestonParams(
            sigma=0.0392, vov=0.1, mr=0.3156, rho=-0.681, theta=0.3156, alpha=0.62
        )
        fwd_rh = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=T)
        phi_rh = lambda u: rough_heston_cf(u, fwd_rh, p_rh)
        cums_rh = rough_heston_cumulants(fwd_rh, p_rh)
        grid_rh = cos_auto_grid(cums_rh, N=256, L=10.0)
        p_atm_rh = cos_prices(phi_rh, fwd_rh, K_atm, grid_rh).call_prices[0]
        p_otm_rh = cos_prices(phi_rh, fwd_rh, K_otm, grid_rh).call_prices[0]
        skew_rh = p_atm_rh - p_otm_rh

        # The two models should give different skews
        assert abs(skew_rh - skew_heston) > 1e-4, (
            f"Rough Heston and standard Heston skews are identical: {skew_rh:.4f} vs {skew_heston:.4f}"
        )
