"""Tests for the GARCH diffusion model (Wu, Ma & Wang 2012).

Layer 1  -  paper formula
    Cross-checks our native MGF implementation against PyFENG's
    ``mgf_logprice_old`` (the slower but NumPy-compatible formula in
    ``GarchFftWuMaWang2012``).  The two formulas implement the same
    Lemma 2.1 and must agree to < 1e-12.

Layer 2  -  cross-engine
    All three Fourier pricers (Lewis, COS, Carr-Madan) must agree
    to 1e-4 (1 bp) against each other.

Layer 3  -  structural / limit
    CF normalization (phi(0)=1), martingale condition (phi(-i)=1),
    modulus bound (|phi(u)| <= 1), positive variance cumulant,
    no-arbitrage price bounds.

PyFENG note
-----------
``pyfeng.GarchFftWuMaWang2012.charfunc_logprice`` and ``.price`` are broken
under NumPy >= 1.25 (``UFuncTypeError`` in ``util.avg_exp``).  We use
``mgf_logprice_old`` as a cross-reference for the MGF values (it avoids
``avg_exp``), and do NOT call the broken methods.

Benchmark prices
----------------
Using sigma=0.06 (v0), mr=20.48 (kappa), theta=0.218, vov=3.20 (nu),
rho=-0.99, spot=100, T=0.5, r=q=0:

    strikes = [95, 100, 105]
    GARCH prices ≈ [14.7732, 12.2470, 10.0435]

Note: PyFENG's docstring for GarchFftWuMaWang2012 shows [11.7235, 8.9978,
6.7091]  -  the same values as Sv32Fft with identical parameters. This is a
copy-paste error in PyFENG; those are the Sv32 prices. The GARCH diffusion
model (nu*v diffusion) gives different prices from the 3/2 SV model
(nu*v^(3/2) diffusion).
"""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.garch_wmw2012 import (
    GarchWMW2012Params,
    garch_wmw2012_cf,
    garch_wmw2012_cumulants,
    _garch_mgf,
)
from foureng.models.base import ForwardSpec
from foureng.pricers.lewis import lewis_call_prices
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.utils.grids import FFTGrid


pytestmark = [pytest.mark.paper, pytest.mark.derived_reference]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STRIKES = np.array([95.0, 100.0, 105.0])
SPOT = 100.0
TEXP = 0.5

# Benchmark parameters (Lewis / PyFENG Sv32 docstring parameters)
_P_BENCH = GarchWMW2012Params(v0=0.06, kappa=20.48, theta=0.218, nu=3.20, rho=-0.99)
_FWD_BENCH = ForwardSpec(S0=SPOT, r=0.0, q=0.0, T=TEXP)

# Computed benchmark prices from our native implementation (Lewis integral,
# 200*4096-point grid, verified to match mgf_logprice_old to 1e-10)
_REF_PRICES = np.array([14.7732, 12.2470, 10.0435])


@pytest.fixture(scope="module")
def bench():
    return _P_BENCH, _FWD_BENCH


@pytest.fixture(scope="module")
def lewis_ref(bench):
    p, fwd = bench
    phi = lambda u: garch_wmw2012_cf(u, fwd, p)
    return lewis_call_prices(phi, STRIKES, spot=fwd.S0, texp=fwd.T,
                             intr=fwd.r, divr=fwd.q)


# ---------------------------------------------------------------------------
# Layer 1: Cross-check native MGF vs PyFENG's mgf_logprice_old
# ---------------------------------------------------------------------------

class TestGarchMGFFormula:
    """Our native MGF must agree with PyFENG's mgf_logprice_old to 1e-12."""

    @pytest.fixture(autouse=True)
    def _pyfeng_old_model(self):
        pf = pytest.importorskip("pyfeng",
            reason="pyfeng not installed  -  GARCH MGF cross-check requires it")
        if not hasattr(pf, "GarchFftWuMaWang2012"):
            pytest.skip("pyfeng.GarchFftWuMaWang2012 not available in this pyfeng version")
        self._pf_model = pf.GarchFftWuMaWang2012(
            sigma=0.06, vov=3.20, mr=20.48, rho=-0.99, theta=0.218,
        )
        if not hasattr(self._pf_model, "mgf_logprice_old"):
            pytest.skip("pyfeng.GarchFftWuMaWang2012.mgf_logprice_old removed in pyfeng>=0.4.0")

    @pytest.mark.parametrize("uu_real", [0.0, 0.25, 0.5, 1.0, 2.0])
    def test_mgf_real_arg_matches_pyfeng_old(self, uu_real):
        """MGF at real uu must match mgf_logprice_old."""
        p = _P_BENCH
        uu = np.array([uu_real + 0.0j])
        native = _garch_mgf(uu, p, TEXP)[0]
        pyfeng = self._pf_model.mgf_logprice_old(uu, TEXP)[0]
        np.testing.assert_allclose(
            native.real, pyfeng.real, rtol=1e-12,
            err_msg=f"MGF real part differs at uu={uu_real}",
        )

    @pytest.mark.parametrize("uu_re,uu_im", [
        (0.5, 0.0), (0.5, 1.0), (0.5, 2.0), (0.5, 5.0), (1.0, 3.0),
    ])
    def test_mgf_complex_arg_matches_pyfeng_old(self, uu_re, uu_im):
        """MGF at complex uu must match mgf_logprice_old (which handles complex)."""
        p = _P_BENCH
        uu = np.array([uu_re + 1j * uu_im])
        native = _garch_mgf(uu, p, TEXP)[0]
        pyfeng = self._pf_model.mgf_logprice_old(uu, TEXP)[0]
        np.testing.assert_allclose(
            abs(native), abs(pyfeng), rtol=1e-12,
            err_msg=f"MGF modulus differs at uu={uu_re}+{uu_im}j",
        )
        np.testing.assert_allclose(
            native.real, pyfeng.real, rtol=1e-12,
            err_msg=f"MGF real part differs at uu={uu_re}+{uu_im}j",
        )

    def test_mgf_martingale_via_pyfeng(self):
        """Both implementations must satisfy MGF(1)=1."""
        p = _P_BENCH
        native_at_1 = _garch_mgf(np.array([1.0]), p, TEXP)[0]
        pyfeng_at_1 = self._pf_model.mgf_logprice_old(np.array([1.0]), TEXP)[0]
        np.testing.assert_allclose(native_at_1.real, 1.0, atol=1e-12)
        np.testing.assert_allclose(pyfeng_at_1.real, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Layer 2: Cross-engine agreement
# ---------------------------------------------------------------------------

class TestGarchCrossEngine:
    """All Fourier pricers using our CF must agree to 1e-4."""

    def test_lewis_matches_ref(self, lewis_ref):
        np.testing.assert_allclose(lewis_ref, _REF_PRICES, atol=1e-3,
                                   err_msg="Lewis vs benchmark prices")

    def test_cos_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: garch_wmw2012_cf(u, fwd, p)
        cums = garch_wmw2012_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=256, L=10.0)
        res = cos_prices(phi, fwd, STRIKES, grid)
        np.testing.assert_allclose(res.call_prices, lewis_ref, atol=1e-4,
                                   err_msg="COS vs Lewis")

    def test_carr_madan_vs_lewis(self, bench, lewis_ref):
        p, fwd = bench
        phi = lambda u: garch_wmw2012_cf(u, fwd, p)
        grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, STRIKES)
        np.testing.assert_allclose(prices, lewis_ref, atol=1e-4,
                                   err_msg="Carr-Madan vs Lewis")

    @pytest.mark.parametrize("K", [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
    def test_cos_vs_lewis_wider_strikes(self, K):
        """COS and Lewis agree to 1e-3 across a wider strike range."""
        p = _P_BENCH
        fwd = _FWD_BENCH
        phi = lambda u: garch_wmw2012_cf(u, fwd, p)
        strikes = np.array([K])
        lewis = lewis_call_prices(phi, strikes, spot=fwd.S0, texp=fwd.T,
                                  intr=fwd.r, divr=fwd.q)
        cums = garch_wmw2012_cumulants(fwd, p)
        grid = cos_auto_grid(cums, N=512, L=12.0)
        cos_res = cos_prices(phi, fwd, strikes, grid)
        np.testing.assert_allclose(cos_res.call_prices, lewis, atol=1e-3,
                                   err_msg=f"COS vs Lewis at K={K}")


# ---------------------------------------------------------------------------
# Layer 3: Structural tests
# ---------------------------------------------------------------------------

class TestGarchStructural:
    """CF normalization and no-arbitrage bounds."""

    def test_cf_at_zero(self, bench):
        """phi(0) = 1."""
        p, fwd = bench
        phi0 = garch_wmw2012_cf(np.array([0.0]), fwd, p)
        np.testing.assert_allclose(abs(phi0[0] - 1.0), 0.0, atol=1e-14)

    def test_martingale_condition(self, bench):
        """phi(-i) = 1 (E[S_T/F_0] = 1)."""
        p, fwd = bench
        phi_neg_i = garch_wmw2012_cf(np.array([-1j]), fwd, p)
        np.testing.assert_allclose(abs(phi_neg_i[0] - 1.0), 0.0, atol=1e-10,
                                   err_msg="GARCH martingale phi(-i)=1")

    def test_cf_modulus_le_one(self, bench):
        """|phi(u)| <= 1 for real u."""
        p, fwd = bench
        u = np.linspace(-20, 20, 401)
        phi = garch_wmw2012_cf(u, fwd, p)
        assert np.all(np.abs(phi) <= 1.0 + 1e-12), \
            f"|phi|>1 at some u: max={np.max(np.abs(phi)):.6f}"

    def test_cumulants_positive_variance(self, bench):
        p, fwd = bench
        c1, c2, c4 = garch_wmw2012_cumulants(fwd, p)
        assert c2 > 0, f"c2 should be positive, got {c2}"
        assert np.isfinite(c1) and np.isfinite(c4)

    def test_call_ge_intrinsic(self, bench, lewis_ref):
        """Call prices must be above intrinsic."""
        p, fwd = bench
        intrinsic = np.maximum(fwd.F0 - STRIKES, 0.0)
        assert np.all(lewis_ref >= intrinsic - 1e-6), \
            "Call prices below intrinsic"

    def test_call_le_spot(self, bench, lewis_ref):
        """Call prices must not exceed spot."""
        p, fwd = bench
        assert np.all(lewis_ref <= fwd.S0 + 1e-8), "Call price exceeds spot"

    def test_price_monotone_in_v0(self):
        """ATM call price must increase with initial variance v0."""
        fwd = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=0.5)
        K_atm = np.array([100.0])
        prices = []
        for v0 in [0.02, 0.06, 0.12, 0.20]:
            p = GarchWMW2012Params(v0=v0, kappa=20.48, theta=0.218, nu=3.20, rho=-0.99)
            phi = lambda u, _p=p: garch_wmw2012_cf(u, fwd, _p)
            cums = garch_wmw2012_cumulants(fwd, p)
            grid = cos_auto_grid(cums, N=256, L=10.0)
            res = cos_prices(phi, fwd, K_atm, grid)
            prices.append(float(res.call_prices[0]))
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1], \
                f"ATM call not increasing in v0: {prices}"

    def test_garch_vs_sv32_prices_differ(self):
        """GARCH diffusion and 3/2 SV must give different prices with same params.

        The PyFENG docstring incorrectly shows identical prices for both models
        (copy-paste). This test verifies they are indeed distinct.
        """
        try:
            import pyfeng as pf
        except ImportError:
            pytest.skip("pyfeng not installed")

        p_garch = GarchWMW2012Params(v0=0.06, kappa=20.48, theta=0.218, nu=3.20, rho=-0.99)
        fwd = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=0.5)
        phi_garch = lambda u: garch_wmw2012_cf(u, fwd, p_garch)
        cums = garch_wmw2012_cumulants(fwd, p_garch)
        grid = cos_auto_grid(cums, N=256, L=10.0)
        garch_prices = cos_prices(phi_garch, fwd, STRIKES, grid).call_prices

        m_sv32 = pf.Sv32Fft(sigma=0.06, vov=3.20, mr=20.48, rho=-0.99, theta=0.218)
        sv32_prices = m_sv32.price(STRIKES, spot=100.0, texp=0.5, cp=1)

        max_diff = np.max(np.abs(garch_prices - sv32_prices))
        assert max_diff > 0.5, (
            f"GARCH and Sv32 prices unexpectedly close: max_diff={max_diff:.4f}. "
            "They should differ by ~3 points for these parameters."
        )
