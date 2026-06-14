"""Test COS digital pricer convergence to BSM closed form.

L5 evidence: COS cash-or-nothing and asset-or-nothing prices converge to
BSM analytic values as N increases.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams, bsm_cf
from foureng.pricers.analytic_bsm import bsm_asset_or_nothing, bsm_cash_or_nothing
from foureng.pricers.cos_digital import cos_digital_prices
from foureng.utils.grids import COSGrid

# Standard test parameters
S = 100.0
K = 100.0
r = 0.05
q = 0.0
sigma = 0.3
T = 1.0

# COS grid with L=8, N=256
GRID_N256 = COSGrid(N=256, a=-8.0, b=8.0)
GRID_N64 = COSGrid(N=64, a=-8.0, b=8.0)
GRID_N1024 = COSGrid(N=1024, a=-8.0, b=8.0)


def _make_phi(fwd: ForwardSpec, params: BsmParams):
    return lambda u: bsm_cf(u, fwd, params)


@pytest.fixture(scope="module")
def bsm_setup():
    fwd = ForwardSpec(S0=S, r=r, q=q, T=T)
    params = BsmParams(sigma=sigma)
    phi = _make_phi(fwd, params)
    return fwd, params, phi


class TestCashOrNothingCOS:
    def test_atm_call_n256(self, bsm_setup):
        """COS cash-or-nothing call matches BSM at N=256 (ATM)."""
        fwd, params, phi = bsm_setup
        cos_price = cos_digital_prices(phi, fwd, np.array([K]), GRID_N256, "cash_or_nothing", cp=1)[
            0
        ]
        bsm_price = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=1)
        assert cos_price == pytest.approx(bsm_price, rel=1e-3)

    def test_itm_call_n256(self, bsm_setup):
        """COS cash-or-nothing call matches BSM for ITM strike (K=80)."""
        fwd, params, phi = bsm_setup
        K_test = 80.0
        cos_price = cos_digital_prices(
            phi, fwd, np.array([K_test]), GRID_N256, "cash_or_nothing", cp=1
        )[0]
        bsm_price = bsm_cash_or_nothing(S, K_test, r, q, sigma, T, cp=1)
        assert cos_price == pytest.approx(bsm_price, rel=1e-3)

    def test_otm_call_n256(self, bsm_setup):
        """COS cash-or-nothing call matches BSM for OTM strike (K=120)."""
        fwd, params, phi = bsm_setup
        K_test = 120.0
        cos_price = cos_digital_prices(
            phi, fwd, np.array([K_test]), GRID_N256, "cash_or_nothing", cp=1
        )[0]
        bsm_price = bsm_cash_or_nothing(S, K_test, r, q, sigma, T, cp=1)
        assert cos_price == pytest.approx(bsm_price, rel=1e-3)

    def test_atm_put_n256(self, bsm_setup):
        """COS cash-or-nothing put matches BSM at N=256 (ATM)."""
        fwd, params, phi = bsm_setup
        cos_price = cos_digital_prices(
            phi, fwd, np.array([K]), GRID_N256, "cash_or_nothing", cp=-1
        )[0]
        bsm_price = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=-1)
        assert cos_price == pytest.approx(bsm_price, rel=1e-3)

    def test_parity_holds_cos(self, bsm_setup):
        """COS cash-or-nothing call + put sums to e^{-rT}."""
        fwd, params, phi = bsm_setup
        strikes = np.array([K])
        cos_call = cos_digital_prices(phi, fwd, strikes, GRID_N256, "cash_or_nothing", cp=1)[0]
        cos_put = cos_digital_prices(phi, fwd, strikes, GRID_N256, "cash_or_nothing", cp=-1)[0]
        assert cos_call + cos_put == pytest.approx(np.exp(-r * T), rel=1e-3)

    def test_multiple_strikes_n256(self, bsm_setup):
        """COS vectorized pricing over multiple strikes matches BSM."""
        fwd, params, phi = bsm_setup
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        cos_prices = cos_digital_prices(phi, fwd, strikes, GRID_N256, "cash_or_nothing", cp=1)
        bsm_prices = np.array(
            [bsm_cash_or_nothing(S, K_i, r, q, sigma, T, cp=1) for K_i in strikes]
        )
        np.testing.assert_allclose(cos_prices, bsm_prices, rtol=1e-3)

    def test_convergence_with_n(self, bsm_setup):
        """Error decreases as N increases (N=64 → N=256)."""
        fwd, params, phi = bsm_setup
        bsm_price = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=1)
        err64 = abs(
            cos_digital_prices(phi, fwd, np.array([K]), GRID_N64, "cash_or_nothing", cp=1)[0]
            - bsm_price
        )
        err256 = abs(
            cos_digital_prices(phi, fwd, np.array([K]), GRID_N256, "cash_or_nothing", cp=1)[0]
            - bsm_price
        )
        # N=256 should be more accurate than N=64
        assert err256 <= err64 + 1e-10  # allow tiny numerical ties


class TestAssetOrNothingCOS:
    def test_atm_call_n256(self, bsm_setup):
        """COS asset-or-nothing call matches BSM at N=256 (ATM)."""
        fwd, params, phi = bsm_setup
        cos_price = cos_digital_prices(
            phi, fwd, np.array([K]), GRID_N256, "asset_or_nothing", cp=1
        )[0]
        bsm_price = bsm_asset_or_nothing(S, K, r, q, sigma, T, cp=1)
        assert cos_price == pytest.approx(bsm_price, rel=1e-3)

    def test_itm_call_n256(self, bsm_setup):
        """COS asset-or-nothing call matches BSM for ITM strike (K=80)."""
        fwd, params, phi = bsm_setup
        K_test = 80.0
        cos_price = cos_digital_prices(
            phi, fwd, np.array([K_test]), GRID_N256, "asset_or_nothing", cp=1
        )[0]
        bsm_price = bsm_asset_or_nothing(S, K_test, r, q, sigma, T, cp=1)
        assert cos_price == pytest.approx(bsm_price, rel=1e-3)

    def test_otm_call_n256(self, bsm_setup):
        """COS asset-or-nothing call matches BSM for OTM strike (K=120)."""
        fwd, params, phi = bsm_setup
        K_test = 120.0
        cos_price = cos_digital_prices(
            phi, fwd, np.array([K_test]), GRID_N256, "asset_or_nothing", cp=1
        )[0]
        bsm_price = bsm_asset_or_nothing(S, K_test, r, q, sigma, T, cp=1)
        assert cos_price == pytest.approx(bsm_price, rel=1e-3)

    def test_atm_put_n256(self, bsm_setup):
        """COS asset-or-nothing put matches BSM at N=256 (ATM)."""
        fwd, params, phi = bsm_setup
        cos_price = cos_digital_prices(
            phi, fwd, np.array([K]), GRID_N256, "asset_or_nothing", cp=-1
        )[0]
        bsm_price = bsm_asset_or_nothing(S, K, r, q, sigma, T, cp=-1)
        assert cos_price == pytest.approx(bsm_price, rel=1e-3)

    def test_parity_holds_cos(self, bsm_setup):
        """COS asset-or-nothing call + put sums to S*e^{-qT}."""
        fwd, params, phi = bsm_setup
        strikes = np.array([K])
        cos_call = cos_digital_prices(phi, fwd, strikes, GRID_N256, "asset_or_nothing", cp=1)[0]
        cos_put = cos_digital_prices(phi, fwd, strikes, GRID_N256, "asset_or_nothing", cp=-1)[0]
        assert cos_call + cos_put == pytest.approx(S * np.exp(-q * T), rel=1e-3)

    def test_multiple_strikes_n256(self, bsm_setup):
        """COS vectorized asset-or-nothing pricing over multiple strikes matches BSM."""
        fwd, params, phi = bsm_setup
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        cos_prices = cos_digital_prices(phi, fwd, strikes, GRID_N256, "asset_or_nothing", cp=1)
        bsm_prices = np.array(
            [bsm_asset_or_nothing(S, K_i, r, q, sigma, T, cp=1) for K_i in strikes]
        )
        np.testing.assert_allclose(cos_prices, bsm_prices, rtol=1e-3)
