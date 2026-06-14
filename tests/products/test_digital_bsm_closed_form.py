"""L5 evidence: exact BSM closed forms for digital options.

Tests the cash-or-nothing and asset-or-nothing formulas in
foureng.pricers.analytic_bsm against known mathematical identities.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.pricers.analytic_bsm import (
    bsm_asset_or_nothing,
    bsm_cash_or_nothing,
    bsm_european,
)

# Standard test parameters
S = 100.0
K = 100.0
r = 0.05
q = 0.02
sigma = 0.2
T = 1.0


class TestCashOrNothing:
    def test_call_formula(self):
        """Cash-or-nothing call = exp(-rT)*N(d2)."""
        from scipy.stats import norm

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        expected = np.exp(-r * T) * norm.cdf(d2)
        result = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=1)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_put_formula(self):
        """Cash-or-nothing put = exp(-rT)*N(-d2)."""
        from scipy.stats import norm

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        expected = np.exp(-r * T) * norm.cdf(-d2)
        result = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=-1)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_call_plus_put_equals_discount(self):
        """Cash-or-nothing call + put = exp(-rT) (parity)."""
        call = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=1)
        put = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=-1)
        expected = np.exp(-r * T)
        assert call + put == pytest.approx(expected, rel=1e-10)

    def test_parity_itm_strike(self):
        """Parity holds for ITM strike."""
        K_itm = 80.0
        call = bsm_cash_or_nothing(S, K_itm, r, q, sigma, T, cp=1)
        put = bsm_cash_or_nothing(S, K_itm, r, q, sigma, T, cp=-1)
        assert call + put == pytest.approx(np.exp(-r * T), rel=1e-10)

    def test_parity_otm_strike(self):
        """Parity holds for OTM strike."""
        K_otm = 120.0
        call = bsm_cash_or_nothing(S, K_otm, r, q, sigma, T, cp=1)
        put = bsm_cash_or_nothing(S, K_otm, r, q, sigma, T, cp=-1)
        assert call + put == pytest.approx(np.exp(-r * T), rel=1e-10)

    def test_price_bounds_call(self):
        """Cash-or-nothing call price in [0, exp(-rT)]."""
        call = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=1)
        assert 0.0 <= call <= np.exp(-r * T)

    def test_price_bounds_put(self):
        """Cash-or-nothing put price in [0, exp(-rT)]."""
        put = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=-1)
        assert 0.0 <= put <= np.exp(-r * T)

    def test_call_itm_greater_than_atm(self):
        """Deep ITM call > ATM call (higher probability of S_T > K)."""
        call_atm = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=1)
        call_itm = bsm_cash_or_nothing(S, 70.0, r, q, sigma, T, cp=1)
        assert call_itm > call_atm

    def test_call_otm_less_than_atm(self):
        """Deep OTM call < ATM call."""
        call_atm = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=1)
        call_otm = bsm_cash_or_nothing(S, 140.0, r, q, sigma, T, cp=1)
        assert call_otm < call_atm

    def test_cash_amount_scaling(self):
        """Price scales linearly with cash amount."""
        cash = 5.0
        result = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=1, cash=cash)
        base = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=1, cash=1.0)
        assert result == pytest.approx(cash * base, rel=1e-10)

    def test_known_atm_value(self):
        """Frozen reference value for S=K=100, r=0.05, q=0.02, sigma=0.2, T=1."""
        result = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=1)
        # exp(-rT)*N(d2): d2 ≈ (0 + (0.03+0.02)*1)/0.2 - 0.2 = 0.25-0.2 = 0.05
        # N(0.05)*e^{-0.05} ≈ 0.5199 * 0.9512 ≈ 0.4945
        assert result == pytest.approx(0.49458, rel=1e-3)


class TestAssetOrNothing:
    def test_call_formula(self):
        """Asset-or-nothing call = S*exp(-qT)*N(d1)."""
        from scipy.stats import norm

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        expected = S * np.exp(-q * T) * norm.cdf(d1)
        result = bsm_asset_or_nothing(S, K, r, q, sigma, T, cp=1)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_put_formula(self):
        """Asset-or-nothing put = S*exp(-qT)*N(-d1)."""
        from scipy.stats import norm

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        expected = S * np.exp(-q * T) * norm.cdf(-d1)
        result = bsm_asset_or_nothing(S, K, r, q, sigma, T, cp=-1)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_call_plus_put_equals_prepaid_forward(self):
        """Asset-or-nothing call + put = S*exp(-qT)."""
        call = bsm_asset_or_nothing(S, K, r, q, sigma, T, cp=1)
        put = bsm_asset_or_nothing(S, K, r, q, sigma, T, cp=-1)
        expected = S * np.exp(-q * T)
        assert call + put == pytest.approx(expected, rel=1e-10)

    def test_parity_various_strikes(self):
        """Asset parity holds for ITM, ATM, OTM."""
        for K_test in [70.0, 100.0, 130.0]:
            call = bsm_asset_or_nothing(S, K_test, r, q, sigma, T, cp=1)
            put = bsm_asset_or_nothing(S, K_test, r, q, sigma, T, cp=-1)
            assert call + put == pytest.approx(S * np.exp(-q * T), rel=1e-10)

    def test_european_decomposition(self):
        """European call = asset_call - K * cash_call.

        Since cash_call = e^{-rT}*N(d2) and asset_call = S*e^{-qT}*N(d1):
          asset_call - K * cash_call
          = S*e^{-qT}*N(d1) - K * e^{-rT}*N(d2)
          = European call.
        Note: K * cash_call, not K*e^{-rT}*cash_call (cash_call already includes discount).
        """
        aon_call = bsm_asset_or_nothing(S, K, r, q, sigma, T, cp=1)
        con_call = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=1)
        euro_call = bsm_european(S, K, r, q, sigma, T, cp=1)
        reconstructed = aon_call - K * con_call
        assert reconstructed == pytest.approx(euro_call, rel=1e-10)

    def test_european_put_decomposition(self):
        """European put = K * cash_put - asset_put."""
        aon_put = bsm_asset_or_nothing(S, K, r, q, sigma, T, cp=-1)
        con_put = bsm_cash_or_nothing(S, K, r, q, sigma, T, cp=-1)
        euro_put = bsm_european(S, K, r, q, sigma, T, cp=-1)
        reconstructed = K * con_put - aon_put
        assert reconstructed == pytest.approx(euro_put, rel=1e-10)

    def test_price_non_negative(self):
        """Asset-or-nothing prices are non-negative."""
        for K_test in [70.0, 100.0, 130.0]:
            assert bsm_asset_or_nothing(S, K_test, r, q, sigma, T, cp=1) >= 0.0
            assert bsm_asset_or_nothing(S, K_test, r, q, sigma, T, cp=-1) >= 0.0
