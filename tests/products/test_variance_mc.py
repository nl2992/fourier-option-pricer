"""MC tests for variance swap and variance option pricers."""

from __future__ import annotations

import numpy as np
import pytest

from foureng.mc.path_engine import GBMPathSpec
from foureng.mc.variance_mc import variance_option_mc, variance_swap_mc

S0 = 100.0
r = 0.05
q = 0.0
sigma = 0.2
T = 1.0

_SPEC = GBMPathSpec(n_paths=40_000, n_steps=252, seed=42, antithetic=True)


class TestVarianceMC:
    def test_variance_swap_rate_matches_sigma_squared(self):
        """Variance swap fair rate E[RV] ≈ sigma^2 under BSM (within 2%)."""
        rv_mean = variance_swap_mc(S0, r, q, sigma, T, 252, _SPEC)
        assert abs(rv_mean - sigma**2) / sigma**2 < 0.02

    def test_variance_option_non_negative(self):
        """Variance option price is non-negative."""
        K_var = sigma**2
        price = variance_option_mc(S0, r, q, sigma, T, K_var, 252, cp=1, spec=_SPEC)
        assert price >= 0.0

    def test_variance_put_non_negative(self):
        """Variance put option (cp=-1) price is non-negative."""
        K_var = sigma**2
        price = variance_option_mc(S0, r, q, sigma, T, K_var, 252, cp=-1, spec=_SPEC)
        assert price >= 0.0

    def test_variance_option_call_deepotm_is_small(self):
        """Variance call with K_var = 5*sigma^2 is deeply OTM; price < ATM call."""
        K_var_atm = sigma**2
        K_var_otm = 5.0 * sigma**2
        spec = GBMPathSpec(n_paths=20_000, n_steps=252, seed=7, antithetic=True)
        price_atm = variance_option_mc(S0, r, q, sigma, T, K_var_atm, 252, cp=1, spec=spec)
        price_otm = variance_option_mc(S0, r, q, sigma, T, K_var_otm, 252, cp=1, spec=spec)
        assert price_otm < price_atm

    def test_variance_parity(self):
        """Variance call - variance put ≈ disc*(E[RV] - K_var) (var swap parity)."""
        spec = GBMPathSpec(n_paths=40_000, n_steps=252, seed=42, antithetic=True)
        K_var = sigma**2
        call = variance_option_mc(S0, r, q, sigma, T, K_var, 252, cp=1, spec=spec)
        put = variance_option_mc(S0, r, q, sigma, T, K_var, 252, cp=-1, spec=spec)
        disc = np.exp(-r * T)
        rv_mean = variance_swap_mc(S0, r, q, sigma, T, 252, spec)
        parity_rhs = disc * (rv_mean - K_var)
        # Same paths → parity holds exactly in MC (up to floating point)
        assert abs((call - put) - parity_rhs) < 1e-10

    @pytest.mark.slow
    def test_variance_option_higher_vol_higher_price(self):
        """Higher underlying vol → higher variance option price (positive vega)."""
        spec = GBMPathSpec(n_paths=20_000, n_steps=252, seed=99, antithetic=True)
        K_var = sigma**2
        price_low = variance_option_mc(S0, r, q, 0.1, T, K_var, 252, cp=1, spec=spec)
        price_high = variance_option_mc(S0, r, q, 0.3, T, K_var, 252, cp=1, spec=spec)
        assert price_high > price_low
