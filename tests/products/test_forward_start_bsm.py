"""L5 evidence: BSM forward-starting option pricing.

Tests foureng.pricers.analytic_bsm.bsm_forward_start against known
mathematical properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.pricers.analytic_bsm import bsm_european, bsm_forward_start

# Standard parameters
S = 100.0
r = 0.05
q = 0.02
sigma = 0.2


class TestForwardStartBSM:
    def test_start_time_zero_equals_vanilla(self):
        """start_time=0 reduces to vanilla BSM with K=alpha*S."""
        maturity = 1.0
        alpha = 1.0
        fs = bsm_forward_start(S, r, q, sigma, start_time=0.0, maturity=maturity, alpha=alpha, cp=1)
        vanilla = bsm_european(S, alpha * S, r, q, sigma, maturity, cp=1)
        assert fs == pytest.approx(vanilla, rel=1e-10)

    def test_start_time_zero_put(self):
        """start_time=0 reduces to vanilla put with K=alpha*S."""
        maturity = 1.0
        alpha = 0.95
        fs = bsm_forward_start(
            S, r, q, sigma, start_time=0.0, maturity=maturity, alpha=alpha, cp=-1
        )
        vanilla = bsm_european(S, alpha * S, r, q, sigma, maturity, cp=-1)
        assert fs == pytest.approx(vanilla, rel=1e-10)

    def test_scale_invariance_call(self):
        """Scale invariance: C(λS, alpha, ...) = λ * C(S, alpha, ...)."""
        start_time = 0.5
        maturity = 1.5
        alpha = 1.0
        lam = 2.0

        fs_S = bsm_forward_start(S, r, q, sigma, start_time, maturity, alpha, cp=1)
        fs_lamS = bsm_forward_start(lam * S, r, q, sigma, start_time, maturity, alpha, cp=1)
        assert fs_lamS == pytest.approx(lam * fs_S, rel=1e-10)

    def test_scale_invariance_put(self):
        """Scale invariance holds for put."""
        start_time = 0.5
        maturity = 1.5
        alpha = 0.9
        lam = 3.0

        fs_S = bsm_forward_start(S, r, q, sigma, start_time, maturity, alpha, cp=-1)
        fs_lamS = bsm_forward_start(lam * S, r, q, sigma, start_time, maturity, alpha, cp=-1)
        assert fs_lamS == pytest.approx(lam * fs_S, rel=1e-10)

    def test_zero_vol_limit_positive_start(self):
        """Zero vol with start_time > 0: deterministic strike, payoff is discounted."""
        sigma_tiny = 1e-10
        start_time = 0.5
        maturity = 1.5
        alpha = 1.0

        fs = bsm_forward_start(S, r, q, sigma_tiny, start_time, maturity, alpha, cp=1)

        # Deterministic: at start_time, S_{t_s} = S*exp((r-q)*t_s), K = alpha * S_{t_s}
        S_ts = S * np.exp((r - q) * start_time)
        K_det = alpha * S_ts
        # S_T = S * exp((r-q)*maturity)
        S_T = S * np.exp((r - q) * maturity)
        expected = np.exp(-r * maturity) * max(S_T - K_det, 0.0)
        assert fs == pytest.approx(expected, rel=1e-4)

    def test_frozen_reference_atm(self):
        """Frozen reference: S=100, r=0.05, q=0.02, sigma=0.2, t_s=0.5, T=1.5, alpha=1."""
        fs = bsm_forward_start(100, 0.05, 0.02, 0.2, 0.5, 1.5, 1.0, cp=1)
        assert fs == pytest.approx(9.13520, rel=1e-3)

    def test_atm_forward_start_formula(self):
        """V = e^{-q*t_s} * S * bsm_european(1, alpha, r, q, sigma, tenor, cp)."""
        start_time = 0.5
        maturity = 1.5
        alpha = 1.0
        tenor = maturity - start_time

        expected = np.exp(-q * start_time) * S * bsm_european(1.0, alpha, r, q, sigma, tenor, cp=1)
        result = bsm_forward_start(S, r, q, sigma, start_time, maturity, alpha, cp=1)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_various_alpha_values(self):
        """Price is monotone decreasing in alpha (higher strike → cheaper call)."""
        start_time = 0.5
        maturity = 1.5
        alphas = [0.8, 0.9, 1.0, 1.1, 1.2]
        prices = [
            bsm_forward_start(S, r, q, sigma, start_time, maturity, alpha, cp=1) for alpha in alphas
        ]
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1]

    def test_call_price_positive(self):
        """Forward start call price is strictly positive."""
        fs = bsm_forward_start(S, r, q, sigma, 0.5, 1.5, 1.0, cp=1)
        assert fs > 0.0

    def test_put_price_positive(self):
        """Forward start put price is strictly positive."""
        fs = bsm_forward_start(S, r, q, sigma, 0.5, 1.5, 1.0, cp=-1)
        assert fs > 0.0

    def test_longer_tenor_increases_price(self):
        """Longer option tenor increases call price (all else equal)."""
        price_short = bsm_forward_start(S, r, q, sigma, 0.5, 1.0, 1.0, cp=1)
        price_long = bsm_forward_start(S, r, q, sigma, 0.5, 2.0, 1.0, cp=1)
        assert price_long > price_short

    def test_higher_vol_increases_price(self):
        """Higher volatility increases forward-start call price."""
        price_low_vol = bsm_forward_start(S, r, q, 0.1, 0.5, 1.5, 1.0, cp=1)
        price_high_vol = bsm_forward_start(S, r, q, 0.4, 0.5, 1.5, 1.0, cp=1)
        assert price_high_vol > price_low_vol
