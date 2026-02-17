"""Parameter validation tests for in-house native-CF model dataclasses.

Every constructor that documents constraints must enforce them by raising
ValueError on invalid inputs.
"""
from __future__ import annotations

import math
import pytest

from foureng.models.fmls import FMLSParams
from foureng.models.meixner import MeixnerParams
from foureng.models.bilateral_gamma import BilateralGammaParams
from foureng.models.generalized_hyperbolic import GHParams
from foureng.models.merton_jd import MertonJDParams


# ---------------------------------------------------------------------------
# FMLSParams
# ---------------------------------------------------------------------------

class TestFMLSValidation:

    def test_valid_params_accepted(self):
        p = FMLSParams(alpha=1.7, sigma=0.3)
        assert p.alpha == 1.7
        assert p.sigma == 0.3

    def test_alpha_exactly_2_accepted(self):
        FMLSParams(alpha=2.0, sigma=0.2)

    @pytest.mark.parametrize("bad_alpha", [0.5, 1.0, 2.1, 3.0, float("nan"), float("inf")])
    def test_rejects_bad_alpha(self, bad_alpha):
        with pytest.raises(ValueError, match="alpha"):
            FMLSParams(alpha=bad_alpha, sigma=0.3)

    @pytest.mark.parametrize("bad_sigma", [0.0, -0.1, float("nan"), float("inf")])
    def test_rejects_bad_sigma(self, bad_sigma):
        with pytest.raises(ValueError, match="sigma"):
            FMLSParams(alpha=1.5, sigma=bad_sigma)


# ---------------------------------------------------------------------------
# MeixnerParams
# ---------------------------------------------------------------------------

class TestMeixnerValidation:

    def test_valid_params_accepted(self):
        p = MeixnerParams(a=0.4, b=-1.1, delta=0.35)
        assert p.a == 0.4

    @pytest.mark.parametrize("bad_a", [0.0, -1.0, float("nan")])
    def test_rejects_bad_a(self, bad_a):
        with pytest.raises(ValueError, match="a must"):
            MeixnerParams(a=bad_a, b=0.0, delta=0.3)

    @pytest.mark.parametrize("bad_b", [math.pi, -math.pi, 4.0, float("nan")])
    def test_rejects_bad_b(self, bad_b):
        with pytest.raises(ValueError, match="b must"):
            MeixnerParams(a=0.4, b=bad_b, delta=0.3)

    @pytest.mark.parametrize("bad_delta", [0.0, -0.5, float("nan")])
    def test_rejects_bad_delta(self, bad_delta):
        with pytest.raises(ValueError, match="delta must"):
            MeixnerParams(a=0.4, b=0.0, delta=bad_delta)


# ---------------------------------------------------------------------------
# BilateralGammaParams
# ---------------------------------------------------------------------------

class TestBilateralGammaValidation:

    def test_valid_params_accepted(self):
        p = BilateralGammaParams(alpha_p=1.0, lambda_p=5.0, alpha_m=0.8, lambda_m=4.0)
        assert p.alpha_p == 1.0

    @pytest.mark.parametrize("bad_ap", [0.0, -1.0, float("nan")])
    def test_rejects_bad_alpha_p(self, bad_ap):
        with pytest.raises(ValueError, match="alpha_p"):
            BilateralGammaParams(alpha_p=bad_ap, lambda_p=5.0, alpha_m=0.8, lambda_m=4.0)

    @pytest.mark.parametrize("bad_lp", [0.5, 1.0, 0.0, -1.0, float("nan")])
    def test_rejects_bad_lambda_p(self, bad_lp):
        with pytest.raises(ValueError, match="lambda_p"):
            BilateralGammaParams(alpha_p=1.0, lambda_p=bad_lp, alpha_m=0.8, lambda_m=4.0)

    @pytest.mark.parametrize("bad_am", [0.0, -0.5, float("nan")])
    def test_rejects_bad_alpha_m(self, bad_am):
        with pytest.raises(ValueError, match="alpha_m"):
            BilateralGammaParams(alpha_p=1.0, lambda_p=5.0, alpha_m=bad_am, lambda_m=4.0)

    @pytest.mark.parametrize("bad_lm", [0.5, 1.0, -1.0, float("nan")])
    def test_rejects_bad_lambda_m(self, bad_lm):
        with pytest.raises(ValueError, match="lambda_m"):
            BilateralGammaParams(alpha_p=1.0, lambda_p=5.0, alpha_m=0.8, lambda_m=bad_lm)


# ---------------------------------------------------------------------------
# GHParams
# ---------------------------------------------------------------------------

class TestGHValidation:

    def test_valid_nig_params_accepted(self):
        p = GHParams(lam=-0.5, alpha=6.1882, beta=-3.8941, delta=0.1622)
        assert p.lam == -0.5

    def test_rejects_non_finite_lam(self):
        with pytest.raises(ValueError, match="lam"):
            GHParams(lam=float("nan"), alpha=6.0, beta=-2.0, delta=0.2)

    @pytest.mark.parametrize("bad_alpha", [0.0, -1.0, float("nan")])
    def test_rejects_bad_alpha(self, bad_alpha):
        with pytest.raises(ValueError, match="alpha"):
            GHParams(lam=-0.5, alpha=bad_alpha, beta=0.0, delta=0.2)

    def test_rejects_beta_ge_alpha(self):
        with pytest.raises(ValueError, match="alpha must be > .beta."):
            GHParams(lam=-0.5, alpha=2.0, beta=2.0, delta=0.2)

    def test_rejects_martingale_violation(self):
        # alpha - |beta| = 3.0 - 2.5 = 0.5 < 1 → martingale condition fails
        with pytest.raises(ValueError, match="martingale"):
            GHParams(lam=-0.5, alpha=3.0, beta=-2.5, delta=0.2)

    @pytest.mark.parametrize("bad_delta", [0.0, -0.1, float("nan")])
    def test_rejects_bad_delta(self, bad_delta):
        with pytest.raises(ValueError, match="delta"):
            GHParams(lam=-0.5, alpha=6.0, beta=-2.0, delta=bad_delta)


# ---------------------------------------------------------------------------
# MertonJDParams
# ---------------------------------------------------------------------------

class TestMertonJDValidation:

    def test_valid_params_accepted(self):
        p = MertonJDParams(sigma=0.2, lam=0.5, muj=-0.1, sigj=0.15)
        assert p.sigma == 0.2

    def test_zero_sigma_accepted(self):
        MertonJDParams(sigma=0.0, lam=1.0, muj=0.0, sigj=0.1)

    def test_zero_lambda_accepted(self):
        MertonJDParams(sigma=0.2, lam=0.0, muj=0.0, sigj=0.0)

    @pytest.mark.parametrize("bad_sigma", [-0.1, float("nan")])
    def test_rejects_bad_sigma(self, bad_sigma):
        with pytest.raises(ValueError, match="sigma"):
            MertonJDParams(sigma=bad_sigma, lam=0.5, muj=0.0, sigj=0.1)

    @pytest.mark.parametrize("bad_lam", [-0.1, float("nan")])
    def test_rejects_bad_lam(self, bad_lam):
        with pytest.raises(ValueError, match="lam"):
            MertonJDParams(sigma=0.2, lam=bad_lam, muj=0.0, sigj=0.1)

    def test_rejects_non_finite_muj(self):
        with pytest.raises(ValueError, match="muj"):
            MertonJDParams(sigma=0.2, lam=0.5, muj=float("inf"), sigj=0.1)

    @pytest.mark.parametrize("bad_sigj", [-0.1, float("nan")])
    def test_rejects_bad_sigj(self, bad_sigj):
        with pytest.raises(ValueError, match="sigj"):
            MertonJDParams(sigma=0.2, lam=0.5, muj=0.0, sigj=bad_sigj)
