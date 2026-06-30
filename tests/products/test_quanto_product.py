"""Tests for the QuantoOption product dataclass."""

from __future__ import annotations

import pytest

from foureng.products.quanto import QuantoOption


BASE_KW = dict(
    S=100.0, K=105.0, T=1.0,
    r_dom=0.05, r_for=0.03, q_for=0.02,
    rho=-0.3, sigma_S=0.20, sigma_X=0.10,
    cp=1,
)


class TestQuantoOption:
    def test_creates_call(self):
        q = QuantoOption(**BASE_KW)
        assert q.cp == 1

    def test_creates_put(self):
        q = QuantoOption(**{**BASE_KW, 'cp': -1})
        assert q.cp == -1

    def test_frozen(self):
        q = QuantoOption(**BASE_KW)
        with pytest.raises((AttributeError, TypeError)):
            q.S = 200.0  # type: ignore[misc]

    def test_zero_rho_allowed(self):
        q = QuantoOption(**{**BASE_KW, 'rho': 0.0})
        assert q.rho == 0.0

    def test_zero_sigma_X_allowed(self):
        q = QuantoOption(**{**BASE_KW, 'sigma_X': 0.0})
        assert q.sigma_X == 0.0

    def test_zero_T_allowed(self):
        q = QuantoOption(**{**BASE_KW, 'T': 0.0})
        assert q.T == 0.0

    def test_raises_negative_S(self):
        with pytest.raises(ValueError, match="S must be positive"):
            QuantoOption(**{**BASE_KW, 'S': -1.0})

    def test_raises_zero_S(self):
        with pytest.raises(ValueError, match="S must be positive"):
            QuantoOption(**{**BASE_KW, 'S': 0.0})

    def test_raises_negative_K(self):
        with pytest.raises(ValueError, match="K must be positive"):
            QuantoOption(**{**BASE_KW, 'K': -5.0})

    def test_raises_negative_T(self):
        with pytest.raises(ValueError, match="T must be non-negative"):
            QuantoOption(**{**BASE_KW, 'T': -0.5})

    def test_raises_negative_sigma_S(self):
        with pytest.raises(ValueError, match="sigma_S must be non-negative"):
            QuantoOption(**{**BASE_KW, 'sigma_S': -0.1})

    def test_raises_negative_sigma_X(self):
        with pytest.raises(ValueError, match="sigma_X must be non-negative"):
            QuantoOption(**{**BASE_KW, 'sigma_X': -0.05})

    def test_raises_rho_greater_than_one(self):
        with pytest.raises(ValueError, match="rho must be in"):
            QuantoOption(**{**BASE_KW, 'rho': 1.5})

    def test_raises_rho_less_than_minus_one(self):
        with pytest.raises(ValueError, match="rho must be in"):
            QuantoOption(**{**BASE_KW, 'rho': -1.1})

    def test_raises_invalid_cp(self):
        with pytest.raises(ValueError, match="cp must be"):
            QuantoOption(**{**BASE_KW, 'cp': 0})

    def test_importable_from_foureng(self):
        import foureng as fe
        assert hasattr(fe, 'QuantoOption')

    def test_constructable_from_foureng(self):
        import foureng as fe
        q = fe.QuantoOption(**BASE_KW)
        assert isinstance(q, fe.QuantoOption)
