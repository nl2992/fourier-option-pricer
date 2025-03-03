"""Tests for the Carr-Madan/FRFT alpha-validity checker.

Verify:
  - Kou: analytic bound eta1-1 is tight.
  - VG:  analytic bound matches runtime phi divergence.
  - Generic runtime check flags bad alphas with NaN/inf phi.
  - assert_alpha_valid raises for bad alpha, passes for good.
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.char_func.base import ForwardSpec
from foureng.char_func.kou import KouParams, kou_cf
from foureng.char_func.variance_gamma import VGParams, vg_cf
from foureng.char_func.heston import HestonParams, heston_cf_form2
from foureng.utils.validity import (
    check_alpha,
    kou_alpha_max,
    vg_alpha_max,
    assert_alpha_valid,
)


def test_kou_alpha_max_analytic():
    """Kou: alpha_max = eta1 - 1, a hard analytic bound on E[S^{alpha+1}].

    Past the bound the CF formula still evaluates to a finite complex number
    (it's just the analytic continuation outside the MGF strip), so the
    runtime phi-finiteness probe does not catch this — but the analytic
    check in assert_alpha_valid does.
    """
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.5)
    p = KouParams(sigma=0.16, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)
    assert kou_alpha_max(p) == pytest.approx(9.0)

    phi = lambda u: kou_cf(u, fwd, p)
    # Inside bound: generic probe passes and assert doesn't raise.
    assert check_alpha(phi, alpha=1.5).ok
    assert_alpha_valid(phi, alpha=1.5, model_params=p)
    assert_alpha_valid(phi, alpha=8.0, model_params=p)
    # Past bound: assert raises even though generic probe might pass.
    with pytest.raises(ValueError, match="Kou"):
        assert_alpha_valid(phi, alpha=9.5, model_params=p)


def test_vg_alpha_max_analytic():
    """VG alpha_max from the positive root of the CF quadratic.

    sigma=0.25, nu=2.0, theta=-0.10: quadratic is 1 + 0.2s - 0.0625 s^2 = 0,
    s_* = (0.2 + sqrt(0.04 + 0.25))/0.125 = (0.2 + 0.53852)/0.125 = 5.9081,
    so alpha_max = 4.9081.
    """
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.03, T=0.25)
    p = VGParams(sigma=0.25, nu=2.0, theta=-0.10)
    amax = vg_alpha_max(p)
    assert amax == pytest.approx(4.9081, abs=1e-3)

    phi = lambda u: vg_cf(u, fwd, p)
    # Inside
    assert_alpha_valid(phi, alpha=amax * 0.5, model_params=p)
    # Past
    with pytest.raises(ValueError, match="VG"):
        assert_alpha_valid(phi, alpha=amax + 0.5, model_params=p)


def test_heston_alpha_generic_probe():
    """For Heston, the strip of analyticity depends on T; use runtime probe."""
    fwd = ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
    p = HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)
    phi = lambda u: heston_cf_form2(u, fwd, p)
    # alpha=1.5 is a standard Carr-Madan default — must pass
    assert check_alpha(phi, alpha=1.5).ok


def test_assert_alpha_valid_raises_kou():
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.5)
    p = KouParams(sigma=0.16, lam=1.0, p=0.4, eta1=3.0, eta2=5.0)
    phi = lambda u: kou_cf(u, fwd, p)
    # eta1=3 => alpha_max=2; alpha=1.5 is fine
    assert_alpha_valid(phi, alpha=1.5, model_params=p)
    # alpha=2.5 should raise
    with pytest.raises(ValueError, match="Kou"):
        assert_alpha_valid(phi, alpha=2.5, model_params=p)


def test_assert_alpha_valid_raises_vg():
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.03, T=0.25)
    p = VGParams(sigma=0.25, nu=2.0, theta=-0.10)
    phi = lambda u: vg_cf(u, fwd, p)
    amax = vg_alpha_max(p)
    # Valid: inside
    assert_alpha_valid(phi, alpha=amax * 0.5, model_params=p)
    # Invalid: past the bound
    with pytest.raises(ValueError, match="VG"):
        assert_alpha_valid(phi, alpha=amax + 0.5, model_params=p)


def test_reject_nonpositive_alpha():
    fwd = ForwardSpec(S0=100.0, r=0.01, q=0.0, T=1.0)
    p = HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)
    phi = lambda u: heston_cf_form2(u, fwd, p)
    assert not check_alpha(phi, alpha=0.0).ok
    assert not check_alpha(phi, alpha=-1.0).ok
