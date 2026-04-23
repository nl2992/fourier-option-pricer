"""Tests for ``foureng.utils.implied_vol.implied_vol_from_prices``.

The utility wraps PyFENG's Brent solver. Three things matter:

  1. **Round-trip on BSM**: price -> IV returns the input vol exactly,
     at every strike and ``cp``.
  2. **Heston-smile shape**: IV backed out of Heston prices is a smooth
     smile with the negative-rho skew present (ATM IV < OTM IV); no
     NaNs on a moderately-wide ITM/OTM strip.
  3. **Guard rails**: prices below intrinsic / above forward return
     NaN rather than raising.
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams
from foureng.models.heston import HestonParams
from foureng.pipeline import price_strip
from foureng.utils.implied_vol import implied_vol_from_prices


pyfeng = pytest.importorskip("pyfeng", reason="utility uses pyfeng.BsmFft")


def test_iv_roundtrip_bsm_calls():
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.25)
    K = np.linspace(70.0, 130.0, 25)
    C = price_strip("bsm", "cos", K, fwd, p)
    iv = implied_vol_from_prices(C, K, fwd, cp=1)
    assert np.all(np.isfinite(iv))
    assert np.max(np.abs(iv - p.sigma)) < 1e-8


def test_iv_roundtrip_bsm_puts():
    """Same round-trip logic on puts (cp=-1). Derive puts via parity."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.30)
    K = np.linspace(70.0, 130.0, 25)
    C = price_strip("bsm", "cos", K, fwd, p)
    # put = call - S0*exp(-qT) + K*exp(-rT)
    P = C - fwd.S0 * np.exp(-fwd.q * fwd.T) + K * np.exp(-fwd.r * fwd.T)
    iv = implied_vol_from_prices(P, K, fwd, cp=-1)
    assert np.all(np.isfinite(iv))
    assert np.max(np.abs(iv - p.sigma)) < 1e-8


def test_iv_heston_smile_shape_is_sane(lewis_heston):
    """Heston with rho < 0 produces a left-skewed smile: ITM IV > OTM IV."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    K = np.linspace(80.0, 120.0, 21)
    C = price_strip("heston", "cos", K, fwd, p)
    iv = implied_vol_from_prices(C, K, fwd, cp=1)

    assert np.all(np.isfinite(iv)), iv
    # Negative-rho Heston: IV(K=80) > IV(K=120). Loose check — depends on
    # parameter regime but is robust for Lewis's canonical set.
    assert iv[0] > iv[-1], f"no left skew: iv[80]={iv[0]}, iv[120]={iv[-1]}"


def test_iv_below_intrinsic_returns_nan():
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    K = np.array([80.0])
    # Call intrinsic on K=80 is (F-K)*disc. Subtract some cents → below.
    F = fwd.F0
    disc = fwd.disc
    P_too_low = (F - K[0]) * disc - 0.10  # well below intrinsic
    iv = implied_vol_from_prices(np.array([P_too_low]), K, fwd, cp=1)
    assert np.isnan(iv[0])


def test_iv_above_forward_returns_nan():
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    K = np.array([100.0])
    P_too_high = fwd.F0 * fwd.disc + 1.0  # above F*disc
    iv = implied_vol_from_prices(np.array([P_too_high]), K, fwd, cp=1)
    assert np.isnan(iv[0])


def test_iv_shape_mismatch_raises():
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    with pytest.raises(ValueError, match="must match"):
        implied_vol_from_prices(
            np.array([1.0, 2.0]), np.array([100.0]), fwd, cp=1
        )
