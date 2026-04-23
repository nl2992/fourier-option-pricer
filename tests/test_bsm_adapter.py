"""BSM adapter — baseline sanity gate for the whole PyFENG integration.

BSM is the only model here with a trivial closed-form CF, so we get
four hard gates for the price of one:

  1. ``bsm_cf`` matches the closed-form Gaussian CF to machine precision.
  2. ``bsm_cumulants`` matches the closed-form cumulants exactly.
  3. Every pricing path (COS / FRFT / CM / pyfeng_fft) agrees with
     Black-Scholes analytic prices on a 21-strike strip.
  4. BSM-implied vol backed out of our prices returns the input sigma
     at every strike (the utility function round-trips).

If any of these breaks, something is wrong at a level below any of the
SV/SVJ models. They are the "if-BSM-fails, nothing-else-matters" gates.
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.char_func.base import ForwardSpec
from foureng.char_func.bsm import BsmParams, bsm_cf, bsm_cumulants
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid
from foureng.utils.implied_vol import implied_vol_from_prices


pyfeng = pytest.importorskip("pyfeng", reason="BSM adapter is PyFENG-backed")


def _bs_call(S, K, r, q, T, sigma):
    """Black-Scholes call — closed form, hand-rolled to stay PyFENG-free."""
    from scipy.stats import norm
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def test_bsm_cf_is_closed_form_gaussian():
    """phi_X(u) = exp(-0.5*sigma^2*T*(u^2 + i*u)) exactly."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.2)
    u = np.linspace(-10.0, 10.0, 41)

    phi_ours = bsm_cf(u, fwd, p)
    sigma2T = p.sigma * p.sigma * fwd.T
    phi_closed = np.exp(-0.5 * sigma2T * (u * u + 1j * u))

    err = float(np.max(np.abs(phi_ours - phi_closed)))
    assert err < 1e-14, f"BSM CF vs closed-form Gaussian: max|err| = {err:.3e}"


def test_bsm_cumulants_closed_form():
    """Gaussian cumulants: c1=-0.5*sigma^2*T, c2=sigma^2*T, c4=0."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.25)
    c1, c2, c4 = bsm_cumulants(fwd, p)
    s2T = p.sigma * p.sigma * fwd.T
    assert c1 == -0.5 * s2T
    assert c2 == s2T
    assert c4 == 0.0


def test_bsm_cf_martingale():
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.2)
    phi0 = bsm_cf(np.array([0.0]), fwd, p)[0]
    assert abs(phi0 - 1.0) < 1e-14


def test_bsm_prices_match_black_scholes_closed_form():
    """Every foureng engine prices BSM to the Black-Scholes analytic."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.2)
    K = np.linspace(80.0, 120.0, 21)
    bs_ref = _bs_call(fwd.S0, K, fwd.r, fwd.q, fwd.T, p.sigma)

    C_cos = price_strip("bsm", "cos", K, fwd, p)
    C_frft = price_strip("bsm", "frft", K, fwd, p,
                          grid=FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5))
    C_cm = price_strip("bsm", "carr_madan", K, fwd, p,
                        grid=FFTGrid(N=4096, eta=0.25, alpha=1.5))
    C_pf = price_strip("bsm", "pyfeng_fft", K, fwd, p)

    # COS on a Gaussian is essentially exact; FFT engines are near their
    # default-grid error floor (~1e-7). Tolerances chosen to catch real
    # regressions without flaking.
    assert np.max(np.abs(C_cos  - bs_ref)) < 1e-10, f"COS vs BS: {np.max(np.abs(C_cos - bs_ref)):.3e}"
    assert np.max(np.abs(C_frft - bs_ref)) < 1e-6
    assert np.max(np.abs(C_cm   - bs_ref)) < 1e-6
    assert np.max(np.abs(C_pf   - bs_ref)) < 1e-10


def test_bsm_implied_vol_roundtrip():
    """Our implied-vol utility recovers the input sigma exactly (Brent)."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.2)
    K = np.linspace(80.0, 120.0, 21)

    C = price_strip("bsm", "cos", K, fwd, p)
    iv = implied_vol_from_prices(C, K, fwd, cp=1)

    assert np.all(np.isfinite(iv)), f"IV has NaNs at some strikes:\n{iv}"
    err = float(np.max(np.abs(iv - p.sigma)))
    assert err < 1e-8, f"IV round-trip error: max|iv - sigma| = {err:.3e}"


def test_bsm_pricing_raises_on_unknown_method():
    """Typos in the method name fail loudly."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.2)
    with pytest.raises(ValueError, match="unknown method"):
        price_strip("bsm", "not-a-method", np.array([100.0]), fwd, p)
