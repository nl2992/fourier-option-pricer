"""Tests for SABR Hagan (2002) implied volatility approximation.

Key invariants:
  1. ATM vol with beta=1, rho=0, nu=0: sigma_SABR ≈ alpha (exact).
  2. ATM vol with beta=0 (normal SABR), rho=0, nu=0: sigma_SABR ≈ alpha/F.
  3. Smile curvature: nu>0 → positive convexity (wings > ATM vol).
  4. Rho tilts the smile: rho>0 → right wing higher than left.
  5. Price positivity and monotone in nu (wider smile → higher OTM prices).
  6. price_strip round-trip: strip prices are non-negative and decrease for OTM calls.
  7. Put-call parity via BSM Hagan vols.
  8. Parameter validation: invalid inputs raise ValueError.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from foureng.models.sabr import SabrParams, sabr_hagan_implied_vol
from foureng.models.base import ForwardSpec
from foureng.pricers.sabr import sabr_hagan_price_at_strikes


# ── helpers ────────────────────────────────────────────────────────────────

F, T = 100.0, 1.0  # forward, expiry
r = 0.05


def _bsm_call_from_vol(F, K, T, r, vol):
    """Black (log-normal) call price from forward."""
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def _bsm_put_from_vol(F, K, T, r, vol):
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


# ── 1. ATM vol, beta=1, nu=0 → sigma_SABR = alpha ────────────────────────


def test_atm_vol_lognormal_backbone():
    alpha = 0.25
    p = SabrParams(alpha=alpha, beta=1.0, rho=0.0, nu=0.0)
    vol_atm = float(sabr_hagan_implied_vol(F, [F], T, p)[0])
    assert abs(vol_atm - alpha) < 1e-6, f"ATM vol {vol_atm:.8f} ≠ alpha {alpha}"


# ── 2. ATM vol, beta=0, nu=0 → sigma_SABR = alpha/F (normal approx) ──────


def test_atm_vol_normal_backbone():
    # For beta=0, nu=0, rho=0, Hagan ATM formula gives:
    # sigma ≈ (alpha/F) * (1 + alpha²*T/(24*F²))  [correction term from Taylor expansion]
    alpha = 20.0
    p = SabrParams(alpha=alpha, beta=0.0, rho=0.0, nu=0.0)
    vol_atm = float(sabr_hagan_implied_vol(F, [F], T, p)[0])
    expected = (alpha / F) * (1.0 + alpha**2 * T / (24.0 * F**2))
    assert abs(vol_atm - expected) < 1e-6, f"ATM vol {vol_atm:.8f} ≠ approx {expected:.8f}"


# ── 3. Smile convexity: wings > ATM when nu > 0 ──────────────────────────


def test_smile_convexity_with_volvol():
    p = SabrParams(alpha=0.3, beta=0.5, rho=0.0, nu=0.5)
    K_otm_call = np.array([F, 110.0, 120.0], dtype=float)
    vols = sabr_hagan_implied_vol(F, K_otm_call, T, p)
    # Wings should have higher vol than ATM
    assert vols[1] > vols[0], "SABR vol should increase for OTM call (wing) with nu>0"
    assert vols[2] > vols[1], "SABR vol should be monotonically higher for deeper OTM"


# ── 4. Rho-induced skew ───────────────────────────────────────────────────


def test_negative_rho_gives_negative_skew():
    """Negative rho tilts smile left: low strike vol > high strike vol."""
    p_neg = SabrParams(alpha=0.3, beta=0.5, rho=-0.5, nu=0.5)
    p_pos = SabrParams(alpha=0.3, beta=0.5, rho=+0.5, nu=0.5)

    K_low = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=float)
    vols_neg = sabr_hagan_implied_vol(F, K_low, T, p_neg)
    vols_pos = sabr_hagan_implied_vol(F, K_low, T, p_pos)

    # Negative rho → higher vol on the left wing vs. right wing
    skew_neg = vols_neg[0] - vols_neg[-1]  # vol[80] - vol[120]
    skew_pos = vols_pos[0] - vols_pos[-1]
    assert skew_neg > skew_pos, (
        f"Negative rho should give more negative skew: {skew_neg:.4f} vs {skew_pos:.4f}"
    )


# ── 5. Monotone OTM prices with nu ────────────────────────────────────────


def test_higher_nu_raises_otm_prices():
    """Higher vol-of-vol makes OTM options more expensive."""
    strikes = np.array([80.0, 120.0], dtype=float)
    fwd = ForwardSpec(S0=F, r=r, q=0.0, T=T)

    p_low_nu  = SabrParams(alpha=0.3, beta=0.5, rho=0.0, nu=0.1)
    p_high_nu = SabrParams(alpha=0.3, beta=0.5, rho=0.0, nu=0.8)

    prices_low  = sabr_hagan_price_at_strikes(fwd, p_low_nu,  strikes, cp=1)
    prices_high = sabr_hagan_price_at_strikes(fwd, p_high_nu, strikes, cp=1)

    for k, pl, ph in zip(strikes, prices_low, prices_high):
        assert ph >= pl - 1e-6, (
            f"Higher nu should raise OTM price at K={k}: {ph:.4f} < {pl:.4f}"
        )


# ── 6. price_strip: call prices ≥ 0, decrease for OTM calls ──────────────


def test_price_strip_sabr_hagan():
    from foureng.pipeline import price_strip

    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=float)
    fwd = ForwardSpec(S0=F, r=r, q=0.0, T=T)
    p = SabrParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)

    prices = price_strip("sabr", "sabr_hagan", strikes, fwd, p, cp=1)

    assert np.all(prices >= -1e-8), "Call prices must be non-negative"
    # ITM → ATM → OTM: prices should decrease for calls as K increases
    assert np.all(np.diff(prices) <= 1e-6), (
        f"Call prices not monotone decreasing in strike: {prices}"
    )


# ── 7. Put-call parity ────────────────────────────────────────────────────


def test_put_call_parity():
    """C(K) - P(K) = F*e^{-rT} - K*e^{-rT} (Black formula PCP)."""
    fwd = ForwardSpec(S0=F, r=r, q=0.0, T=T)
    p = SabrParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
    strikes = np.array([90.0, 100.0, 110.0], dtype=float)

    calls = sabr_hagan_price_at_strikes(fwd, p, strikes, cp=1)
    puts  = sabr_hagan_price_at_strikes(fwd, p, strikes, cp=-1)

    # Forward price = S0 * e^{(r-q)*T} = S0 * e^{r*T} here (q=0)
    # PCP: C - P = (F0 - K) * e^{-rT}
    F0 = F  # since S0=F and q=0, forward = S0*e^{r*T}; but fwd.F0 uses q
    # ForwardSpec.F0 = S0 * exp((r-q)*T)
    fwd_price = fwd.S0 * np.exp((fwd.r - fwd.q) * fwd.T)  # = F * e^{rT}
    pcp = (fwd_price - strikes) * np.exp(-r * T)

    for k, c, pu, expected in zip(strikes, calls, puts, pcp):
        assert abs(c - pu - expected) < 1e-4, (
            f"PCP failed at K={k}: C={c:.6f}, P={pu:.6f}, C-P={c-pu:.6f}, expected={expected:.6f}"
        )


# ── 8. Parameter validation ───────────────────────────────────────────────


def test_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        SabrParams(alpha=-0.1, beta=0.5, rho=0.0, nu=0.3)


def test_invalid_beta():
    with pytest.raises(ValueError, match="beta"):
        SabrParams(alpha=0.3, beta=1.5, rho=0.0, nu=0.3)


def test_invalid_rho():
    with pytest.raises(ValueError, match="rho"):
        SabrParams(alpha=0.3, beta=0.5, rho=1.0, nu=0.3)


def test_invalid_nu():
    with pytest.raises(ValueError, match="nu"):
        SabrParams(alpha=0.3, beta=0.5, rho=0.0, nu=-0.1)


def test_invalid_forward():
    p = SabrParams(alpha=0.3, beta=0.5, rho=0.0, nu=0.3)
    with pytest.raises(ValueError):
        sabr_hagan_implied_vol(0.0, [100.0], 1.0, p)


def test_invalid_strike():
    p = SabrParams(alpha=0.3, beta=0.5, rho=0.0, nu=0.3)
    with pytest.raises(ValueError):
        sabr_hagan_implied_vol(100.0, [-10.0], 1.0, p)
