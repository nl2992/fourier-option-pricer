"""Equity + Hull-White stochastic-rate hybrid validation.

Independent references:
- BSM base: the hybrid is again lognormal with total variance
  sigma^2 T + V_P, so it must equal BSM at the effective vol exactly
  (Merton 1973 stochastic-rate Black-Scholes);
- sigma_r = 0 collapses to the base CF identically;
- a -> 0 recovers the Ho-Lee limit V_P = sigma_r^2 T^3 / 3;
- Kou base: Monte Carlo with an exact independent Gaussian rate term.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.models.hull_white_hybrid import (
    HullWhiteHybridParams,
    hw_bond_variance,
    hw_hybrid_cf,
)

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
_STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
_U = np.linspace(-40.0, 40.0, 161)
_KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.6, eta1=25.0, eta2=15.0)
_HW = dict(mean_reversion=0.1, sigma_r=0.015)


def test_bond_variance_exact_and_holee_limit():
    # closed form vs numerical quadrature
    a, sr, T = 0.3, 0.02, 2.0
    s_grid = np.linspace(0.0, T, 200_001)
    sig_p = (sr / a) * (1.0 - np.exp(-a * (T - s_grid)))
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x compat
    v_quad = float(trapz(sig_p**2, s_grid))
    assert hw_bond_variance(a, sr, T) == pytest.approx(v_quad, rel=1e-8)
    # Ho-Lee limit continuity at the a*T branch point
    v_small = hw_bond_variance(1e-5, sr, T)
    assert v_small == pytest.approx(sr**2 * T**3 / 3.0, rel=1e-3)


@pytest.mark.parametrize("cp", [1, -1])
def test_bsm_base_equals_effective_vol_closed_form(cp):
    sigma = 0.2
    p = HullWhiteHybridParams("bsm", fe.BsmParams(sigma=sigma), **_HW)
    v_p = hw_bond_variance(_HW["mean_reversion"], _HW["sigma_r"], _FWD.T)
    sigma_eff = np.sqrt(sigma**2 + v_p / _FWD.T)
    hybrid = fe.price_strip("hw_hybrid", "cos", _STRIKES, _FWD, p, cp=cp)
    bsm_eff = fe.price_strip("bsm", "cos", _STRIKES, _FWD, fe.BsmParams(sigma=sigma_eff), cp=cp)
    np.testing.assert_allclose(hybrid, bsm_eff, atol=1e-8, rtol=0.0)


def test_zero_rate_vol_collapses_to_base():
    p = HullWhiteHybridParams("kou", _KOU, mean_reversion=0.1, sigma_r=0.0)
    np.testing.assert_allclose(
        hw_hybrid_cf(_U, _FWD, p), fe.kou_cf(_U, _FWD, _KOU), atol=0.0, rtol=0.0
    )


def test_martingale_normalization():
    p = HullWhiteHybridParams("kou", _KOU, **_HW)
    phi_minus_i = hw_hybrid_cf(np.array([-1j]), _FWD, p)
    np.testing.assert_allclose(phi_minus_i, [1.0 + 0.0j], atol=1e-12)


def test_cumulant_additivity():
    p = HullWhiteHybridParams("kou", _KOU, **_HW)
    v_p = hw_bond_variance(_HW["mean_reversion"], _HW["sigma_r"], _FWD.T)
    c1_h, c2_h, c4_h = fe.hw_hybrid_cumulants(_FWD, p)
    c1_b, c2_b, c4_b = fe.kou_cumulants(_FWD, _KOU)
    assert c1_h == pytest.approx(c1_b - 0.5 * v_p, abs=1e-15)
    assert c2_h == pytest.approx(c2_b + v_p, abs=1e-15)
    assert c4_h == pytest.approx(c4_b, abs=1e-15)


def test_rate_vol_increases_option_value():
    base = fe.price_strip("kou", "cos", _STRIKES, _FWD, _KOU)
    p = HullWhiteHybridParams("kou", _KOU, mean_reversion=0.05, sigma_r=0.03)
    hybrid = fe.price_strip("hw_hybrid", "cos", _STRIKES, _FWD, p)
    assert np.all(hybrid > base)


def test_cross_engine_and_parity():
    p = HullWhiteHybridParams("kou", _KOU, **_HW)
    cos = fe.price_strip("hw_hybrid", "cos", _STRIKES, _FWD, p)
    hil = fe.price_strip("hw_hybrid", "hilbert", _STRIKES, _FWD, p)
    np.testing.assert_allclose(cos, hil, atol=2e-6, rtol=0.0)
    puts = fe.price_strip("hw_hybrid", "cos", _STRIKES, _FWD, p, cp=-1)
    np.testing.assert_allclose(cos - puts, _FWD.disc * (_FWD.F0 - _STRIKES), atol=1e-8)


def test_kou_base_matches_monte_carlo():
    """Exact simulation: Kou increments + one independent Gaussian for the
    rate integral -- both legs are drawn from their exact laws."""
    rng = np.random.default_rng(29)
    n = 250_000
    sig, lam, p_up, e1, e2 = _KOU.sigma, _KOU.lam, _KOU.p, _KOU.eta1, _KOU.eta2
    zeta = p_up * e1 / (e1 - 1.0) + (1.0 - p_up) * e2 / (e2 + 1.0) - 1.0
    T = _FWD.T

    y = (-0.5 * sig * sig - lam * zeta) * T + sig * np.sqrt(T) * rng.standard_normal(n)
    n_jumps = rng.poisson(lam * T, size=n)
    for j in range(1, int(n_jumps.max()) + 1):
        mask = n_jumps >= j
        n_active = int(mask.sum())
        up = rng.random(n_active) < p_up
        jump = np.where(
            up,
            rng.exponential(1.0 / e1, size=n_active),
            -rng.exponential(1.0 / e2, size=n_active),
        )
        y[mask] += jump

    v_p = hw_bond_variance(_HW["mean_reversion"], _HW["sigma_r"], T)
    z = -0.5 * v_p + np.sqrt(v_p) * rng.standard_normal(n)
    s_t = _FWD.F0 * np.exp(y + z)

    K = 100.0
    payoff = _FWD.disc * np.maximum(s_t - K, 0.0)
    mc, se = float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n))
    p = HullWhiteHybridParams("kou", _KOU, **_HW)
    cf_price = float(fe.price_strip("hw_hybrid", "cos", np.array([K]), _FWD, p)[0])
    assert abs(cf_price - mc) < 4.0 * se, f"CF {cf_price:.4f} vs MC {mc:.4f} +/- {se:.4f}"


def test_heston_base_works_through_pipeline():
    """Independence argument is model-agnostic: a Heston base must price and
    stay above the pure-Heston value."""
    heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.35, rho=-0.6, v0=0.04)
    p = HullWhiteHybridParams("heston", heston, mean_reversion=0.1, sigma_r=0.02)
    hybrid = fe.price_strip("hw_hybrid", "cos", _STRIKES, _FWD, p)
    base = fe.price_strip("heston", "cos", _STRIKES, _FWD, heston)
    assert np.all(hybrid > base)


def test_param_validation():
    with pytest.raises(ValueError):
        HullWhiteHybridParams("hw_hybrid", _KOU, 0.1, 0.01)
    with pytest.raises(TypeError):
        HullWhiteHybridParams("kou", {"sigma": 0.2}, 0.1, 0.01)
    with pytest.raises(ValueError):
        HullWhiteHybridParams("kou", _KOU, -0.1, 0.01)
    with pytest.raises(ValueError):
        HullWhiteHybridParams("kou", _KOU, 0.1, -0.01)
    bad = HullWhiteHybridParams("no_such_model", _KOU, 0.1, 0.01)
    with pytest.raises(ValueError):
        hw_hybrid_cf(_U, _FWD, bad)
