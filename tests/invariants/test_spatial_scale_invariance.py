"""Spatial scale (linear homogeneity) invariance — Buckingham-pi Phase 3.4.

For any option pricing model, scaling both spot and strike by λ must scale
the call price by λ:

    C(λ·S₀, λ·K, r, q, T, params) = λ · C(S₀, K, r, q, T, params)

This follows from the linearity of the payoff max(S_T - K, 0) in (S_T, K)
and the fact that the log-return distribution is scale-invariant.

Models tested: BSM, Heston, VG, CGMY, NIG, Kou, MertonJD, Bates,
               HestonKou, HestonCGMY, DoubleHeston, VGSA  (12 total)
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe


# ── model fixture catalogue ────────────────────────────────────────────────

ALL_MODELS = [
    ("bsm",          fe.BsmParams(sigma=0.20)),
    ("heston",       fe.HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04)),
    ("vg",           fe.VGParams(sigma=0.12, nu=0.3, theta=-0.14)),
    ("cgmy",         fe.CgmyParams(C=1.0, G=5.0, M=5.0, Y=1.5)),
    ("nig",          fe.NigParams(sigma=0.15, nu=0.3, theta=-0.14)),
    ("kou",          fe.KouParams(sigma=0.15, lam=0.5, p=0.4, eta1=10.0, eta2=5.0)),
    ("merton_jd",    fe.MertonJDParams(sigma=0.15, lam=0.5, muj=-0.1, sigj=0.10)),
    ("bates",        fe.BatesParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04,
                                    lam_j=0.5, mu_j=-0.1, sigma_j=0.1)),
    ("heston_kou",   fe.HestonKouParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04,
                                         lam_j=0.5, p_j=0.4, eta1=10.0, eta2=5.0)),
    ("heston_cgmy",  fe.HestonCGMYParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.5, v0=0.04,
                                          C=0.5, G=5.0, M=5.0, Y=1.5)),
    ("double_heston", fe.DoubleHestonParams(kappa1=2.0, theta1=0.02, nu1=0.3, rho1=-0.4, v01=0.02,
                                             kappa2=1.0, theta2=0.02, nu2=0.2, rho2=-0.3, v02=0.02)),
    ("vgsa",         fe.VGSAParams(C=1.0, G=5.0, M=5.0, kappa=3.0, eta=1.0, lam=1.0)),
]

_BASE_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
_SCALES = [0.5, 2.0, 5.0]


# ── helpers ────────────────────────────────────────────────────────────────

def _price(model: str, params, fwd: fe.ForwardSpec, K: float) -> float:
    return float(fe.price_strip(model, "cos_improved", [K], fwd, params)[0])


# ── ATM scale invariance ───────────────────────────────────────────────────

@pytest.mark.parametrize("model,params", ALL_MODELS)
@pytest.mark.parametrize("lam", _SCALES)
def test_atm_spatial_scale(model, params, lam):
    """C(λS₀, λK, r, q, T) = λ · C(S₀, K, r, q, T) at-the-money."""
    K0 = _BASE_FWD.S0
    fwd_scaled = fe.ForwardSpec(S0=lam * _BASE_FWD.S0, r=_BASE_FWD.r,
                                q=_BASE_FWD.q, T=_BASE_FWD.T)
    p_base = _price(model, params, _BASE_FWD, K0)
    p_scaled = _price(model, params, fwd_scaled, lam * K0)
    tol = max(1e-4 * lam * p_base, 1e-8)
    assert abs(p_scaled - lam * p_base) < tol, (
        f"{model} lam={lam}: {p_scaled:.8f} != {lam:.1f} * {p_base:.8f}"
    )


# ── OTM / ITM scale invariance ─────────────────────────────────────────────

@pytest.mark.parametrize("model,params", ALL_MODELS)
@pytest.mark.parametrize("K0", [85.0, 110.0])
def test_otm_itm_spatial_scale(model, params, K0):
    """Scale invariance holds for OTM and ITM strikes too (λ=2)."""
    lam = 2.0
    fwd_scaled = fe.ForwardSpec(S0=lam * _BASE_FWD.S0, r=_BASE_FWD.r,
                                q=_BASE_FWD.q, T=_BASE_FWD.T)
    p_base = _price(model, params, _BASE_FWD, K0)
    p_scaled = _price(model, params, fwd_scaled, lam * K0)
    tol = max(1e-4 * lam * p_base, 1e-8)
    assert abs(p_scaled - lam * p_base) < tol, (
        f"{model} K={K0} lam={lam}: {p_scaled:.8f} != {lam:.1f} * {p_base:.8f}"
    )


# ── strip-level scale invariance ──────────────────────────────────────────

@pytest.mark.parametrize("model,params", ALL_MODELS)
def test_strip_spatial_scale(model, params):
    """Vectorised price_strip respects scale invariance across a strike strip."""
    strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
    lam = 3.0
    fwd_scaled = fe.ForwardSpec(S0=lam * _BASE_FWD.S0, r=_BASE_FWD.r,
                                q=_BASE_FWD.q, T=_BASE_FWD.T)
    p_base = fe.price_strip(model, "cos_improved", strikes, _BASE_FWD, params)
    p_scaled = fe.price_strip(model, "cos_improved", lam * strikes, fwd_scaled, params)
    np.testing.assert_allclose(
        p_scaled, lam * p_base, rtol=1e-4,
        err_msg=f"{model}: strip scale invariance failed"
    )
