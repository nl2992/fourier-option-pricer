"""Tests for the five new models added in the batch: heston_vg, svjj,
bns_gamma_ou, nts, cgmysa.

Each model gets:
  1. CF structure checks  (phi(0)=1, phi(-i)=1, |phi|<=1, conjugate symmetry)
  2. Cross-engine consistency (COS vs Carr-Madan, atol=1e-3)
  3. Structural / no-arbitrage (positive, monotone, PCP)
  4. Parameter validation  (boundary violations raise ValueError)
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.bns_gamma_ou import BNSGammaOUParams, bns_gamma_ou_cf
from foureng.models.cgmysa import CGMYSAParams, cgmysa_cf
from foureng.models.heston_vg import HestonVGParams, heston_vg_cf
from foureng.models.nts import NTSParams, nts_cf
from foureng.models.svjj import SVJJParams, svjj_cf
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid

_FWD = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
_STRIKES = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
_UGRID = np.linspace(0.5, 5.0, 20)
_GRID_CM = FFTGrid(N=4096, eta=0.25, alpha=1.5)

# ---------------------------------------------------------------------------
# Canonical parameters for each model
# ---------------------------------------------------------------------------

P_HV = HestonVGParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.7, v0=0.04,
                       vg_sigma=0.15, vg_nu=0.5, vg_theta=-0.1)

P_SVJJ = SVJJParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.7, v0=0.04,
                     lam=0.5, mu_J=-0.05, sigma_J=0.1, mu_V=0.05, rho_J=0.5)

P_BNS = BNSGammaOUParams(V0=0.04, lambda_ou=2.0, rho_l=-0.5, a=2.0, b=10.0)

P_NTS = NTSParams(lam=1.0, theta=0.0, sigma=0.2, alpha=0.5)

P_CGMYSA = CGMYSAParams(C0=1.0, C=0.5, G=5.0, M=5.0, Y=0.5,
                         kappa=2.0, eta=1.0, lam=0.5)

_ALL = [
    ("heston_vg",   P_HV,     heston_vg_cf),
    ("svjj",        P_SVJJ,   svjj_cf),
    ("bns_gamma_ou",P_BNS,    bns_gamma_ou_cf),
    ("nts",         P_NTS,    nts_cf),
    ("cgmysa",      P_CGMYSA, cgmysa_cf),
]


# ===========================================================================
# 1. CF structure
# ===========================================================================

@pytest.mark.parametrize("name,params,cf_fn", _ALL)
def test_phi_at_zero(name, params, cf_fn):
    """phi(0) = 1."""
    val = cf_fn(np.array([0.0]), _FWD, params)[0]
    assert abs(val - 1.0) < 1e-10, f"{name}: phi(0)={val}"


@pytest.mark.parametrize("name,params,cf_fn", _ALL)
def test_phi_at_minus_i(name, params, cf_fn):
    """phi(-i) = 1 (log-forward martingale convention)."""
    val = cf_fn(np.array([-1j]), _FWD, params)[0]
    assert abs(val - 1.0) < 1e-6, f"{name}: phi(-i)={val:.8f}"


@pytest.mark.parametrize("name,params,cf_fn", _ALL)
def test_phi_modulus_le_one(name, params, cf_fn):
    """|phi(u)| <= 1 for all real u."""
    vals = cf_fn(_UGRID, _FWD, params)
    assert np.all(np.abs(vals) <= 1.0 + 1e-10), \
        f"{name}: |phi| exceeds 1 at some u"


@pytest.mark.parametrize("name,params,cf_fn", _ALL)
def test_conjugate_symmetry(name, params, cf_fn):
    """phi(-u) = conj(phi(u)) for real u."""
    vals_pos = cf_fn(_UGRID, _FWD, params)
    vals_neg = cf_fn(-_UGRID, _FWD, params)
    err = np.max(np.abs(vals_neg - np.conj(vals_pos)))
    assert err < 1e-10, f"{name}: conjugate symmetry max err={err:.2e}"


# ===========================================================================
# 2. Cross-engine consistency
# ===========================================================================

@pytest.mark.parametrize("name,params,cf_fn", _ALL)
def test_cos_vs_carr_madan(name, params, cf_fn):
    """COS and Carr-Madan agree to 1e-3."""
    calls_cos = price_strip(name, "cos", _STRIKES, _FWD, params)
    calls_cm = price_strip(name, "carr_madan", _STRIKES, _FWD, params,
                           grid=_GRID_CM)
    err = float(np.max(np.abs(calls_cos - calls_cm)))
    assert err < 1e-3, f"{name}: COS vs CM max|err|={err:.2e}"


# ===========================================================================
# 3. Structural / no-arbitrage properties
# ===========================================================================

@pytest.mark.parametrize("name,params,cf_fn", _ALL)
def test_calls_positive_and_monotone(name, params, cf_fn):
    """Call prices positive and strictly decreasing in strike."""
    calls = price_strip(name, "cos", _STRIKES, _FWD, params)
    assert np.all(calls > 0), f"{name}: non-positive call"
    assert np.all(np.diff(calls) < 0), f"{name}: non-monotone calls"


@pytest.mark.parametrize("name,params,cf_fn", _ALL)
def test_put_call_parity(name, params, cf_fn):
    """Puts implied by PCP are non-negative."""
    calls = price_strip(name, "cos", _STRIKES, _FWD, params)
    df = np.exp(-_FWD.r * _FWD.T)
    puts = calls - df * (_FWD.F0 - _STRIKES)
    assert np.all(puts >= -1e-6), f"{name}: negative implied put"


# ===========================================================================
# 4. Model-specific properties
# ===========================================================================

class TestHestonVGSpecific:

    def test_vg_jump_raises_smile(self):
        """Adding VG jumps to Heston increases OTM put value vs pure Heston."""
        from foureng.models.heston import HestonParams
        p_heston = HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.7, v0=0.04)
        K_otm = np.array([85.0])
        c_h = price_strip("heston", "cos", K_otm, _FWD, p_heston)[0]
        c_hv = price_strip("heston_vg", "cos", K_otm, _FWD, P_HV)[0]
        assert c_hv > c_h, "VG jumps should increase deep OTM call price"

    def test_existence_condition_rejected(self):
        """VG existence condition violation raises ValueError."""
        with pytest.raises(ValueError, match="existence condition"):
            HestonVGParams(kappa=2, theta=0.04, nu=0.3, rho=-0.5, v0=0.04,
                           vg_sigma=1.0, vg_nu=5.0, vg_theta=0.0)


class TestSVJJSpecific:

    def test_zero_lambda_recovers_heston(self):
        """SVJJ with lam=0 matches Heston prices."""
        from foureng.models.heston import HestonParams
        p_h = HestonParams(kappa=2, theta=0.04, nu=0.3, rho=-0.7, v0=0.04)
        p_svjj0 = SVJJParams(kappa=2, theta=0.04, nu=0.3, rho=-0.7, v0=0.04,
                              lam=0.0, mu_J=0.0, sigma_J=0.1, mu_V=0.0, rho_J=0.0)
        calls_h = price_strip("heston", "cos", _STRIKES, _FWD, p_h)
        calls_s = price_strip("svjj", "cos", _STRIKES, _FWD, p_svjj0)
        err = np.max(np.abs(calls_h - calls_s))
        assert err < 1e-4, f"SVJJ(lam=0) vs Heston: max|err|={err:.2e}"

    def test_higher_lambda_raises_otm(self):
        """More jump activity raises OTM call price."""
        K_otm = np.array([115.0])
        p_lo = SVJJParams(kappa=2, theta=0.04, nu=0.3, rho=-0.5, v0=0.04,
                          lam=0.1, mu_J=-0.05, sigma_J=0.1, mu_V=0.0, rho_J=0.0)
        p_hi = SVJJParams(kappa=2, theta=0.04, nu=0.3, rho=-0.5, v0=0.04,
                          lam=1.0, mu_J=-0.05, sigma_J=0.1, mu_V=0.0, rho_J=0.0)
        c_lo = price_strip("svjj", "cos", K_otm, _FWD, p_lo)[0]
        c_hi = price_strip("svjj", "cos", K_otm, _FWD, p_hi)[0]
        assert c_hi > c_lo, "Higher jump intensity should raise OTM call"

    def test_invalid_existence_condition(self):
        with pytest.raises(ValueError, match="existence condition"):
            SVJJParams(kappa=2, theta=0.04, nu=0.3, rho=-0.5, v0=0.04,
                       lam=0.5, mu_J=0.0, sigma_J=0.1, mu_V=2.0, rho_J=1.0)


class TestBNSSpecific:

    def test_higher_activity_raises_atm(self):
        """More Gamma activity (larger a) raises ATM price."""
        K_atm = np.array([_FWD.F0])
        p_lo = BNSGammaOUParams(V0=0.04, lambda_ou=2.0, rho_l=0.0, a=0.5, b=5.0)
        p_hi = BNSGammaOUParams(V0=0.04, lambda_ou=2.0, rho_l=0.0, a=4.0, b=5.0)
        c_lo = price_strip("bns_gamma_ou", "cos", K_atm, _FWD, p_lo)[0]
        c_hi = price_strip("bns_gamma_ou", "cos", K_atm, _FWD, p_hi)[0]
        assert c_hi > c_lo, "Higher Gamma activity should raise ATM price"

    def test_rho_l_ge_b_rejected(self):
        with pytest.raises(ValueError, match="rho_l must be < b"):
            BNSGammaOUParams(V0=0.04, lambda_ou=2.0, rho_l=10.0, a=1.0, b=5.0)


class TestNTSSpecific:

    def test_alpha_near_one_agrees_structurally(self):
        """NTS with alpha close to 1 (BSM-like regime) prices calls correctly."""
        p = NTSParams(lam=1.0, theta=0.0, sigma=0.2, alpha=0.9)
        calls = price_strip("nts", "cos", _STRIKES, _FWD, p)
        assert np.all(calls > 0) and np.all(np.diff(calls) < 0)

    def test_higher_lam_raises_atm(self):
        """Larger lam (more activity) raises ATM call."""
        K_atm = np.array([_FWD.F0])
        p_lo = NTSParams(lam=0.5, theta=0.0, sigma=0.2, alpha=0.5)
        p_hi = NTSParams(lam=2.0, theta=0.0, sigma=0.2, alpha=0.5)
        c_lo = price_strip("nts", "cos", K_atm, _FWD, p_lo)[0]
        c_hi = price_strip("nts", "cos", K_atm, _FWD, p_hi)[0]
        assert c_hi > c_lo, "Larger NTS lam should raise ATM call"

    def test_existence_violated_rejected(self):
        with pytest.raises(ValueError, match="existence condition"):
            NTSParams(lam=1.0, theta=-2.0, sigma=0.1, alpha=0.5)

    def test_alpha_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            NTSParams(lam=1.0, theta=0.0, sigma=0.2, alpha=1.5)


class TestCGMYSASpecific:

    def test_deterministic_clock_matches_cgmy(self):
        """CGMY-SA with lam=0 (deterministic clock at eta=1) matches CGMY."""
        from foureng.models.cgmy import CgmyParams
        p_cgmy = CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.5)
        # C0=eta=1, lam=0 → clock Y_T = T deterministically
        p_sa = CGMYSAParams(C0=1.0, C=0.5, G=5.0, M=5.0, Y=0.5,
                             kappa=1.0, eta=1.0, lam=0.0)
        calls_cgmy = price_strip("cgmy", "cos", _STRIKES, _FWD, p_cgmy)
        calls_sa = price_strip("cgmysa", "cos", _STRIKES, _FWD, p_sa)
        err = np.max(np.abs(calls_cgmy - calls_sa))
        assert err < 1e-3, f"CGMYSA(lam=0,C0=eta=1) vs CGMY: err={err:.2e}"

    def test_stochastic_clock_changes_prices(self):
        """Turning on the stochastic clock (lam>0) changes prices."""
        p_det = CGMYSAParams(C0=1.0, C=0.5, G=5.0, M=5.0, Y=0.5,
                              kappa=2.0, eta=1.0, lam=0.0)
        p_sto = CGMYSAParams(C0=1.0, C=0.5, G=5.0, M=5.0, Y=0.5,
                              kappa=2.0, eta=1.0, lam=1.0)
        calls_det = price_strip("cgmysa", "cos", _STRIKES, _FWD, p_det)
        calls_sto = price_strip("cgmysa", "cos", _STRIKES, _FWD, p_sto)
        assert not np.allclose(calls_det, calls_sto, atol=1e-6)

    def test_bad_Y_rejected(self):
        with pytest.raises(ValueError, match="Y must be in"):
            CGMYSAParams(C0=1.0, C=0.5, G=5.0, M=5.0, Y=1.0,
                         kappa=2.0, eta=1.0, lam=0.5)
