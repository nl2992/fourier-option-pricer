"""Snapshot tests — the public API surface must not silently shrink.

These tests lock in the set of names exported from the top-level `foureng`
package at the v0.5.1 baseline.  Adding new public names is fine; removing
existing ones requires an explicit snapshot update here.
"""

from __future__ import annotations

import foureng

# Baseline public API captured at v0.5.1-baseline (2026-05-29).
# fmt: off
_BASELINE_API = {
    "BSInputs", "BatesParams", "BilateralGammaParams", "BsmParams",
    "COSFilterSpec", "COSGreeks", "COSGrid", "COSGridPolicy",
    "COSPolicyDecision", "COSResult", "CVResult", "CalibrationResult",
    "CgmyParams", "CharFunc", "DoubleHestonParams", "FFTGrid",
    "FMLSParams", "FRFTGrid", "FilteredCOSDecision", "ForwardSpec",
    "GHParams", "GarchWMW2012Params", "HestonCGMYParams", "HestonKouParams",
    "HestonMCScheme", "HestonParams", "KouParams", "MCSpec", "MeixnerParams",
    "MertonJDParams", "ModelSpec", "NigParams", "OusvParams",
    "RoughHestonParams", "SurfaceSpec", "Sv32Params", "VGParams",
    "VGSAParams",
    "bates_cf", "bates_cumulants", "bilateral_gamma_cf",
    "bilateral_gamma_cumulants", "bs_call_cv", "bs_price_from_fwd",
    "bsm_cf", "bsm_cumulants", "calibrate_heston", "calibrate_kou",
    "calibrate_vg", "carr_madan_fft_prices", "carr_madan_price_at_strikes",
    "cgmy_cf", "cgmy_cumulants", "cos_adaptive_decision", "cos_auto_grid",
    "cos_delta_gamma", "cos_filter_weights", "cos_improved_grid",
    "cos_parameter_sensitivity", "cos_price_and_greeks", "cos_prices",
    "double_heston_cf", "double_heston_cumulants", "european_call_mc",
    "filtered_cos_prices", "fmls_cf", "fmls_cumulants",
    "frft_price_at_strikes", "frft_prices", "garch_wmw2012_cf",
    "garch_wmw2012_cumulants", "gh_cf", "gh_cumulants",
    "heston_call_bs_control", "heston_cf_form2", "heston_cgmy_cf",
    "heston_cgmy_cumulants", "heston_conditional_mc_calls", "heston_cumulants",
    "heston_kou_cf", "heston_kou_cumulants", "heston_riccati_cd",
    "implied_vol_brent", "implied_vol_newton_safeguarded", "kou_cf",
    "kou_cumulants", "lewis_call_prices", "lewis_prices", "meixner_cf",
    "meixner_cumulants", "merton_jd_cf", "merton_jd_cumulants",
    "model_iv_surface", "model_price_surface", "nig_cf", "nig_cumulants",
    "ousv_cf", "ousv_cumulants", "price_strip", "recommended_cos_policy",
    "rough_heston_cf", "rough_heston_cumulants", "sv32_cf", "sv32_cumulants",
    "vg_cf", "vg_cumulants", "vgsa_cf", "vgsa_cumulants",
}
# fmt: on


def _public(module) -> set[str]:
    return {name for name in dir(module) if not name.startswith("_")}


def test_no_baseline_names_removed():
    """Nothing from the v0.5.1 baseline was silently deleted."""
    current = _public(foureng)
    missing = _BASELINE_API - current
    assert not missing, f"Baseline API names removed: {sorted(missing)}"


def test_public_api_is_not_empty():
    assert len(_public(foureng)) > 50


def test_price_strip_callable():
    assert callable(foureng.price_strip)


def test_forward_spec_constructable():
    fwd = foureng.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    assert fwd.S0 == 100.0
