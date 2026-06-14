"""Snapshot tests for the public API surface.

These tests lock in the set of names exported from the top-level `foureng`
package. Adding new public names is fine; removing existing ones requires an
explicit snapshot update here.
"""

from __future__ import annotations

import foureng

# Public API captured against the current release (2026-06-10).
# fmt: off
_BASELINE_API = {
    # version
    "__version__",
    # base types
    "BSInputs", "CharFunc", "ForwardSpec", "ModelSpec",
    # model parameter dataclasses
    "BatesParams", "BilateralGammaParams", "BsmParams", "CgmyParams",
    "DoubleHestonParams", "FMLSParams", "GHParams", "GarchWMW2012Params",
    "HestonCGMYParams", "HestonKouParams", "HestonParams", "KouParams",
    "MeixnerParams", "MertonJDParams", "NigParams", "OusvParams",
    "RoughHestonParams", "SabrParams", "Sv32Params", "VGParams",
    "VGSAParams",
    # characteristic functions
    "bates_cf", "bilateral_gamma_cf", "bsm_cf", "cgmy_cf",
    "double_heston_cf", "fmls_cf", "garch_wmw2012_cf", "gh_cf",
    "heston_cf_form2", "heston_cgmy_cf", "heston_kou_cf", "kou_cf",
    "meixner_cf", "merton_jd_cf", "nig_cf", "ousv_cf",
    "rough_heston_cf", "sv32_cf", "vg_cf", "vgsa_cf",
    # cumulant functions
    "bates_cumulants", "bilateral_gamma_cumulants", "bsm_cumulants",
    "cgmy_cumulants", "double_heston_cumulants", "fmls_cumulants",
    "garch_wmw2012_cumulants", "gh_cumulants", "heston_cgmy_cumulants",
    "heston_cumulants", "heston_kou_cumulants", "kou_cumulants",
    "meixner_cumulants", "merton_jd_cumulants", "nig_cumulants",
    "ousv_cumulants", "rough_heston_cumulants", "sv32_cumulants",
    "vg_cumulants", "vgsa_cumulants",
    # misc model helpers
    "heston_riccati_cd", "sabr_hagan_implied_vol",
    # pipeline dispatchers
    "price", "price_strip",
    # grid objects
    "CONVGrid", "COSGrid", "COSGridPolicy", "FFTGrid", "FRFTGrid",
    "LatticeGrid", "PDEGrid", "ProjGrid",
    # grid builders
    "cos_auto_grid", "cos_improved_grid", "proj_auto_grid",
    "recommended_cos_policy",
    # COS pricing
    "COSPolicyDecision", "COSResult", "cos_adaptive_decision",
    "cos_bermudan_price", "cos_bermudan_price_strip",
    "cos_digital_price", "cos_digital_price_strip",
    "cos_prices",
    # Fourier pricers
    "carr_madan_fft_prices", "carr_madan_price_at_strikes",
    "conv_price_at_strikes", "filtered_cos_prices",
    "frft_price_at_strikes", "frft_prices",
    "lewis_call_prices", "lewis_prices",
    "mellin_price_at_strikes",
    # PROJ pricers
    "proj_bermudan_put", "proj_european_price_at_strikes",
    "proj_price_at_strikes",
    # BSM lattice / PDE
    "bsm_lattice_price", "bsm_lattice_price_at_strikes",
    "bsm_pde_fd_price", "bsm_pde_fd_price_at_strikes",
    # SABR pricer
    "sabr_hagan_price_at_strikes",
    # filtered COS helpers
    "COSFilterSpec", "FilteredCOSDecision", "cos_filter_weights",
    # implied volatility
    "bs_price_from_fwd", "implied_vol_brent",
    "implied_vol_newton_safeguarded",
    # surface and calibration
    "CalibrationResult", "SurfaceSpec",
    "calibrate_cgmy", "calibrate_heston", "calibrate_kou",
    "calibrate_nig", "calibrate_vg",
    "model_iv_surface", "model_price_surface",
    # Greeks
    "COSGreeks", "cos_delta_gamma", "cos_parameter_sensitivity",
    "cos_price_and_greeks",
    # Monte Carlo
    "CVResult", "HestonMCScheme", "MCResult", "MCSpec",
    "bs_call_cv", "bs_price_from_fwd", "european_call_mc",
    "heston_call_bs_control", "heston_conditional_mc_calls",
    "mc_price",
    # BSM analytics
    "bsm_asset_or_nothing", "bsm_barrier_price", "bsm_cash_or_nothing",
    "bsm_discrete_geometric_asian", "bsm_forward_start", "bsm_gap_call",
    "bsm_geometric_asian", "bsm_geometric_asian_parity",
    "bsm_lookback_floating", "bsm_variance_option_integrated",
    "bsm_variance_swap",
    # multi-asset analytics
    "kirk_spread", "margrabe_exchange",
}
# fmt: on


def _public(module) -> set[str]:
    return {name for name in dir(module) if not name.startswith("_")}


def test_no_baseline_names_removed():
    """No name from the baseline was silently removed."""
    current = _public(foureng)
    missing = _BASELINE_API - current
    assert not missing, f"Baseline API names removed: {sorted(missing)}"


def test_public_api_is_not_empty():
    assert len(_public(foureng)) > 50


def test_price_strip_callable():
    assert callable(foureng.price_strip)


def test_price_callable():
    assert callable(foureng.price)


def test_forward_spec_constructable():
    fwd = foureng.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    assert fwd.S0 == 100.0
