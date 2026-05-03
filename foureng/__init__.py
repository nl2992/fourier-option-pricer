"""fourier-option-pricer: Fourier methods for European option pricing.

Public API (stable surface for external use):

    from foureng import (
        ForwardSpec, HestonParams, VGParams, KouParams,
        heston_cf_form2, vg_cf, kou_cf,
        heston_cumulants, vg_cumulants, kou_cumulants,
        cos_prices, cos_auto_grid, cos_improved_grid,
        carr_madan_price_at_strikes, frft_price_at_strikes,
        COSGrid, COSGridPolicy, FFTGrid, FRFTGrid,
        implied_vol_newton_safeguarded, BSInputs, bs_price_from_fwd,
        SurfaceSpec, model_iv_surface, model_price_surface,
        calibrate_heston, calibrate_vg, calibrate_kou,
        cos_price_and_greeks, cos_delta_gamma, cos_parameter_sensitivity,
        bs_call_cv, heston_call_bs_control,
    )

Submodules (``foureng.pricers``, ``foureng.models``, ``foureng.mc``,
``foureng.iv``, ``foureng.surface``, ``foureng.greeks``, ``foureng.utils``)
remain importable for finer-grained access. ``foureng.models`` is the
canonical location of the characteristic-function layer — this used to
live at ``foureng.char_func`` before the Pass-1 PyFENG-compat rename.
"""
from __future__ import annotations

__version__ = "0.3.1"

from .models.base import ForwardSpec, CharFunc, ModelSpec
from .models.heston import HestonParams, heston_cf_form2, heston_cumulants
from .models.variance_gamma import VGParams, vg_cf, vg_cumulants
from .models.kou import KouParams, kou_cf, kou_cumulants
from .models.sv32 import Sv32Params, sv32_cf, sv32_cumulants
from .models.garch_wmw2012 import GarchWMW2012Params, garch_wmw2012_cf, garch_wmw2012_cumulants
from .models.rough_heston import RoughHestonParams, rough_heston_cf, rough_heston_cumulants
from .models.merton_jd import MertonJDParams, merton_jd_cf, merton_jd_cumulants
from .models.meixner import MeixnerParams, meixner_cf, meixner_cumulants
from .models.bilateral_gamma import BilateralGammaParams, bilateral_gamma_cf, bilateral_gamma_cumulants
from .models.generalized_hyperbolic import GHParams, gh_cf, gh_cumulants
from .models.fmls import FMLSParams, fmls_cf, fmls_cumulants

from .utils.grids import COSGrid, COSGridPolicy, FFTGrid, FRFTGrid

from .pricers.cos import (
    COSPolicyDecision,
    COSResult,
    cos_adaptive_decision,
    cos_auto_grid,
    cos_improved_grid,
    cos_prices,
    recommended_cos_policy,
)
from .pricers.carr_madan import carr_madan_price_at_strikes, carr_madan_fft_prices
from .pricers.frft import frft_price_at_strikes, frft_prices
from .pricers.filtered_cos import FilteredCOSDecision, filtered_cos_prices
from .utils.spectral_filters import COSFilterSpec, cos_filter_weights

from .iv.implied_vol import (
    BSInputs,
    bs_price_from_fwd,
    implied_vol_brent,
    implied_vol_newton_safeguarded,
)

from .surface import (
    SurfaceSpec,
    model_iv_surface,
    model_price_surface,
    CalibrationResult,
    calibrate_heston,
    calibrate_vg,
    calibrate_kou,
)

from .greeks import (
    COSGreeks,
    cos_delta_gamma,
    cos_price_and_greeks,
    cos_parameter_sensitivity,
)

from .mc.black_scholes_mc import european_call_mc, MCSpec
from .mc.heston_conditional_mc import heston_conditional_mc_calls, HestonMCScheme
from .mc.control_variate import bs_call_cv, heston_call_bs_control, CVResult

__all__ = [
    "__version__",
    # char funcs
    "ForwardSpec", "CharFunc", "ModelSpec",
    "HestonParams", "heston_cf_form2", "heston_cumulants",
    "VGParams", "vg_cf", "vg_cumulants",
    "KouParams", "kou_cf", "kou_cumulants",
    "Sv32Params", "sv32_cf", "sv32_cumulants",
    "GarchWMW2012Params", "garch_wmw2012_cf", "garch_wmw2012_cumulants",
    "RoughHestonParams", "rough_heston_cf", "rough_heston_cumulants",
    "MertonJDParams", "merton_jd_cf", "merton_jd_cumulants",
    "MeixnerParams", "meixner_cf", "meixner_cumulants",
    "BilateralGammaParams", "bilateral_gamma_cf", "bilateral_gamma_cumulants",
    "GHParams", "gh_cf", "gh_cumulants",
    "FMLSParams", "fmls_cf", "fmls_cumulants",
    # grids
    "COSGrid", "COSGridPolicy", "FFTGrid", "FRFTGrid",
    # pricers
    "cos_prices", "cos_auto_grid", "cos_improved_grid",
    "recommended_cos_policy", "cos_adaptive_decision",
    "COSResult", "COSPolicyDecision",
    "carr_madan_price_at_strikes", "carr_madan_fft_prices",
    "frft_price_at_strikes", "frft_prices",
    # filtered COS extension
    "COSFilterSpec", "cos_filter_weights",
    "filtered_cos_prices", "FilteredCOSDecision",
    # iv
    "BSInputs", "bs_price_from_fwd",
    "implied_vol_brent", "implied_vol_newton_safeguarded",
    # surface + calibration
    "SurfaceSpec", "model_iv_surface", "model_price_surface",
    "CalibrationResult", "calibrate_heston", "calibrate_vg", "calibrate_kou",
    # greeks
    "COSGreeks", "cos_delta_gamma", "cos_price_and_greeks", "cos_parameter_sensitivity",
    # mc
    "european_call_mc", "MCSpec",
    "heston_conditional_mc_calls", "HestonMCScheme",
    "bs_call_cv", "heston_call_bs_control", "CVResult",
]
