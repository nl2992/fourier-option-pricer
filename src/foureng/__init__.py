"""fourier-option-pricer: Fourier methods for European option pricing.

Public API (stable surface for external use):

    from foureng import (
        ForwardSpec, HestonParams, VGParams, KouParams,
        heston_cf_form2, vg_cf, kou_cf,
        heston_cumulants, vg_cumulants, kou_cumulants,
        cos_prices, cos_auto_grid,
        carr_madan_price_at_strikes, frft_price_at_strikes,
        COSGrid, FFTGrid, FRFTGrid,
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

__version__ = "0.2.0"

from .models.base import ForwardSpec, CharFunc, ModelSpec
from .models.heston import HestonParams, heston_cf_form2, heston_cumulants
from .models.variance_gamma import VGParams, vg_cf, vg_cumulants
from .models.kou import KouParams, kou_cf, kou_cumulants

from .utils.grids import COSGrid, FFTGrid, FRFTGrid

from .pricers.cos import cos_prices, cos_auto_grid, COSResult
from .pricers.carr_madan import carr_madan_price_at_strikes, carr_madan_fft_prices
from .pricers.frft import frft_price_at_strikes, frft_prices

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
    # grids
    "COSGrid", "FFTGrid", "FRFTGrid",
    # pricers
    "cos_prices", "cos_auto_grid", "COSResult",
    "carr_madan_price_at_strikes", "carr_madan_fft_prices",
    "frft_price_at_strikes", "frft_prices",
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
