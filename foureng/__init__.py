"""fourier-option-pricer: Fourier methods for European option pricing.

All 20 model dataclasses, characteristic functions, and cumulants are
available directly from this top-level package:

    import foureng as fe

    fwd    = fe.ForwardSpec(S0=100, r=0.01, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)
    phi    = lambda u: fe.heston_cf_form2(u, fwd, params)
    grid   = fe.cos_auto_grid(fe.heston_cumulants(fwd, params), N=256, L=10.0)
    result = fe.cos_prices(phi, fwd, [90, 100, 110], grid)

    # or use the unified dispatcher
    prices = fe.price_strip("heston", "cos", [90, 100, 110], fwd, params)

Submodules ``foureng.models``, ``foureng.pricers``, ``foureng.mc``,
``foureng.iv``, ``foureng.surface``, and ``foureng.greeks`` are also
importable for finer-grained access.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError as _PNF
    from importlib.metadata import version as _pkg_version

    __version__: str = _pkg_version("fourier-option-pricer")
except _PNF:  # editable install without metadata yet
    __version__ = "0.4.1"

from .analytics.bsm_asian import (
    bsm_discrete_geometric_asian,
    bsm_geometric_asian,
    bsm_geometric_asian_parity,
)
from .analytics.bsm_barrier import bsm_barrier_price
from .analytics.bsm_exotics import (
    bsm_forward_start,
    bsm_gap_call,
    bsm_lookback_floating,
    margrabe_exchange,
)
from .analytics.bsm_variance import bsm_variance_option_integrated, bsm_variance_swap
from .greeks import (
    COSGreeks,
    cos_delta_gamma,
    cos_parameter_sensitivity,
    cos_price_and_greeks,
)
from .iv.implied_vol import (
    BSInputs,
    bs_price_from_fwd,
    implied_vol_brent,
    implied_vol_newton_safeguarded,
)
from .mc.black_scholes_mc import european_call_mc
from .mc.control_variate import CVResult, bs_call_cv, heston_call_bs_control
from .mc.engine import MCResult, MCSpec, mc_price
from .mc.heston_conditional_mc import HestonMCScheme, heston_conditional_mc_calls
from .models.base import CharFunc, ForwardSpec, ModelSpec
from .models.bates import BatesParams, bates_cf, bates_cumulants
from .models.bilateral_gamma import (
    BilateralGammaParams,
    bilateral_gamma_cf,
    bilateral_gamma_cumulants,
)
from .models.bsm import (
    BsmParams,
    bsm_asset_or_nothing,
    bsm_cash_or_nothing,
    bsm_cf,
    bsm_cumulants,
)
from .models.cgmy import CgmyParams, cgmy_cf, cgmy_cumulants
from .models.double_heston import DoubleHestonParams, double_heston_cf, double_heston_cumulants
from .models.fmls import FMLSParams, fmls_cf, fmls_cumulants
from .models.garch_wmw2012 import GarchWMW2012Params, garch_wmw2012_cf, garch_wmw2012_cumulants
from .models.generalized_hyperbolic import GHParams, gh_cf, gh_cumulants
from .models.heston import HestonParams, heston_cf_form2, heston_cumulants, heston_riccati_cd
from .models.heston_cgmy import HestonCGMYParams, heston_cgmy_cf, heston_cgmy_cumulants
from .models.heston_kou import HestonKouParams, heston_kou_cf, heston_kou_cumulants
from .models.kou import KouParams, kou_cf, kou_cumulants
from .models.meixner import MeixnerParams, meixner_cf, meixner_cumulants
from .models.merton_jd import MertonJDParams, merton_jd_cf, merton_jd_cumulants
from .models.nig import NigParams, nig_cf, nig_cumulants
from .models.ousv import OusvParams, ousv_cf, ousv_cumulants
from .models.rough_heston import RoughHestonParams, rough_heston_cf, rough_heston_cumulants
from .models.sabr import SabrParams, sabr_hagan_implied_vol
from .models.sv32 import Sv32Params, sv32_cf, sv32_cumulants
from .models.variance_gamma import VGParams, vg_cf, vg_cumulants
from .models.vgsa import VGSAParams, vgsa_cf, vgsa_cumulants
from .pipeline import price, price_strip
from .pricers.carr_madan import carr_madan_fft_prices, carr_madan_price_at_strikes
from .pricers.conv import conv_price_at_strikes
from .pricers.cos import (
    COSPolicyDecision,
    COSResult,
    cos_adaptive_decision,
    cos_auto_grid,
    cos_improved_grid,
    cos_prices,
    recommended_cos_policy,
)
from .pricers.cos_bermudan import cos_bermudan_price, cos_bermudan_price_strip
from .pricers.cos_digital import cos_digital_price, cos_digital_price_strip
from .pricers.filtered_cos import FilteredCOSDecision, filtered_cos_prices
from .pricers.frft import frft_price_at_strikes, frft_prices
from .pricers.lattice import LatticeGrid, bsm_lattice_price, bsm_lattice_price_at_strikes
from .pricers.lewis import lewis_call_prices, lewis_prices
from .pricers.mellin import mellin_price_at_strikes
from .pricers.pde_fd import PDEGrid, bsm_pde_fd_price, bsm_pde_fd_price_at_strikes
from .pricers.proj import proj_european_price_at_strikes
from .pricers.sabr import sabr_hagan_price_at_strikes
from .surface import (
    CalibrationResult,
    SurfaceSpec,
    calibrate_cgmy,
    calibrate_heston,
    calibrate_kou,
    calibrate_nig,
    calibrate_vg,
    model_iv_surface,
    model_price_surface,
)
from .utils.grids import CONVGrid, COSGrid, COSGridPolicy, FFTGrid, FRFTGrid
from .utils.spectral_filters import COSFilterSpec, cos_filter_weights

__all__ = [
    "__version__",
    # base
    "ForwardSpec",
    "CharFunc",
    "ModelSpec",
    # models  -  params, CF, cumulants (all 20)
    "BsmParams",
    "bsm_cf",
    "bsm_cumulants",
    "HestonParams",
    "heston_cf_form2",
    "heston_cumulants",
    "OusvParams",
    "ousv_cf",
    "ousv_cumulants",
    "VGParams",
    "vg_cf",
    "vg_cumulants",
    "CgmyParams",
    "cgmy_cf",
    "cgmy_cumulants",
    "NigParams",
    "nig_cf",
    "nig_cumulants",
    "KouParams",
    "kou_cf",
    "kou_cumulants",
    "BatesParams",
    "bates_cf",
    "bates_cumulants",
    "HestonKouParams",
    "heston_kou_cf",
    "heston_kou_cumulants",
    "HestonCGMYParams",
    "heston_cgmy_cf",
    "heston_cgmy_cumulants",
    "Sv32Params",
    "sv32_cf",
    "sv32_cumulants",
    "GarchWMW2012Params",
    "garch_wmw2012_cf",
    "garch_wmw2012_cumulants",
    "RoughHestonParams",
    "rough_heston_cf",
    "rough_heston_cumulants",
    "MertonJDParams",
    "merton_jd_cf",
    "merton_jd_cumulants",
    "MeixnerParams",
    "meixner_cf",
    "meixner_cumulants",
    "BilateralGammaParams",
    "bilateral_gamma_cf",
    "bilateral_gamma_cumulants",
    "GHParams",
    "gh_cf",
    "gh_cumulants",
    "FMLSParams",
    "fmls_cf",
    "fmls_cumulants",
    "DoubleHestonParams",
    "double_heston_cf",
    "double_heston_cumulants",
    "heston_riccati_cd",
    "VGSAParams",
    "vgsa_cf",
    "vgsa_cumulants",
    "SabrParams",
    "sabr_hagan_implied_vol",
    "bsm_variance_swap",
    "bsm_variance_option_integrated",
    # pipeline
    "price",
    "price_strip",
    # grids
    "COSGrid",
    "COSGridPolicy",
    "FFTGrid",
    "FRFTGrid",
    "CONVGrid",
    "LatticeGrid",
    "PDEGrid",
    # pricers
    "cos_prices",
    "cos_auto_grid",
    "cos_improved_grid",
    "recommended_cos_policy",
    "cos_adaptive_decision",
    "COSResult",
    "COSPolicyDecision",
    "carr_madan_price_at_strikes",
    "carr_madan_fft_prices",
    "conv_price_at_strikes",
    "cos_bermudan_price",
    "cos_bermudan_price_strip",
    "frft_price_at_strikes",
    "frft_prices",
    "lewis_call_prices",
    "lewis_prices",
    "mellin_price_at_strikes",
    "proj_european_price_at_strikes",
    "sabr_hagan_price_at_strikes",
    "bsm_lattice_price",
    "bsm_lattice_price_at_strikes",
    "bsm_pde_fd_price",
    "bsm_pde_fd_price_at_strikes",
    # filtered COS extension
    "COSFilterSpec",
    "cos_filter_weights",
    "filtered_cos_prices",
    "FilteredCOSDecision",
    # iv
    "BSInputs",
    "bs_price_from_fwd",
    "implied_vol_brent",
    "implied_vol_newton_safeguarded",
    # surface + calibration
    "SurfaceSpec",
    "model_iv_surface",
    "model_price_surface",
    "CalibrationResult",
    "calibrate_heston",
    "calibrate_cgmy",
    "calibrate_vg",
    "calibrate_kou",
    "calibrate_nig",
    # greeks
    "COSGreeks",
    "cos_delta_gamma",
    "cos_price_and_greeks",
    "cos_parameter_sensitivity",
    # mc
    "european_call_mc",
    "MCSpec",
    "MCResult",
    "mc_price",
    "heston_conditional_mc_calls",
    "HestonMCScheme",
    "bs_call_cv",
    "heston_call_bs_control",
    "CVResult",
    # analytics
    "bsm_discrete_geometric_asian",
    "bsm_geometric_asian",
    "bsm_geometric_asian_parity",
    "bsm_barrier_price",
    "bsm_forward_start",
    "bsm_gap_call",
    "bsm_lookback_floating",
    "margrabe_exchange",
    # digital
    "bsm_asset_or_nothing",
    "bsm_cash_or_nothing",
    "cos_digital_price",
    "cos_digital_price_strip",
]
