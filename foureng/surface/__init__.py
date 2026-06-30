"""Implied-vol surface construction and model calibration.

model_price_surface and model_iv_surface compute (nT, nK) grids over a
SurfaceSpec using the COS pricer plus safeguarded Newton IV inversion.
calibrate_heston, calibrate_vg, and calibrate_kou fit model parameters
to a grid of market implied vols via Nelder-Mead minimisation on a
sum-of-squared-IV-residuals objective.
"""

from .calibration import (
    CalibrationResult,
    SabrSmileCalibResult,
    calibrate_cgmy,
    calibrate_heston,
    calibrate_kou,
    calibrate_nig,
    calibrate_sabr_smile,
    calibrate_vg,
)
from .svi import SVIFitResult, SVIParams, fit_svi_smile, svi_butterfly_density
from .svi import svi_check_butterfly_arbitrage, svi_implied_vol, svi_total_variance
from .vol_surface import SurfaceSpec, model_iv_surface, model_price_surface

__all__ = [
    "SurfaceSpec",
    "model_iv_surface",
    "model_price_surface",
    "CalibrationResult",
    "SabrSmileCalibResult",
    "calibrate_heston",
    "calibrate_vg",
    "calibrate_kou",
    "calibrate_cgmy",
    "calibrate_nig",
    "calibrate_sabr_smile",
    "SVIParams",
    "SVIFitResult",
    "svi_total_variance",
    "svi_implied_vol",
    "svi_butterfly_density",
    "svi_check_butterfly_arbitrage",
    "fit_svi_smile",
]
