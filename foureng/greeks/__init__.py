"""Fourier-method Greeks via analytical differentiation of the COS expansion.

cos_price_and_greeks computes price, Delta, and Gamma in one COS sweep by
differentiating the put payoff coefficients analytically with respect to F_0.
cos_parameter_sensitivity computes dC/dtheta for any scalar model parameter
by replacing phi with dphi/dtheta in the same COS sum (useful for Vega-style
Greeks when the derivative of the CF is available in closed form).
"""

from .cos_greeks import (
    COSGreeks,
    cos_delta_gamma,
    cos_parameter_sensitivity,
    cos_price_and_greeks,
)

__all__ = [
    "COSGreeks",
    "cos_delta_gamma",
    "cos_price_and_greeks",
    "cos_parameter_sensitivity",
]
