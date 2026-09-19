"""Model restrictions for methods that only support a specific Levy family.

These methods gate on a model set defined in their own pricer module
(the L10 set in ``cos_bermudan._SUPPORTED_MODELS``, the L8 ``LEVY_*_MODELS``
sets, or ``MELLIN_SUPPORTED_MODELS``). explain_capability() must agree with
that gate: heston (a stochastic-volatility model, in none of these sets)
should be "Not supported", and vg (in every one of these sets) should be
"Supported".
"""

from __future__ import annotations

import pytest

from foureng.core.capabilities import explain_capability

# (method, product) pairs gated on cos_bermudan._SUPPORTED_MODELS (the L10 set).
_L10_METHOD_PRODUCT = [
    ("hilbert_barrier", "barrier"),
    ("hilbert_lookback", "lookback"),
    ("proj", "bermudan"),
    ("proj_barrier", "barrier"),
    ("proj_double_barrier", "double_barrier"),
    ("proj_asian", "asian"),
    ("proj_step", "step"),
    ("proj_swing", "swing"),
    ("asian_cos", "asian"),
    ("cos_bermudan", "bermudan"),
    ("cos_american", "american"),
]

# (method, product) pairs gated on the smaller L8 LEVY_*_MODELS sets.
_L8_METHOD_PRODUCT = [
    ("asian_cf", "asian"),
    ("forward_start_cf", "forward_start"),
    ("cliquet_cf", "cliquet"),
    ("fader_cf", "fader"),
    ("variance_levy_analytic", "variance_swap"),
]

# (method, product) pairs gated on MELLIN_SUPPORTED_MODELS.
_MELLIN_METHOD_PRODUCT = [("mellin", "european")]

_ALL_LEVY_ONLY = _L10_METHOD_PRODUCT + _L8_METHOD_PRODUCT + _MELLIN_METHOD_PRODUCT


@pytest.mark.parametrize("method,product", _ALL_LEVY_ONLY)
def test_heston_not_supported_for_levy_only_methods(method, product):
    result = explain_capability("heston", product, method)
    assert result.startswith("Not supported:"), (
        f"Expected Not supported for (heston, {product}, {method}), got: {result}"
    )


@pytest.mark.parametrize("method,product", _ALL_LEVY_ONLY)
def test_vg_supported_for_levy_only_methods(method, product):
    result = explain_capability("vg", product, method)
    assert result.startswith("Supported:"), (
        f"Expected Supported for (vg, {product}, {method}), got: {result}"
    )
