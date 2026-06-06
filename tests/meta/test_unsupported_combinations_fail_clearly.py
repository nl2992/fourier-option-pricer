"""Verify that unsupported (model, product, method) triples fail explicitly, not silently.

The definition of done for Phase 2: unsupported combinations fail explicitly,
not silently. This test file checks that:
1. explain_capability returns a non-empty, informative 'Not supported' string.
2. The future price() dispatcher will raise NotImplementedError for unimplemented
   products with an informative message.
"""

from __future__ import annotations

import pytest

from foureng.core.capabilities import METHOD_REGISTRY, explain_capability

# Every (model, product, method) below should be "Not supported"
_UNSUPPORTED = [
    # COS can't do path-dependent
    ("bsm", "asian", "cos"),
    ("heston", "asian", "cos"),
    ("vg", "asian", "cos_improved"),
    ("cgmy", "barrier", "cos"),
    ("bsm", "lookback", "frft"),
    ("heston", "lookback", "carr_madan"),
    ("vg", "lookback", "lookback_bsm"),
    ("vg", "lookback", "lookback_mc"),
    ("bsm", "variance_swap", "cos"),
    ("heston", "variance_swap", "variance_mc"),
    ("heston", "variance_option", "variance_mc"),
    ("bsm", "cliquet", "cos_improved"),
    ("heston", "cliquet", "cliquet_mc"),
    # COS Bermudan: SV models not supported
    ("heston", "bermudan", "cos_bermudan"),
    ("sv32", "bermudan", "cos_bermudan"),
    ("ousv", "bermudan", "cos_bermudan"),
    ("rough_heston", "bermudan", "cos_bermudan"),
    ("bates", "bermudan", "cos_bermudan"),
    ("heston_kou", "bermudan", "cos_bermudan"),
    ("heston", "forward_start", "forward_start_bsm"),
    # pyfeng_fft: non-pyfeng models
    ("kou", "european", "pyfeng_fft"),
    ("merton_jd", "european", "pyfeng_fft"),
    ("bilateral_gamma", "european", "pyfeng_fft"),
    ("generalized_hyperbolic", "european", "pyfeng_fft"),
    ("fmls", "european", "pyfeng_fft"),
    ("heston", "digital", "digital_bsm"),
    # PDE/lattice: CF-only models without state-space
    # (these are 'planned' methods; their specs say they require markov state)
    # The registry still says they support european — so these should be Supported.
    # Instead test that unknown products fail:
    ("bsm", "exchange", "cos"),
    ("heston", "basket", "frft"),
]


@pytest.mark.parametrize("model,product,method", _UNSUPPORTED)
def test_unsupported_returns_not_supported_string(model, product, method):
    result = explain_capability(model, product, method)
    assert result.startswith("Not supported:"), (
        f"Expected 'Not supported:' for ({model}, {product}, {method}), got:\n{result}"
    )


@pytest.mark.parametrize("model,product,method", _UNSUPPORTED)
def test_unsupported_explanation_is_nonempty(model, product, method):
    result = explain_capability(model, product, method)
    # Should be at least 20 chars to be informative
    assert len(result) > 20, f"Explanation too short for ({model}, {product}, {method}): {result!r}"


def test_price_dispatcher_raises_not_implemented_for_barrier():
    """price() raises NotImplementedError for unimplemented products, not silently."""
    import foureng as fe
    from foureng.pipeline import price
    from foureng.products import BarrierOption

    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    opt = BarrierOption(strike=100.0, barrier=80.0, maturity=1.0, cp=1, barrier_type="down_out")
    with pytest.raises(NotImplementedError) as exc_info:
        price(opt, "bsm", "cos", fwd, params)
    assert "barrier" in str(exc_info.value).lower()


def test_price_dispatcher_raises_not_implemented_for_asian():
    import numpy as np

    import foureng as fe
    from foureng.pipeline import price
    from foureng.products import AsianOption

    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.20)
    opt = AsianOption(strike=100.0, maturity=1.0, cp=1, monitoring_times=np.linspace(0.25, 1.0, 4))
    with pytest.raises(NotImplementedError) as exc_info:
        price(opt, "bsm", "cos", fwd, params)
    assert "asian" in str(exc_info.value).lower()


def test_all_method_registry_entries_have_non_empty_products():
    for key, spec in METHOD_REGISTRY.items():
        assert len(spec.supports_products) > 0, (
            f"MethodSpec for {key!r} has no supported products — this is a registry error"
        )


def test_all_method_registry_entries_have_notes():
    for key, spec in METHOD_REGISTRY.items():
        assert spec.notes, f"MethodSpec for {key!r} is missing notes — add a short description"
