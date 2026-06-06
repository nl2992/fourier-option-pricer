"""Tests for the explain_capability() function and capability matrix logic."""

from __future__ import annotations

import pytest

from foureng.core.capabilities import explain_capability

# ── Supported combinations ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,product,method",
    [
        ("heston", "european", "cos"),
        ("bsm", "european", "cos_improved"),
        ("vg", "european", "carr_madan"),
        ("cgmy", "european", "frft"),
        ("bsm", "european", "pyfeng_fft"),
        ("heston", "european", "pyfeng_fft"),
        ("bsm", "bermudan", "cos_bermudan"),
        ("bsm", "digital", "digital_bsm"),
        ("heston", "digital", "cos_digital"),
        ("vg", "bermudan", "cos_bermudan"),
        ("cgmy", "bermudan", "cos_bermudan"),
        ("kou", "bermudan", "cos_bermudan"),
        ("bsm", "european", "monte_carlo"),
        ("heston", "asian", "monte_carlo"),
        ("bsm", "barrier", "pde_fd"),
        ("bsm", "american", "lattice"),
        ("bsm", "forward_start", "forward_start_bsm"),
        ("bsm", "lookback", "lookback_bsm"),
        ("bsm", "lookback", "lookback_mc"),
        ("bsm", "variance_swap", "variance_mc"),
        ("bsm", "variance_option", "variance_mc"),
        ("bsm", "cliquet", "cliquet_mc"),
        ("vg", "european", "proj"),
    ],
)
def test_supported_combinations_return_supported(model, product, method):
    result = explain_capability(model, product, method)
    assert result.startswith("Supported:"), (
        f"Expected Supported for ({model}, {product}, {method}), got: {result}"
    )


# ── Unsupported combinations ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,product,method",
    [
        ("heston", "asian", "cos"),
        ("bsm", "barrier", "cos"),
        ("heston", "lookback", "cos_improved"),
        ("bsm", "bermudan", "cos"),
        ("heston", "bermudan", "cos_bermudan"),  # SV model — 2-D extension needed
        ("bates", "bermudan", "cos_bermudan"),  # SVJD — not supported
        ("kou", "asian", "cos"),
        ("bsm", "cliquet", "frft"),
        ("heston", "digital", "digital_bsm"),
    ],
)
def test_unsupported_combinations_return_not_supported(model, product, method):
    result = explain_capability(model, product, method)
    assert result.startswith("Not supported:"), (
        f"Expected Not supported for ({model}, {product}, {method}), got: {result}"
    )


# ── Reason quality ─────────────────────────────────────────────────────────


def test_asian_cos_explanation_mentions_path_dependent():
    result = explain_capability("heston", "asian", "cos")
    assert "path-dependent" in result.lower() or "path" in result.lower()


def test_heston_bermudan_cos_bermudan_mentions_2d():
    result = explain_capability("heston", "bermudan", "cos_bermudan")
    assert "2-D" in result or "two-dimensional" in result.lower() or "dimension" in result.lower()


def test_unknown_method_returns_not_supported():
    result = explain_capability("bsm", "european", "nonexistent_method")
    assert result.startswith("Not supported:")
    assert "method" in result.lower()


def test_unknown_model_returns_not_supported():
    result = explain_capability("nonexistent_model", "european", "cos")
    assert result.startswith("Not supported:")


def test_pyfeng_fft_unsupported_model():
    # kou doesn't have pyfeng_fft
    result = explain_capability("kou", "european", "pyfeng_fft")
    assert result.startswith("Not supported:")
    assert "PyFENG" in result or "pyfeng" in result.lower()


def test_barrier_cos_suggests_alternatives():
    result = explain_capability("bsm", "barrier", "cos")
    assert any(word in result.lower() for word in ["pde", "lattice", "mc", "monte"])


def test_digital_bsm_explanation_mentions_bsm_only():
    result = explain_capability("heston", "digital", "digital_bsm")
    assert "model='bsm'" in result


# ── Parity: same output regardless of how registry is imported ─────────────


def test_pricers_registry_explain_capability_same():
    from foureng.pricers.registry import explain_capability as ec2

    assert explain_capability is ec2
