"""Tests for the method registry (MethodSpec entries)."""

from __future__ import annotations

import pytest

from foureng.core.capabilities import METHOD_REGISTRY, MethodSpec

_BASELINE_METHODS = {
    "cos",
    "cos_improved",
    "cos_filtered",
    "carr_madan",
    "frft",
    "pyfeng_fft",
    "conv",
    "barrier_bsm",
    "asian_bsm",
    "asian_mc",
    "double_barrier_mc",
    "forward_start_bsm",
    "lookback_bsm",
    "lookback_mc",
    "variance_analytic_bsm",
    "variance_mc",
    "cliquet_mc",
    "cos_bermudan",
    "mellin",
    "proj",
    "sabr_hagan",
    "digital_bsm",
    "cos_digital",
}

_PLANNED_METHODS = {
    "monte_carlo",
    "pde_fd",
    "lattice",
    "ctmc",
}


def test_all_baseline_methods_registered():
    missing = _BASELINE_METHODS - set(METHOD_REGISTRY)
    assert not missing, f"Missing baseline methods: {sorted(missing)}"


def test_all_planned_methods_registered():
    missing = _PLANNED_METHODS - set(METHOD_REGISTRY)
    assert not missing, f"Missing planned methods: {sorted(missing)}"


def test_every_method_has_capability_spec():
    for key, spec in METHOD_REGISTRY.items():
        assert isinstance(spec, MethodSpec), f"{key!r} entry is not a MethodSpec"


def test_every_method_supports_at_least_one_product():
    for key, spec in METHOD_REGISTRY.items():
        assert len(spec.supports_products) >= 1, f"{key!r}: supports_products must be non-empty"


def test_every_method_supports_at_least_one_exercise_style():
    for key, spec in METHOD_REGISTRY.items():
        assert len(spec.supports_exercise) >= 1, f"{key!r}: supports_exercise must be non-empty"


def test_cf_methods_require_cf():
    cf_methods = {
        "cos",
        "cos_improved",
        "cos_filtered",
        "carr_madan",
        "frft",
        "pyfeng_fft",
        "conv",
        "cos_bermudan",
        "mellin",
        "proj",
        "cos_digital",
    }
    for m in cf_methods:
        assert METHOD_REGISTRY[m].requires_cf is True, f"{m!r}: requires_cf should be True"


def test_monte_carlo_requires_simulation():
    assert METHOD_REGISTRY["monte_carlo"].requires_simulation is True


def test_monte_carlo_is_path_dependent():
    assert METHOD_REGISTRY["monte_carlo"].supports_path_dependent is True


def test_cos_methods_not_path_dependent():
    for m in ("cos", "cos_improved", "cos_filtered"):
        assert METHOD_REGISTRY[m].supports_path_dependent is False


def test_cos_bermudan_supports_bermudan():
    assert "bermudan" in METHOD_REGISTRY["cos_bermudan"].supports_products


def test_monte_carlo_covers_exotic_products():
    mc = METHOD_REGISTRY["monte_carlo"]
    exotic = {"barrier", "asian", "lookback", "variance_swap", "cliquet"}
    missing = exotic - mc.supports_products
    assert not missing, f"MC should support exotic products: {missing}"


def test_all_entries_are_frozen():
    spec = METHOD_REGISTRY["cos"]
    with pytest.raises((AttributeError, TypeError)):
        spec.requires_cf = False  # type: ignore[misc]


def test_import_from_pricers_registry():
    from foureng.pricers.registry import METHOD_REGISTRY as MR2

    assert MR2 is METHOD_REGISTRY
