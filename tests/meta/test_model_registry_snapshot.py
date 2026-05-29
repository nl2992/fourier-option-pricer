"""Snapshot tests — the model registry must contain all 26 baseline models."""

from __future__ import annotations

import pytest

from foureng.models.registry import MODEL_REGISTRY

_BASELINE_26 = {
    "bsm", "heston", "ousv", "vg", "cgmy", "nig", "sv32", "rough_heston",
    "kou", "bates", "heston_kou", "heston_cgmy", "garch_wmw2012",
    "merton_jd", "meixner", "bilateral_gamma", "generalized_hyperbolic",
    "fmls", "double_heston", "vgsa", "heston_nig", "heston_vg",
    "svjj", "bns_gamma_ou", "nts", "cgmysa",
}

_PYFENG_BACKED = {"bsm", "heston", "ousv", "vg", "cgmy", "nig", "sv32", "rough_heston"}


def test_all_26_models_registered():
    assert len(MODEL_REGISTRY) >= 26


def test_baseline_model_keys_present():
    missing = _BASELINE_26 - set(MODEL_REGISTRY)
    assert not missing, f"Missing models: {sorted(missing)}"


def test_pyfeng_backed_flags():
    for key in _PYFENG_BACKED:
        assert MODEL_REGISTRY[key].pyfeng_fft is True, f"{key!r} should have pyfeng_fft=True"


def test_non_pyfeng_models_flagged_false():
    non_pyfeng = _BASELINE_26 - _PYFENG_BACKED
    for key in non_pyfeng:
        assert MODEL_REGISTRY[key].pyfeng_fft is False, f"{key!r} should have pyfeng_fft=False"


def test_every_entry_has_cf_and_cumulants():
    for key, entry in MODEL_REGISTRY.items():
        assert callable(entry.cf), f"{key!r}: cf must be callable"
        assert callable(entry.cumulants), f"{key!r}: cumulants must be callable"


def test_every_entry_has_params_cls():
    for key, entry in MODEL_REGISTRY.items():
        assert entry.params_cls is not None, f"{key!r}: params_cls is None"


def test_all_entries_stable():
    for key, entry in MODEL_REGISTRY.items():
        assert entry.status == "stable", f"{key!r}: status={entry.status!r}"


def test_no_duplicate_keys():
    keys = list(MODEL_REGISTRY.keys())
    assert len(keys) == len(set(keys))
