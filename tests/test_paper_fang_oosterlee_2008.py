"""Fang & Oosterlee (2008) paper-facing tests and replication assets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.paper_replications.fo2008_cos.params import CASES
from benchmarks.paper_replications.manifest import (
    FO2008_EXPECTED_CASE_IDS,
    FO2008_EXPECTED_FAMILIES,
    PAPER_OUTPUT_BUNDLES,
    PUBLISHED_PAPER_ANCHOR_KEYS,
)
from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cf_form2, heston_cumulants
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.refs.paper_refs import PAPER_ANCHORS
from tests._paper_test_support import (
    ROOT,
    FO2008_REPLAY_DF,
    assert_close,
    assert_valid_png,
    fo2008_row_error,
    row_id,
)


def test_fo2008_anchor_registry_stays_explicit():
    assert tuple(PUBLISHED_PAPER_ANCHOR_KEYS) == (
        "fo2008_heston_atm",
        "lewis_heston_strip",
        "cm1999_vg_case4",
    )
    assert set(PUBLISHED_PAPER_ANCHOR_KEYS).issubset(PAPER_ANCHORS)


def test_cos_heston_fo2008(fo2008_heston):
    """FO2008 Table 1 Heston ATM call."""
    d = fo2008_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"]
    )
    phi = lambda u: heston_cf_form2(u, fwd, params)

    grid = cos_auto_grid(heston_cumulants(fwd, params), N=256, L=10.0)
    price = cos_prices(phi, fwd, np.array([d["K"]]), grid).call_prices[0]
    err = abs(price - d["ref_call"])
    assert err < 1e-6, f"FO2008 COS N=256 err = {err:.3e}"


def test_cos_heston_fo2008_n_convergence(fo2008_heston):
    """FO2008 Heston case should show monotone N-convergence."""
    d = fo2008_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"]
    )
    phi = lambda u: heston_cf_form2(u, fwd, params)

    errs = {}
    cums = heston_cumulants(fwd, params)
    for N in (64, 128, 192, 256):
        price = cos_prices(phi, fwd, np.array([d["K"]]), cos_auto_grid(cums, N=N, L=10.0)).call_prices[0]
        errs[N] = abs(price - d["ref_call"])

    assert errs[128] < errs[64], f"not monotone: {errs}"
    assert errs[192] < errs[128], f"not monotone: {errs}"
    assert errs[256] < errs[192], f"not monotone: {errs}"
    assert errs[256] < errs[64] * 1e-4, f"too-slow convergence: {errs}"


def test_cos_heston_fo2008_L_sensitivity(fo2008_heston):
    """FO2008 recommends L=10; the recommended-or-wider band should be stable."""
    d = fo2008_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = HestonParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"]
    )
    phi = lambda u: heston_cf_form2(u, fwd, params)

    cums = heston_cumulants(fwd, params)
    prices = {}
    for L in (8.0, 10.0, 12.0):
        prices[L] = float(
            cos_prices(phi, fwd, np.array([d["K"]]), cos_auto_grid(cums, N=512, L=L)).call_prices[0]
        )
    spread = max(prices.values()) - min(prices.values())
    assert spread < 2e-6, f"L-sensitivity too high at N=512: {prices} spread={spread:.3e}"


@pytest.mark.parametrize(
    "row",
    FO2008_REPLAY_DF.to_dict("records"),
    ids=[row_id(row) for row in FO2008_REPLAY_DF.to_dict("records")],
)
def test_fo2008_replay_matches_saved_error_rows(row: dict):
    got = fo2008_row_error(str(row["case_id"]), str(row["method"]), int(row["N"]))
    expected = float(row["our_error"])
    assert_close(
        got,
        expected,
        rtol=1e-5,
        atol=max(1e-11, abs(expected) * 1e-8),
        label=row_id(row),
    )


def test_fo2008_replay_registry_is_exhaustive():
    assert tuple(CASES) == FO2008_EXPECTED_CASE_IDS
    families = tuple(dict.fromkeys(case.model for case in CASES.values()))
    assert families == FO2008_EXPECTED_FAMILIES
    csv_case_ids = tuple(dict.fromkeys(FO2008_REPLAY_DF["case_id"].tolist()))
    assert csv_case_ids == FO2008_EXPECTED_CASE_IDS


def test_fo2008_output_bundles_have_required_core_files():
    for bundle in PAPER_OUTPUT_BUNDLES:
        outdir = ROOT / bundle.output_dir
        assert outdir.is_dir(), f"missing output dir: {outdir}"

        summary = outdir / bundle.summary_file
        assert summary.is_file(), f"missing summary: {summary}"

        for rel in bundle.csv_files:
            path = outdir / rel
            assert path.is_file(), f"missing CSV: {path}"
            assert path.stat().st_size > 0, f"empty CSV: {path}"


def test_fo2008_output_bundles_have_required_figures():
    for bundle in PAPER_OUTPUT_BUNDLES:
        outdir = ROOT / bundle.output_dir
        figure_paths = [
            outdir / bundle.family_figure_pattern.format(family=family)
            for family in FO2008_EXPECTED_FAMILIES
        ]
        figure_paths.extend(outdir / rel for rel in bundle.extra_figures)

        for path in figure_paths:
            assert path.is_file(), f"missing figure: {path}"
            assert_valid_png(path)


def test_fo2008_output_summaries_cover_all_tables():
    for bundle in PAPER_OUTPUT_BUNDLES:
        summary = (ROOT / bundle.output_dir / bundle.summary_file).read_text()
        for label in bundle.expected_table_labels:
            assert label in summary, f"{label} missing from {bundle.name} summary"
