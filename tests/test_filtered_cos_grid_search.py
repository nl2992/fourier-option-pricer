"""Tests for foureng/experiments/cos_filter_grid_search.py.

Uses a tiny CI-safe candidate set (2 candidates, n_repeat=1) so the suite
runs quickly.  Covers:

- run_filtered_cos_grid_search returns a pd.DataFrame.
- All required columns are present.
- select_fastest_under_tolerance returns a row with status == "ok".
- Selector chooses the fastest candidate when multiple pass the tolerance.
- Selector falls back to lowest error when none pass the tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.variance_gamma import VGParams
from foureng.pipeline import price_strip
from foureng.utils.grids import COSGridPolicy
from foureng.utils.spectral_filters import COSFilterSpec
from foureng.experiments.cos_filter_grid_search import (
    FilterGridCandidate,
    run_filtered_cos_grid_search,
    select_fastest_under_tolerance,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vg_setup():
    fwd = ForwardSpec(S0=100.0, r=0.1, q=0.0, T=0.1)
    p = VGParams(sigma=0.12, nu=0.2, theta=-0.14)
    strikes = np.array([90.0, 100.0, 110.0])
    ref = price_strip("vg", "cos_improved", strikes, fwd, p)
    return fwd, p, strikes, ref


@pytest.fixture()
def small_policy():
    return COSGridPolicy(
        mode="benchmark",
        truncation="tolerance",
        centered=True,
        dx_target=0.01,
        L=10.0,
        eps_trunc=1e-10,
        max_N=4096,
        width_fallback=0.0,
    )


# ---------------------------------------------------------------------------
# Grid-search returns correct DataFrame structure
# ---------------------------------------------------------------------------

def test_filtered_cos_grid_search_returns_candidate_table(vg_setup, small_policy):
    """Grid search must return a DataFrame with required columns and 2 rows."""
    fwd, p, strikes, ref = vg_setup

    candidates = [
        FilterGridCandidate("junike_no_filter", small_policy, None),
        FilterGridCandidate("junike_exp_filter", small_policy, COSFilterSpec("exponential", order=8)),
    ]

    df = run_filtered_cos_grid_search(
        model="vg",
        strikes=strikes,
        fwd=fwd,
        params=p,
        reference=ref,
        candidates=candidates,
        tol=1e-6,
        n_repeat=1,
    )

    required = {
        "method_label", "runtime_ms", "max_abs_err",
        "mean_abs_err", "passes_tol", "status",
        "truncation", "dx_target", "L", "eps_trunc",
        "filter", "filter_order",
    }
    assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
    assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
    assert required.issubset(df.columns), (
        f"Missing columns: {required - set(df.columns)}"
    )


def test_grid_search_status_ok_for_valid_candidates(vg_setup, small_policy):
    """All valid candidates should have status='ok'."""
    fwd, p, strikes, ref = vg_setup

    candidates = [
        FilterGridCandidate("junike_no_filter", small_policy, None),
        FilterGridCandidate("junike_fejer", small_policy, COSFilterSpec("fejer")),
    ]

    df = run_filtered_cos_grid_search(
        model="vg", strikes=strikes, fwd=fwd, params=p,
        reference=ref, candidates=candidates, tol=1e-6, n_repeat=1,
    )

    assert (df["status"] == "ok").all(), (
        f"Some candidates failed: {df[df['status'] != 'ok']['status'].tolist()}"
    )


def test_grid_search_runtime_positive(vg_setup, small_policy):
    """Successful candidates must have positive runtime_ms."""
    fwd, p, strikes, ref = vg_setup

    candidates = [FilterGridCandidate("c", small_policy, None)]
    df = run_filtered_cos_grid_search(
        model="vg", strikes=strikes, fwd=fwd, params=p,
        reference=ref, candidates=candidates, tol=1e-6, n_repeat=1,
    )
    ok = df[df["status"] == "ok"]
    assert (ok["runtime_ms"] > 0).all()


def test_grid_search_filter_column_values(vg_setup, small_policy):
    """Filter column should be 'none' for no-filter, 'exponential' for exp filter."""
    fwd, p, strikes, ref = vg_setup

    candidates = [
        FilterGridCandidate("no_filt", small_policy, None),
        FilterGridCandidate("exp_filt", small_policy, COSFilterSpec("exponential", order=8)),
    ]
    df = run_filtered_cos_grid_search(
        model="vg", strikes=strikes, fwd=fwd, params=p,
        reference=ref, candidates=candidates, tol=1e-6, n_repeat=1,
    )

    assert set(df["filter"]) == {"none", "exponential"}


# ---------------------------------------------------------------------------
# Selector: fastest under tolerance
# ---------------------------------------------------------------------------

def test_selector_returns_ok_candidate(vg_setup, small_policy):
    """Selector must return a row with status='ok'."""
    fwd, p, strikes, ref = vg_setup

    candidates = [
        FilterGridCandidate("junike_no_filter", small_policy, None),
        FilterGridCandidate("junike_exp_filter", small_policy, COSFilterSpec("exponential", order=8)),
    ]

    df = run_filtered_cos_grid_search(
        model="vg", strikes=strikes, fwd=fwd, params=p,
        reference=ref, candidates=candidates, tol=1e-6, n_repeat=1,
    )

    best = select_fastest_under_tolerance(df, tol=1e-6)
    assert best["status"] == "ok"


def test_selector_chooses_fastest_under_tol():
    """Selector must prefer the fastest row whose max_abs_err ≤ tol."""
    df = pd.DataFrame([
        {"method_label": "slow_accurate", "status": "ok", "runtime_ms": 10.0, "max_abs_err": 1e-10},
        {"method_label": "fast_good",     "status": "ok", "runtime_ms":  1.0, "max_abs_err": 1e-7},
        {"method_label": "bad",           "status": "ok", "runtime_ms":  0.5, "max_abs_err": 1e-2},
    ])
    best = select_fastest_under_tolerance(df, tol=1e-6)
    assert best["method_label"] == "fast_good", (
        f"Expected 'fast_good', got '{best['method_label']}'"
    )


def test_selector_falls_back_to_lowest_error_when_none_pass():
    """When no candidate passes tol, selector should pick lowest max_abs_err."""
    df = pd.DataFrame([
        {"method_label": "a", "status": "ok", "runtime_ms": 1.0, "max_abs_err": 1e-2},
        {"method_label": "b", "status": "ok", "runtime_ms": 5.0, "max_abs_err": 1e-4},
    ])
    best = select_fastest_under_tolerance(df, tol=1e-8)
    assert best["method_label"] == "b", (
        f"Expected 'b' (lowest error), got '{best['method_label']}'"
    )


def test_selector_handles_all_failures():
    """Selector should still return a row even if all candidates failed."""
    df = pd.DataFrame([
        {"method_label": "x", "status": "fail: ValueError: oops", "runtime_ms": np.nan, "max_abs_err": np.inf},
        {"method_label": "y", "status": "fail: ValueError: oops", "runtime_ms": np.nan, "max_abs_err": np.inf},
    ])
    best = select_fastest_under_tolerance(df, tol=1e-6)
    assert best is not None


def test_selector_exact_boundary():
    """max_abs_err exactly equal to tol should be treated as passing."""
    df = pd.DataFrame([
        {"method_label": "exact_tol",   "status": "ok", "runtime_ms": 2.0, "max_abs_err": 1e-6},
        {"method_label": "below_tol",   "status": "ok", "runtime_ms": 3.0, "max_abs_err": 1e-7},
        {"method_label": "above_tol",   "status": "ok", "runtime_ms": 0.5, "max_abs_err": 1e-5},
    ])
    best = select_fastest_under_tolerance(df, tol=1e-6)
    # Both "exact_tol" and "below_tol" pass; "exact_tol" is faster
    assert best["method_label"] == "exact_tol"


# ---------------------------------------------------------------------------
# Error measurement
# ---------------------------------------------------------------------------

def test_max_abs_err_computed_correctly(vg_setup, small_policy):
    """max_abs_err in the table must agree with manual computation."""
    fwd, p, strikes, ref = vg_setup

    candidates = [FilterGridCandidate("junike_no_filter", small_policy, None)]
    df = run_filtered_cos_grid_search(
        model="vg", strikes=strikes, fwd=fwd, params=p,
        reference=ref, candidates=candidates, tol=1e-6, n_repeat=1,
    )

    row = df.iloc[0]
    assert row["status"] == "ok"
    # Manual recompute
    prices = price_strip("vg", "cos_improved", strikes, fwd, p, grid=small_policy)
    expected = float(np.max(np.abs(prices - ref)))
    assert abs(row["max_abs_err"] - expected) < 1e-12 * max(expected, 1e-15)
