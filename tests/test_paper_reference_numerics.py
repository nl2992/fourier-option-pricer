"""Exhaustive numerical paper-reference suite.

This module complements the more targeted adapter / method tests with one
consolidated paper-facing sweep:

- every published anchor in :mod:`foureng.refs.paper_refs`,
- every FO2008 replay case in the saved replication CSV,
- every Junike/improved-COS summary case,
- every frozen regression strip in :data:`foureng.refs.paper_refs.REGRESSION_STRIPS`.

The goal is breadth rather than novelty: when a refactor changes a paper-backed
result, this suite should be the place that fails first.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from benchmarks.paper_replications.fo2008_cos.params import CASES, PaperCase
from benchmarks.paper_replications.manifest import (
    FO2008_EXPECTED_CASE_IDS,
    JUNIKE_SUMMARY_CASE_IDS,
    PUBLISHED_PAPER_ANCHOR_KEYS,
    REGRESSION_STRIP_MODELS,
)
from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams, bsm_cf, bsm_cumulants
from foureng.models.cgmy import CgmyParams, cgmy_cf, cgmy_cumulants
from foureng.models.heston import HestonParams, heston_cf, heston_cumulants
from foureng.models.variance_gamma import VGParams, vg_cf, vg_cumulants
from foureng.pipeline import price_strip
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.cos import (
    cos_adaptive_decision,
    cos_auto_grid,
    cos_prices,
    recommended_cos_policy,
)
from foureng.pricers.lewis import lewis_call_prices
from foureng.refs.paper_refs import PAPER_ANCHORS, REGRESSION_STRIPS
from foureng.utils.grids import COSGrid, FFTGrid, FRFTGrid


pytest.importorskip(
    "pyfeng",
    reason="paper-reference numerics rely on PyFENG-backed CFs and CI installs pyfeng",
)


ROOT = Path(__file__).resolve().parents[1]
FO2008_REPLAY_DF = pd.read_csv(
    ROOT / "benchmarks/paper_replications/fo2008_cos/outputs/fo2008_replication_errors.csv"
)
JUNIKE_COMPARE_DF = pd.read_csv(
    ROOT / "benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv"
).set_index("case_id")

_REGRESSION_CM_GRID = FFTGrid(N=32768, eta=0.10, alpha=1.5)
_REGRESSION_FRFT_GRID = FRFTGrid(N=16384, eta=0.10, lam=0.0025, alpha=1.5)


def _row_id(row: dict) -> str:
    return f"{row['case_id']}[{row['method']}]-N{int(row['N'])}"


@lru_cache(maxsize=None)
def _case_context(case_id: str):
    case = CASES[case_id]
    fwd = ForwardSpec(
        S0=case.forward,
        r=case.params["r"],
        q=case.params["q"],
        T=case.maturity,
    )
    strikes = np.asarray(case.strikes, dtype=float)

    if case.model == "bsm":
        params = BsmParams(sigma=case.params["sigma"])
        phi = lambda u: bsm_cf(u, fwd, params)
        cums = bsm_cumulants(fwd, params)
    elif case.model == "heston":
        params = HestonParams(
            kappa=case.params["kappa"],
            theta=case.params["theta"],
            nu=case.params["nu"],
            rho=case.params["rho"],
            v0=case.params["v0"],
        )
        phi = lambda u: heston_cf(u, fwd, params)
        cums = heston_cumulants(fwd, params)
    elif case.model == "vg":
        params = VGParams(
            sigma=case.params["sigma"],
            nu=case.params["nu"],
            theta=case.params["theta"],
        )
        phi = lambda u: vg_cf(u, fwd, params)
        cums = vg_cumulants(fwd, params)
    elif case.model == "cgmy":
        params = CgmyParams(
            C=case.params["C"],
            G=case.params["G"],
            M=case.params["M"],
            Y=case.params["Y"],
        )
        phi = lambda u: cgmy_cf(u, fwd, params)
        cums = cgmy_cumulants(fwd, params)
    else:  # pragma: no cover - the FO2008 registry is fixed
        raise ValueError(f"unsupported FO2008 model: {case.model}")

    return case, phi, fwd, params, cums, strikes


@lru_cache(maxsize=None)
def _reference_for_case(case_id: str) -> np.ndarray:
    case, phi, fwd, _params, cums, strikes = _case_context(case_id)
    if case.reference_source == "FO2008_strip":
        grid_ref = cos_auto_grid(cums, N=case.extras["reference_N"], L=case.extras["L"])
        return np.asarray(cos_prices(phi, fwd, strikes, grid_ref).call_prices, dtype=float)
    return np.atleast_1d(np.asarray(case.reference_values, dtype=float))


@lru_cache(maxsize=None)
def _improved_reference_for_case(case_id: str) -> np.ndarray:
    case, phi, fwd, params, _cums, strikes = _case_context(case_id)

    if case.model == "bsm":
        vol = params.sigma * np.sqrt(fwd.T)
        d1 = (np.log(fwd.F0 / strikes) + 0.5 * params.sigma * params.sigma * fwd.T) / vol
        d2 = d1 - vol
        return np.asarray(
            fwd.disc * (fwd.F0 * norm.cdf(d1) - strikes * norm.cdf(d2)),
            dtype=float,
        )

    if case.model == "heston":
        return np.asarray(
            lewis_call_prices(
                phi,
                strikes,
                spot=fwd.S0,
                texp=fwd.T,
                intr=fwd.r,
                divr=fwd.q,
                method="trapz",
                u_max=250.0,
                n_u=8192,
            ),
            dtype=float,
        )

    return np.atleast_1d(np.asarray(case.reference_values, dtype=float))


@lru_cache(maxsize=None)
def _paper_cos_prices(case_id: str, n_terms: int) -> np.ndarray:
    case, phi, fwd, _params, cums, strikes = _case_context(case_id)
    if case.model == "cgmy" and "trunc_ab" in case.extras:
        a, b = case.extras["trunc_ab"]
        grid = COSGrid(N=int(n_terms), a=float(a), b=float(b), label="paper")
    else:
        grid = cos_auto_grid(cums, N=int(n_terms), L=float(case.extras.get("L", 10.0)))
    return np.asarray(cos_prices(phi, fwd, strikes, grid).call_prices, dtype=float)


def _max_abs_err(prices, ref) -> float:
    return float(np.max(np.abs(np.asarray(prices, dtype=float) - np.asarray(ref, dtype=float))))


def _fo2008_row_error(case_id: str, method: str, n_terms: int) -> float:
    case, phi, fwd, _params, _cums, strikes = _case_context(case_id)
    ref = _reference_for_case(case_id)

    if method == "cos":
        prices = _paper_cos_prices(case_id, int(n_terms))
        return _max_abs_err(prices, ref)

    if method == "carr_madan":
        if case_id != "bsm_table2":
            raise AssertionError(f"unexpected Carr-Madan FO2008 row for {case_id}")
        grid = FFTGrid(N=max(int(n_terms), 64), eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, strikes)
        return _max_abs_err(prices, ref)

    raise AssertionError(f"unsupported FO2008 replay method: {method}")


def _assert_close(actual: float, expected: float, *, rtol: float, atol: float, label: str) -> None:
    assert np.isclose(actual, expected, rtol=rtol, atol=atol), (
        f"{label}: got {actual:.12e}, expected {expected:.12e}, "
        f"diff={abs(actual - expected):.3e}"
    )


@pytest.mark.parametrize(
    "anchor_key",
    PUBLISHED_PAPER_ANCHOR_KEYS,
    ids=list(PUBLISHED_PAPER_ANCHOR_KEYS),
)
def test_published_anchor_cases_reproduce_reference_prices(anchor_key: str):
    anchor = PAPER_ANCHORS[anchor_key]

    if anchor_key == "fo2008_heston_atm":
        phi = lambda u: heston_cf(u, anchor.fwd, anchor.params)
        grid = cos_auto_grid(heston_cumulants(anchor.fwd, anchor.params), N=256, L=10.0)
        got = cos_prices(phi, anchor.fwd, anchor.strikes, grid).call_prices
        err = _max_abs_err(got, anchor.prices)
        assert err < 1e-6, f"{anchor.name}: COS max|err| = {err:.3e}"
        return

    if anchor_key == "lewis_heston_strip":
        phi = lambda u: heston_cf(u, anchor.fwd, anchor.params)
        grid = cos_auto_grid(heston_cumulants(anchor.fwd, anchor.params), N=128, L=10.0)
        got = cos_prices(phi, anchor.fwd, anchor.strikes, grid).call_prices
        err = _max_abs_err(got, anchor.prices)
        assert err < 1e-6, f"{anchor.name}: COS max|err| = {err:.3e}"
        return

    if anchor_key == "cm1999_vg_case4":
        phi = lambda u: vg_cf(u, anchor.fwd, anchor.params)
        calls = carr_madan_price_at_strikes(
            phi,
            anchor.fwd,
            FFTGrid(N=4096, eta=0.25, alpha=1.5),
            anchor.strikes,
        )
        puts = calls - anchor.fwd.S0 * np.exp(-anchor.fwd.q * anchor.fwd.T) + anchor.strikes * np.exp(-anchor.fwd.r * anchor.fwd.T)
        err = _max_abs_err(puts, anchor.prices)
        assert err < 2e-4, f"{anchor.name}: Carr-Madan max|err| = {err:.3e}"
        return

    raise AssertionError(f"unhandled paper anchor: {anchor_key}")


@pytest.mark.parametrize(
    "row",
    FO2008_REPLAY_DF.to_dict("records"),
    ids=[_row_id(row) for row in FO2008_REPLAY_DF.to_dict("records")],
)
def test_fo2008_replay_matches_saved_error_rows(row: dict):
    got = _fo2008_row_error(str(row["case_id"]), str(row["method"]), int(row["N"]))
    expected = float(row["our_error"])
    _assert_close(
        got,
        expected,
        rtol=1e-5,
        atol=max(1e-11, abs(expected) * 1e-8),
        label=_row_id(row),
    )


def test_fo2008_registry_is_fully_replayed_by_saved_csv():
    csv_case_ids = tuple(dict.fromkeys(FO2008_REPLAY_DF["case_id"].tolist()))
    assert csv_case_ids == FO2008_EXPECTED_CASE_IDS


@pytest.mark.parametrize(
    "case_id",
    JUNIKE_SUMMARY_CASE_IDS,
    ids=list(JUNIKE_SUMMARY_CASE_IDS),
)
def test_junike_saved_summary_cases_replay_cleanly(case_id: str):
    case, phi, fwd, params, cums, strikes = _case_context(case_id)
    ref = _improved_reference_for_case(case_id)

    default_prices = price_strip(case.model, "cos", strikes, fwd, params)
    paper_prices = _paper_cos_prices(case_id, max(case.Ns))
    policy = recommended_cos_policy(case.model, params, mode="benchmark")
    decision = cos_adaptive_decision(cums, model=case.model, params=params, policy=policy, strike_count=len(strikes))
    improved_prices = price_strip(case.model, "cos_improved", strikes, fwd, params, grid=policy)

    expected = JUNIKE_COMPARE_DF.loc[case_id]

    _assert_close(
        _max_abs_err(default_prices, ref),
        float(expected["default_err"]),
        rtol=1e-5,
        atol=1e-10,
        label=f"{case_id} default_err",
    )
    _assert_close(
        _max_abs_err(paper_prices, ref),
        float(expected["paper_grid_err"]),
        rtol=1e-5,
        atol=1e-10,
        label=f"{case_id} paper_grid_err",
    )
    _assert_close(
        _max_abs_err(improved_prices, ref),
        float(expected["improved_err"]),
        rtol=1e-5,
        atol=1e-10,
        label=f"{case_id} improved_err",
    )
    assert decision.method == expected["improved_method"]
    assert decision.grid.N == int(expected["improved_N"])


@pytest.mark.parametrize(
    "case_id",
    FO2008_EXPECTED_CASE_IDS,
    ids=list(FO2008_EXPECTED_CASE_IDS),
)
def test_cos_improved_stays_within_reasonable_full_fo2008_case_bounds(case_id: str):
    case, _phi, fwd, params, cums, strikes = _case_context(case_id)
    ref = _reference_for_case(case_id)

    default_prices = price_strip(case.model, "cos", strikes, fwd, params)
    paper_prices = _paper_cos_prices(case_id, max(case.Ns))
    policy = recommended_cos_policy(case.model, params, mode="benchmark")
    decision = cos_adaptive_decision(cums, model=case.model, params=params, policy=policy, strike_count=len(strikes))
    improved_prices = price_strip(case.model, "cos_improved", strikes, fwd, params, grid=policy)

    default_err = _max_abs_err(default_prices, ref)
    paper_err = _max_abs_err(paper_prices, ref)
    improved_err = _max_abs_err(improved_prices, ref)

    assert np.all(np.isfinite(improved_prices)), f"{case_id}: non-finite improved prices"
    assert np.all(np.asarray(improved_prices) >= -1e-12), f"{case_id}: negative improved prices"

    allowed_ratio = 5.0 if case_id == "cgmy_table10_y198" else 1.05
    baseline = max(default_err, paper_err)
    assert improved_err <= baseline * allowed_ratio + 1e-12, (
        f"{case_id}: improved error drifted too far. "
        f"default={default_err:.3e}, paper={paper_err:.3e}, improved={improved_err:.3e}, "
        f"method={decision.method}, N={decision.grid.N}"
    )


@pytest.mark.parametrize(
    "model",
    REGRESSION_STRIP_MODELS,
    ids=list(REGRESSION_STRIP_MODELS),
)
@pytest.mark.parametrize(
    ("method", "grid", "tol"),
    (
        ("carr_madan", _REGRESSION_CM_GRID, 1e-12),
        ("cos", None, 1e-7),
        ("frft", _REGRESSION_FRFT_GRID, 1e-7),
    ),
    ids=("carr_madan", "cos", "frft"),
)
def test_all_regression_strip_cases_replay_seeded_oracles(model: str, method: str, grid, tol: float):
    ref = REGRESSION_STRIPS[model]
    if grid is None:
        got = price_strip(ref.model, method, ref.strikes, ref.fwd, ref.params)
    else:
        got = price_strip(ref.model, method, ref.strikes, ref.fwd, ref.params, grid=grid)
    err = _max_abs_err(got, ref.prices)
    assert err < tol, f"{ref.name}: {method} max|err| = {err:.3e}"
