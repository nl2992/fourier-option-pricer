"""Shared helpers for paper-backed regression tests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from benchmarks.paper_replications.fo2008_cos.params import CASES
from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams, bsm_cf, bsm_cumulants
from foureng.models.cgmy import CgmyParams, cgmy_cf, cgmy_cumulants
from foureng.models.heston import HestonParams, heston_cf, heston_cumulants
from foureng.models.variance_gamma import VGParams, vg_cf, vg_cumulants
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.pricers.lewis import lewis_call_prices
from foureng.utils.grids import COSGrid, FFTGrid

pytest.importorskip(
    "pyfeng",
    reason="paper-backed tests rely on PyFENG-backed CFs and CI installs pyfeng",
)


ROOT = Path(__file__).resolve().parents[1]
FO2008_REPLAY_DF = pd.read_csv(
    ROOT / "benchmarks/paper_replications/fo2008_cos/outputs/fo2008_replication_errors.csv"
)
JUNIKE_COMPARE_DF = pd.read_csv(
    ROOT / "benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv"
).set_index("case_id")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def row_id(row: dict) -> str:
    return f"{row['case_id']}[{row['method']}]-N{int(row['N'])}"


@lru_cache(maxsize=None)
def case_context(case_id: str):
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
    else:  # pragma: no cover - fixed registry
        raise ValueError(f"unsupported FO2008 model: {case.model}")

    return case, phi, fwd, params, cums, strikes


@lru_cache(maxsize=None)
def reference_for_case(case_id: str) -> np.ndarray:
    case, phi, fwd, _params, cums, strikes = case_context(case_id)
    if case.reference_source == "FO2008_strip":
        grid_ref = cos_auto_grid(cums, N=case.extras["reference_N"], L=case.extras["L"])
        return np.asarray(cos_prices(phi, fwd, strikes, grid_ref).call_prices, dtype=float)
    return np.atleast_1d(np.asarray(case.reference_values, dtype=float))


@lru_cache(maxsize=None)
def improved_reference_for_case(case_id: str) -> np.ndarray:
    case, phi, fwd, params, _cums, strikes = case_context(case_id)

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
def paper_cos_prices(case_id: str, n_terms: int) -> np.ndarray:
    case, phi, fwd, _params, cums, strikes = case_context(case_id)
    if case.model == "cgmy" and "trunc_ab" in case.extras:
        a, b = case.extras["trunc_ab"]
        grid = COSGrid(N=int(n_terms), a=float(a), b=float(b), label="paper")
    else:
        grid = cos_auto_grid(cums, N=int(n_terms), L=float(case.extras.get("L", 10.0)))
    return np.asarray(cos_prices(phi, fwd, strikes, grid).call_prices, dtype=float)


def max_abs_err(prices, ref) -> float:
    return float(np.max(np.abs(np.asarray(prices, dtype=float) - np.asarray(ref, dtype=float))))


def fo2008_row_error(case_id: str, method: str, n_terms: int) -> float:
    case, phi, fwd, _params, _cums, strikes = case_context(case_id)
    ref = reference_for_case(case_id)

    if method == "cos":
        return max_abs_err(paper_cos_prices(case_id, int(n_terms)), ref)

    if method == "carr_madan":
        if case_id != "bsm_table2":
            raise AssertionError(f"unexpected Carr-Madan FO2008 row for {case_id}")
        grid = FFTGrid(N=max(int(n_terms), 64), eta=0.25, alpha=1.5)
        prices = carr_madan_price_at_strikes(phi, fwd, grid, strikes)
        return max_abs_err(prices, ref)

    raise AssertionError(f"unsupported FO2008 replay method: {method}")


def assert_close(actual: float, expected: float, *, rtol: float, atol: float, label: str) -> None:
    assert np.isclose(actual, expected, rtol=rtol, atol=atol), (
        f"{label}: got {actual:.12e}, expected {expected:.12e}, diff={abs(actual - expected):.3e}"
    )


def assert_valid_png(path: Path) -> None:
    payload = path.read_bytes()
    assert payload.startswith(PNG_SIGNATURE), f"{path} is not a valid PNG"
    assert len(payload) > 256, f"{path} looks unexpectedly small"
