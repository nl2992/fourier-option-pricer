"""Junike-style improved-COS policy tests."""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.paper_replications.manifest import FO2008_EXPECTED_CASE_IDS, JUNIKE_SUMMARY_CASE_IDS
from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams, bsm_cf, bsm_cumulants
from foureng.models.heston import HestonParams, heston_cf, heston_cumulants
from foureng.models.variance_gamma import VGParams
from foureng.pipeline import price_strip
from foureng.pricers.cos import cos_adaptive_decision, cos_auto_grid, cos_prices, recommended_cos_policy
from foureng.pricers.lewis import lewis_call_prices
from foureng.utils.cumulants import Cumulants, cos_centered_half_width
from foureng.utils.grids import COSGrid, COSGridPolicy
from tests._paper_test_support import (
    JUNIKE_COMPARE_DF,
    assert_close,
    case_context,
    improved_reference_for_case,
    max_abs_err,
    paper_cos_prices,
)


pytestmark = [pytest.mark.paper, pytest.mark.derived_reference, pytest.mark.numerical_stability]


def test_centered_cos_grid_matches_uncentered_bsm():
    """Centered COS should match the legacy uncentered grid on BSM."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.5)
    params = BsmParams(sigma=0.25)
    strikes = np.array([80.0, 100.0, 120.0])
    cums = bsm_cumulants(fwd, params)
    phi = lambda u: bsm_cf(u, fwd, params)

    old_calls = cos_prices(phi, fwd, strikes, cos_auto_grid(cums, N=512, L=10.0)).call_prices

    c1, c2, c4 = cums
    half_width = cos_centered_half_width(Cumulants(c1=c1, c2=c2, c4=c4), L=10.0)
    centered_grid = COSGrid(N=512, a=-half_width, b=half_width, center=c1, label="centered")
    centered_calls = cos_prices(phi, fwd, strikes, centered_grid).call_prices

    assert np.allclose(centered_calls, old_calls, atol=1e-12, rtol=0.0)


def test_cos_adaptive_decision_scales_n_with_interval_width():
    """Wider intervals should force larger N under the adaptive policy."""
    policy = COSGridPolicy(
        mode="benchmark",
        truncation="heuristic",
        centered=True,
        dx_target=0.05,
        L=10.0,
        width_fallback=0.0,
    )
    d_small = cos_adaptive_decision((0.0, 0.04, 0.0), model="bsm", policy=policy)
    d_large = cos_adaptive_decision((0.0, 1.00, 0.0), model="bsm", policy=policy)

    assert d_large.grid.N > d_small.grid.N
    assert d_small.grid.dx <= 0.05
    assert d_large.grid.dx <= 0.05


def test_price_strip_cos_improved_can_fall_back_to_lewis():
    """Wide Heston intervals should be allowed to route through Lewis."""
    fwd = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=10.0)
    params = HestonParams(kappa=1.5768, theta=0.0398, nu=0.5751, rho=-0.5711, v0=0.0175)
    strikes = np.array([100.0])

    policy = COSGridPolicy(
        mode="benchmark",
        truncation="paper",
        centered=True,
        paper_L=32.0,
        dx_target=0.02,
        width_fallback=10.0,
        fallback_method="lewis",
    )
    decision = cos_adaptive_decision(
        heston_cumulants(fwd, params),
        model="heston",
        params=params,
        policy=policy,
    )
    assert decision.method == "lewis"

    got = price_strip("heston", "cos_improved", strikes, fwd, params, grid=policy)
    ref = lewis_call_prices(
        lambda u: heston_cf(u, fwd, params),
        strikes,
        spot=fwd.S0,
        texp=fwd.T,
        intr=fwd.r,
        divr=fwd.q,
        method="trapz",
        u_max=200.0,
        n_u=max(4096, decision.grid.N),
    )
    assert np.allclose(got, ref, atol=1e-11, rtol=0.0)


def test_price_strip_cos_improved_uses_stable_direct_call_regime_for_heston_strip():
    """Centered Heston intervals should avoid parity-cancellation issues."""
    fwd = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=1.0)
    params = HestonParams(kappa=1.5768, theta=0.0398, nu=0.5751, rho=-0.5711, v0=0.0175)
    strikes = np.arange(50.0, 151.0, 5.0)

    got = price_strip("heston", "cos_improved", strikes, fwd, params)
    ref = lewis_call_prices(
        lambda u: heston_cf(u, fwd, params),
        strikes,
        spot=fwd.S0,
        texp=fwd.T,
        intr=fwd.r,
        divr=fwd.q,
        method="trapz",
        u_max=250.0,
        n_u=8192,
    )
    assert np.allclose(got, ref, atol=1e-8, rtol=0.0)


def test_price_strip_cos_improved_short_maturity_vg_is_tightly_resolved():
    """Short-maturity VG benchmark case should stay tightly resolved."""
    fwd = ForwardSpec(S0=100.0, r=0.1, q=0.0, T=0.1)
    params = VGParams(sigma=0.12, nu=0.2, theta=-0.14)
    strikes = np.array([90.0])

    got = price_strip("vg", "cos_improved", strikes, fwd, params)
    ref = np.array([10.993703187])
    assert np.allclose(got, ref, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "case_id",
    JUNIKE_SUMMARY_CASE_IDS,
    ids=list(JUNIKE_SUMMARY_CASE_IDS),
)
def test_junike_saved_summary_cases_replay_cleanly(case_id: str):
    case, _phi, fwd, params, cums, strikes = case_context(case_id)
    ref = improved_reference_for_case(case_id)

    default_prices = price_strip(case.model, "cos", strikes, fwd, params)
    paper_prices = paper_cos_prices(case_id, max(case.Ns))
    policy = recommended_cos_policy(case.model, params, mode="benchmark")
    decision = cos_adaptive_decision(
        cums,
        model=case.model,
        params=params,
        policy=policy,
        strike_count=len(strikes),
    )
    improved_prices = price_strip(case.model, "cos_improved", strikes, fwd, params, grid=policy)

    expected = JUNIKE_COMPARE_DF.loc[case_id]

    assert_close(
        max_abs_err(default_prices, ref),
        float(expected["default_err"]),
        rtol=1e-5,
        atol=1e-10,
        label=f"{case_id} default_err",
    )
    assert_close(
        max_abs_err(paper_prices, ref),
        float(expected["paper_grid_err"]),
        rtol=1e-5,
        atol=1e-10,
        label=f"{case_id} paper_grid_err",
    )
    assert_close(
        max_abs_err(improved_prices, ref),
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
    case, _phi, fwd, params, cums, strikes = case_context(case_id)
    ref = improved_reference_for_case(case_id if case_id in JUNIKE_COMPARE_DF.index else case_id)

    default_prices = price_strip(case.model, "cos", strikes, fwd, params)
    paper_prices = paper_cos_prices(case_id, max(case.Ns))
    policy = recommended_cos_policy(case.model, params, mode="benchmark")
    decision = cos_adaptive_decision(
        cums,
        model=case.model,
        params=params,
        policy=policy,
        strike_count=len(strikes),
    )
    improved_prices = price_strip(case.model, "cos_improved", strikes, fwd, params, grid=policy)

    default_err = max_abs_err(default_prices, ref)
    paper_err = max_abs_err(paper_prices, ref)
    improved_err = max_abs_err(improved_prices, ref)

    assert np.all(np.isfinite(improved_prices)), f"{case_id}: non-finite improved prices"
    assert np.all(np.asarray(improved_prices) >= -1e-12), f"{case_id}: negative improved prices"

    allowed_ratio = 5.0 if case_id == "cgmy_table10_y198" else 1.05
    baseline = max(default_err, paper_err)
    assert improved_err <= baseline * allowed_ratio + 1e-12, (
        f"{case_id}: improved error drifted too far. "
        f"default={default_err:.3e}, paper={paper_err:.3e}, improved={improved_err:.3e}, "
        f"method={decision.method}, N={decision.grid.N}"
    )
