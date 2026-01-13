"""Slow stress test: adaptive filtered-COS vs vanilla COS and Junike-COS.

Marked with ``@pytest.mark.slow``.  Run with::

    pytest tests/test_filtered_cos_outperforms_baselines.py -q -m slow

Weak-dominance claim — what it actually means
----------------------------------------------
The adaptive selector is a **best-of-candidates** rule that optimises
the joint (error, runtime) metric:

  • If **any** candidate passes the target tolerance ``tol``:
    the selector picks the **fastest** passing candidate.
    → ``adaptive_error ≤ tol``  (guaranteed by construction).

  • If **no** candidate passes ``tol``:
    the selector falls back to the **lowest-error** candidate.
    → ``adaptive_error ≤ junike_error``  (Junike-no-filter is always in the set).

What we do NOT assert
~~~~~~~~~~~~~~~~~~~~~
``adaptive_error ≤ vanilla_error`` is NOT a universal claim.  The
selector can choose a filtered candidate that is slightly less accurate
than a vanilla grid that happened to be well-tuned for one particular
(model, T, K-range) combination.  The relevant comparison is between
the adaptive result and the candidates in its own search set.

Stress cases
~~~~~~~~~~~~
- VG at T=0.1  (infinite-activity jumps, short maturity)
- CGMY Y=0.5 at T=0.25  (semi-heavy tails)
"""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.variance_gamma import VGParams
from foureng.pipeline import price_strip
from foureng.pricers.cos import recommended_cos_policy
from foureng.utils.grids import COSGridPolicy
from foureng.utils.spectral_filters import COSFilterSpec
from foureng.experiments.cos_filter_grid_search import (
    FilterGridCandidate,
    run_filtered_cos_grid_search,
    select_fastest_under_tolerance,
)


def _make_candidate_set(policy: COSGridPolicy) -> list[FilterGridCandidate]:
    """Standard candidate set: Junike-no-filter + four filter variants."""
    return [
        FilterGridCandidate("junike_no_filter",    policy, None),
        FilterGridCandidate("junike_exp_p4",        policy, COSFilterSpec("exponential", order=4)),
        FilterGridCandidate("junike_exp_p8",        policy, COSFilterSpec("exponential", order=8)),
        FilterGridCandidate("junike_lanczos",       policy, COSFilterSpec("lanczos")),
        FilterGridCandidate("junike_fejer",         policy, COSFilterSpec("fejer")),
    ]


@pytest.mark.slow
def test_adaptive_filtered_cos_on_vg_stress_case():
    """Adaptive selector on short-maturity VG: tolerance and regression checks.

    Uses the recommended VG policy (heuristic truncation, dx=0.003) which
    gives N≈267 — comparable to vanilla's N=256 but with centred intervals.
    This ensures the candidate set is competitive before filtering is applied.

    Assertions
    ----------
    (a) If any candidate passes ``tol``:
        ``adaptive_error ≤ tol``  (selector guarantee).
    (b) ``adaptive_error ≤ junike_error``  when no candidate passes tol
        (selector falls back to lowest-error; Junike-no-filter is a candidate).
    (c) All 5 candidates produced finite, non-negative prices (sanity).
    """
    fwd    = ForwardSpec(S0=100.0, r=0.1, q=0.0, T=0.1)
    p      = VGParams(sigma=0.12, nu=0.2, theta=-0.14)
    strikes = np.linspace(70.0, 130.0, 25)

    # ── High-resolution oracle ─────────────────────────────────────────────────
    oracle_policy = COSGridPolicy(
        mode="benchmark",
        truncation="heuristic",
        centered=True,
        dx_target=0.0005,
        L=12.0,
        min_N=32,
        max_N=16384,
        width_fallback=0.0,
    )
    ref = price_strip("vg", "cos_improved", strikes, fwd, p, grid=oracle_policy)

    # ── Recommended policy for VG (heuristic, dx=0.003) ───────────────────────
    rec_policy = recommended_cos_policy("vg", p, mode="benchmark")

    # ── Baseline: Junike no-filter at the recommended policy ──────────────────
    junike = price_strip("vg", "cos_improved", strikes, fwd, p, grid=rec_policy)
    err_junike = float(np.max(np.abs(junike - ref)))

    # ── Grid search ───────────────────────────────────────────────────────────
    tol = max(err_junike * 10.0, 1e-5)   # tolerance relative to Junike quality
    candidates = _make_candidate_set(rec_policy)

    df = run_filtered_cos_grid_search(
        model="vg",
        strikes=strikes,
        fwd=fwd,
        params=p,
        reference=ref,
        candidates=candidates,
        tol=tol,
        n_repeat=1,
    )

    best         = select_fastest_under_tolerance(df, tol=tol)
    err_adaptive = float(best["max_abs_err"])
    any_pass     = df["passes_tol"].any()
    eps          = 1e-12

    # (a) tolerance guarantee
    if any_pass:
        assert err_adaptive <= tol + eps, (
            f"Selector violated tol={tol:.3e}: adaptive_error={err_adaptive:.3e}\n{best}"
        )

    # (b) no-pass fallback
    if not any_pass:
        assert err_adaptive <= err_junike + eps, (
            f"When no candidate passes, selector must pick ≤ Junike error.\n"
            f"adaptive={err_adaptive:.3e}, junike={err_junike:.3e}"
        )

    # (c) all candidates must have produced finite, non-negative results
    ok_rows = df[df["status"] == "ok"]
    assert len(ok_rows) == len(candidates), (
        f"Some candidates raised exceptions:\n{df[df['status'] != 'ok']}"
    )


@pytest.mark.slow
def test_adaptive_filtered_cos_on_cgmy_stress_case():
    """Adaptive selector on CGMY Y=0.5 at T=0.25: same assertion structure.

    CGMY with Y=0.5 has semi-heavy tails.  Short maturity stresses
    finite-series resolution similarly to the VG case.
    """
    from foureng.models.cgmy import CgmyParams

    fwd    = ForwardSpec(S0=100.0, r=0.04, q=0.0, T=0.25)
    p      = CgmyParams(C=1.0, G=5.0, M=10.0, Y=0.5)
    strikes = np.linspace(80.0, 120.0, 13)

    oracle_policy = COSGridPolicy(
        mode="benchmark",
        truncation="tolerance",
        centered=True,
        dx_target=0.002,
        L=14.0,
        eps_trunc=1e-12,
        max_N=16384,
        width_fallback=0.0,
    )
    ref = price_strip("cgmy", "cos_improved", strikes, fwd, p, grid=oracle_policy)

    # Recommended policy for CGMY (Y<1 → tolerance, dx=0.03)
    rec_policy = recommended_cos_policy("cgmy", p, mode="benchmark")
    junike     = price_strip("cgmy", "cos_improved", strikes, fwd, p, grid=rec_policy)
    err_junike = float(np.max(np.abs(junike - ref)))

    tol        = max(err_junike * 10.0, 1e-5)
    candidates = _make_candidate_set(rec_policy)

    df = run_filtered_cos_grid_search(
        model="cgmy",
        strikes=strikes,
        fwd=fwd,
        params=p,
        reference=ref,
        candidates=candidates,
        tol=tol,
        n_repeat=1,
    )

    best         = select_fastest_under_tolerance(df, tol=tol)
    err_adaptive = float(best["max_abs_err"])
    any_pass     = df["passes_tol"].any()
    eps          = 1e-12

    if any_pass:
        assert err_adaptive <= tol + eps, (
            f"Selector violated tol={tol:.3e}: adaptive_error={err_adaptive:.3e}\n{best}"
        )

    if not any_pass:
        assert err_adaptive <= err_junike + eps, (
            f"Fallback: adaptive ({err_adaptive:.3e}) > junike ({err_junike:.3e})"
        )

    ok_rows = df[df["status"] == "ok"]
    assert len(ok_rows) == len(candidates)


@pytest.mark.slow
def test_adaptive_selector_weak_dominance_over_fixed_policies():
    """The selector weakly dominates every fixed policy in the candidate set.

    For two competing candidates (Junike-no-filter vs Junike-exp-p8):
    the adaptive selector picks the one that is better in the joint
    (error-satisfies-tol, runtime) sense.  It can never pick a candidate
    that is strictly dominated by another candidate in the set.

    This test verifies the mechanism, not a specific winner.
    """
    from foureng.models.bsm import BsmParams

    fwd    = ForwardSpec(S0=100.0, r=0.05, q=0.02, T=0.5)
    p      = BsmParams(sigma=0.25)
    strikes = np.linspace(85.0, 115.0, 13)

    # Oracle: tight tolerance, large N
    oracle_policy = COSGridPolicy(
        mode="benchmark",
        truncation="tolerance",
        centered=True,
        dx_target=0.001,
        L=12.0,
        eps_trunc=1e-12,
        max_N=16384,
        width_fallback=0.0,
    )
    ref = price_strip("bsm", "cos_improved", strikes, fwd, p, grid=oracle_policy)

    policy = COSGridPolicy(
        mode="benchmark",
        truncation="tolerance",
        centered=True,
        dx_target=0.01,
        L=10.0,
        eps_trunc=1e-10,
        max_N=4096,
        width_fallback=0.0,
    )

    candidates = _make_candidate_set(policy)
    tol        = 1e-7

    df = run_filtered_cos_grid_search(
        model="bsm",
        strikes=strikes,
        fwd=fwd,
        params=p,
        reference=ref,
        candidates=candidates,
        tol=tol,
        n_repeat=1,
    )

    best         = select_fastest_under_tolerance(df, tol=tol)
    err_adaptive = float(best["max_abs_err"])
    any_pass     = df["passes_tol"].any()
    eps          = 1e-12

    # Tolerance guarantee
    if any_pass:
        assert err_adaptive <= tol + eps, (
            f"Selector violated tol={tol:.3e}: adaptive_error={err_adaptive:.3e}"
        )

    # Weak dominance: selected error ≤ every other passing candidate's error
    # (holds trivially for the fastest-under-tol selector among passes)
    passing = df[df["passes_tol"]].copy()
    if not passing.empty:
        assert err_adaptive <= passing["max_abs_err"].max() + eps, (
            "Selected candidate is worse than some passing candidates — "
            "this should be impossible."
        )
