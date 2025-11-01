"""Deterministic adaptive COS policy grid-search selector.

This module implements a grid-search over a set of numerical COS policies,
comparing:

  1. Vanilla COS (classic heuristic policy, no filter)
  2. Junike-style COS / cos_improved (tolerance-based truncation, no filter)
  3. Filtered Junike-COS (tolerance-based truncation + spectral filter)

For each candidate the selector records pricing error against a reference
solution and wall-clock runtime.  The selection rule is:

    fastest candidate satisfying ``max_abs_err ≤ tol``

or, if none passes:

    candidate with lowest ``max_abs_err`` among successful runs.

This is a **deterministic adaptive numerical-policy selector**, not a
black-box ML model.  Its dominance claim is intentionally conservative:
it weakly dominates every fixed policy in the candidate set, because
no-filter and Junike-only policies are always included as candidates.

Design references
-----------------
- Junike, G. and Pankrashkin, K. (2022), "Precise option pricing by the COS
  method—How to choose the truncation range," Applied Mathematics and
  Computation, 421, 126935.

- Ruijter, M. J., Versteegh, M. and Oosterlee, C. W. (2015), "On the
  application of spectral filters in a Fourier option pricing technique,"
  Journal of Computational Finance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from foureng.pipeline import price_strip
from foureng.utils.grids import COSGridPolicy
from foureng.utils.spectral_filters import COSFilterSpec


# ---------------------------------------------------------------------------
# Candidate dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterGridCandidate:
    """A single (policy, filter) candidate for the grid search.

    Parameters
    ----------
    method_label :
        Human-readable label stored in the results table.
    policy :
        COS adaptive-grid policy (:class:`~foureng.utils.grids.COSGridPolicy`).
    filter_spec :
        Spectral filter to apply, or ``None`` for no filtering.
    """

    method_label: str
    policy: COSGridPolicy
    filter_spec: COSFilterSpec | None


def policy_filter_candidates(
    policy: COSGridPolicy,
    *,
    label_prefix: str = "",
    no_filter_label: str | None = None,
) -> list[FilterGridCandidate]:
    """Build the compact candidate set used in the notebooks and demos."""
    prefix = f"{label_prefix}_" if label_prefix else ""
    none_label = no_filter_label or f"{prefix}no_filter"

    candidates = [FilterGridCandidate(none_label, policy, None)]
    specs = [
        ("fejer", COSFilterSpec("fejer")),
        ("lanczos", COSFilterSpec("lanczos")),
        ("raised_cosine", COSFilterSpec("raised_cosine")),
        ("exp_p4", COSFilterSpec("exponential", order=4)),
        ("exp_p8", COSFilterSpec("exponential", order=8)),
        ("exp_p12", COSFilterSpec("exponential", order=12)),
    ]
    for label, spec in specs:
        candidates.append(FilterGridCandidate(f"{prefix}{label}", policy, spec))
    return candidates


def filter_spec_from_result(row: pd.Series | dict[str, Any]) -> COSFilterSpec | None:
    """Recover a ``COSFilterSpec`` from a grid-search result row."""
    filter_name = str(row["filter"])
    if filter_name == "none":
        return None
    if filter_name == "exponential":
        order = int(row["filter_order"])
        return COSFilterSpec("exponential", order=order)
    return COSFilterSpec(filter_name)


def describe_filter_result(
    row: pd.Series | dict[str, Any],
    *,
    none_label: str = "no filter",
) -> str:
    """Human-readable filter label for notebook captions and legends."""
    spec = filter_spec_from_result(row)
    if spec is None:
        return none_label
    if spec.name == "exponential":
        return f"exponential p={int(spec.order)}"
    return spec.name.replace("_", " ")


# ---------------------------------------------------------------------------
# Internal timing helper
# ---------------------------------------------------------------------------

def _time_call(fn, n_repeat: int = 3):
    """Run ``fn()`` up to ``n_repeat`` times and return ``(result, best_ms)``.

    ``best_ms`` is the minimum elapsed wall-clock time in milliseconds.
    Taking the minimum (not the mean) removes OS scheduling noise that would
    unfairly penalise fast candidates.
    """
    best = float("inf")
    out = None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = fn()
        elapsed = (time.perf_counter() - t0) * 1e3
        if elapsed < best:
            best = elapsed
    return out, best


# ---------------------------------------------------------------------------
# Default candidate set
# ---------------------------------------------------------------------------

def default_filtered_cos_candidates() -> list[FilterGridCandidate]:
    """Build the default grid-search candidate set.

    Includes:
    - Classic COS policies (heuristic truncation, no filter)
    - Junike policies (tolerance truncation, no filter)
    - Junike + each spectral filter variant

    Returns
    -------
    list[FilterGridCandidate]
        All candidates to search.
    """
    base_policies = [
        COSGridPolicy(
            mode="benchmark",
            truncation="heuristic",
            centered=True,
            dx_target=dx,
            L=L,
            min_N=32,
            max_N=4096,
            width_fallback=0.0,
        )
        for dx in [0.02, 0.01, 0.005]
        for L in [8.0, 10.0, 12.0, 16.0]
    ]

    improved_policies = [
        COSGridPolicy(
            mode="benchmark",
            truncation="tolerance",
            centered=True,
            dx_target=dx,
            L=L,
            eps_trunc=eps,
            min_N=32,
            max_N=8192,
            width_fallback=0.0,
        )
        for dx in [0.02, 0.01, 0.005, 0.003]
        for L in [8.0, 10.0, 12.0]
        for eps in [1e-8, 1e-10, 1e-12]
    ]

    out: list[FilterGridCandidate] = []

    # Classic policies — no filter (Junike is not applied here)
    for policy in base_policies:
        out.append(FilterGridCandidate("classic_cos_policy", policy, None))

    # Improved (Junike) policies — no filter + each filter variant
    for policy in improved_policies:
        out.extend(
            policy_filter_candidates(
                policy,
                label_prefix="junike",
                no_filter_label="junike_policy",
            )
        )

    return out


# ---------------------------------------------------------------------------
# Grid-search runner
# ---------------------------------------------------------------------------

def run_filtered_cos_grid_search(
    *,
    model: str,
    strikes: np.ndarray,
    fwd: Any,
    params: Any,
    reference: np.ndarray,
    candidates: list[FilterGridCandidate] | None = None,
    tol: float = 1e-8,
    n_repeat: int = 3,
) -> pd.DataFrame:
    """Run all candidates and return a results table sorted by quality.

    For each candidate the function records:

    - ``runtime_ms``    — minimum wall-clock time over ``n_repeat`` runs.
    - ``max_abs_err``   — max|price − reference| across all strikes.
    - ``mean_abs_err``  — mean|price − reference|.
    - ``passes_tol``    — whether ``max_abs_err ≤ tol``.
    - ``status``        — ``"ok"`` or ``"fail: <ExcType>: <message>"``.

    The returned DataFrame is sorted: passing candidates first (by runtime),
    then failing candidates (by error).

    Parameters
    ----------
    model : str
        Model name string (e.g. ``"vg"``, ``"heston"``).
    strikes : np.ndarray
        Strike array used for pricing and error measurement.
    fwd :
        :class:`~foureng.models.base.ForwardSpec`.
    params :
        Model parameter dataclass.
    reference : np.ndarray
        High-resolution reference prices, same shape as ``strikes``.
    candidates :
        List of :class:`FilterGridCandidate`.  Defaults to
        :func:`default_filtered_cos_candidates`.
    tol : float
        Target error tolerance (used for ``passes_tol`` flag).
    n_repeat : int
        Number of timing repetitions; minimum time is reported.

    Returns
    -------
    pd.DataFrame
    """
    if candidates is None:
        candidates = default_filtered_cos_candidates()

    rows = []

    for cand in candidates:
        try:
            # Pick method and grid arg based on whether a filter is requested
            if cand.filter_spec is None:
                method = "cos_improved"
                grid_arg = cand.policy
            else:
                method = "cos_filtered"
                grid_arg = (cand.policy, cand.filter_spec)

            prices, runtime_ms = _time_call(
                lambda m=method, g=grid_arg: price_strip(
                    model, m, strikes, fwd, params, grid=g
                ),
                n_repeat=n_repeat,
            )

            err = np.asarray(prices, dtype=float) - np.asarray(reference, dtype=float)
            max_abs_err = float(np.max(np.abs(err)))
            mean_abs_err = float(np.mean(np.abs(err)))

            rows.append({
                "method_label":  cand.method_label,
                "truncation":    cand.policy.truncation,
                "dx_target":     cand.policy.dx_target,
                "L":             cand.policy.L,
                "eps_trunc":     cand.policy.eps_trunc,
                "filter":        "none" if cand.filter_spec is None else cand.filter_spec.name,
                "filter_order":  np.nan if cand.filter_spec is None else cand.filter_spec.order,
                "runtime_ms":    runtime_ms,
                "max_abs_err":   max_abs_err,
                "mean_abs_err":  mean_abs_err,
                "passes_tol":    max_abs_err <= tol,
                "status":        "ok",
            })

        except Exception as exc:
            rows.append({
                "method_label":  cand.method_label,
                "truncation":    getattr(cand.policy, "truncation", None),
                "dx_target":     getattr(cand.policy, "dx_target", None),
                "L":             getattr(cand.policy, "L", None),
                "eps_trunc":     getattr(cand.policy, "eps_trunc", None),
                "filter":        "none" if cand.filter_spec is None else cand.filter_spec.name,
                "filter_order":  np.nan if cand.filter_spec is None else cand.filter_spec.order,
                "runtime_ms":    np.nan,
                "max_abs_err":   np.inf,
                "mean_abs_err":  np.inf,
                "passes_tol":    False,
                "status":        f"fail: {type(exc).__name__}: {exc}",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values(
        ["passes_tol", "max_abs_err", "runtime_ms"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Best-candidate selector
# ---------------------------------------------------------------------------

def select_fastest_under_tolerance(df: pd.DataFrame, tol: float) -> pd.Series:
    """Select the fastest candidate satisfying ``max_abs_err ≤ tol``.

    If no candidate satisfies the tolerance, fall back to the candidate
    with the lowest ``max_abs_err`` among those with ``status == "ok"``.
    If all runs failed, return the row with the lowest ``max_abs_err``
    regardless of status.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`run_filtered_cos_grid_search`.
    tol : float
        Error tolerance threshold.

    Returns
    -------
    pd.Series
        The selected row.
    """
    ok = df[(df["status"] == "ok") & (df["max_abs_err"] <= tol)].copy()
    if not ok.empty:
        return ok.sort_values(["runtime_ms", "max_abs_err"]).iloc[0]

    # No candidate passes — take lowest error among successful runs
    valid = df[df["status"] == "ok"].copy()
    if not valid.empty:
        return valid.sort_values(["max_abs_err", "runtime_ms"]).iloc[0]

    # All failed — report the "least bad" regardless
    return df.sort_values("max_abs_err").iloc[0]
