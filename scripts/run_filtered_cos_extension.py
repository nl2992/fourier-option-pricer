#!/usr/bin/env python
"""Standalone runner for the adaptive filtered-COS stress-case grid search.

Callable from the command line or from the demo notebook.  Writes:

  - benchmarks/mc_vs_fourier_methods/outputs/cos_policy_search_showcase.csv
  - benchmarks/mc_vs_fourier_methods/outputs/adaptive_filtered_cos_model_zoo.csv

Figures are also written if matplotlib is available.

Usage (CLI)
-----------
    python scripts/run_filtered_cos_extension.py
    python scripts/run_filtered_cos_extension.py --no-figures

Usage (notebook)
----------------
    import importlib, runpy
    runpy.run_path("scripts/run_filtered_cos_extension.py")

The script is NOT required by any test.  Tests use the
``foureng.experiments.cos_filter_grid_search`` module directly with a
tiny candidate set so CI runs fast.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

# ── Ensure foureng is importable when run from the repo root ────────────────
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams
from foureng.models.heston import HestonParams
from foureng.models.variance_gamma import VGParams
from foureng.models.cgmy import CgmyParams
from foureng.models.kou import KouParams
from foureng.pipeline import price_strip
from foureng.pricers.cos import recommended_cos_policy
from foureng.utils.grids import COSGridPolicy
from foureng.experiments.cos_filter_grid_search import (
    policy_filter_candidates,
    run_filtered_cos_grid_search,
    select_fastest_under_tolerance,
)

# ── Output paths ─────────────────────────────────────────────────────────────
_OUT = _ROOT / "benchmarks" / "mc_vs_fourier_methods" / "outputs"
_FIG = _OUT / "figures"
_OUT.mkdir(parents=True, exist_ok=True)
_FIG.mkdir(parents=True, exist_ok=True)

TOL = 1e-6


# ── Model-zoo stress cases ────────────────────────────────────────────────────

STRESS_CASES = [
    {
        "case_name": "BSM_T1",
        "model":     "bsm",
        "fwd":       ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0),
        "params":    BsmParams(sigma=0.25),
        "strikes":   np.linspace(80.0, 120.0, 13),
    },
    {
        "case_name": "Heston_T1",
        "model":     "heston",
        "fwd":       ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0),
        "params":    HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.7, v0=0.04),
        "strikes":   np.linspace(80.0, 120.0, 13),
    },
    {
        "case_name": "VG_T01_short",
        "model":     "vg",
        "fwd":       ForwardSpec(S0=100.0, r=0.1, q=0.0, T=0.1),
        "params":    VGParams(sigma=0.12, nu=0.2, theta=-0.14),
        "strikes":   np.linspace(70.0, 130.0, 25),
    },
    {
        "case_name": "VG_T1",
        "model":     "vg",
        "fwd":       ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0),
        "params":    VGParams(sigma=0.12, nu=0.2, theta=-0.14),
        "strikes":   np.linspace(80.0, 120.0, 13),
    },
    {
        "case_name": "CGMY_T025",
        "model":     "cgmy",
        "fwd":       ForwardSpec(S0=100.0, r=0.04, q=0.0, T=0.25),
        "params":    CgmyParams(C=1.0, G=5.0, M=10.0, Y=0.5),
        "strikes":   np.linspace(80.0, 120.0, 13),
    },
    {
        "case_name": "Kou_T05",
        "model":     "kou",
        "fwd":       ForwardSpec(S0=100.0, r=0.04, q=0.0, T=0.5),
        "params":    KouParams(sigma=0.16, lam=0.30, p=0.40, eta1=10.0, eta2=6.0),
        "strikes":   np.linspace(80.0, 120.0, 13),
    },
]


def _oracle_policy(model: str, params) -> COSGridPolicy:
    """High-resolution reference policy."""
    return COSGridPolicy(
        mode="benchmark",
        truncation="tolerance" if model not in {"vg"} else "heuristic",
        centered=True,
        dx_target=0.001,
        L=14.0,
        eps_trunc=1e-12,
        min_N=64,
        max_N=16384,
        width_fallback=0.0,
    )


def _candidate_set(policy: COSGridPolicy):
    return policy_filter_candidates(policy, label_prefix="junike")


# ── Showcase table ─────────────────────────────────────────────────────────────

def build_showcase(write_figures: bool = True) -> pd.DataFrame:
    """Run the policy grid search for all stress cases and build the showcase table."""
    rows = []

    for case in STRESS_CASES:
        name    = case["case_name"]
        model   = case["model"]
        fwd     = case["fwd"]
        params  = case["params"]
        strikes = case["strikes"]

        print(f"  [{name}] computing oracle ...", flush=True)
        ref = price_strip(model, "cos_improved", strikes, fwd, params,
                          grid=_oracle_policy(model, params))

        # vanilla COS
        van = price_strip(model, "cos", strikes, fwd, params)
        err_van = float(np.max(np.abs(van - ref)))

        # timing for vanilla
        import time
        t0 = time.perf_counter()
        _ = price_strip(model, "cos", strikes, fwd, params)
        rt_van = (time.perf_counter() - t0) * 1e3

        rec_policy = recommended_cos_policy(model, params, mode="benchmark")

        # Junike COS (recommended policy, no filter)
        jun = price_strip(model, "cos_improved", strikes, fwd, params, grid=rec_policy)
        err_jun = float(np.max(np.abs(jun - ref)))
        t0 = time.perf_counter()
        _ = price_strip(model, "cos_improved", strikes, fwd, params, grid=rec_policy)
        rt_jun = (time.perf_counter() - t0) * 1e3

        # Adaptive filtered search
        print(f"  [{name}] running grid search ({len(_candidate_set(rec_policy))} candidates)...", flush=True)
        df_search = run_filtered_cos_grid_search(
            model=model,
            strikes=strikes,
            fwd=fwd,
            params=params,
            reference=ref,
            candidates=_candidate_set(rec_policy),
            tol=TOL,
            n_repeat=3,
        )

        best     = select_fastest_under_tolerance(df_search, tol=TOL)
        err_adap = float(best["max_abs_err"])
        rt_adap  = float(best["runtime_ms"])

        # Determine adaptive_result_label
        if err_adap < err_jun - 1e-12 and err_adap < err_van - 1e-12:
            label = "strictly improves error"
        elif err_adap <= TOL and rt_adap < rt_jun:
            label = "matches tolerance faster"
        elif err_adap <= TOL:
            label = "matches Junike within tolerance"
        else:
            label = "no improvement found"

        rows.append({
            "case_name":                  name,
            "model":                      model,
            "maturity":                   fwd.T,
            "strike_count":               len(strikes),
            "vanilla_cos_error":          err_van,
            "vanilla_cos_runtime_ms":     rt_van,
            "junike_cos_error":           err_jun,
            "junike_cos_runtime_ms":      rt_jun,
            "adaptive_filtered_error":    err_adap,
            "adaptive_filtered_runtime_ms": rt_adap,
            "selected_filter":            best["filter"],
            "selected_policy":            f"dx={best['dx_target']},L={best['L']}",
            "adaptive_result_label":      label,
        })

        # Per-case figure
        if write_figures:
            _write_case_scatter(name, df_search, _FIG)

    df = pd.DataFrame(rows)
    csv_path = _OUT / "cos_policy_search_showcase.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved showcase → {csv_path}")
    return df


def _write_case_scatter(case_name: str, df: pd.DataFrame, fig_dir: pathlib.Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return

    colours = {
        "classic_cos_policy": "C7",
        "junike_policy":      "C0",
    }
    markers = {
        "none":          "o",
        "fejer":         "s",
        "lanczos":       "^",
        "raised_cosine": "D",
        "exponential":   "*",
    }

    for _, row in ok.iterrows():
        ml  = row["method_label"].split("_")[0] + "_" + row["method_label"].split("_")[1] \
              if len(row["method_label"].split("_")) > 1 else row["method_label"]
        col = colours.get(
            "classic_cos_policy" if "classic" in row["method_label"] else "junike_policy",
            "C3",
        )
        mk  = markers.get(row["filter"], "o")
        ax.scatter(
            row["runtime_ms"],
            max(row["max_abs_err"], 1e-16),
            c=col, marker=mk, s=60, alpha=0.6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("runtime (ms)")
    ax.set_ylabel("max|error|")
    ax.set_title(f"COS policy search — {case_name}")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    path = fig_dir / f"cos_policy_search_{case_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── Model-zoo rerun table ──────────────────────────────────────────────────────

def build_model_zoo_table(showcase_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the showcase into the model-zoo rerun format."""
    rows = []
    for _, r in showcase_df.iterrows():
        err_v   = r["vanilla_cos_error"]
        err_j   = r["junike_cos_error"]
        err_a   = r["adaptive_filtered_error"]

        beats_v = err_a < err_v
        beats_j = err_a < err_j
        within  = err_a <= TOL

        if beats_v and beats_j:
            status = "adaptive best"
        elif within and beats_v:
            status = "adaptive matches best within tolerance"
        elif not beats_v and not beats_j:
            status = "Junike still best"
        elif err_v < err_j and err_v < err_a:
            status = "vanilla still best"
        else:
            status = "adaptive matches best within tolerance"

        rows.append({
            "case_name":                     r["case_name"],
            "model":                         r["model"],
            "maturity":                      r["maturity"],
            "vanilla_cos_error":             err_v,
            "junike_cos_error":              err_j,
            "adaptive_filtered_error":       err_a,
            "vanilla_cos_runtime_ms":        r["vanilla_cos_runtime_ms"],
            "junike_cos_runtime_ms":         r["junike_cos_runtime_ms"],
            "adaptive_filtered_runtime_ms":  r["adaptive_filtered_runtime_ms"],
            "selected_filter":               r["selected_filter"],
            "selected_policy":               r["selected_policy"],
            "adaptive_beats_vanilla":        beats_v,
            "adaptive_beats_junike":         beats_j,
            "adaptive_matches_best_within_tolerance": within,
            "final_status":                  status,
        })

    df = pd.DataFrame(rows)
    csv_path = _OUT / "adaptive_filtered_cos_model_zoo.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved model-zoo → {csv_path}")
    return df


def write_model_zoo_figures(df: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cases  = df["case_name"].tolist()
    x      = np.arange(len(cases))
    width  = 0.25

    # ── Error bar chart ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(cases) * 1.4), 5))
    ax.bar(x - width, np.clip(df["vanilla_cos_error"],   1e-16, None), width, label="Vanilla COS",   color="C7")
    ax.bar(x,          np.clip(df["junike_cos_error"],    1e-16, None), width, label="Junike COS",    color="C0")
    ax.bar(x + width,  np.clip(df["adaptive_filtered_error"], 1e-16, None), width, label="Adaptive filtered", color="C2")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=30, ha="right")
    ax.set_ylabel("max|error|")
    ax.set_title("Adaptive filtered-COS: model-zoo error comparison")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3, axis="y")
    fig.tight_layout()
    path = _FIG / "adaptive_filtered_cos_model_zoo_errors.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved figure → {path}")

    # ── Runtime bar chart ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(cases) * 1.4), 5))
    ax.bar(x - width, df["vanilla_cos_runtime_ms"],          width, label="Vanilla COS",   color="C7")
    ax.bar(x,          df["junike_cos_runtime_ms"],           width, label="Junike COS",    color="C0")
    ax.bar(x + width,  df["adaptive_filtered_runtime_ms"],   width, label="Adaptive filtered", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=30, ha="right")
    ax.set_ylabel("runtime (ms)")
    ax.set_title("Adaptive filtered-COS: model-zoo runtime comparison")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3, axis="y")
    fig.tight_layout()
    path = _FIG / "adaptive_filtered_cos_model_zoo_runtime.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved figure → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(write_figures: bool = True):
    print("═" * 60)
    print("  Adaptive filtered-COS extension runner")
    print("═" * 60)

    print("\nPhase 1: Showcase stress-case grid search")
    showcase = build_showcase(write_figures=write_figures)

    print("\nPhase 2: Model-zoo rerun summary table")
    zoo = build_model_zoo_table(showcase)

    if write_figures:
        print("\nPhase 3: Generating model-zoo figures")
        write_model_zoo_figures(zoo)

    # Print summary
    print("\n── Summary ────────────────────────────────────────────────")
    for _, r in zoo.iterrows():
        print(
            f"  {r['case_name']:<22} | "
            f"vanilla={r['vanilla_cos_error']:.2e}  "
            f"junike={r['junike_cos_error']:.2e}  "
            f"adaptive={r['adaptive_filtered_error']:.2e}  "
            f"→ {r['final_status']}"
        )

    print("\n  Done.\n")
    return showcase, zoo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run adaptive filtered-COS extension")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip matplotlib figure generation")
    args = parser.parse_args()
    main(write_figures=not args.no_figures)
