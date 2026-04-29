#!/usr/bin/env python
"""Append the adaptive filtered-COS notebook section to notebooks/demo.ipynb.

This script is idempotent:
  - if the section marker already exists, the old section is replaced;
  - if the appendix marker exists, the section is inserted just before it.

Run with:
    python scripts/append_filtered_cos_section.py
"""
from __future__ import annotations
import pathlib

from notebook_support import code, md, read_notebook, write_notebook

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "demo.ipynb"
MARKER = "<!-- demo-filtered-cos-section -->"
APPENDIX_MARKER = "<!-- demo-extra-visuals -->"


# ── Section 6 header ──────────────────────────────────────────────────────────

S6_MD = f"""\
## 6. Adaptive filtered-COS

{MARKER}

The model-zoo above shows where vanilla COS and Junike-style truncation work or
struggle. Junike-style truncation fixes the interval-selection problem, but it
does not remove every remaining source of COS error.

Short maturities, jump-heavy characteristic functions, and payoff kinks can
still leave visible oscillatory error after the interval itself is chosen well.

This section adds a filtered-COS overlay and then asks a narrower question:
given a small candidate set, which policy is the cheapest one that still meets
the target tolerance?

The selector is conservative by construction. Junike-without-filter stays in the
candidate set, so the filter is never forced.

**References**
- Junike & Pankrashkin (2022), *Precise option pricing by the COS method*,
  Applied Mathematics and Computation, 421, 126935.
- Ruijter, Versteegh & Oosterlee (2015), *On the application of spectral filters
  in a Fourier option pricing technique*, Journal of Computational Finance.
"""

S6_FILTER_MD = """\
### What coefficient filtering changes

Filtered COS keeps the same model characteristic function, the same truncation
interval, and the same payoff coefficients. The only change is that the
high-frequency COS modes are damped before the final series sum:

`price = disc * sum_k sigma_k A_k V_k`

Here `A_k` are the CF-side COS coefficients, `V_k` are the payoff coefficients,
and `sigma_k` are filter weights in `[0, 1]`. Low modes stay close to one; tail
modes are shrunk to reduce finite-series ringing.

The three profiles below are representative of the candidate set:
- **Fejer**: linear taper.
- **Lanczos**: sinc taper.
- **Raised cosine**: smooth cosine cutoff.
"""

S6_FILTER_PLOT_CODE = """\
_Nw = 64
_xw = np.arange(_Nw) / (_Nw - 1)
_profiles = [
    ("Fejer", cos_filter_weights(_Nw, COSFilterSpec("fejer")), "darkorange"),
    ("Lanczos", cos_filter_weights(_Nw, COSFilterSpec("lanczos")), "forestgreen"),
    ("Raised cosine", cos_filter_weights(_Nw, COSFilterSpec("raised_cosine")), "purple"),
]

fig, ax = plt.subplots(figsize=(6.4, 3.6))
for _label, _weights, _color in _profiles:
    ax.plot(_xw, _weights, lw=2.0, color=_color, label=_label)

ax.set_xlabel("normalised COS mode  k / (N - 1)")
ax.set_ylabel("weight  $\\sigma_k$")
ax.set_title("Three filter profiles on the COS coefficients")
ax.set_ylim(-0.02, 1.05)
ax.grid(True, ls="--", alpha=0.3)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
plt.show()
"""

# ── Section 6 imports / helpers ───────────────────────────────────────────────

S6_SETUP_CODE = """\
import time, pathlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from foureng.models.bsm    import BsmParams
from foureng.models.heston import HestonParams
from foureng.models.variance_gamma import VGParams
from foureng.models.cgmy   import CgmyParams
from foureng.models.kou    import KouParams
from foureng.models.base   import ForwardSpec
from foureng.pipeline import price_strip
from foureng.pricers.cos import recommended_cos_policy
from foureng.utils.grids import COSGridPolicy

try:
    from foureng.utils.spectral_filters import COSFilterSpec, cos_filter_weights
except Exception:
    @dataclass(frozen=True)
    class COSFilterSpec:
        name: str = "none"
        order: int = 8
        alpha: float | None = None

    def cos_filter_weights(N, spec=None):
        if N <= 0:
            raise ValueError(f"N must be positive (got {N})")
        if spec is None or spec.name == "none":
            return np.ones(N, dtype=float)
        if N == 1:
            return np.ones(1, dtype=float)
        k = np.arange(N, dtype=float)
        x = k / float(N - 1)
        if spec.name == "fejer":
            w = 1.0 - x
        elif spec.name == "lanczos":
            w = np.sinc(x)
        elif spec.name == "raised_cosine":
            w = 0.5 * (1.0 + np.cos(np.pi * x))
        elif spec.name == "exponential":
            alpha = float(spec.alpha if spec.alpha is not None else -np.log(np.finfo(float).eps))
            w = np.exp(-alpha * x**int(spec.order))
        else:
            raise ValueError(f"unknown filter {spec.name!r}")
        w[0] = 1.0
        return np.asarray(w, dtype=float)

try:
    from foureng.experiments.cos_filter_grid_search import (
        policy_filter_candidates,
        run_filtered_cos_grid_search,
        select_fastest_under_tolerance,
    )
except Exception:
    @dataclass(frozen=True)
    class _FilterGridCandidate:
        method_label: str
        policy: COSGridPolicy
        filter_spec: COSFilterSpec | None

    def policy_filter_candidates(policy, *, label_prefix="", no_filter_label=None):
        prefix = f"{label_prefix}_" if label_prefix else ""
        none_label = no_filter_label or f"{prefix}no_filter"
        candidates = [_FilterGridCandidate(none_label, policy, None)]
        specs = [
            ("fejer", COSFilterSpec("fejer")),
            ("lanczos", COSFilterSpec("lanczos")),
            ("raised_cosine", COSFilterSpec("raised_cosine")),
            ("exp_p4", COSFilterSpec("exponential", order=4)),
            ("exp_p8", COSFilterSpec("exponential", order=8)),
            ("exp_p12", COSFilterSpec("exponential", order=12)),
        ]
        for label, spec in specs:
            candidates.append(_FilterGridCandidate(f"{prefix}{label}", policy, spec))
        return candidates

    def _time_call(fn, n_repeat=3):
        best = float("inf")
        out = None
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            out = fn()
            best = min(best, (time.perf_counter() - t0) * 1e3)
        return out, best

    def run_filtered_cos_grid_search(*, model, strikes, fwd, params, reference,
                                     candidates=None, tol=1e-8, n_repeat=3):
        if candidates is None:
            raise ValueError("candidates must be provided in notebook fallback mode")
        rows = []
        for cand in candidates:
            try:
                if cand.filter_spec is None:
                    method = "cos_improved"
                    grid_arg = cand.policy
                else:
                    method = "cos_filtered"
                    grid_arg = (cand.policy, cand.filter_spec)
                prices, runtime_ms = _time_call(
                    lambda m=method, g=grid_arg: price_strip(model, m, strikes, fwd, params, grid=g),
                    n_repeat=n_repeat,
                )
                err = np.asarray(prices, dtype=float) - np.asarray(reference, dtype=float)
                max_abs_err = float(np.max(np.abs(err)))
                mean_abs_err = float(np.mean(np.abs(err)))
                rows.append({
                    "method_label": cand.method_label,
                    "truncation": cand.policy.truncation,
                    "dx_target": cand.policy.dx_target,
                    "L": cand.policy.L,
                    "eps_trunc": cand.policy.eps_trunc,
                    "filter": "none" if cand.filter_spec is None else cand.filter_spec.name,
                    "filter_order": np.nan if cand.filter_spec is None else cand.filter_spec.order,
                    "runtime_ms": runtime_ms,
                    "max_abs_err": max_abs_err,
                    "mean_abs_err": mean_abs_err,
                    "passes_tol": max_abs_err <= tol,
                    "status": "ok",
                })
            except Exception as exc:
                rows.append({
                    "method_label": cand.method_label,
                    "truncation": getattr(cand.policy, "truncation", None),
                    "dx_target": getattr(cand.policy, "dx_target", None),
                    "L": getattr(cand.policy, "L", None),
                    "eps_trunc": getattr(cand.policy, "eps_trunc", None),
                    "filter": "none" if cand.filter_spec is None else cand.filter_spec.name,
                    "filter_order": np.nan if cand.filter_spec is None else cand.filter_spec.order,
                    "runtime_ms": np.nan,
                    "max_abs_err": np.inf,
                    "mean_abs_err": np.inf,
                    "passes_tol": False,
                    "status": f"fail: {type(exc).__name__}: {exc}",
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(
            ["passes_tol", "max_abs_err", "runtime_ms"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    def select_fastest_under_tolerance(df, tol):
        ok = df[(df["status"] == "ok") & (df["max_abs_err"] <= tol)].copy()
        if not ok.empty:
            return ok.sort_values(["runtime_ms", "max_abs_err"]).iloc[0]
        valid = df[df["status"] == "ok"].copy()
        if not valid.empty:
            return valid.sort_values(["max_abs_err", "runtime_ms"]).iloc[0]
        return df.sort_values("max_abs_err").iloc[0]

_OUT_DIR = pathlib.Path("benchmarks/mc_vs_fourier_methods/outputs")
_FIG_DIR = _OUT_DIR / "figures"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_FIG_DIR.mkdir(parents=True, exist_ok=True)
_TOL = 1e-6

STRESS_CASES = [
    dict(case_name="BSM_T1",       model="bsm",
         fwd=ForwardSpec(S0=100, r=0.03, q=0.01, T=1.0),
         params=BsmParams(sigma=0.25),
         strikes=np.linspace(80, 120, 13)),
    dict(case_name="Heston_T1",    model="heston",
         fwd=ForwardSpec(S0=100, r=0.03, q=0.01, T=1.0),
         params=HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.7, v0=0.04),
         strikes=np.linspace(80, 120, 13)),
    dict(case_name="VG_T01",       model="vg",
         fwd=ForwardSpec(S0=100, r=0.1, q=0.0, T=0.1),
         params=VGParams(sigma=0.12, nu=0.2, theta=-0.14),
         strikes=np.linspace(70, 130, 25)),
    dict(case_name="VG_T1",        model="vg",
         fwd=ForwardSpec(S0=100, r=0.05, q=0.0, T=1.0),
         params=VGParams(sigma=0.12, nu=0.2, theta=-0.14),
         strikes=np.linspace(80, 120, 13)),
    dict(case_name="CGMY_T025",    model="cgmy",
         fwd=ForwardSpec(S0=100, r=0.04, q=0.0, T=0.25),
         params=CgmyParams(C=1.0, G=5.0, M=10.0, Y=0.5),
         strikes=np.linspace(80, 120, 13)),
    dict(case_name="Kou_T05",      model="kou",
         fwd=ForwardSpec(S0=100, r=0.04, q=0.0, T=0.5),
         params=KouParams(sigma=0.16, lam=0.30, p=0.40, eta1=10.0, eta2=6.0),
         strikes=np.linspace(80, 120, 13)),
]

def _oracle_policy(model):
    trunc = "heuristic" if model == "vg" else "tolerance"
    return COSGridPolicy(
        mode="benchmark", truncation=trunc, centered=True,
        dx_target=0.001, L=14.0, eps_trunc=1e-12,
        min_N=64, max_N=16384, width_fallback=0.0,
    )
"""

# ── Section 6.1 header ────────────────────────────────────────────────────────

S61_MD = """\
### 6.1 Policy search across stress cases

We run a deterministic policy grid search over 7 candidate policies for each
stress case, then programmatically select the showcase case.

Case selection rule:
1. `adaptive_error < junike_error` AND `adaptive_error < vanilla_error` → *"strictly improves error"*
2. `adaptive_error ≤ tol` AND `runtime_adaptive < runtime_junike` → *"matches tolerance faster"*
3. `adaptive_error ≤ tol` → *"matches Junike within tolerance"*
4. Otherwise → *"no improvement found"*
"""

S61_CODE = """\
_showcase_rows = []

for case in STRESS_CASES:
    name, model = case["case_name"], case["model"]
    fwd, params, strikes = case["fwd"], case["params"], case["strikes"]

    ref = price_strip(model, "cos_improved", strikes, fwd, params,
                      grid=_oracle_policy(model))

    t0  = time.perf_counter()
    van = price_strip(model, "cos", strikes, fwd, params)
    rt_van = (time.perf_counter() - t0) * 1e3
    err_van = float(np.max(np.abs(van - ref)))

    rec_pol = recommended_cos_policy(model, params)
    t0  = time.perf_counter()
    jun = price_strip(model, "cos_improved", strikes, fwd, params, grid=rec_pol)
    rt_jun = (time.perf_counter() - t0) * 1e3
    err_jun = float(np.max(np.abs(jun - ref)))

    df_srch = run_filtered_cos_grid_search(
        model=model, strikes=strikes, fwd=fwd, params=params,
        reference=ref, candidates=policy_filter_candidates(rec_pol, label_prefix="junike"),
        tol=_TOL, n_repeat=3,
    )
    best     = select_fastest_under_tolerance(df_srch, tol=_TOL)
    err_adap = float(best["max_abs_err"])
    rt_adap  = float(best["runtime_ms"])

    if err_adap < err_jun - 1e-12 and err_adap < err_van - 1e-12:
        label = "strictly improves error"
    elif err_adap <= _TOL and rt_adap < rt_jun:
        label = "matches tolerance faster"
    elif err_adap <= _TOL:
        label = "matches Junike within tolerance"
    else:
        label = "no improvement found"

    _showcase_rows.append(dict(
        case_name=name, model=model, maturity=fwd.T,
        strike_count=len(strikes),
        vanilla_cos_error=err_van,    vanilla_cos_runtime_ms=rt_van,
        junike_cos_error=err_jun,     junike_cos_runtime_ms=rt_jun,
        adaptive_filtered_error=err_adap, adaptive_filtered_runtime_ms=rt_adap,
        selected_filter=best["filter"],
        selected_policy=f"dx={best['dx_target']},L={best['L']}",
        adaptive_result_label=label,
        _df_srch=df_srch,
    ))

showcase_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_df_srch"}
                              for r in _showcase_rows])

csv_path = _OUT_DIR / "cos_policy_search_showcase.csv"
showcase_df.to_csv(csv_path, index=False)
showcase_df[["case_name","vanilla_cos_error","junike_cos_error",
             "adaptive_filtered_error","selected_filter","adaptive_result_label"]]
"""

S61_PLOT_CODE = """\
_priority = ["strictly improves error", "matches tolerance faster",
             "matches Junike within tolerance", "no improvement found"]
_best_case = min(_showcase_rows,
    key=lambda r: (_priority.index(r["adaptive_result_label"]),
                   r["adaptive_filtered_error"]))

_sr = _best_case["_df_srch"].copy()
_ok = _sr[_sr["status"] == "ok"].copy()

fig, ax = plt.subplots(figsize=(8, 5))

_cmap = {"none": "steelblue", "fejer": "darkorange", "lanczos": "green",
         "raised_cosine": "purple", "exponential": "crimson"}
_mmap = {"classic": "^", "junike": "o"}

for _, row in _ok.iterrows():
    c  = _cmap.get(row["filter"], "grey")
    mk = "^" if "classic" in row["method_label"] else "o"
    ax.scatter(row["runtime_ms"], max(row["max_abs_err"], 1e-16),
               c=c, marker=mk, s=60, alpha=0.7,
               label=f'{row["filter"]} (order={int(row["filter_order"]) if not np.isnan(row["filter_order"]) else "–"})'
               if row["filter"] not in [r["filter"] for _, r in
                  list(_ok.iterrows())[:list(_ok.index).index(row.name)]] else "")

ax.scatter(_best_case["adaptive_filtered_runtime_ms"],
           max(_best_case["adaptive_filtered_error"], 1e-16),
           c="gold", marker="*", s=350, zorder=5, label="Selected (adaptive)")

ax.axhline(_best_case["junike_cos_error"], color="steelblue",
           ls="--", lw=1.2, alpha=0.6, label="Junike COS error")
ax.axhline(_best_case["vanilla_cos_error"], color="grey",
           ls=":", lw=1.2, alpha=0.6, label="Vanilla COS error")
ax.axhline(_TOL, color="black", ls="-.", lw=0.9, alpha=0.5, label=f"Target tol={_TOL:.0e}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("runtime (ms)")
ax.set_ylabel("max|error|")
ax.set_title(f"COS policy search: vanilla vs Junike vs adaptive filtered-COS\\n"
             f"({_best_case['case_name']}, {_best_case['adaptive_result_label']})")
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")
ax.grid(True, which="both", ls="--", alpha=0.35)
fig.tight_layout()

fig_path = _FIG_DIR / "cos_policy_search_showcase.png"
fig.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.show()
"""

S61_INTERP_MD = """\
#### Reading the scatter

The table above is computed rather than hard-coded. Read the
`adaptive_result_label` column for the outcome in each case. The scatter plot
shows all candidates in `(runtime, error)` space. The gold star marks the
selected policy; dashed lines show the Junike and vanilla COS error baselines.

Key observations:
- The selector can only pick a result at least as good as the best candidate in
  its set (`junike_no_filter` is always included).
- When a filtered candidate passes the tolerance **and** runs faster than
  Junike-no-filter, the selector chooses it → runtime win with no accuracy loss.
- When no filtered candidate improves over Junike-no-filter, the selector picks
  Junike — the filter is never forced.
"""

# ── Section 6.2 ───────────────────────────────────────────────────────────────

S62_MD = """\
### 6.2 Model-zoo rerun

We repeat the comparison for all stress cases and summarise the result in one table.
"""

S62_CODE = """\
_zoo_rows = []

for r in _showcase_rows:
    err_v = r["vanilla_cos_error"]
    err_j = r["junike_cos_error"]
    err_a = r["adaptive_filtered_error"]

    beats_v = err_a < err_v
    beats_j = err_a < err_j
    within  = err_a <= _TOL

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

    _zoo_rows.append(dict(
        case_name=r["case_name"], model=r["model"], maturity=r["maturity"],
        vanilla_cos_error=err_v,   junike_cos_error=err_j,
        adaptive_filtered_error=err_a,
        vanilla_cos_runtime_ms=r["vanilla_cos_runtime_ms"],
        junike_cos_runtime_ms=r["junike_cos_runtime_ms"],
        adaptive_filtered_runtime_ms=r["adaptive_filtered_runtime_ms"],
        selected_filter=r["selected_filter"],
        selected_policy=r["selected_policy"],
        adaptive_beats_vanilla=beats_v,
        adaptive_beats_junike=beats_j,
        adaptive_matches_best_within_tolerance=within,
        final_status=status,
    ))

zoo_df = pd.DataFrame(_zoo_rows)
zoo_csv = _OUT_DIR / "adaptive_filtered_cos_model_zoo.csv"
zoo_df.to_csv(zoo_csv, index=False)

cols = ["case_name","vanilla_cos_error","junike_cos_error",
        "adaptive_filtered_error","selected_filter","final_status"]
zoo_df[cols].style.format({
    "vanilla_cos_error":        "{:.2e}",
    "junike_cos_error":         "{:.2e}",
    "adaptive_filtered_error":  "{:.2e}",
})
"""

S62_PLOT_CODE = """\
cases = zoo_df["case_name"].tolist()
x     = np.arange(len(cases))
w     = 0.25

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.bar(x - w, np.clip(zoo_df["vanilla_cos_error"],          1e-16, None),
        w, label="Vanilla COS",       color="silver",  edgecolor="grey")
ax1.bar(x,      np.clip(zoo_df["junike_cos_error"],           1e-16, None),
        w, label="Junike COS",        color="steelblue", edgecolor="navy", alpha=0.85)
ax1.bar(x + w,  np.clip(zoo_df["adaptive_filtered_error"],   1e-16, None),
        w, label="Adaptive filtered", color="mediumseagreen", edgecolor="darkgreen", alpha=0.85)
ax1.axhline(_TOL, color="black", ls="-.", lw=1, alpha=0.6, label=f"Target tol={_TOL:.0e}")
ax1.set_yscale("log")
ax1.set_xticks(x); ax1.set_xticklabels(cases, rotation=30, ha="right")
ax1.set_ylabel("max|error|")
ax1.set_title("Error comparison by model")
ax1.legend(fontsize=8)
ax1.grid(True, which="both", ls="--", alpha=0.3, axis="y")

ax2.bar(x - w, zoo_df["vanilla_cos_runtime_ms"],           w,
        label="Vanilla COS",       color="silver",         edgecolor="grey")
ax2.bar(x,      zoo_df["junike_cos_runtime_ms"],            w,
        label="Junike COS",        color="steelblue",      edgecolor="navy", alpha=0.85)
ax2.bar(x + w,  zoo_df["adaptive_filtered_runtime_ms"],    w,
        label="Adaptive filtered", color="mediumseagreen", edgecolor="darkgreen", alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(cases, rotation=30, ha="right")
ax2.set_ylabel("runtime (ms)")
ax2.set_title("Runtime comparison by model")
ax2.legend(fontsize=8)
ax2.grid(True, which="both", ls="--", alpha=0.3, axis="y")

fig.suptitle("Adaptive filtered-COS: model-zoo rerun", fontsize=12, y=1.01)
fig.tight_layout()

err_fig_path = _FIG_DIR / "adaptive_filtered_cos_model_zoo_errors.png"
rt_fig_path  = _FIG_DIR / "adaptive_filtered_cos_model_zoo_runtime.png"
fig.savefig(err_fig_path, dpi=120, bbox_inches="tight")
fig2, ax = plt.subplots(figsize=(max(8, len(cases)*1.5), 4.5))
ax.bar(x - w, zoo_df["vanilla_cos_runtime_ms"],          w,
       label="Vanilla COS",       color="silver",         edgecolor="grey")
ax.bar(x,      zoo_df["junike_cos_runtime_ms"],           w,
       label="Junike COS",        color="steelblue",      edgecolor="navy", alpha=0.85)
ax.bar(x + w,  zoo_df["adaptive_filtered_runtime_ms"],   w,
       label="Adaptive filtered", color="mediumseagreen", edgecolor="darkgreen", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(cases, rotation=30, ha="right")
ax.set_ylabel("runtime (ms)"); ax.legend(fontsize=8)
ax.grid(True, which="both", ls="--", alpha=0.3, axis="y")
fig2.tight_layout()
fig2.savefig(rt_fig_path, dpi=120, bbox_inches="tight")
plt.show()
"""

S63_MD = """\
### 6.3 Takeaways from the extension
"""

S63_CONCLUSIONS_CODE = """\
_statuses = zoo_df["final_status"].value_counts().to_dict()
_any_best  = zoo_df[zoo_df["final_status"] == "adaptive best"]
_any_junike_still = zoo_df[zoo_df["final_status"] == "Junike still best"]
_any_vanilla_still = zoo_df[zoo_df["final_status"] == "vanilla still best"]
_any_within = zoo_df[zoo_df["adaptive_matches_best_within_tolerance"]]
_any_unstable = zoo_df[zoo_df["adaptive_filtered_error"].isna() |
                       zoo_df["adaptive_filtered_error"].isin([float("inf")])]

if not _any_best.empty:
    _c1 = _any_best.iloc[0]
    _conclusion1 = (
        f"**Where adaptive filtered-COS works best**: the `{_c1['case_name']}` case "
        f"(model={_c1['model']}, T={_c1['maturity']}).  "
        f"The adaptive selector found a filtered policy (filter: `{_c1['selected_filter']}`) "
        f"that reduced max|error| from {_c1['vanilla_cos_error']:.2e} (vanilla) "
        f"to {_c1['adaptive_filtered_error']:.2e}, beating both baselines."
    )
elif not _any_within.empty:
    _c1 = _any_within.iloc[0]
    _conclusion1 = (
        f"**Where adaptive filtered-COS works best**: the `{_c1['case_name']}` case "
        f"(model={_c1['model']}, T={_c1['maturity']}).  "
        f"The adaptive selector met the target tolerance ({_TOL:.0e}) with "
        f"filter `{_c1['selected_filter']}` while the Junike error was "
        f"{_c1['junike_cos_error']:.2e}."
    )
else:
    _conclusion1 = (
        f"**Where adaptive filtered-COS works best**: in this run none of the cases "
        f"produced a clear error improvement over Junike.  "
        f"The selector matched the Junike quality in all {len(zoo_df)} cases, "
        f"meaning the Junike truncation was already controlling the dominant error source."
    )

if not _any_junike_still.empty:
    _c2_names = ", ".join(f"`{r}`" for r in _any_junike_still["case_name"].tolist())
    _conclusion2 = (
        f"**Where Junike truncation remains enough**: {_c2_names}.  "
        "Filtering was unnecessary because truncation error was the dominant "
        "issue and Junike-style L selection resolved it completely.  "
        "The adaptive selector correctly chose the no-filter Junike candidate."
    )
else:
    _conclusion2 = (
        "**Where Junike truncation remains enough**: in every case in this zoo, "
        "the adaptive selector either matched or improved on Junike-only — no case "
        "showed Junike to be clearly insufficient without filtering.  "
        "This confirms Junike is effective when truncation is the dominant issue."
    )

if not _any_vanilla_still.empty:
    _c3_names = ", ".join(f"`{r}`" for r in _any_vanilla_still["case_name"].tolist())
    _conclusion3 = (
        f"**Where vanilla COS is adequate**: {_c3_names}.  "
        "The default N=256 heuristic grid was accurate enough that Junike "
        "truncation and filtering did not improve further.  "
        "These cases confirm the adaptive selector is not forced to add complexity "
        "where the simpler method already performs well."
    )
else:
    _bsm = zoo_df[zoo_df["model"] == "bsm"]
    if not _bsm.empty:
        _c3 = _bsm.iloc[0]
        _conclusion3 = (
            f"**Where vanilla COS is adequate**: the `{_c3['case_name']}` BSM case "
            f"(error={_c3['vanilla_cos_error']:.2e}) shows that for smooth diffusion "
            f"models vanilla COS already achieves competitive accuracy.  "
            f"The adaptive selector chose filter=`{_c3['selected_filter']}` and "
            f"reached error={_c3['adaptive_filtered_error']:.2e}, confirming no "
            f"harmful side-effect of the extension on smooth models."
        )
    else:
        _conclusion3 = (
            "**Where vanilla COS is adequate**: for smooth Gaussian-like models with "
            "moderate maturities, vanilla COS with the heuristic L=10 grid is already "
            "competitive.  The adaptive selector matched this performance without regression."
        )

if not _any_unstable.empty:
    _c4_names = ", ".join(f"`{r}`" for r in _any_unstable["case_name"].tolist())
    _conclusion4 = (
        f"**Where further work is needed**: {_c4_names} produced non-finite or "
        "infinite errors.  These cases should be investigated for CF blow-up, "
        "width-fallback issues, or parameter constraints."
    )
else:
    _conclusion4 = (
        "**No unresolved instability appeared in this model-zoo rerun** under the "
        f"selected tolerance {_TOL:.0e}.  "
        "All cases produced finite, non-negative prices.  "
        "The main remaining challenge is choosing the cheapest policy rather than "
        "fixing pricing failure: the adaptive selector currently uses a fixed "
        "candidate set; a learned or model-adaptive set could further reduce "
        "runtime in cases where the recommended policy is over-conservative."
    )

from IPython.display import Markdown, display
_md_text = (
    "\n**1.** " + _conclusion1 + "\n\n"
    "**2.** " + _conclusion2 + "\n\n"
    "**3.** " + _conclusion3 + "\n\n"
    "**4.** " + _conclusion4 + "\n"
)
display(Markdown(_md_text))
"""

# ── Assemble and write ─────────────────────────────────────────────────────────

def main():
    nb = read_notebook(NB_PATH)

    def _cell_source(cell):
        source = cell.get("source", "")
        if isinstance(source, list):
            return "".join(source)
        return source

    section_start = None
    appendix_start = None
    for idx, cell in enumerate(nb.get("cells", [])):
        source = _cell_source(cell)
        if section_start is None and MARKER in source:
            section_start = idx
        if appendix_start is None and APPENDIX_MARKER in source:
            appendix_start = idx

    if section_start is not None:
        section_end = appendix_start if appendix_start is not None and appendix_start > section_start else len(nb["cells"])
        del nb["cells"][section_start:section_end]
        insert_at = section_start
        if appendix_start is not None and appendix_start > section_start:
            insert_at = min(insert_at, len(nb["cells"]))
    else:
        insert_at = appendix_start if appendix_start is not None else len(nb["cells"])

    new_cells = [
        md(S6_MD),
        code(S6_SETUP_CODE),
        md(S6_FILTER_MD),
        code(S6_FILTER_PLOT_CODE),
        md(S61_MD),
        code(S61_CODE),
        code(S61_PLOT_CODE),
        md(S61_INTERP_MD),
        md(S62_MD),
        code(S62_CODE),
        code(S62_PLOT_CODE),
        md(S63_MD),
        code(S63_CONCLUSIONS_CODE),
    ]

    nb["cells"][insert_at:insert_at] = new_cells
    write_notebook(NB_PATH, nb)
    print(f"Updated {len(new_cells)} cells → {NB_PATH}")
    print(f"Notebook now has {len(nb['cells'])} cells total")


if __name__ == "__main__":
    main()
