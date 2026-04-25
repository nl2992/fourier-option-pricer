"""Build ``notebooks/cos_method_improved.ipynb`` with Columbia styling.

Run from the repo root:
    python3 scripts/build_cos_method_improved_notebook.py

Then execute:
    jupyter nbconvert --to notebook --execute \
        notebooks/cos_method_improved.ipynb \
        --output cos_method_improved.ipynb
"""
from __future__ import annotations
import json
from itertools import count

from pathlib import Path

try:
    import nbformat as nbf
except ImportError:  # pragma: no cover - optional builder convenience only
    nbf = None


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "cos_method_improved.ipynb"
CELL_IDS = count()


def md(text: str):
    if nbf is not None:
        return nbf.v4.new_markdown_cell(text.strip("\n"))
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": f"cell-{next(CELL_IDS)}",
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(src: str):
    if nbf is not None:
        return nbf.v4.new_code_cell(src.strip("\n"))
    return {
        "cell_type": "code",
        "metadata": {},
        "id": f"cell-{next(CELL_IDS)}",
        "execution_count": None,
        "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True),
    }


INTRO_MD = r"""
# The COS Method · Improved Truncation

**Columbia University · MAFN · MATH 5030 · Spring 2026**

*Fang–Oosterlee (2008) revisited through Junike–Pankrashkin (2022) and Junike (2024). Same pricing formula; *better support and series-truncation policies*.*

*Engine: `foureng` · Instructor: Prof. Jaehyuk Choi*

---

This notebook studies the adaptive `cos_improved` path as a numerical policy
layer around COS:

- same pricing formula,
- better support selection,
- better resolution selection,
- clearer fallback logic in hostile regimes.

The goal is not to claim a universally smaller benchmark error. The goal is to
separate the numerical failure modes cleanly enough that the method is easier
to explain, validate, and ship.
"""


THEORY_MD = r"""
## 1 · Framing the upgrade

> **FO2008 cumulant rule**
>
> Pick $[a,b]$ from low-order cumulants with a safety multiplier $L$:

$$
[a,b] = \left[c_1 - L\sqrt{c_2 + \sqrt{|c_4|}},\; c_1 + L\sqrt{c_2 + \sqrt{|c_4|}}\right].
$$

Convenient and effective on diffusive models — but still a heuristic. If
$[a,b]$ is too short, COS discards tail mass *before* the cosine expansion
starts; raising $N$ then refines the approximation on a truncated support and
cannot recover the missing tail.

> ⚠️ **Junike–Pankrashkin tolerance rule**
>
> Choose a centre $m$ and require absolute moments outside $[m-M, m+M]$ to fall below a tolerance. By Markov's inequality:

$$
\mathbb{P}(|X-m| \ge M) \le \frac{\mathbb{E}[|X-m|^n]}{M^n},
\qquad
M \ge \left(\frac{\mathbb{E}[|X-m|^n]}{\varepsilon}\right)^{1/n},
\qquad
[a,b] = [m-M,\; m+M].
$$

**Junike (2024)** then handles the second knob: once support is fixed,
choose enough cosine modes to resolve it. That separation — support
truncation, then series truncation, then payoff representation — is the
logic behind `COSGridPolicy`, `cos_centered_half_width`, and the adaptive
`cos_improved` path used below.
"""


IMPORTS_CODE = r"""
# Cell 1 — Imports, paths, styling
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)

def _iter_repo_candidates():
    seen = set()

    def _expand(pathlike):
        if not pathlike:
            return []
        p = pathlib.Path(pathlike).expanduser()
        try:
            p = p.resolve()
        except Exception:
            pass
        return [p, *p.parents]

    candidates = []
    for raw in (os.environ.get("PWD"), os.environ.get("OLDPWD")):
        candidates.extend(_expand(raw))
    try:
        candidates.extend(_expand(pathlib.Path.cwd()))
    except FileNotFoundError:
        pass
    for raw in sys.path:
        if raw:
            candidates.extend(_expand(raw))
    spec = importlib.util.find_spec("foureng")
    if spec is not None and spec.origin:
        candidates.extend(_expand(spec.origin))

    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


REPO_ROOT = None
for candidate in _iter_repo_candidates():
    if (candidate / "foureng").exists() and (candidate / "benchmarks").exists():
        REPO_ROOT = candidate
        break
if REPO_ROOT is None:
    for base in [pathlib.Path.home() / name for name in ("Desktop", "Documents", "Projects", "Code")]:
        if not base.exists():
            continue
        try:
            for match in base.rglob("foureng/__init__.py"):
                candidate = match.parent.parent
                if (candidate / "benchmarks").exists():
                    REPO_ROOT = candidate
                    break
        except Exception:
            continue
        if REPO_ROOT is not None:
            break
if REPO_ROOT is None:
    raise RuntimeError(
        "Could not locate repo root. Launch the notebook from inside the project "
        "or set PWD to the repo path."
    )

for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUTDIR = REPO_ROOT / "benchmarks" / "cos_method_improved" / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

from benchmarks.paper_replications.fo2008_cos.params import CASES, PaperCase
from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams, bsm_cf, bsm_cumulants
from foureng.models.cgmy import CgmyParams, cgmy_cf, cgmy_cumulants
from foureng.models.heston import HestonParams, heston_cf, heston_cumulants
from foureng.models.variance_gamma import VGParams, vg_cf, vg_cumulants
from foureng.pipeline import price_strip
from foureng.pricers.cos import (
    cos_adaptive_decision,
    cos_auto_grid,
    cos_prices,
    recommended_cos_policy,
)
from foureng.pricers.lewis import lewis_call_prices
from foureng.utils.cumulants import Cumulants, cos_centered_half_width
from foureng.utils.grids import COSGrid, COSGridPolicy
from foureng.viz import (
    apply_columbia_style,
    NAVY,
    COLUMBIA_BLUE,
    DARK,
    SLATE,
    ORANGE,
    GREEN,
    PANEL,
    CLOUD,
)

apply_columbia_style()
CU_BLUE = NAVY
CU_LIGHT = COLUMBIA_BLUE
CU_ORANGE = ORANGE
CU_GREEN = GREEN
CU_GREY = SLATE
CU_BG = PANEL
CU_ACC = CLOUD

pd.set_option("display.float_format", lambda x: f"{x: .6e}")
warnings.filterwarnings("ignore", category=RuntimeWarning)
print("repo root:", REPO_ROOT)
print("output dir:", OUTDIR)
"""


HELPERS_CODE = r"""
# Cell 2 — Helpers
N_REP = 5


def timed_median_ms(fn, *args, n_rep=N_REP, **kwargs):
    fn(*args, **kwargs)
    times = []
    out = None
    for _ in range(n_rep):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return out, float(np.median(times))


def style_table(df, caption, *, gradient_cols=None):
    styler = (
        df.style
        .format(precision=4, thousands=",")
        .set_caption(caption)
        .set_table_styles([
            {"selector": "caption", "props": [("color", CU_BLUE), ("font-weight", "700"),
                                                ("font-size", "13.5px"), ("padding", "4px 0 10px 0"),
                                                ("text-align", "left"), ("caption-side", "top")]},
            {"selector": "table", "props": [("border-collapse", "separate"), ("border-spacing", "0"),
                                              ("margin", "4px 0 10px 0"), ("font-size", "12.5px"),
                                              ("border", "1px solid #CAD4E3"), ("border-radius", "6px"),
                                              ("overflow", "hidden")]},
            {"selector": "thead th", "props": [("background", CU_BLUE), ("color", "white"),
                                                 ("font-weight", "700"), ("padding", "8px 12px"),
                                                 ("border", "none"), ("text-align", "center")]},
            {"selector": "tbody td", "props": [("padding", "6px 12px"), ("border-bottom", "1px solid #E1E7EF")]},
            {"selector": "tbody tr:nth-child(odd) td", "props": [("background", "#FFFFFF")]},
            {"selector": "tbody tr:nth-child(even) td", "props": [("background", "#EAF0F8")]},
        ])
    )
    if gradient_cols:
        styler = styler.background_gradient(cmap="Blues", subset=gradient_cols, axis=None, low=0.02, high=0.85)
    return styler


def fwd_for(case: PaperCase) -> ForwardSpec:
    r = case.params.get("r", 0.0)
    q = case.params.get("q", 0.0)
    return ForwardSpec(S0=case.forward, r=r, q=q, T=case.maturity)


def params_for(case: PaperCase):
    p = {k: v for k, v in case.params.items() if k not in ("r", "q")}
    if case.model == "bsm":
        return BsmParams(**p)
    if case.model == "heston":
        return HestonParams(**p)
    if case.model == "vg":
        return VGParams(**p)
    if case.model == "cgmy":
        return CgmyParams(**p)
    raise ValueError(case.model)


def cf_for(case: PaperCase):
    fwd = fwd_for(case)
    p = params_for(case)
    if case.model == "bsm":
        return lambda u: bsm_cf(u, fwd, p), fwd, p
    if case.model == "heston":
        return lambda u: heston_cf(u, fwd, p), fwd, p
    if case.model == "vg":
        return lambda u: vg_cf(u, fwd, p), fwd, p
    if case.model == "cgmy":
        return lambda u: cgmy_cf(u, fwd, p), fwd, p
    raise ValueError(case.model)


def cumulants_for(case: PaperCase):
    fwd = fwd_for(case)
    p = params_for(case)
    if case.model == "bsm":
        return bsm_cumulants(fwd, p)
    if case.model == "heston":
        return heston_cumulants(fwd, p)
    if case.model == "vg":
        return vg_cumulants(fwd, p)
    if case.model == "cgmy":
        return cgmy_cumulants(fwd, p)
    raise ValueError(case.model)


def bs_call_exact(strikes, fwd: ForwardSpec, sigma: float):
    strikes = np.asarray(strikes, dtype=float)
    vol = sigma * np.sqrt(fwd.T)
    d1 = (np.log(fwd.F0 / strikes) + 0.5 * sigma * sigma * fwd.T) / vol
    d2 = d1 - vol
    return fwd.disc * (fwd.F0 * norm.cdf(d1) - strikes * norm.cdf(d2))


def reference_for_case(case: PaperCase):
    phi, fwd, p = cf_for(case)
    K = np.asarray(case.strikes, dtype=float)
    if case.model == "bsm":
        return np.asarray(bs_call_exact(K, fwd, p.sigma), dtype=float)
    if case.model == "heston":
        return np.asarray(
            lewis_call_prices(
                phi,
                K,
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
    ref = np.asarray(case.reference_values, dtype=float)
    return ref.reshape(-1)


def max_abs_err(prices, ref):
    return float(np.max(np.abs(np.asarray(prices, dtype=float) - np.asarray(ref, dtype=float))))


def paper_grid_for(case: PaperCase):
    cums = cumulants_for(case)
    if case.model == "cgmy" and "trunc_ab" in case.extras:
        a, b = case.extras["trunc_ab"]
        return COSGrid(N=max(case.Ns), a=float(a), b=float(b), label="paper")
    L = float(case.extras.get("L", 10.0))
    return cos_auto_grid(cums, N=max(case.Ns), L=L)
"""


CASE_GRID_MD = r"""
## 2 · Three pricing strategies, side-by-side

> **What we run**
>
> Eight FO2008 paper cases (Tables 2, 4, 5, 6, 7, 8, 10), each priced three ways and scored against the same trusted reference (analytic for BSM, fine Lewis integral for Heston, paper reference values otherwise).

| Variant | Support rule | Series rule | Why we keep it |
|---|---|---|---|
| **default** | repo's pre-Junike default | repo's pre-Junike default | regression baseline |
| **paper-grid** | FO2008 paper $[a,b]$ recipe with case-specific $L$ | $N = \max(\text{case.Ns})$ | direct paper replay |
| **improved** | adaptive Junike-style tolerance with cumulant fallback | Junike series-truncation rule | proposed default |

The improved variant is **not** uniformly better. It is a robustness fix.
The right scoreboard is `vs default`, `vs paper-grid replay`, and
`vs paper best reported` — kept side-by-side in the second table.
"""


SUMMARY_CODE = r"""
# Cell 3 — Fixed COS vs paper-grid COS vs improved adaptive COS
CASE_IDS = [
    "bsm_table2",
    "heston_table4_t1",
    "heston_table5_t10",
    "heston_table6_strip",
    "vg_table7_t01",
    "vg_table7_t1",
    "cgmy_table8_y05",
    "cgmy_table10_y198",
]


def cmp_status(lhs, rhs, *, rtol=1e-12, atol=1e-18):
    if pd.isna(lhs) or pd.isna(rhs):
        return "n/a"
    if np.isclose(lhs, rhs, rtol=rtol, atol=atol):
        return "match"
    return "better" if lhs < rhs else "worse"


rows = []
compare_rows = []
for cid in CASE_IDS:
    case = CASES[cid]
    phi, fwd, p = cf_for(case)
    K = np.asarray(case.strikes, dtype=float)
    ref = reference_for_case(case)
    cums = cumulants_for(case)

    def _default_cos():
        return price_strip(case.model, "cos", K, fwd, p)

    def _paper_cos():
        return cos_prices(phi, fwd, K, paper_grid_for(case)).call_prices

    policy = recommended_cos_policy(case.model, p, mode="benchmark")
    decision = cos_adaptive_decision(cums, model=case.model, params=p, policy=policy, strike_count=len(K))

    def _improved_cos():
        return price_strip(case.model, "cos_improved", K, fwd, p, grid=policy)

    default_prices, default_ms = timed_median_ms(_default_cos)
    paper_prices, paper_ms = timed_median_ms(_paper_cos)
    improved_prices, improved_ms = timed_median_ms(_improved_cos)

    default_err = max_abs_err(default_prices, ref)
    paper_grid_err = max_abs_err(paper_prices, ref)
    improved_err = max_abs_err(improved_prices, ref)

    rows.append({
        "case_id": cid,
        "model": case.model,
        "n_strikes": len(K),
        "default_err": default_err,
        "paper_grid_err": paper_grid_err,
        "improved_err": improved_err,
        "default_ms": default_ms,
        "paper_grid_ms": paper_ms,
        "improved_ms": improved_ms,
        "improved_method": decision.method,
        "improved_N": decision.grid.N,
        "improved_width": decision.grid.width,
        "improved_dx": decision.grid.dx,
        "improved_center": decision.grid.center,
        "improved_L": decision.L_used,
        "tail_proxy": decision.tail_proxy,
        "tail_family": decision.tail_family,
        "decision_reason": decision.reason,
    })

    paper_best_idx = int(np.argmin(case.paper_errors)) if case.paper_errors is not None else None
    paper_best_n = case.Ns[paper_best_idx] if paper_best_idx is not None else np.nan
    paper_best_err = case.paper_errors[paper_best_idx] if paper_best_idx is not None else np.nan
    paper_best_ms = (
        case.paper_times_ms[paper_best_idx]
        if paper_best_idx is not None and case.paper_times_ms is not None
        else np.nan
    )

    compare_rows.append({
        "case_id": cid,
        "paper_best_N": paper_best_n,
        "paper_best_err": paper_best_err,
        "paper_best_ms_historical": paper_best_ms,
        "default_err": default_err,
        "paper_grid_err": paper_grid_err,
        "improved_method": decision.method,
        "improved_N": decision.grid.N,
        "improved_err": improved_err,
        "improved_ms": improved_ms,
        "vs_default": cmp_status(improved_err, default_err),
        "vs_paper_grid": cmp_status(improved_err, paper_grid_err),
        "vs_paper_best": cmp_status(improved_err, paper_best_err),
    })

SUMMARY_DF = pd.DataFrame(rows)
SUMMARY_DF["gain_vs_default"] = SUMMARY_DF["default_err"] / SUMMARY_DF["improved_err"]
SUMMARY_DF["gain_vs_paper_grid"] = SUMMARY_DF["paper_grid_err"] / SUMMARY_DF["improved_err"]
SUMMARY_DF.to_csv(OUTDIR / "cos_method_improved_summary.csv", index=False)

PAPER_COMPARE_DF = pd.DataFrame(compare_rows)
PAPER_COMPARE_DF.to_csv(OUTDIR / "cos_method_improved_paper_compare.csv", index=False)

display(style_table(
    SUMMARY_DF.sort_values("improved_err")[
        ["case_id", "model", "default_err", "paper_grid_err", "improved_err",
         "improved_method", "improved_N", "improved_width", "gain_vs_default"]
    ],
    "Adaptive policy vs default and paper-grid replay",
    gradient_cols=["default_err", "paper_grid_err", "improved_err"],
))
display(style_table(
    PAPER_COMPARE_DF[
        ["case_id", "paper_best_N", "paper_best_err", "default_err",
         "paper_grid_err", "improved_method", "improved_N", "improved_err",
         "vs_default", "vs_paper_grid", "vs_paper_best"]
    ],
    "Comparison against paper-reported benchmark points",
    gradient_cols=["paper_best_err", "default_err", "paper_grid_err", "improved_err"],
))
print("wrote", OUTDIR / "cos_method_improved_summary.csv")
print("wrote", OUTDIR / "cos_method_improved_paper_compare.csv")
"""


DIAGNOSTIC_MD = r"""
## 3 · Heston $T{=}10$ — three error sources isolated

> **The microscope**
>
> Long-maturity Heston is the canonical case where lumping every numerical effect into a single "COS error" hides the actual problem. We measure each source on its own grid.

* **Support error** — sweep $L$ at fixed $N{=}4096$. The error floor here is the
  tail mass *outside* $[a,b]$. Raising $N$ cannot beat this floor.
* **Series error** — fix paper-wide $L{=}32$ and sweep $N$. This is the cosine
  truncation error on a known-good support. **Put + parity** vs **direct
  call** payoff coefficients are compared — the direct $\chi$ recurrence has
  an $e^b$ factor that loses precision catastrophically when $b \approx 35$.
* **Policy error** — three competing $[a,b]$ rules (`heuristic`, `tolerance`,
  `paper`) on the same put-and-parity pricer.

The three CSVs `heston_t10_{policy_comparison, support_error,
series_vs_coeff_error}.csv` are written so the README and the FO2008
replication summary can pick them up.
"""


DIAGNOSTIC_CODE = r"""
# Cell 4 — Heston T=10 diagnostics: support error, series error, coefficient error
case = CASES["heston_table5_t10"]
phi, fwd, p = cf_for(case)
K = np.asarray(case.strikes, dtype=float)
ref = float(reference_for_case(case)[0])
cums = cumulants_for(case)
c1, c2, c4 = cums
cobj = Cumulants(c1=c1, c2=c2, c4=c4)

series_rows = []
for N in [64, 128, 256, 512, 1024, 2048, 4096]:
    half_width = cos_centered_half_width(cobj, L=32.0)
    grid = COSGrid(N=N, a=-half_width, b=half_width, center=c1, label="paper_centered")
    p_put = float(cos_prices(phi, fwd, K, grid, payoff_mode="put_parity").call_prices[0])
    p_call = float(cos_prices(phi, fwd, K, grid, payoff_mode="call_direct").call_prices[0])
    series_rows.append({
        "N": N,
        "width": grid.width,
        "dx": grid.dx,
        "put_parity_err": abs(p_put - ref),
        "call_direct_err": abs(p_call - ref),
        "direct_minus_put": abs(p_call - p_put),
    })
SERIES_DF = pd.DataFrame(series_rows)
SERIES_DF.to_csv(OUTDIR / "heston_t10_series_vs_coeff_error.csv", index=False)

support_rows = []
for L in [6, 8, 10, 12, 16, 24, 32]:
    pol = COSGridPolicy(
        mode="benchmark",
        truncation="heuristic",
        centered=True,
        L=float(L),
        fixed_N=4096,
        width_fallback=0.0,
    )
    dec = cos_adaptive_decision(cums, model=case.model, params=p, policy=pol)
    price = float(cos_prices(phi, fwd, K, dec.grid, payoff_mode="put_parity").call_prices[0])
    support_rows.append({
        "L": float(L),
        "width": dec.grid.width,
        "tail_proxy": dec.tail_proxy,
        "err": abs(price - ref),
    })
SUPPORT_DF = pd.DataFrame(support_rows)
SUPPORT_DF.to_csv(OUTDIR / "heston_t10_support_error.csv", index=False)

policy_rows = []
for name, pol in [
    ("heuristic", COSGridPolicy(mode="benchmark", truncation="heuristic", centered=True, L=10.0, width_fallback=0.0)),
    ("tolerance", COSGridPolicy(mode="benchmark", truncation="tolerance", centered=True, eps_trunc=1e-10, width_fallback=0.0)),
    ("paper", COSGridPolicy(mode="benchmark", truncation="paper", centered=True, paper_L=32.0, width_fallback=0.0)),
]:
    dec = cos_adaptive_decision(cums, model=case.model, params=p, policy=pol)
    price = float(cos_prices(phi, fwd, K, dec.grid, payoff_mode="put_parity").call_prices[0])
    policy_rows.append({
        "policy": name,
        "N": dec.grid.N,
        "width": dec.grid.width,
        "dx": dec.grid.dx,
        "L_used": dec.L_used,
        "tail_proxy": dec.tail_proxy,
        "err": abs(price - ref),
    })
POLICY_DF = pd.DataFrame(policy_rows)
POLICY_DF.to_csv(OUTDIR / "heston_t10_policy_comparison.csv", index=False)

display(style_table(POLICY_DF, "Policy comparison on Heston T=10", gradient_cols=["width", "dx", "tail_proxy", "err"]))
display(style_table(SUPPORT_DF, "Support-width sensitivity on Heston T=10", gradient_cols=["width", "tail_proxy", "err"]))
display(style_table(SERIES_DF, "Series and coefficient diagnostics on Heston T=10", gradient_cols=["put_parity_err", "call_direct_err", "direct_minus_put"]))
"""


PLOTS_CODE = r"""
# Cell 5 — Plots (Columbia palette)
from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

# (a) Support truncation — flattens to a tail-mass floor below L≈12
ax = axes[0]
ax.plot(SUPPORT_DF["L"], SUPPORT_DF["err"], marker="o", color=CU_BLUE, lw=2.0, ms=7)
ax.fill_between(SUPPORT_DF["L"],
                np.maximum(SUPPORT_DF["err"].values, 1e-18),
                1e-18, color=CU_BLUE, alpha=0.08)
ax.axvline(10, color=CU_GREY, ls=":", lw=1, alpha=0.7)
ax.set_yscale("log")
ax.set_xlabel(r"truncation multiplier  $L$")
ax.set_ylabel("|abs error|")
ax.set_title("(a) Support truncation — Heston $T{=}10$, $N{=}4096$")
ax.text(10.2, 0.5*max(SUPPORT_DF["err"].values), r"$L{=}10$ (default)",
        color=CU_GREY, fontsize=9, va="top")

# (b) Series truncation — put+parity converges; direct call diverges as e^b
ax = axes[1]
ax.plot(SERIES_DF["N"], np.maximum(SERIES_DF["put_parity_err"], 1e-18),
        marker="o", color=CU_BLUE,   lw=2.0, ms=7, label="put + parity")
ax.plot(SERIES_DF["N"], np.maximum(SERIES_DF["call_direct_err"], 1e-18),
        marker="s", color=CU_ORANGE, lw=2.0, ms=7,
        label=r"direct call ($\chi$ has $e^{b}$)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("cosine terms  $N$")
ax.set_ylabel("|abs error|")
ax.set_title(r"(b) Series truncation — paper $L{=}32$, $b\approx 35$")
ax.legend(loc="upper right")

# (c) Default vs improved scatter — colour by who wins
ax = axes[2]
x = np.maximum(SUMMARY_DF["default_err"].to_numpy(),  1e-18)
y = np.maximum(SUMMARY_DF["improved_err"].to_numpy(), 1e-18)
colors = [CU_BLUE if (yi <= xi) else CU_ORANGE for xi, yi in zip(x, y)]
ax.scatter(x, y, c=colors, s=70, edgecolors="white", linewidths=1.2, zorder=3)
for _, row in SUMMARY_DF.iterrows():
    ax.annotate(row["case_id"].replace("_", " "),
                (max(row["default_err"], 1e-18), max(row["improved_err"], 1e-18)),
                fontsize=8, color=CU_GREY, alpha=0.85,
                xytext=(5, 4), textcoords="offset points")
lim_lo = min(x.min(), y.min()) * 0.5
lim_hi = max(x.max(), y.max()) * 2.0
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
        ls="--", color=CU_GREY, lw=1.0, alpha=0.7)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
ax.set_xlabel("default abs error")
ax.set_ylabel("improved abs error")
ax.set_title("(c) Default vs improved (lower-left = adaptive wins)")
ax.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor=CU_BLUE,   markersize=8, label=r"adaptive $\leq$ default"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=CU_ORANGE, markersize=8, label=r"adaptive $>$ default"),
    Line2D([0],[0], color=CU_GREY, ls="--", label=r"$y=x$"),
], loc="lower right", fontsize=9)

fig.suptitle("COS Method Improved · Heston $T{=}10$ microscope + cross-case scatter",
             color=CU_BLUE, fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "cos_method_improved_diagnostics.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
print("wrote", OUTDIR / "cos_method_improved_diagnostics.png")
"""


FIGURE_EXPORT_MD = r"""
## 4 · Visual diagnostics

> **Three panels, one figure**
>
> (a) Support-side error vs $L$ — flattens to a tail-mass floor below $L \approx 12$.
> (b) Series-side error vs $N$ — put + parity decays to machine precision while the direct-call recurrence diverges as $e^b$ floods the answer.
> (c) Default vs improved error per case — points below the $y{=}x$ line are wins for the adaptive path.

The figure is exported to `benchmarks/cos_method_improved/outputs/cos_method_improved_diagnostics.png`
so README references keep working.
"""


TAKEAWAYS_CODE = r"""
# Cell 6 — Styled takeaways: ranked summary table + Columbia scoreboard card
from IPython.display import HTML

best = SUMMARY_DF[["case_id", "improved_method", "improved_N", "improved_width",
                   "improved_err", "gain_vs_default"]].copy()
best["improved_width"]   = best["improved_width"].map(lambda v: f"{v: .3g}")
best["improved_err"]     = best["improved_err"].map(lambda v: f"{v: .3e}")
best["gain_vs_default"]  = best["gain_vs_default"].map(lambda v: f"{v: .3g}")

display(style_table(best, "Improved-COS summary, ranked by improved error"))

n_total          = len(PAPER_COMPARE_DF)
n_better_default = int((PAPER_COMPARE_DF["vs_default"]    == "better").sum())
n_better_paper   = int((PAPER_COMPARE_DF["vs_paper_grid"] == "better").sum())
n_better_best    = int((PAPER_COMPARE_DF["vs_paper_best"] == "better").sum())

scoreboard_html = (
    '<div class="cu-card">'
    '<h3>Scoreboard</h3>'
    f'<p><span class="cu-flag">vs default</span> &nbsp; adaptive wins on '
    f'<b>{n_better_default}/{n_total}</b> cases</p>'
    f'<p><span class="cu-flag">vs paper-grid replay</span> &nbsp; adaptive wins on '
    f'<b>{n_better_paper}/{n_total}</b> cases</p>'
    f'<p><span class="cu-flag">vs paper best reported</span> &nbsp; adaptive wins on '
    f'<b>{n_better_best}/{n_total}</b> cases</p>'
    '<p style="margin-top:10px; color:#5E6B7A;"><em>Reading note.</em> '
    'BSM is reference-limited (we are within 1e-14 of the analytic price already, '
    'so a 4× factor on a 4·10<sup>-15</sup> error has no practical meaning). '
    'CGMY Y=1.98 is treated as a hostile fallback regime — the adaptive layer '
    'escalates to Lewis when the cumulant width threshold is exceeded.</p>'
    '</div>'
)
display(HTML(scoreboard_html))
"""


OUTRO_MD = r"""
## 5 · Bottom line

> **What the numbers say**
>
> The improved policy **does not** uniformly dominate the pre-Junike default. It **does** reliably beat a literal FO2008 paper-grid replay, and it cleans up the regimes where the cumulant rule has no honest justification (long-maturity Heston, the FO2008 strip, heavy-tail CGMY $Y \to 2$).
>
> Treat `cos_improved` as a *robustness fix and stability layer*, not a universal speedup. The paper milliseconds in the comparison table are historical context, not portable timing claims.

---
*Columbia University · MAFN · MATH 5030 Spring 2026 · Instructor: Prof. Jaehyuk Choi*
"""


def build() -> None:
    cells = [
        md(INTRO_MD),
        md(THEORY_MD),
        code(IMPORTS_CODE),
        code(HELPERS_CODE),
        md(CASE_GRID_MD),
        code(SUMMARY_CODE),
        md(DIAGNOSTIC_MD),
        code(DIAGNOSTIC_CODE),
        code(PLOTS_CODE),
        md(FIGURE_EXPORT_MD),
        code(TAKEAWAYS_CODE),
        md(OUTRO_MD),
    ]
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    if nbf is not None:
        nb = nbf.v4.new_notebook(cells=cells, metadata=metadata)
        nbf.write(nb, OUT)
    else:
        nb = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
        OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT}   ({len(cells)} cells)")


if __name__ == "__main__":
    build()
