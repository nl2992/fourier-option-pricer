#!/usr/bin/env python3
"""Append extra surface + cross-strike diagnostics to notebooks/demo.ipynb.

This script is additive and idempotent:
  - if the appendix marker already exists, the old appendix is removed;
  - a fresh 3-cell appendix is then appended to the end of demo.ipynb.

Run from repo root:
    python3 scripts/append_demo_extra_visuals.py
"""
from __future__ import annotations

import pathlib

from notebook_support import code, md, read_notebook, write_notebook


ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "demo.ipynb"
MARKER = "<!-- demo-extra-visuals -->"

APPENDIX_MD = f"""## Appendix A. Extra diagnostics

{MARKER}

Two short add-ons:
- a Heston implied-volatility surface built from COS prices;
- a dense short-maturity VG strip, where the filtered-COS differences are easier to see than they are under Heston.
"""

SURFACE_MD = """### A.1 Heston implied-volatility surface

This surface is built from COS prices and Black-style implied-vol inversion on a small maturity/strike grid.
"""

DENSE_MD = """### A.2 Dense VG strip

This is the same short-maturity VG setting used in the stress tests, plotted more densely across strikes so the COS variants separate visibly.
"""


SURFACE_CODE = """from mpl_toolkits import mplot3d
from foureng.surface import SurfaceSpec, model_iv_surface

SURF_MATS = np.array([0.25, 0.50, 1.00, 1.50, 2.00])
SURF_STRIKES = np.linspace(80.0, 120.0, 25)
SURF_SPEC = SurfaceSpec(
    S0=HESTON_FWD.S0,
    r=HESTON_FWD.r,
    q=HESTON_FWD.q,
    maturities=SURF_MATS,
    strikes=SURF_STRIKES,
)

SURF_IVS = model_iv_surface(
    SURF_SPEC,
    cf_factory=lambda fwd: (lambda u: heston_cf(u, fwd, HESTON_PARAMS)),
    cumulant_factory=lambda fwd: heston_cumulants(fwd, HESTON_PARAMS),
    N=256,
    L=10.0,
)

K_mesh, T_mesh = np.meshgrid(SURF_STRIKES, SURF_MATS)

fig = plt.figure(figsize=(10.5, 5.2))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(
    K_mesh,
    T_mesh,
    SURF_IVS,
    cmap='Blues',
    edgecolor='none',
    alpha=0.95,
)
ax.set_title('Heston implied-vol surface from COS prices')
ax.set_xlabel('strike K')
ax.set_ylabel('maturity T')
ax.set_zlabel('implied vol')
fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08, label='implied vol')
fig.tight_layout()
plt.show()
"""


COS_COMPARE_CODE = """try:
    from foureng.experiments.cos_filter_grid_search import (
        describe_filter_result,
        filter_spec_from_result,
        policy_filter_candidates,
        run_filtered_cos_grid_search,
        select_fastest_under_tolerance,
    )
except Exception:
    def filter_spec_from_result(row):
        filter_name = str(row["filter"])
        if filter_name == "none":
            return None
        if filter_name == "exponential":
            return COSFilterSpec("exponential", order=int(row["filter_order"]))
        return COSFilterSpec(filter_name)

    def describe_filter_result(row, *, none_label="no filter"):
        spec = filter_spec_from_result(row)
        if spec is None:
            return none_label
        if spec.name == "exponential":
            return f"exponential p={int(spec.order)}"
        return spec.name.replace("_", " ")

    if "policy_filter_candidates" not in globals():
        from dataclasses import dataclass

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

try:
    from foureng.viz.notebook_runtime import error_zoom_bounds
except Exception:
    def error_zoom_bounds(*arrays, pad_frac=0.08, min_pad=5e-6):
        values = [
            np.ravel(np.asarray(arr, dtype=float))
            for arr in arrays
            if np.asarray(arr).size
        ]
        if not values:
            return 0.0, 1.0
        merged = np.concatenate(values)
        lo = float(np.nanmin(merged))
        hi = float(np.nanmax(merged))
        pad = max(pad_frac * (hi - lo), min_pad)
        return max(0.0, lo - pad), hi + pad

VG_DENSE_FWD = ForwardSpec(S0=100.0, r=0.10, q=0.0, T=0.10)
VG_DENSE_PARAMS = VGParams(sigma=0.12, nu=0.2, theta=-0.14)
VG_DENSE_K = np.linspace(70.0, 130.0, 61)
VG_DENSE_REF, _ = timeit_strip(
    price_strip,
    'vg', 'pyfeng_fft',
    VG_DENSE_K, VG_DENSE_FWD, VG_DENSE_PARAMS,
    n_repeat=2,
)

VG_DENSE_CLASSIC, _ = timeit_strip(
    price_strip,
    'vg', 'cos',
    VG_DENSE_K, VG_DENSE_FWD, VG_DENSE_PARAMS,
    n_repeat=2,
)

VG_DENSE_IMPROVED_POLICY = recommended_cos_policy('vg', VG_DENSE_PARAMS, mode='benchmark')
VG_DENSE_IMPROVED, _ = timeit_strip(
    price_strip,
    'vg', 'cos_improved',
    VG_DENSE_K, VG_DENSE_FWD, VG_DENSE_PARAMS,
    grid=VG_DENSE_IMPROVED_POLICY,
    n_repeat=2,
)

dense_candidates = policy_filter_candidates(VG_DENSE_IMPROVED_POLICY)
dense_search = run_filtered_cos_grid_search(
    model='vg',
    strikes=VG_DENSE_K,
    fwd=VG_DENSE_FWD,
    params=VG_DENSE_PARAMS,
    reference=VG_DENSE_REF,
    candidates=dense_candidates,
    tol=1e-6,
    n_repeat=2,
)
dense_best = select_fastest_under_tolerance(dense_search, tol=1e-6)

dense_filter = filter_spec_from_result(dense_best)
if dense_filter is None:
    VG_DENSE_ADAPTIVE = VG_DENSE_IMPROVED
    adaptive_label = 'Adaptive choice (no filter)'
else:
    VG_DENSE_ADAPTIVE, _ = timeit_strip(
        price_strip,
        'vg', 'cos_filtered',
        VG_DENSE_K, VG_DENSE_FWD, VG_DENSE_PARAMS,
        grid=(VG_DENSE_IMPROVED_POLICY, dense_filter),
        n_repeat=2,
    )
    adaptive_label = f"Adaptive choice ({describe_filter_result(dense_best)})"

VG_ERR_CLASSIC = np.abs(VG_DENSE_CLASSIC - VG_DENSE_REF)
VG_ERR_IMPROVED = np.abs(VG_DENSE_IMPROVED - VG_DENSE_REF)
VG_ERR_ADAPTIVE = np.abs(VG_DENSE_ADAPTIVE - VG_DENSE_REF)

VG_GAP_IMPROVED = 1e5 * (VG_ERR_IMPROVED - VG_ERR_CLASSIC)
VG_GAP_ADAPTIVE = 1e5 * (VG_ERR_ADAPTIVE - VG_ERR_CLASSIC)

err_lo, err_hi = error_zoom_bounds(VG_ERR_CLASSIC, VG_ERR_IMPROVED, VG_ERR_ADAPTIVE)

fig, axes = plt.subplots(
    1, 3, figsize=(16.2, 4.6),
    gridspec_kw={'width_ratios': [1.25, 1.0, 1.0]},
)

axes[0].plot(VG_DENSE_K, VG_DENSE_REF, color=DARK, linewidth=2.5, label='PyFENG FFT ref')
axes[0].plot(VG_DENSE_K, VG_DENSE_CLASSIC, color=CB_STEEL, linestyle='--', linewidth=1.8,
             marker='o', markevery=5, ms=3.5,
             label='COS classic')
axes[0].plot(VG_DENSE_K, VG_DENSE_IMPROVED, color=NAVY, linestyle='-.', linewidth=1.8,
             marker='s', markevery=5, ms=3.5,
             label='COS improved')
axes[0].plot(VG_DENSE_K, VG_DENSE_ADAPTIVE, color=COLUMBIA_BLUE, linewidth=1.8,
             marker='^', markevery=5, ms=3.5,
             label=adaptive_label)
axes[0].set_title('Short-maturity VG strip: price level')
axes[0].set_xlabel('strike K')
axes[0].set_ylabel('call price')
axes[0].legend(frameon=False, fontsize=8, loc='upper right')
axes[0].grid(True, alpha=0.18, axis='y')

axes[1].plot(VG_DENSE_K, VG_ERR_CLASSIC, color=CB_STEEL, linestyle='--', linewidth=1.8,
             marker='o', markevery=5, ms=3.5,
             label='COS classic')
axes[1].plot(VG_DENSE_K, VG_ERR_IMPROVED, color=NAVY, linestyle='-.', linewidth=1.8,
             marker='s', markevery=5, ms=3.5,
             label='COS improved')
axes[1].plot(VG_DENSE_K, VG_ERR_ADAPTIVE, color=COLUMBIA_BLUE, linewidth=1.8,
             marker='^', markevery=5, ms=3.5,
             label=adaptive_label)
axes[1].set_title('Absolute error (tight linear zoom)')
axes[1].set_xlabel('strike K')
axes[1].set_ylabel('abs error')
axes[1].set_ylim(err_lo, err_hi)
axes[1].legend(frameon=False, fontsize=8, loc='upper left')
axes[1].grid(True, alpha=0.22)

axes[2].axhline(0.0, color='#94a3b8', linestyle=':', linewidth=1.2)
axes[2].plot(VG_DENSE_K, VG_GAP_IMPROVED, color=NAVY, linestyle='-.', linewidth=1.8,
             marker='s', markevery=5, ms=3.5,
             label='improved - classic')
axes[2].plot(VG_DENSE_K, VG_GAP_ADAPTIVE, color=COLUMBIA_BLUE, linewidth=1.8,
             marker='^', markevery=5, ms=3.5,
             label='adaptive - classic')
axes[2].fill_between(
    VG_DENSE_K,
    VG_GAP_ADAPTIVE,
    0.0,
    where=VG_GAP_ADAPTIVE < 0.0,
    color=COLUMBIA_BLUE,
    alpha=0.10,
)
axes[2].fill_between(
    VG_DENSE_K,
    VG_GAP_IMPROVED,
    0.0,
    where=VG_GAP_IMPROVED < 0.0,
    color=NAVY,
    alpha=0.08,
)
axes[2].set_title('Error gap vs COS classic')
axes[2].set_xlabel('strike K')
axes[2].set_ylabel('delta abs error x 1e5')
axes[2].legend(frameon=False, fontsize=8, loc='upper left')
axes[2].grid(True, alpha=0.22)

stats_text = (
    f"classic  max={VG_ERR_CLASSIC.max():.3e}\\n"
    f"improved max={VG_ERR_IMPROVED.max():.3e}\\n"
    f"adaptive max={VG_ERR_ADAPTIVE.max():.3e}\\n"
    f"best filter: {describe_filter_result(dense_best)}"
)
axes[2].text(
    0.03, 0.03, stats_text,
    transform=axes[2].transAxes,
    fontsize=8,
    va='bottom',
    ha='left',
    bbox=dict(facecolor='white', edgecolor='#94a3b8', alpha=0.9, boxstyle='round,pad=0.3'),
)

fig.tight_layout()
plt.show()

dense_search[['method_label', 'filter', 'filter_order', 'runtime_ms', 'max_abs_err', 'passes_tol']].head(7)
"""


def main() -> None:
    nb = read_notebook(NB_PATH)

    append_start = None
    for idx, cell in enumerate(nb.get("cells", [])):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if MARKER in source:
            append_start = idx
            break

    if append_start is not None:
        nb["cells"] = nb["cells"][:append_start]

    nb["cells"].extend([
        md(APPENDIX_MD),
        md(SURFACE_MD),
        code(SURFACE_CODE),
        md(DENSE_MD),
        code(COS_COMPARE_CODE),
    ])

    write_notebook(NB_PATH, nb)
    print(f"Updated → {NB_PATH}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
