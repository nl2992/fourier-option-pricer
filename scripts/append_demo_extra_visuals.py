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

APPENDIX_MD = f"""## Appendix — Two extra checks

{MARKER}

Two additional plots:

1. A Heston implied-volatility surface built from COS prices.
2. A dense short-maturity VG strip, which is a cleaner place to see the filter choice.

The second plot is intentionally not Heston. In Heston the COS variants are already close enough that the error curves largely overlap. Short-maturity VG separates them enough to make the comparison visible.
"""


SURFACE_CODE = """from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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


COS_COMPARE_CODE = """from foureng.experiments.cos_filter_grid_search import (
    describe_filter_result,
    filter_spec_from_result,
    policy_filter_candidates,
    run_filtered_cos_grid_search,
    select_fastest_under_tolerance,
)
from foureng.viz.notebook_runtime import error_zoom_bounds

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

print('Adaptive dense-strip selection:')
print(
    dense_search[['method_label', 'filter', 'filter_order', 'runtime_ms', 'max_abs_err', 'passes_tol']]
    .head(7)
    .to_string(index=False)
)
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
        code(SURFACE_CODE),
        code(COS_COMPARE_CODE),
    ])

    write_notebook(NB_PATH, nb)
    print(f"Updated → {NB_PATH}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
