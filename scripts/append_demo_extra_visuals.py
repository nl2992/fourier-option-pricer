#!/usr/bin/env python3
"""Append extra surface + cross-strike diagnostics to notebooks/demo.ipynb.

This script is additive and idempotent:
  - if the appendix marker already exists, the old appendix is removed;
  - a fresh 3-cell appendix is then appended to the end of demo.ipynb.

Run from repo root:
    python3 scripts/append_demo_extra_visuals.py
"""
from __future__ import annotations

import json
import pathlib
import uuid


ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "demo.ipynb"
MARKER = "<!-- demo-extra-visuals -->"


def _id() -> str:
    return uuid.uuid4().hex[:8]


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": src,
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _id(),
        "metadata": {},
        "outputs": [],
        "source": src,
    }


APPENDIX_MD = f"""## Appendix — Surface and cross-strike diagnostics

{MARKER}

Two extra visuals round out the notebook:

1. A **Heston implied-volatility surface** generated from COS prices and Black-76 inversion.
2. A **short-maturity VG cross-strike comparison** of `COS classic`, `COS improved`, and the adaptive filtered-COS overlay against the same `PyFENG FFT` reference strip.

These are diagnostic visuals rather than new benchmarks: they reuse the same model family already introduced above and make the smile/surface geometry easier to see at a glance.
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
    FilterGridCandidate,
    run_filtered_cos_grid_search,
    select_fastest_under_tolerance,
)
from foureng.utils.spectral_filters import COSFilterSpec

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

dense_candidates = [
    FilterGridCandidate('no_filter', VG_DENSE_IMPROVED_POLICY, None),
    FilterGridCandidate('fejer', VG_DENSE_IMPROVED_POLICY, COSFilterSpec('fejer')),
    FilterGridCandidate('lanczos', VG_DENSE_IMPROVED_POLICY, COSFilterSpec('lanczos')),
    FilterGridCandidate('raised_cosine', VG_DENSE_IMPROVED_POLICY, COSFilterSpec('raised_cosine')),
    FilterGridCandidate('exp_p4', VG_DENSE_IMPROVED_POLICY, COSFilterSpec('exponential', order=4)),
    FilterGridCandidate('exp_p8', VG_DENSE_IMPROVED_POLICY, COSFilterSpec('exponential', order=8)),
    FilterGridCandidate('exp_p12', VG_DENSE_IMPROVED_POLICY, COSFilterSpec('exponential', order=12)),
]
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

if dense_best['filter'] == 'none':
    VG_DENSE_ADAPTIVE = VG_DENSE_IMPROVED
    adaptive_label = 'Adaptive selector (no filter)'
else:
    if dense_best['filter'] == 'exponential':
        dense_filter = COSFilterSpec('exponential', order=int(dense_best['filter_order']))
        filter_text = f"exponential p={int(dense_best['filter_order'])}"
    elif dense_best['filter'] == 'raised_cosine':
        dense_filter = COSFilterSpec('raised_cosine')
        filter_text = 'raised cosine'
    else:
        dense_filter = COSFilterSpec(str(dense_best['filter']))
        filter_text = str(dense_best['filter'])
    VG_DENSE_ADAPTIVE, _ = timeit_strip(
        price_strip,
        'vg', 'cos_filtered',
        VG_DENSE_K, VG_DENSE_FWD, VG_DENSE_PARAMS,
        grid=(VG_DENSE_IMPROVED_POLICY, dense_filter),
        n_repeat=2,
    )
    adaptive_label = f'Adaptive filtered ({filter_text})'

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

axes[0].plot(VG_DENSE_K, VG_DENSE_REF, color=DARK, linewidth=2.4, label='PyFENG FFT ref')
axes[0].plot(VG_DENSE_K, VG_DENSE_CLASSIC, color=CB_STEEL, linestyle='--', linewidth=1.9,
             label='COS classic')
axes[0].plot(VG_DENSE_K, VG_DENSE_IMPROVED, color=NAVY, linestyle='-.', linewidth=1.9,
             label='COS improved')
axes[0].plot(VG_DENSE_K, VG_DENSE_ADAPTIVE, color=COLUMBIA_BLUE, linewidth=1.9,
             label=adaptive_label)
axes[0].set_title('Short-maturity VG strip: price levels by COS variant')
axes[0].set_xlabel('strike K')
axes[0].set_ylabel('call price')
axes[0].legend(frameon=False, fontsize=8)

axes[1].plot(VG_DENSE_K, np.abs(VG_DENSE_CLASSIC - VG_DENSE_REF), color=CB_STEEL,
             linestyle='--', linewidth=1.9, label='COS classic')
axes[1].plot(VG_DENSE_K, np.abs(VG_DENSE_IMPROVED - VG_DENSE_REF), color=NAVY,
             linestyle='-.', linewidth=1.9, label='COS improved')
axes[1].plot(VG_DENSE_K, np.abs(VG_DENSE_ADAPTIVE - VG_DENSE_REF), color=COLUMBIA_BLUE,
             linewidth=1.9, label=adaptive_label)
axes[1].set_yscale('log')
axes[1].set_title('Cross-strike abs error vs PyFENG FFT')
axes[1].set_xlabel('strike K')
axes[1].set_ylabel('abs error')
axes[1].legend(frameon=False, fontsize=8)

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
    nb = json.loads(NB_PATH.read_text())

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

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Updated → {NB_PATH}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
