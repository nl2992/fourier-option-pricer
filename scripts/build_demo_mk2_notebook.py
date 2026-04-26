"""Build notebooks/demo.ipynb — clean, self-contained PyPI demo.

Imports foureng as a published PyPI package (pip install fourier-option-pricer),
inlines HESTON_TABLE5_T10 (no benchmarks/ dependency), and drops file saves.

Run from repo root:
    python3 scripts/build_demo_mk2_notebook.py
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "notebooks" / "demo.ipynb"


# ---------------------------------------------------------------------------
# Notebook helpers
# ---------------------------------------------------------------------------

def _id() -> str:
    """Short unique cell id (8 hex chars) required by nbformat >= 5.1."""
    return uuid.uuid4().hex[:8]

def md(source: str) -> dict:
    return {"id": _id(), "cell_type": "markdown", "metadata": {}, "source": source}

def code(source: str) -> dict:
    return {"id": _id(), "cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}

def nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Cell 0 — Title
# ---------------------------------------------------------------------------

TITLE_MD = r"""# Fourier Methods for European Option Pricing

**Columbia University · MAFN · MATH 5030 · Spring 2026**

*MC baseline → Carr–Madan FFT → PyFENG Lewis → COS → Junike-style adaptive truncation*

*Demo notebook · `foureng` package · Instructor: Prof. Jaehyuk Choi*

[![PyPI](https://img.shields.io/pypi/v/fourier-option-pricer)](https://pypi.org/project/fourier-option-pricer/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nl2992/fourier-option-pricer/blob/main/notebooks/demo.ipynb)

---

### Characteristic-function backbone

All methods share the same core input: the characteristic function of the terminal log-forward return.

$$\varphi_T(u) = \mathbb{E}^{\mathbb{Q}}\!\left[e^{iu X_T}\right], \qquad X_T = \log\!\left(\frac{S_T}{F_0}\right)$$

Many models admit a closed-form $\varphi_T$ even when the density is not available in closed form. Fourier methods exploit this by recovering prices from $\varphi_T$ directly.

This notebook pits five numerical engines against published references for European option pricing:

1. **Monte Carlo** — flexible baseline; shows the $O(n^{-1/2})$ convergence bottleneck.
2. **Carr–Madan FFT** — the 1999 workhorse; damped-call transform on a uniform frequency grid.
3. **PyFENG Lewis** — external Heston benchmark exposed through the PyFENG backend.
4. **COS (Fang–Oosterlee 2008)** — primary pricer; spectral convergence on a cumulant-truncated interval.
5. **COS + Junike adaptive truncation** — tolerance-driven widening for long-maturity stress cases.

A **cross-model diagnostic** closes the notebook, sweeping all nine supported models.
"""

# ---------------------------------------------------------------------------
# Cell 1 — Install (own cell so Jupyter flushes the package before imports)
# ---------------------------------------------------------------------------

INSTALL_CODE = r"""%pip install -q -U fourier-option-pricer
"""

# ---------------------------------------------------------------------------
# Cell 2 — Imports + inline data + colours + helpers
# ---------------------------------------------------------------------------

SETUP_CODE = r"""# ── Ensure fourier-option-pricer is installed and importable ─────────────────
# This block is self-healing: it works whether or not the install cell above
# has already been run, handles editable-install conflicts, and gives a clear
# error if the package is genuinely missing after all attempts.
import sys, importlib, importlib.util

if importlib.util.find_spec("foureng") is None:
    # Not yet importable — run pip now (safe to call even if already installed)
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "fourier-option-pricer"],
        check=True,
    )
    # Reload site so Python picks up packages installed in this session
    import site
    importlib.reload(site)
    importlib.invalidate_caches()

    # Belt-and-suspenders: add every site-packages dir that pip knows about
    if importlib.util.find_spec("foureng") is None:
        import sysconfig
        for _p in sysconfig.get_paths().values():
            if "site-packages" in _p and _p not in sys.path:
                sys.path.insert(0, _p)
        importlib.invalidate_caches()

if importlib.util.find_spec("foureng") is None:
    raise ImportError(
        "foureng is not importable even after installation.\n"
        "→ Restart the kernel (Kernel ▸ Restart) and run all cells from the top."
    )

# ── Standard imports ──────────────────────────────────────────────────────────
import time, types, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from IPython.display import display
from typing import Callable, Iterable
warnings.filterwarnings("ignore")

# ── foureng ───────────────────────────────────────────────────────────────────
from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd
from foureng.mc.black_scholes_mc import MCSpec, european_call_mc
from foureng.mc.heston_conditional_mc import HestonMCScheme, heston_conditional_mc_calls
from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams
from foureng.models.heston import HestonParams, heston_cf, heston_cumulants
from foureng.models.variance_gamma import VGParams
from foureng.pipeline import price_strip
from foureng.pricers.cos import cos_adaptive_decision, cos_auto_grid, recommended_cos_policy
from foureng.refs.paper_refs import (
    HESTON_PUBLISHED_STRIP,
    OUSV_REGRESSION_STRIP_V1,
    CGMY_REGRESSION_STRIP_V1,
    NIG_REGRESSION_STRIP_V1,
    BATES_REGRESSION_STRIP_V1,
    HESTON_KOU_REGRESSION_STRIP_V1,
    HESTON_CGMY_REGRESSION_STRIP_V1,
)
from foureng.utils.grids import COSGridPolicy, FFTGrid
from foureng.viz.columbia import apply_columbia_style, NAVY, COLUMBIA_BLUE, DARK
apply_columbia_style()
pd.options.display.float_format = lambda x: f'{x:,.6g}'

# ── FO2008 Table 5 stress case — inlined, no benchmarks/ dependency ───────────
HESTON_TABLE5_T10 = types.SimpleNamespace(
    strikes=[100.0],
    maturity=10.0,
    params={"kappa": 1.5768, "theta": 0.0398, "nu": 0.5751,
            "rho": -0.5711, "v0": 0.0175, "r": 0.0, "q": 0.0},
    forward=100.0,
    reference_values=22.318945791,
    Ns=[40, 65, 90, 115, 140],
    extras={"L": 32.0},
)

# ── Colour palette ────────────────────────────────────────────────────────────
CB_LIGHT   = '#EAF4FB'
CB_SOFT    = '#D7EAF5'
CB_MID     = '#7FA3C7'
CB_STEEL   = '#5B8FB9'
CB_DEEP    = '#1F5CA6'
CB_PALETTE = [DARK, NAVY, CB_DEEP, CB_STEEL, CB_MID, COLUMBIA_BLUE]
PYFENG_LEWIS_LABEL = 'PyFENG Lewis'


# ── Helper: timed strip execution ─────────────────────────────────────────────
def timeit_strip(fn, *args, n_repeat: int = 3, warmup: bool = True, **kwargs):
    if warmup:
        fn(*args, **kwargs)
    best_ms = float('inf')
    out = None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best_ms = min(best_ms, (time.perf_counter() - t0) * 1e3)
    return out, best_ms


def sci(x: float) -> str:
    if pd.isna(x):
        return '--'
    return f'{x:.2e}'


# ── Helper: dark-theme table styler ───────────────────────────────────────────
def night_style(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    formats: dict[str, str | Callable] | None = None,
    highlight_min: Iterable[str] | None = None,
    highlight_max: Iterable[str] | None = None,
    hide_index: bool = True,
):
    styler = df.style
    if formats:
        styler = styler.format(formats)
    if highlight_min:
        styler = styler.highlight_min(
            subset=list(highlight_min), color=CB_SOFT,
            props='color: #0B1F3A; font-weight: bold;')
    if highlight_max:
        styler = styler.highlight_max(
            subset=list(highlight_max), color=CB_MID,
            props='color: #0B1F3A; font-weight: bold;')
    styler = styler.set_properties(**{
        'background-color': DARK,
        'color': '#F8FAFC',
        'border': '1px solid #4B6FA8',
        'font-size': '11px',
    })
    styler = styler.set_table_styles([
        {'selector': 'table',
         'props': [('border-collapse', 'collapse'), ('width', '100%'),
                   ('font-family', 'Menlo, Monaco, monospace')]},
        {'selector': 'caption',
         'props': [('caption-side', 'top'), ('color', NAVY),
                   ('font-size', '13px'), ('font-weight', 'bold'), ('padding', '6px 0')]},
        {'selector': 'th',
         'props': [('background-color', NAVY), ('color', '#F8FAFC'),
                   ('border', '1px solid #4B6FA8'), ('padding', '6px 8px')]},
        {'selector': 'td', 'props': [('padding', '6px 8px')]},
        {'selector': 'tbody tr:nth-child(even)',
         'props': [('background-color', '#10294B')]},
        {'selector': 'tbody tr:nth-child(odd)',
         'props': [('background-color', '#0B1F3A')]},
        {'selector': 'tbody tr:hover',
         'props': [('background-color', CB_DEEP)]},
    ], overwrite=False)
    if caption is not None:
        styler = styler.set_caption(caption)
    if hide_index:
        styler = styler.hide(axis='index')
    return styler


# ── Helper: overview diagram ───────────────────────────────────────────────────
def draw_overview_diagram():
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis('off')
    boxes = [
        (0.02, 0.22, 0.18, 0.56, 'Contract + model\nheld fixed per section'),
        (0.24, 0.22, 0.18, 0.56, 'Reference price\nclosed form or oracle'),
        (0.46, 0.22, 0.18, 0.56, 'Method family\nMC, FFT, PyFENG, COS'),
        (0.68, 0.22, 0.13, 0.56, 'Error\nmax abs'),
        (0.84, 0.22, 0.13, 0.56, 'Runtime\nms'),
    ]
    colors = [CB_LIGHT, CB_SOFT, COLUMBIA_BLUE, CB_MID, NAVY]
    text_colors = [DARK, DARK, DARK, DARK, '#F8FAFC']
    for (x, y, w, h, text), color, text_color in zip(boxes, colors, text_colors):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.04',
            facecolor=color, edgecolor=NAVY, linewidth=1.6)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text,
                ha='center', va='center', fontsize=11, color=text_color)
    for x0, x1 in [(0.20, 0.24), (0.42, 0.46), (0.64, 0.68), (0.81, 0.84)]:
        ax.annotate('', xy=(x1, 0.5), xytext=(x0, 0.5),
                    arrowprops=dict(arrowstyle='->', lw=1.8, color=NAVY))
    ax.set_title(
        'Notebook structure: one benchmark policy, then method-by-method diagnostics',
        color=NAVY, pad=10)
    fig.tight_layout()
    return fig


# ── Helper: heatmap ───────────────────────────────────────────────────────────
def heatmap(ax, pivot: pd.DataFrame, *, title: str, cmap: str,
            annotation: str = 'sci', cbar_label: str = '',
            transform: str | None = None):
    values = pivot.to_numpy(dtype=float)
    mask = np.isnan(values)
    plot_values = values.copy()
    if transform == 'log10':
        plot_values = np.where(mask, np.nan, np.log10(np.maximum(values, 1e-18)))
    elif transform == 'neglog10':
        plot_values = np.where(mask, np.nan, -np.log10(np.maximum(values, 1e-18)))
    finite = plot_values[np.isfinite(plot_values)]
    midpoint = float(np.median(finite)) if finite.size else 0.0
    im = ax.imshow(plot_values, aspect='auto', cmap=cmap)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            text = ('--' if np.isnan(val)
                    else (f'{val:.2f}' if annotation == 'float' else f'{val:.1e}'))
            tone = plot_values[i, j]
            text_color = ('#F8FAFC' if np.isfinite(tone) and tone >= midpoint
                          else DARK)
            ax.text(j, i, text, ha='center', va='center',
                    color=text_color if not np.isnan(val) else '#94a3b8',
                    fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbar_label, rotation=90)


# ── Helper: method frontier scatter ───────────────────────────────────────────
def method_frontier(ax, df: pd.DataFrame, *, x: str, y: str,
                    label_col: str, title: str):
    markers = {
        'Monte Carlo': 'o', 'Carr-Madan FFT': 's',
        PYFENG_LEWIS_LABEL: 'D', 'COS classic': 'P', 'COS improved': 'X',
    }
    colors = {
        'Monte Carlo': DARK, 'Carr-Madan FFT': NAVY,
        PYFENG_LEWIS_LABEL: COLUMBIA_BLUE,
        'COS classic': CB_STEEL, 'COS improved': CB_MID,
    }
    for label, sub in df.groupby(label_col):
        ax.plot(sub[x], sub[y],
                marker=markers.get(label, 'o'),
                label=label,
                color=colors.get(label, NAVY),
                linewidth=1.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('runtime (ms)')
    ax.set_ylabel('max abs error')
    ax.set_title(title)
    ax.legend(frameon=False, loc='best')
"""

# ---------------------------------------------------------------------------
# Cell 2 — §0 intro
# ---------------------------------------------------------------------------

S0_MD = r"""## 0. Problem setup and benchmark policy

Each section uses one contract, one model, one reference source, one error metric, and one runtime metric. The comparisons only change method settings, not the benchmark itself."""

# ---------------------------------------------------------------------------
# Cell 3 — §0 benchmark policy table
# ---------------------------------------------------------------------------

S0_POLICY_CODE = r"""benchmark_policy = pd.DataFrame([
    {
        'section': 'MC vs Carr-Madan',
        'contract / model': '41-strike BSM call strip',
        'reference': 'Black-Scholes closed form',
        'error metric': 'max abs error across strip',
        'runtime metric': 'best strip runtime (ms)',
    },
    {
        'section': 'Carr-Madan vs PyFENG Lewis',
        'contract / model': 'published Heston 5-strike strip',
        'reference': 'published 15-digit Heston prices',
        'error metric': 'max abs error across strip',
        'runtime metric': 'best strip runtime (ms)',
    },
    {
        'section': 'Plain COS replication',
        'contract / model': 'same published Heston 5-strike strip',
        'reference': 'same published Heston strip',
        'error metric': 'max abs error across strip',
        'runtime metric': 'best strip runtime (ms)',
    },
    {
        'section': 'COS truncation stress test',
        'contract / model': 'FO2008 Heston ATM, T=10',
        'reference': 'FO2008 Table 5 reference price',
        'error metric': 'abs error at K=100',
        'runtime metric': 'best price runtime (ms)',
    },
    {
        'section': 'Cross-model diagnostics',
        'contract / model': 'canonical 41-strike strips by model family',
        'reference': 'PyFENG FFT for supported models; frozen hi-res Fourier oracle for unsupported hybrids',
        'error metric': 'max abs error on canonical strip',
        'runtime metric': 'best strip runtime (ms)',
    },
])
display(night_style(
    benchmark_policy,
    caption='Benchmark policy used throughout the notebook',
    hide_index=True,
))
"""

# ---------------------------------------------------------------------------
# Cell 4 — §0 overview diagram
# ---------------------------------------------------------------------------

S0_DIAGRAM_CODE = r"""fig = draw_overview_diagram()
plt.show()
"""

# ---------------------------------------------------------------------------
# Cell 5 — §1 intro
# ---------------------------------------------------------------------------

S1_MD = r"""## 1. Monte Carlo baseline vs Carr-Madan

Monte Carlo is a flexible baseline, but for plain European options it can converge slowly relative to transform methods. To make that visible without changing the target, this section uses a Black-Scholes call strip and the closed-form strip price as the single source of truth for both methods."""

# ---------------------------------------------------------------------------
# Cell 6 — §1 MC sweep + CM sweep + table
# ---------------------------------------------------------------------------

S1_CODE = r"""BSM_K = np.linspace(80.0, 120.0, 41)
BSM_FWD = ForwardSpec(S0=100.0, r=0.03, q=0.0, T=1.0)
BSM_PARAMS = BsmParams(sigma=0.20)
BSM_REF = np.array([
    bs_price_from_fwd(
        BSM_PARAMS.sigma,
        BSInputs(F0=BSM_FWD.F0, K=float(k), T=BSM_FWD.T,
                 r=BSM_FWD.r, q=BSM_FWD.q, is_call=True),
    )
    for k in BSM_K
])

mc_rows = []
for n_paths in [5_000, 20_000, 50_000, 100_000, 200_000]:
    prices, runtime_ms = timeit_strip(
        european_call_mc,
        S0=BSM_FWD.S0,
        K=BSM_K,
        T=BSM_FWD.T,
        r=BSM_FWD.r,
        q=BSM_FWD.q,
        vol=BSM_PARAMS.sigma,
        mc=MCSpec(n_paths=n_paths, seed=7),
        n_repeat=2,
    )
    mc_rows.append({
        'method': 'Monte Carlo',
        'setting': f'n_paths={n_paths:,}',
        'runtime_ms': runtime_ms,
        'max_abs_err': float(np.max(np.abs(prices - BSM_REF))),
    })

cm_rows = []
for N in [256, 512, 1024, 2048, 4096]:
    prices, runtime_ms = timeit_strip(
        price_strip,
        'bsm',
        'carr_madan',
        BSM_K,
        BSM_FWD,
        BSM_PARAMS,
        grid=FFTGrid(N=N, eta=0.10, alpha=1.5),
    )
    cm_rows.append({
        'method': 'Carr-Madan FFT',
        'setting': f'N={N}, eta=0.10',
        'runtime_ms': runtime_ms,
        'max_abs_err': float(np.max(np.abs(prices - BSM_REF))),
    })

bsm_frontier = pd.DataFrame(mc_rows + cm_rows)

display(night_style(
    bsm_frontier,
    caption='Black-Scholes strip: Monte Carlo vs Carr-Madan under one closed-form benchmark',
    formats={'runtime_ms': '{:.3f}', 'max_abs_err': sci},
    highlight_min=['runtime_ms', 'max_abs_err'],
))

BSM_MC_REP = european_call_mc(
    S0=BSM_FWD.S0, K=BSM_K, T=BSM_FWD.T, r=BSM_FWD.r, q=BSM_FWD.q,
    vol=BSM_PARAMS.sigma, mc=MCSpec(n_paths=200_000, seed=7),
)
BSM_CM_REP = price_strip(
    'bsm', 'carr_madan', BSM_K, BSM_FWD, BSM_PARAMS,
    grid=FFTGrid(N=2048, eta=0.10, alpha=1.5),
)
"""

# ---------------------------------------------------------------------------
# Cell 7 — §1 plots
# ---------------------------------------------------------------------------

S1_PLOT_CODE = r"""fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
method_frontier(axes[0], bsm_frontier, x='runtime_ms', y='max_abs_err',
                label_col='method', title='BSM: error vs runtime')
axes[0].set_ylabel('max abs error vs closed form')

axes[1].plot(BSM_K, BSM_REF, color=DARK, linewidth=2.2, label='closed form')
axes[1].plot(BSM_K, BSM_MC_REP, color=NAVY, linestyle='--', label='MC (200k paths)')
axes[1].plot(BSM_K, BSM_CM_REP, color=COLUMBIA_BLUE, linestyle='-.', label='Carr-Madan (N=2048)')
axes[1].set_title('BSM strip: representative prices')
axes[1].set_xlabel('strike K')
axes[1].set_ylabel('call price')
axes[1].legend(frameon=False)

fig.tight_layout()
plt.show()
"""

# ---------------------------------------------------------------------------
# Cell 8 — §2 intro
# ---------------------------------------------------------------------------

S2_MD = r"""## 2. Carr-Madan vs PyFENG Lewis

From here on the contract and benchmark stay fixed: the published five-strike Heston strip. The implementation detail worth keeping explicit is that our dispatch key is still `pyfeng_fft`, but the wrapped PyFENG call is `HestonFft.price()`, so the external comparison leg shown below is labelled as `PyFENG Lewis`."""

# ---------------------------------------------------------------------------
# Cell 9 — §2 CM sweep + PyFENG + summary table
# ---------------------------------------------------------------------------

S2_CODE = r"""HESTON_FWD = HESTON_PUBLISHED_STRIP.fwd
HESTON_PARAMS = HESTON_PUBLISHED_STRIP.params
HESTON_K = HESTON_PUBLISHED_STRIP.strikes
HESTON_REF = HESTON_PUBLISHED_STRIP.prices

cm_sweep_rows = []
for N in [512, 1024, 2048, 4096]:
    for eta in [0.05, 0.10, 0.25]:
        prices, runtime_ms = timeit_strip(
            price_strip,
            'heston',
            'carr_madan',
            HESTON_K,
            HESTON_FWD,
            HESTON_PARAMS,
            grid=FFTGrid(N=N, eta=eta, alpha=1.5),
        )
        cm_sweep_rows.append({
            'method': 'Carr-Madan FFT',
            'N': N,
            'aux': eta,
            'runtime_ms': runtime_ms,
            'max_abs_err': float(np.max(np.abs(prices - HESTON_REF))),
        })
cm_sweep = pd.DataFrame(cm_sweep_rows)

pyfeng_prices, pyfeng_runtime_ms = timeit_strip(
    price_strip,
    'heston',
    'pyfeng_fft',
    HESTON_K,
    HESTON_FWD,
    HESTON_PARAMS,
)

cm_best_row = cm_sweep.sort_values(['max_abs_err', 'runtime_ms']).iloc[0]
transform_summary = pd.DataFrame([
    {
        'method': 'Carr-Madan FFT',
        'configuration': f"N={int(cm_best_row['N'])}, eta={cm_best_row['aux']:.2f}",
        'runtime_ms': float(cm_best_row['runtime_ms']),
        'max_abs_err': float(cm_best_row['max_abs_err']),
    },
    {
        'method': PYFENG_LEWIS_LABEL,
        'configuration': 'repo backend default',
        'runtime_ms': pyfeng_runtime_ms,
        'max_abs_err': float(np.max(np.abs(pyfeng_prices - HESTON_REF))),
    },
])

cm_best = cm_sweep.sort_values(['max_abs_err', 'runtime_ms']).head(6).copy()
cm_best['label'] = cm_best.apply(
    lambda r: f"N={int(r['N'])}, eta={r['aux']:.2f}", axis=1)

display(night_style(
    transform_summary,
    caption='Heston published strip: transform comparison',
    formats={'runtime_ms': '{:.3f}', 'max_abs_err': sci},
    highlight_min=['runtime_ms', 'max_abs_err'],
))
"""

# ---------------------------------------------------------------------------
# Cell 10 — §2 bar charts
# ---------------------------------------------------------------------------

S2_BAR_CODE = r"""fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

cm_plot = cm_best.sort_values('max_abs_err', ascending=True).copy()
cm_plot['score'] = -np.log10(cm_plot['max_abs_err'])
axes[0].barh(cm_plot['label'], cm_plot['score'], color=CB_STEEL, edgecolor=NAVY)
axes[0].set_title('Carr-Madan: best tested settings')
axes[0].set_xlabel('-log10(max abs error)  |  higher is better')
for y, (_, row) in enumerate(cm_plot.iterrows()):
    axes[0].text(row['score'] + 0.08, y,
                 f"{row['max_abs_err']:.1e} | {row['runtime_ms']:.3f} ms",
                 va='center', fontsize=8, color=DARK)

method_compare = transform_summary.copy().sort_values('max_abs_err', ascending=False)
method_compare['score'] = -np.log10(method_compare['max_abs_err'])
axes[1].barh(method_compare['method'], method_compare['score'],
             color=[NAVY, COLUMBIA_BLUE], edgecolor=DARK)
axes[1].set_title('Method comparison on the published strip')
axes[1].set_xlabel('-log10(max abs error)  |  higher is better')
for y, (_, row) in enumerate(method_compare.iterrows()):
    axes[1].text(row['score'] + 0.08, y,
                 f"{row['max_abs_err']:.1e} | {row['runtime_ms']:.3f} ms",
                 va='center', fontsize=8, color=DARK)

fig.tight_layout()
plt.show()
"""

# ---------------------------------------------------------------------------
# Cell 11 — §2 frontier
# ---------------------------------------------------------------------------

S2_FRONTIER_CODE = r"""frontier_transform = pd.concat([
    cm_sweep[['method', 'runtime_ms', 'max_abs_err']],
    pd.DataFrame([{
        'method': PYFENG_LEWIS_LABEL,
        'runtime_ms': pyfeng_runtime_ms,
        'max_abs_err': float(np.max(np.abs(pyfeng_prices - HESTON_REF))),
    }]),
], ignore_index=True)

fig, ax = plt.subplots(figsize=(7.0, 4.4))
method_frontier(ax, frontier_transform, x='runtime_ms', y='max_abs_err',
                label_col='method', title='Heston published strip: transform frontier')
fig.tight_layout()
plt.show()
"""

# ---------------------------------------------------------------------------
# Cell 12 — §3 intro
# ---------------------------------------------------------------------------

S3_MD = r"""## 3. Plain COS replication

This section keeps the same Heston strip and uses the plain Fang-Oosterlee setup: classical cumulant interval, fixed `L=10`, and a sweep over the cosine term count `N`. This is the baseline COS story before changing the interval policy."""

# ---------------------------------------------------------------------------
# Cell 13 — §3 plain COS table
# ---------------------------------------------------------------------------

S3_CODE = r"""HESTON_CUMS = heston_cumulants(HESTON_FWD, HESTON_PARAMS)
cos_rows = []
for N in [32, 64, 128, 256, 512, 1024]:
    grid = cos_auto_grid(HESTON_CUMS, N=N, L=10.0)
    prices, runtime_ms = timeit_strip(
        price_strip,
        'heston',
        'cos',
        HESTON_K,
        HESTON_FWD,
        HESTON_PARAMS,
        grid=grid,
    )
    cos_rows.append({
        'N': N,
        'a': grid.a,
        'b': grid.b,
        'width': grid.width,
        'runtime_ms': runtime_ms,
        'max_abs_err': float(np.max(np.abs(prices - HESTON_REF))),
    })
cos_plain = pd.DataFrame(cos_rows)

display(night_style(
    cos_plain,
    caption='Plain COS on the published Heston strip',
    formats={'a': '{:.3f}', 'b': '{:.3f}', 'width': '{:.3f}',
             'runtime_ms': '{:.3f}', 'max_abs_err': sci},
    highlight_min=['runtime_ms', 'max_abs_err'],
))

COS_STD_GRID   = cos_auto_grid(HESTON_CUMS, N=256, L=10.0)
COS_STD_PRICES = price_strip('heston', 'cos', HESTON_K, HESTON_FWD, HESTON_PARAMS,
                              grid=COS_STD_GRID)
cos_price_panel = pd.DataFrame({
    'K': HESTON_K,
    'Published ref': HESTON_REF,
    'COS N=256': COS_STD_PRICES,
    'abs err': np.abs(COS_STD_PRICES - HESTON_REF),
})
display(night_style(
    cos_price_panel,
    caption='COS N=256 vs published reference (Heston strip)',
    formats={'K': '{:.1f}', 'Published ref': '{:.8f}',
             'COS N=256': '{:.8f}', 'abs err': sci},
))
"""

# ---------------------------------------------------------------------------
# Cell 14 — §3 plots
# ---------------------------------------------------------------------------

S3_PLOT_CODE = r"""fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
axes[0].plot(cos_plain['N'], cos_plain['max_abs_err'], marker='o', color=CB_DEEP)
axes[0].set_xscale('log', base=2)
axes[0].set_yscale('log')
axes[0].set_title('Plain COS: error decay with N')
axes[0].set_xlabel('N')
axes[0].set_ylabel('max abs error vs published ref')

axes[1].plot(HESTON_K, HESTON_REF, marker='o', color=DARK, label='Published ref')
axes[1].plot(HESTON_K, COS_STD_PRICES, marker='s', linestyle='--', color=CB_DEEP,
             label='COS N=256')
axes[1].bar(HESTON_K, np.abs(COS_STD_PRICES - HESTON_REF), alpha=0.25,
            color=COLUMBIA_BLUE, label='|error|')
axes[1].set_title('Plain COS at the standard N=256 setting')
axes[1].set_xlabel('strike K')
axes[1].set_ylabel('price / abs error')
axes[1].legend(frameon=False, loc='best')

fig.tight_layout()
plt.show()
"""

# ---------------------------------------------------------------------------
# Cell 15 — §4 intro
# ---------------------------------------------------------------------------

S4_MD = r"""## 4. COS with improved truncation

Junike-style improvements are best read as an interval-selection layer around COS, not as a separate pricing family. To make the effect visible, this section uses the long-maturity FO2008 Heston stress case where the classical interval becomes very wide and resolution is easily wasted."""

# ---------------------------------------------------------------------------
# Cell 16 — §4 stress case table
# ---------------------------------------------------------------------------

S4_CODE = r"""CASE = HESTON_TABLE5_T10
STRESS_FWD = ForwardSpec(S0=CASE.forward, r=CASE.params['r'], q=CASE.params['q'],
                         T=CASE.maturity)
STRESS_PARAMS = HestonParams(
    kappa=CASE.params['kappa'],
    theta=CASE.params['theta'],
    nu=CASE.params['nu'],
    rho=CASE.params['rho'],
    v0=CASE.params['v0'],
)
STRESS_K    = np.asarray(CASE.strikes, dtype=float)
STRESS_REF  = np.asarray([CASE.reference_values], dtype=float)
STRESS_CUMS = heston_cumulants(STRESS_FWD, STRESS_PARAMS)

stress_rows = []
for N in CASE.Ns:
    classic_grid = cos_auto_grid(STRESS_CUMS, N=N, L=CASE.extras['L'])
    classic_prices, classic_ms = timeit_strip(
        price_strip,
        'heston', 'cos',
        STRESS_K, STRESS_FWD, STRESS_PARAMS,
        grid=classic_grid,
    )
    improved_policy = COSGridPolicy(
        mode='benchmark', truncation='tolerance',
        centered=True, fixed_N=N, eps_trunc=1e-10,
    )
    improved_decision = cos_adaptive_decision(
        STRESS_CUMS, model='heston', params=STRESS_PARAMS,
        policy=improved_policy,
    )
    improved_prices, improved_ms = timeit_strip(
        price_strip,
        'heston', 'cos_improved',
        STRESS_K, STRESS_FWD, STRESS_PARAMS,
        grid=improved_policy,
    )
    stress_rows.append({
        'N': N,
        'classic_width': classic_grid.width,
        'improved_width': improved_decision.grid.width,
        'classic_runtime_ms': classic_ms,
        'improved_runtime_ms': improved_ms,
        'classic_err': float(np.max(np.abs(classic_prices - STRESS_REF))),
        'improved_err': float(np.max(np.abs(improved_prices - STRESS_REF))),
        'tail_proxy': improved_decision.tail_proxy,
        'improvement_ratio': (
            float(np.max(np.abs(classic_prices - STRESS_REF)))
            / max(float(np.max(np.abs(improved_prices - STRESS_REF))), 1e-18)
        ),
    })

stress_df = pd.DataFrame(stress_rows)
display(night_style(
    stress_df,
    caption='Long-maturity Heston stress case: classical vs improved truncation',
    formats={
        'classic_width': '{:.3f}',
        'improved_width': '{:.3f}',
        'classic_runtime_ms': '{:.3f}',
        'improved_runtime_ms': '{:.3f}',
        'classic_err': sci,
        'improved_err': sci,
        'tail_proxy': sci,
        'improvement_ratio': '{:.2f}',
    },
    highlight_min=['classic_runtime_ms', 'improved_runtime_ms',
                   'classic_err', 'improved_err'],
))

STRESS_CLASSIC_140 = cos_auto_grid(STRESS_CUMS, N=140, L=CASE.extras['L'])
STRESS_IMPROVED_140 = cos_adaptive_decision(
    STRESS_CUMS, model='heston', params=STRESS_PARAMS,
    policy=COSGridPolicy(mode='benchmark', truncation='tolerance',
                         centered=True, fixed_N=140, eps_trunc=1e-10),
)
"""

# ---------------------------------------------------------------------------
# Cell 17 — §4 plots
# ---------------------------------------------------------------------------

S4_PLOT_CODE = r"""fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
axes[0].plot(stress_df['N'], stress_df['classic_err'],
             marker='o', color=CB_STEEL, label='classical interval')
axes[0].plot(stress_df['N'], stress_df['improved_err'],
             marker='s', color=NAVY, label='improved interval')
axes[0].set_yscale('log')
axes[0].set_title('FO2008 stress case: error vs N')
axes[0].set_xlabel('N')
axes[0].set_ylabel('abs error at K=100')
axes[0].legend(frameon=False)

axes[1].axvline(0.0, color='#94a3b8', linestyle=':', linewidth=1.2)
axes[1].hlines(1.0,
               STRESS_CLASSIC_140.a - STRESS_CLASSIC_140.center,
               STRESS_CLASSIC_140.b - STRESS_CLASSIC_140.center,
               color=CB_STEEL, linewidth=8,
               label=f'classical width={STRESS_CLASSIC_140.width:.1f}')
axes[1].hlines(0.4,
               STRESS_IMPROVED_140.grid.a, STRESS_IMPROVED_140.grid.b,
               color=NAVY, linewidth=8,
               label=f'improved width={STRESS_IMPROVED_140.grid.width:.1f}')
axes[1].set_ylim(0, 1.4)
axes[1].set_yticks([0.4, 1.0])
axes[1].set_yticklabels(['improved', 'classical'])
axes[1].set_title('Interval geometry at N=140')
axes[1].set_xlabel('centered state variable')
axes[1].legend(frameon=False, loc='upper left')

fig.tight_layout()
plt.show()
"""

# ---------------------------------------------------------------------------
# Cell 18 — §5 intro
# ---------------------------------------------------------------------------

S5_MD = r"""## 5. Cross-model diagnostics

The final panel compares **our three in-house pricers** against a model-by-model benchmark. For `BSM`, `Heston`, `OUSV`, `VG`, `CGMY`, and `NIG`, the benchmark is PyFENG's native FFT pricer. For `Bates`, `Heston-Kou`, and `Heston-CGMY`, PyFENG has no native pricer, so the fallback benchmark is the frozen high-resolution Fourier oracle already used in the regression tests.

This keeps the external benchmark where it exists, while still giving the hybrid jump models a defensible comparison target."""

# ---------------------------------------------------------------------------
# Cell 19 — §5 cross-model setup + tables
# ---------------------------------------------------------------------------

S5_CODE = r"""COMMON_K   = np.linspace(80.0, 120.0, 41)
COMMON_FWD = ForwardSpec(S0=100.0, r=0.03, q=0.0, T=1.0)

PYFENG_BENCHMARK_MODELS = {'bsm', 'heston', 'ousv', 'vg', 'cgmy', 'nig'}
PYFENG_BENCHMARK_LABEL  = 'PyFENG FFT benchmark'
HYBRID_BENCHMARK_LABEL  = 'Frozen hi-res Fourier oracle'


def benchmark_for_case(model, strikes, fwd, params, *,
                        fallback_prices=None, fallback_source=None):
    if model in PYFENG_BENCHMARK_MODELS:
        ref = np.asarray(price_strip(model, 'pyfeng_fft', strikes, fwd, params),
                         dtype=float)
        return ref, 'External PyFENG FFT', PYFENG_BENCHMARK_LABEL
    if fallback_prices is None or fallback_source is None:
        raise ValueError(f'No benchmark configured for model={model!r}')
    return (
        np.asarray(fallback_prices, dtype=float),
        'Internal hi-res oracle',
        f'{HYBRID_BENCHMARK_LABEL}: {fallback_source}',
    )


cross_cases = []

bsm_params = BsmParams(sigma=0.20)
bsm_ref, bsm_bt, bsm_bs = benchmark_for_case('bsm', COMMON_K, COMMON_FWD, bsm_params)
cross_cases.append({
    'name': 'BSM', 'model': 'bsm', 'family': 'Diffusion', 'jumps': 'No',
    'fwd': COMMON_FWD, 'params': bsm_params, 'strikes': COMMON_K,
    'benchmark': bsm_ref, 'benchmark type': bsm_bt, 'benchmark source': bsm_bs,
})

heston_params = HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.7, v0=0.04)
heston_ref, heston_bt, heston_bs = benchmark_for_case(
    'heston', COMMON_K, COMMON_FWD, heston_params)
cross_cases.append({
    'name': 'Heston', 'model': 'heston', 'family': 'Stochastic vol', 'jumps': 'No',
    'fwd': COMMON_FWD, 'params': heston_params, 'strikes': COMMON_K,
    'benchmark': heston_ref, 'benchmark type': heston_bt, 'benchmark source': heston_bs,
})

vg_params = VGParams(sigma=0.12, nu=0.2, theta=-0.14)
vg_ref, vg_bt, vg_bs = benchmark_for_case('vg', COMMON_K, COMMON_FWD, vg_params)
cross_cases.append({
    'name': 'VG', 'model': 'vg', 'family': 'Pure jump',
    'jumps': 'Infinite activity',
    'fwd': COMMON_FWD, 'params': vg_params, 'strikes': COMMON_K,
    'benchmark': vg_ref, 'benchmark type': vg_bt, 'benchmark source': vg_bs,
})

for anchor, name, family, jumps in [
    (OUSV_REGRESSION_STRIP_V1,       'OUSV',        'Stochastic vol',  'No'),
    (CGMY_REGRESSION_STRIP_V1,       'CGMY',        'Pure jump',       'Infinite activity'),
    (NIG_REGRESSION_STRIP_V1,        'NIG',         'Pure jump',       'Infinite activity'),
    (BATES_REGRESSION_STRIP_V1,      'Bates',       'SV + jumps',      'Finite activity'),
    (HESTON_KOU_REGRESSION_STRIP_V1, 'Heston-Kou',  'SV + jumps',      'Finite activity (double-exp)'),
    (HESTON_CGMY_REGRESSION_STRIP_V1,'Heston-CGMY', 'SV + jumps',      'Infinite activity'),
]:
    ref, bt, bs = benchmark_for_case(
        anchor.model,
        anchor.strikes,
        anchor.fwd,
        anchor.params,
        fallback_prices=anchor.prices,
        fallback_source=anchor.ref_method,
    )
    cross_cases.append({
        'name': name, 'model': anchor.model,
        'family': family, 'jumps': jumps,
        'fwd': anchor.fwd, 'params': anchor.params,
        'strikes': anchor.strikes,
        'benchmark': ref, 'benchmark type': bt, 'benchmark source': bs,
    })

taxonomy = pd.DataFrame([
    {
        'model': c['name'],
        'family': c['family'],
        'jump structure': c['jumps'],
        'benchmark type': c['benchmark type'],
        'benchmark source': c['benchmark source'],
        'strip points': len(c['strikes']),
    }
    for c in cross_cases
])
display(night_style(
    taxonomy,
    caption='Model taxonomy and benchmark source used in the cross-model diagnostic panel',
    hide_index=True,
))

cross_rows = []
for case in cross_cases:
    model   = case['model']
    fwd     = case['fwd']
    params  = case['params']
    strikes = case['strikes']
    ref     = case['benchmark']
    methods = [
        ('Carr-Madan FFT',
         lambda model=model, strikes=strikes, fwd=fwd, params=params:
             price_strip(model, 'carr_madan', strikes, fwd, params,
                         grid=FFTGrid(2048, 0.10, 1.5))),
        ('COS classic',
         lambda model=model, strikes=strikes, fwd=fwd, params=params:
             price_strip(model, 'cos', strikes, fwd, params)),
        ('COS improved',
         lambda model=model, strikes=strikes, fwd=fwd, params=params:
             price_strip(model, 'cos_improved', strikes, fwd, params,
                         grid=recommended_cos_policy(model, params, mode='surface'))),
    ]
    for label, fn in methods:
        try:
            prices, runtime_ms = timeit_strip(fn, n_repeat=2)
            err = float(np.max(np.abs(np.asarray(prices) - ref)))
        except Exception:
            runtime_ms = np.nan
            err = np.nan
        cross_rows.append({
            'model': case['name'],
            'family': case['family'],
            'jumps': case['jumps'],
            'benchmark type': case['benchmark type'],
            'benchmark source': case['benchmark source'],
            'method': label,
            'runtime_ms': runtime_ms,
            'max_abs_err': err,
        })

cross_results = pd.DataFrame(cross_rows)

best_rows = []
for model_name, sub in cross_results.groupby('model'):
    sub_valid = sub.dropna(subset=['runtime_ms', 'max_abs_err'])
    best_err  = sub_valid.sort_values(['max_abs_err', 'runtime_ms']).iloc[0]
    fastest   = sub_valid.sort_values(['runtime_ms', 'max_abs_err']).iloc[0]
    best_rows.append({
        'model': model_name,
        'benchmark source': sub_valid['benchmark source'].iloc[0],
        'best by error': best_err['method'],
        'best error': best_err['max_abs_err'],
        'fastest': fastest['method'],
        'fastest runtime_ms': fastest['runtime_ms'],
    })

best_summary = pd.DataFrame(best_rows)
display(night_style(
    best_summary,
    caption='Best-performing in-house method by model under the benchmark-aware cross-model settings',
    formats={'best error': sci, 'fastest runtime_ms': '{:.3f}'},
    hide_index=True,
))
"""

# ---------------------------------------------------------------------------
# Cell 20 — §5 heatmaps + bar charts + family tables
# ---------------------------------------------------------------------------

S5_PLOT_CODE = r"""family_order  = ['Diffusion', 'Stochastic vol', 'Pure jump', 'SV + jumps']
family_labels = {
    'Diffusion':     'Non-jump diffusion',
    'Stochastic vol':'Stochastic vol (no jumps)',
    'Pure jump':     'Pure jump',
    'SV + jumps':    'Hybrid stoch vol + jumps',
}
family_names = [family_labels[n] for n in family_order]


def describe_model(row):
    family = row['family']
    jump_structure = row['jump structure']
    if family == 'Diffusion':
        return 'non-jump diffusion'
    if family == 'Stochastic vol':
        return 'stochastic vol, no jumps'
    if family == 'Pure jump':
        return f'pure jump, {jump_structure.lower()}'
    return f'hybrid stoch vol + jumps, {jump_structure.lower()}'


def add_family_guides(ax, boundaries):
    for b in boundaries:
        ax.axhline(b - 0.5, color=DARK, lw=1.1, alpha=0.35)


taxonomy_plot = taxonomy.copy()
taxonomy_plot['family_group'] = pd.Categorical(
    taxonomy_plot['family'].map(family_labels),
    categories=family_names, ordered=True,
)
taxonomy_plot['display_label'] = taxonomy_plot.apply(
    lambda row: f"{row['model']} ({describe_model(row)})", axis=1)
taxonomy_plot = taxonomy_plot.sort_values(['family_group', 'model']).reset_index(drop=True)

model_order   = taxonomy_plot['model'].tolist()
display_order = taxonomy_plot['display_label'].tolist()
family_sizes  = (taxonomy_plot.groupby('family_group', observed=True)
                 .size().reindex(family_names, fill_value=0))
family_breaks = family_sizes.cumsum().tolist()[:-1]
method_order  = ['Carr-Madan FFT', 'COS classic', 'COS improved']

error_pivot   = (cross_results.pivot(index='model', columns='method', values='max_abs_err')
                 .reindex(index=model_order, columns=method_order))
runtime_pivot = (cross_results.pivot(index='model', columns='method', values='runtime_ms')
                 .reindex(index=model_order, columns=method_order))
error_pivot.index   = display_order
runtime_pivot.index = display_order

fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.1))
heatmap(axes[0], error_pivot,
        title='Cross-model max abs error vs benchmark',
        cmap='Blues',
        cbar_label='-log10 max abs error (higher is better)',
        transform='neglog10')
heatmap(axes[1], runtime_pivot,
        title='Cross-model runtime for in-house pricers',
        cmap='Blues',
        cbar_label='log10 runtime (ms)',
        annotation='float',
        transform='log10')
for ax in axes:
    add_family_guides(ax, family_breaks)
axes[1].set_yticklabels([])
axes[1].tick_params(axis='y', length=0)
fig.subplots_adjust(left=0.36, right=0.98, wspace=0.18)
plt.show()

method_colors = {
    'Carr-Madan FFT': NAVY,
    'COS classic':    CB_STEEL,
    'COS improved':   COLUMBIA_BLUE,
}

best_error = (
    cross_results.dropna(subset=['max_abs_err', 'runtime_ms'])
    .sort_values(['model', 'max_abs_err', 'runtime_ms'])
    .groupby('model', as_index=False).first()
)
fastest_runtime = (
    cross_results.dropna(subset=['max_abs_err', 'runtime_ms'])
    .sort_values(['model', 'runtime_ms', 'max_abs_err'])
    .groupby('model', as_index=False).first()
)
best_error['model']      = pd.Categorical(best_error['model'],      categories=model_order, ordered=True)
fastest_runtime['model'] = pd.Categorical(fastest_runtime['model'], categories=model_order, ordered=True)
best_error      = best_error.sort_values('model')
fastest_runtime = fastest_runtime.sort_values('model')
y_pos = np.arange(len(display_order))

fig2, axes2 = plt.subplots(1, 2, figsize=(15.6, 6.0), sharey=True)
axes2[0].barh(y_pos, best_error['max_abs_err'],
              color=[method_colors[m] for m in best_error['method']],
              edgecolor=DARK)
axes2[0].set_xscale('log')
axes2[0].set_title('Best error by model (vs benchmark)')
axes2[0].set_xlabel('max abs error')
axes2[0].set_yticks(y_pos)
axes2[0].set_yticklabels(display_order)
axes2[0].invert_yaxis()

axes2[1].barh(y_pos, fastest_runtime['runtime_ms'],
              color=[method_colors[m] for m in fastest_runtime['method']],
              edgecolor=DARK)
axes2[1].set_xscale('log')
axes2[1].set_title('Fastest in-house method by model')
axes2[1].set_xlabel('runtime (ms)')
axes2[1].tick_params(axis='y', left=False, labelleft=False)
axes2[1].invert_yaxis()

for ax in axes2:
    add_family_guides(ax, family_breaks)

legend_handles = [Patch(facecolor=c, edgecolor=DARK, label=l)
                  for l, c in method_colors.items()]
axes2[1].legend(handles=legend_handles, frameon=False, loc='lower right',
                title='Winning method')
fig2.subplots_adjust(left=0.36, right=0.98, wspace=0.14)
plt.show()

family_compare = (
    cross_results
    .merge(taxonomy_plot[['model', 'family_group']], on='model', how='left')
    .groupby(['family_group', 'method'], observed=True)
    .agg(median_error=('max_abs_err', 'median'),
         median_runtime_ms=('runtime_ms', 'median'))
    .reset_index()
)
family_error_compare = (
    family_compare.pivot(index='family_group', columns='method', values='median_error')
    .reindex(index=family_names, columns=method_order)
    .reset_index()
    .rename(columns={'family_group': 'model family'})
)
family_runtime_compare = (
    family_compare.pivot(index='family_group', columns='method', values='median_runtime_ms')
    .reindex(index=family_names, columns=method_order)
    .reset_index()
    .rename(columns={'family_group': 'model family'})
)
display(night_style(
    family_error_compare,
    caption='Median max abs error by model family under the benchmark-aware comparison',
    formats={m: sci for m in method_order},
    hide_index=True,
))
display(night_style(
    family_runtime_compare,
    caption='Median runtime by model family (ms)',
    formats={m: '{:.3f}' for m in method_order},
    hide_index=True,
))
"""

# ---------------------------------------------------------------------------
# Cell 21 — Conclusions
# ---------------------------------------------------------------------------

CONCLUSIONS_MD = r"""## General conclusions from the notebook

1. **COS classic is the strongest default pricer in the demo.**
   - On the published Heston strip, plain COS reaches reference-level accuracy quickly and remains stable as the number of terms increases.
   - In the benchmark-aware cross-model panel, it is the most consistent method across diffusion, stochastic-volatility, pure-jump, and hybrid-jump families.
   - The main practical result is that COS classic gives the best overall balance of accuracy, runtime, and robustness.

2. **COS improved is a robustness upgrade, not a blanket replacement.**
   - In the long-maturity Heston stress case, the improved truncation helps when the classical interval becomes too wide and starts wasting resolution.
   - In the cross-model panel, it is strongest in selected regimes, such as Heston and Heston-CGMY, and it often wins on speed even when it does not win on error.
   - When the classical interval is already well calibrated, the gain from the improved policy is naturally small and can sometimes reverse.

3. **The benchmark design supports the pricing conclusions.**
   - Monte Carlo remains a useful baseline, but for vanilla European strips it is clearly dominated by Fourier and spectral methods.
   - For `BSM`, `Heston`, `OUSV`, `VG`, `CGMY`, and `NIG`, the notebook compares our pricers against an external PyFENG FFT benchmark.
   - For `Bates`, `Heston-Kou`, and `Heston-CGMY`, where PyFENG has no native pricer, the comparison falls back to the frozen high-resolution Fourier oracle already cross-verified in the tests.

4. **The project takeaway is clear.**
   - `COS classic` should be treated as the main production-quality pricer in this notebook.
   - `COS improved` should be presented as a targeted safeguard for truncation-sensitive regimes rather than as a universal upgrade.
   - `Carr-Madan FFT` remains valuable as a classical Fourier reference and validation tool, but it is more sensitive to numerical tuning than COS.

---
*Package*: `pip install fourier-option-pricer` ·
*Repo*: [github.com/nl2992/fourier-option-pricer](https://github.com/nl2992/fourier-option-pricer) ·
*Author*: Nigel Li · Columbia MAFN 2026
"""

# ---------------------------------------------------------------------------
# Assemble and write
# ---------------------------------------------------------------------------

cells = [
    md(TITLE_MD),
    code(INSTALL_CODE),
    code(SETUP_CODE),
    md(S0_MD),
    code(S0_POLICY_CODE),
    code(S0_DIAGRAM_CODE),
    md(S1_MD),
    code(S1_CODE),
    code(S1_PLOT_CODE),
    md(S2_MD),
    code(S2_CODE),
    code(S2_BAR_CODE),
    code(S2_FRONTIER_CODE),
    md(S3_MD),
    code(S3_CODE),
    code(S3_PLOT_CODE),
    md(S4_MD),
    code(S4_CODE),
    code(S4_PLOT_CODE),
    md(S5_MD),
    code(S5_CODE),
    code(S5_PLOT_CODE),
    md(CONCLUSIONS_MD),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb(cells), indent=1))
print(f"Written → {OUT}  ({len(cells)} cells)")
