#!/usr/bin/env python3
"""Patch static notebooks so they bootstrap cleanly on Colab."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

INSTALL_SOURCE = "%pip install -q -U fourier-option-pricer\n"


def bootstrap_source(outdir_rel: str) -> str:
    return f"""from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import subprocess
import sys
import time
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

if importlib.util.find_spec('foureng') is None:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '-U', 'fourier-option-pricer'],
        check=True,
    )
    importlib.invalidate_caches()


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
    for raw in (os.environ.get('PWD'), os.environ.get('OLDPWD')):
        candidates.extend(_expand(raw))
    try:
        candidates.extend(_expand(pathlib.Path.cwd()))
    except FileNotFoundError:
        pass
    for raw in sys.path:
        if raw:
            candidates.extend(_expand(raw))
    spec = importlib.util.find_spec('foureng')
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
    if (candidate / 'foureng').exists() and (candidate / 'benchmarks').exists():
        REPO_ROOT = candidate
        break
if REPO_ROOT is None:
    clone_root = pathlib.Path.cwd() / 'fourier-option-pricer'
    repo_url = os.environ.get(
        'FOURENG_REPO_URL',
        'https://github.com/nl2992/fourier-option-pricer.git',
    )
    if not clone_root.exists():
        subprocess.run(
            [
                'git',
                'clone',
                '--depth',
                '1',
                '--quiet',
                repo_url,
                str(clone_root),
            ],
            check=True,
        )
    REPO_ROOT = clone_root.resolve()

for path in (REPO_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUTDIR = REPO_ROOT / {outdir_rel!r}
FIGDIR = OUTDIR / 'figures'
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams, bsm_cf, bsm_cumulants
from foureng.models.heston import HestonParams, heston_cf, heston_cumulants
from foureng.models.variance_gamma import VGParams, vg_cf, vg_cumulants
from foureng.models.cgmy import CgmyParams, cgmy_cf, cgmy_cumulants
from foureng.pricers.cos import cos_prices, cos_auto_grid
from foureng.pricers.carr_madan import carr_madan_price_at_strikes, carr_madan_fft_prices
from foureng.pricers.frft import frft_price_at_strikes
from foureng.utils.grids import FFTGrid, FRFTGrid, COSGrid
from foureng.utils.cumulants import Cumulants, cos_truncation_interval
from foureng.pipeline import price_strip

from benchmarks.paper_replications.fo2008_cos.params import CASES, PaperCase

pd.set_option('display.float_format', lambda x: f'{{x: .6e}}')
warnings.filterwarnings('ignore', category=RuntimeWarning)

CU_BLUE = '#012169'
CU_MID = '#2b4a8b'
CU_LIGHT = '#7aa0d1'
CU_ACC = '#B9D9EB'
CU_ORANGE = '#d9622c'
CU_GREY = '#404040'
CU_TEXT = '#243447'
CU_BG = '#f7fafd'
CU_GRID = '#dfe6ee'

mpl.rcParams.update({{
    'axes.edgecolor': CU_GREY,
    'axes.labelcolor': CU_BLUE,
    'axes.titlecolor': CU_BLUE,
    'axes.titleweight': 'semibold',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.grid': True,
    'grid.color': CU_GRID,
    'grid.linestyle': ':',
    'grid.linewidth': 0.8,
    'xtick.color': CU_GREY,
    'ytick.color': CU_GREY,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.size': 10.5,
    'legend.frameon': False,
    'legend.fontsize': 9.5,
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
}})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[CU_BLUE, CU_MID, CU_LIGHT, CU_ORANGE, '#5b87b2'])

CASE_LABELS = {{
    'bsm_table2': 'Table 2 · BSM',
    'heston_table4_t1': 'Table 4 · Heston (T=1)',
    'heston_table5_t10': 'Table 5 · Heston (T=10)',
    'heston_table6_strip': 'Table 6 · Heston strip',
    'vg_table7_t01': 'Table 7 · VG (T=0.1)',
    'vg_table7_t1': 'Table 7 · VG (T=1.0)',
    'cgmy_table8_y05': 'Table 8 · CGMY (Y=0.5)',
    'cgmy_table9_y15': 'Table 9 · CGMY (Y=1.5)',
    'cgmy_table10_y198': 'Table 10 · CGMY (Y=1.98)',
}}
METHOD_LABELS = {{
    'cos': 'COS',
    'carr_madan': 'Carr–Madan',
    'frft': 'FRFT',
    'pyfeng_fft': 'PyFENG FFT reference',
}}
METHOD_COLORS = {{
    'cos': CU_BLUE,
    'carr_madan': CU_MID,
    'frft': CU_LIGHT,
    'pyfeng_fft': '#5b87b2',
}}


def human_case(case_id):
    return CASE_LABELS.get(case_id, str(case_id).replace('_', ' '))


def human_method(method):
    return METHOD_LABELS.get(method, str(method).replace('_', ' '))


def _format_map(df: pd.DataFrame):
    fmt = {{}}
    for col in df.columns:
        key = str(col).lower()
        series = df[col]
        if pd.api.types.is_integer_dtype(series):
            fmt[col] = '{{:,.0f}}'
        elif pd.api.types.is_float_dtype(series):
            if 'price' in key:
                fmt[col] = '{{:.8f}}'
            elif 'ms' in key or 'time' in key:
                fmt[col] = '{{:.4f}}'
            else:
                fmt[col] = '{{:.2e}}'
    return fmt


def _status_style(value):
    palette = {{
        'better': 'background-color:#e8f5ee; color:#0b5d3f; font-weight:600;',
        'match': 'background-color:#edf3fb; color:#012169; font-weight:600;',
        'worse': 'background-color:#fff1eb; color:#a63c06; font-weight:600;',
        'not comparable': 'background-color:#f1f3f5; color:#495057; font-weight:600;',
    }}
    return palette.get(str(value), '')


def style_table(df: pd.DataFrame, caption: str):
    styler = (
        df.style
          .hide(axis='index')
          .format(_format_map(df), na_rep='—')
          .set_caption(caption)
          .set_table_styles([
              {{'selector': 'caption', 'props': [('color', CU_BLUE), ('font-weight', '600'),
                                                ('font-size', '13px'), ('padding', '4px 0 10px 2px'),
                                                ('text-align', 'left')]}},
              {{'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%'),
                                              ('background', 'white'), ('border', '1px solid #d9e2ef'),
                                              ('font-size', '12px')]}},
              {{'selector': 'thead th', 'props': [('background', CU_BLUE), ('color', 'white'),
                                                 ('font-weight', '600'), ('padding', '7px 10px'),
                                                 ('border', '1px solid #16346b')]}},
              {{'selector': 'tbody td', 'props': [('padding', '6px 10px'), ('border', '1px solid #e2e8f0'),
                                                 ('color', CU_TEXT)]}},
              {{'selector': 'tbody tr:nth-child(even)', 'props': [('background', CU_BG)]}},
          ])
    )
    status_cols = [col for col in df.columns if 'status' in str(col).lower()]
    if status_cols:
        styler = styler.map(_status_style, subset=status_cols)
    return styler


def show_table(df: pd.DataFrame, caption: str):
    display(style_table(df, caption))


def finish_axes(ax, *, title, xlabel, ylabel, xscale='linear', yscale='linear'):
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_facecolor(CU_BG)
    ax.grid(True, which='major', alpha=0.85)
    ax.grid(True, which='minor', alpha=0.20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

print('imports OK; repo root:', REPO_ROOT)
print('outputs:', OUTDIR)
print('cases:', list(CASES))
"""


def patch_notebook(name: str, outdir_rel: str) -> None:
    path = NB_DIR / name
    nb = nbformat.read(path, as_version=4)

    install_idx = 1
    if len(nb.cells) <= install_idx or INSTALL_SOURCE not in nb.cells[install_idx].source:
        nb.cells.insert(install_idx, new_code_cell(INSTALL_SOURCE))

    bootstrap_idx = install_idx + 1
    nb.cells[bootstrap_idx].source = bootstrap_source(outdir_rel)
    nbformat.write(nb, path)
    print(f"patched {path}")


def main() -> None:
    patch_notebook(
        "cosPaper_Replication.ipynb",
        "benchmarks/paper_replications/cos_paper_replication/outputs",
    )
    patch_notebook(
        "fo2008_replication.ipynb",
        "benchmarks/paper_replications/fo2008_cos/outputs",
    )


if __name__ == "__main__":
    main()
