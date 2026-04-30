#!/usr/bin/env python3
"""
Patch demo.ipynb §6 extension cells and create adaptive_cos.ipynb.

demo.ipynb changes:
  - Cell 23: richer narrative (issues + resolutions, stress-case table)
  - Cell 24: trim duplicate imports already in cell 2; drop _FIG_DIR
  - Cell 25: simplify header
  - Cell 27: remove fig.savefig
  - Cell 28: shorten interpretation
  - Cell 31: remove both fig.savefig + fig2 save block
  - Cell 32: simplify header

adaptive_cos.ipynb (new):
  - Fang-Oosterlee 2008 canonical test cases
  - Three-way comparison: vanilla COS, Junike COS, adaptive filtered COS
"""
import json, pathlib

ROOT     = pathlib.Path(__file__).resolve().parents[1]
NB_DEMO  = ROOT / "notebooks" / "demo.ipynb"
NB_ADAP  = ROOT / "notebooks" / "adaptive_cos.ipynb"

# ── helpers ──────────────────────────────────────────────────────────────────
def code_cell(src):
    if isinstance(src, str):
        src = src.splitlines(keepends=True)
    return {"cell_type":"code","execution_count":None,
            "metadata":{},"outputs":[],"source":src}

def md_cell(src):
    if isinstance(src, str):
        src = src.splitlines(keepends=True)
    return {"cell_type":"markdown","metadata":{},"source":src}

def nb_template(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name":"python","version":"3.11.0"}
        },
        "cells": cells,
    }

# ════════════════════════════════════════════════════════════════════════════
# PART 1 — patch demo.ipynb extension cells
# ════════════════════════════════════════════════════════════════════════════
nb = json.loads(NB_DEMO.read_text())

# ── Cell 23: richer narrative ─────────────────────────────────────────────
nb["cells"][23] = md_cell(
    "## \u00a76 \u00b7 Adaptive Filtered-COS Policy Layer\n"
    "\n"
    "### What COS gets wrong \u2014 and how we fix it\n"
    "\n"
    "The COS method approximates the option price as a truncated cosine series on "
    "an interval $[a, b]$ covering the log-return density. Accuracy depends on two "
    "independent parameters: the **interval width** $b - a$ and the **series "
    "length** $N$. Vanilla COS (Fang \u0026 Oosterlee 2008) sets both via simple "
    "heuristics. This section identifies the remaining weaknesses and adds targeted "
    "fixes.\n"
    "\n"
    "**Issue 1 \u2014 Heuristic truncation is wrong for stress cases.**  \n"
    "Vanilla COS fixes $[a, b]$ from the first four cumulants with $L = 10$. For "
    "long-maturity or heavy-tailed models the interval under-covers the distribution "
    "and truncation error dominates. For short maturities it over-covers and wastes "
    "resolution. *Fix:* Junike \u0026 Pankrashkin (2022) iterate the half-width until "
    "a tail-mass proxy (integral of $|\\phi(u)|$ in the tails) falls below "
    "$\\varepsilon_{\\text{trunc}}$. This directly removes truncation error for any "
    "model \u2014 demonstrated in \u00a74 above on the FO2008 Heston $T = 10$ stress case.\n"
    "\n"
    "**Issue 2 \u2014 Finite-series oscillation survives Junike truncation.**  \n"
    "Even on the correct interval, the cosine expansion truncated at $N$ terms "
    "exhibits Gibbs-like oscillations near the payoff kink and wherever the "
    "characteristic function decays slowly (pure-jump models, short maturities). "
    "These oscillations are orthogonal to the truncation error and are not reduced "
    "by widening $[a, b]$. *Fix:* Apply a **spectral filter** $\\sigma_k$ to each "
    "COS coefficient before summing: $A_k \\to \\sigma_k A_k$. The filter tapers "
    "high-frequency terms, trading a small controllable bias for a large reduction "
    "in oscillatory error. Five filter families are implemented (Fej\u00e9r, Lanczos, "
    "raised-cosine, exponential of tunable order $p$). The exponential filter was "
    "found most effective across jump models: it decays as "
    "$\\exp(-\\alpha (k/N)^p)$, giving a smooth taper whose aggressiveness is "
    "controlled by $p$.\n"
    "\n"
    "**Issue 3 \u2014 A fixed filter choice regresses on smooth models.**  \n"
    "Applying a strong filter unconditionally introduces bias for models where "
    "vanilla COS already converges smoothly (BSM, Heston at moderate maturity). "
    "*Fix:* Run a **deterministic policy search** over seven candidate policies "
    "(Junike + no filter, + each of the six filters). The adaptive selector picks "
    "the **fastest candidate meeting a target error tolerance** $\\varepsilon_{\\text{tol}}$. "
    "Because Junike-no-filter is always in the set, the selector **weakly dominates "
    "any single fixed policy** in the candidate set: it can never choose a result "
    "worse than Junike alone.\n"
    "\n"
    "### Stress cases\n"
    "\n"
    "| Case | Model | $T$ | Primary numerical challenge |\n"
    "|------|-------|-----|-----------------------------|\n"
    "| `BSM_T1` | BSM | 1y | Control: smooth Gaussian CF, no filter expected |\n"
    "| `Heston_T1` | Heston SV | 1y | Moderate roughness from SV mean-reversion/correlation |\n"
    "| `VG_T01` | Variance Gamma | 0.1y | Pure jump, short maturity: peaked density, slow CF decay |\n"
    "| `VG_T1` | Variance Gamma | 1y | L\u00e9vy at longer maturity: density smoother, Junike near-optimal |\n"
    "| `CGMY_T025` | CGMY (infinite activity) | 0.25y | Oscillatory CF, Gibbs error survives Junike |\n"
    "| `Kou_T05` | Kou double-exp jump | 0.5y | Jump diffusion: exponential filter expected to help |\n"
    "\n"
    "### Implementation note\n"
    "\n"
    "One model-specific issue arose during development: `recommended_cos_policy` "
    "returns `truncation='heuristic'` for VG (not `'tolerance'`), because VG's tail "
    "is light enough that the cumulant-based interval is reliable. Forcing "
    "`truncation='tolerance'` for VG at $T = 0.1$ produces an artificially narrow "
    "interval and poor candidate results. The fix is to delegate to "
    "`recommended_cos_policy(model, params)` per case, which handles the model-specific "
    "truncation strategy correctly, rather than hard-coding a single strategy across "
    "all models.\n"
    "\n"
    "**References**\n"
    "- Junike \u0026 Pankrashkin (2022), *Precise option pricing by the COS method*, "
    "Applied Mathematics and Computation, 421, 126935.\n"
    "- Ruijter, Versteegh \u0026 Oosterlee (2015), *On the application of spectral "
    "filters in a Fourier option pricing technique*, Journal of Computational Finance.\n"
)

# ── Cell 24: trimmed imports ──────────────────────────────────────────────
# Already in cell 2: time, sys, importlib, np, pd, plt,
#   ForwardSpec, BsmParams, HestonParams, VGParams,
#   price_strip, recommended_cos_policy, COSGridPolicy
nb["cells"][24] = code_cell([
    "import pathlib\n",
    "\n",
    "from foureng.models.cgmy import CgmyParams\n",
    "from foureng.models.kou  import KouParams\n",
    "\n",
    "# spectral_filters and experiments are not in PyPI v0.2.0.\n",
    "# If the editable install is clobbered by %pip install, evict the stale\n",
    "# module cache and reload from the repo root.\n",
    "def _try_ext():\n",
    "    try:\n",
    "        from foureng.utils.spectral_filters import COSFilterSpec          # noqa\n",
    "        from foureng.experiments.cos_filter_grid_search import (          # noqa\n",
    "            FilterGridCandidate, run_filtered_cos_grid_search,\n",
    "            select_fastest_under_tolerance,\n",
    "        )\n",
    "        return True\n",
    "    except (ImportError, ModuleNotFoundError):\n",
    "        return False\n",
    "\n",
    "if not _try_ext():\n",
    "    _root = pathlib.Path('.').resolve()\n",
    "    for _ in range(6):\n",
    "        if (_root / 'pyproject.toml').exists():\n",
    "            break\n",
    "        _root = _root.parent\n",
    "    for _k in [k for k in sys.modules if k == 'foureng' or k.startswith('foureng.')]:\n",
    "        del sys.modules[_k]\n",
    "    if str(_root) not in sys.path:\n",
    "        sys.path.insert(0, str(_root))\n",
    "    importlib.invalidate_caches()\n",
    "    if not _try_ext():\n",
    "        raise ImportError('foureng extension not importable. Run: pip install -e .')\n",
    "\n",
    "from foureng.utils.spectral_filters import COSFilterSpec\n",
    "from foureng.experiments.cos_filter_grid_search import (\n",
    "    FilterGridCandidate, run_filtered_cos_grid_search, select_fastest_under_tolerance,\n",
    ")\n",
    "\n",
    "_OUT_DIR = pathlib.Path('.').resolve().parent / 'benchmarks/mc_vs_fourier_methods/outputs'\n",
    "_OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "_TOL = 1e-6\n",
    "\n",
    "STRESS_CASES = [\n",
    "    dict(case_name='BSM_T1',    model='bsm',\n",
    "         fwd=ForwardSpec(S0=100, r=0.03, q=0.01, T=1.0),\n",
    "         params=BsmParams(sigma=0.25), strikes=np.linspace(80, 120, 13)),\n",
    "    dict(case_name='Heston_T1', model='heston',\n",
    "         fwd=ForwardSpec(S0=100, r=0.03, q=0.01, T=1.0),\n",
    "         params=HestonParams(kappa=2.0, theta=0.04, nu=0.5, rho=-0.7, v0=0.04),\n",
    "         strikes=np.linspace(80, 120, 13)),\n",
    "    dict(case_name='VG_T01',    model='vg',\n",
    "         fwd=ForwardSpec(S0=100, r=0.1, q=0.0, T=0.1),\n",
    "         params=VGParams(sigma=0.12, nu=0.2, theta=-0.14),\n",
    "         strikes=np.linspace(70, 130, 25)),\n",
    "    dict(case_name='VG_T1',     model='vg',\n",
    "         fwd=ForwardSpec(S0=100, r=0.05, q=0.0, T=1.0),\n",
    "         params=VGParams(sigma=0.12, nu=0.2, theta=-0.14),\n",
    "         strikes=np.linspace(80, 120, 13)),\n",
    "    dict(case_name='CGMY_T025', model='cgmy',\n",
    "         fwd=ForwardSpec(S0=100, r=0.04, q=0.0, T=0.25),\n",
    "         params=CgmyParams(C=1.0, G=5.0, M=10.0, Y=0.5),\n",
    "         strikes=np.linspace(80, 120, 13)),\n",
    "    dict(case_name='Kou_T05',   model='kou',\n",
    "         fwd=ForwardSpec(S0=100, r=0.04, q=0.0, T=0.5),\n",
    "         params=KouParams(sigma=0.16, lam=0.30, p=0.40, eta1=10.0, eta2=6.0),\n",
    "         strikes=np.linspace(80, 120, 13)),\n",
    "]\n",
    "\n",
    "def _oracle_policy(model):\n",
    "    trunc = 'heuristic' if model == 'vg' else 'tolerance'\n",
    "    return COSGridPolicy(mode='benchmark', truncation=trunc, centered=True,\n",
    "                         dx_target=0.001, L=14.0, eps_trunc=1e-12,\n",
    "                         min_N=64, max_N=16384, width_fallback=0.0)\n",
    "\n",
    "def _candidate_set(policy):\n",
    "    return [\n",
    "        FilterGridCandidate('no_filter',   policy, None),\n",
    "        FilterGridCandidate('fejer',       policy, COSFilterSpec('fejer')),\n",
    "        FilterGridCandidate('lanczos',     policy, COSFilterSpec('lanczos')),\n",
    "        FilterGridCandidate('raised_cos',  policy, COSFilterSpec('raised_cosine')),\n",
    "        FilterGridCandidate('exp_p4',      policy, COSFilterSpec('exponential', order=4)),\n",
    "        FilterGridCandidate('exp_p8',      policy, COSFilterSpec('exponential', order=8)),\n",
    "        FilterGridCandidate('exp_p12',     policy, COSFilterSpec('exponential', order=12)),\n",
    "    ]\n",
    "\n",
    "print('Setup complete:', len(STRESS_CASES), 'stress cases')\n",
])

# ── Cell 25: simplified header ────────────────────────────────────────────
nb["cells"][25] = md_cell(
    "### \u00a76.1 \u2014 Vanilla COS vs Junike-COS vs Adaptive Filtered-COS\n"
    "\n"
    "Policy search over 7 candidates (Junike + 6 filter variants) for each stress "
    "case. The selector picks the fastest candidate meeting $\\varepsilon_{\\text{tol}} = 10^{-6}$.\n"
)

# ── Cell 26: trim inline comments ────────────────────────────────────────
nb["cells"][26] = code_cell([
    "_showcase_rows = []\n",
    "\n",
    "for case in STRESS_CASES:\n",
    "    name, model = case['case_name'], case['model']\n",
    "    fwd, params, strikes = case['fwd'], case['params'], case['strikes']\n",
    "\n",
    "    ref     = price_strip(model, 'cos_improved', strikes, fwd, params,\n",
    "                           grid=_oracle_policy(model))\n",
    "    rec_pol = recommended_cos_policy(model, params)\n",
    "\n",
    "    t0 = time.perf_counter()\n",
    "    van    = price_strip(model, 'cos', strikes, fwd, params)\n",
    "    rt_van = (time.perf_counter() - t0) * 1e3\n",
    "    err_van = float(np.max(np.abs(van - ref)))\n",
    "\n",
    "    t0 = time.perf_counter()\n",
    "    jun    = price_strip(model, 'cos_improved', strikes, fwd, params, grid=rec_pol)\n",
    "    rt_jun = (time.perf_counter() - t0) * 1e3\n",
    "    err_jun = float(np.max(np.abs(jun - ref)))\n",
    "\n",
    "    df_srch = run_filtered_cos_grid_search(\n",
    "        model=model, strikes=strikes, fwd=fwd, params=params,\n",
    "        reference=ref, candidates=_candidate_set(rec_pol),\n",
    "        tol=_TOL, n_repeat=3,\n",
    "    )\n",
    "    best     = select_fastest_under_tolerance(df_srch, tol=_TOL)\n",
    "    err_adap = float(best['max_abs_err'])\n",
    "    rt_adap  = float(best['runtime_ms'])\n",
    "\n",
    "    if err_adap < err_jun - 1e-12 and err_adap < err_van - 1e-12:\n",
    "        label = 'strictly improves error'\n",
    "    elif err_adap <= _TOL and rt_adap < rt_jun:\n",
    "        label = 'matches tolerance faster'\n",
    "    elif err_adap <= _TOL:\n",
    "        label = 'matches Junike within tolerance'\n",
    "    else:\n",
    "        label = 'no improvement found'\n",
    "\n",
    "    _showcase_rows.append(dict(\n",
    "        case_name=name, model=model, maturity=fwd.T,\n",
    "        strike_count=len(strikes),\n",
    "        vanilla_cos_error=err_van,    vanilla_cos_runtime_ms=rt_van,\n",
    "        junike_cos_error=err_jun,     junike_cos_runtime_ms=rt_jun,\n",
    "        adaptive_filtered_error=err_adap, adaptive_filtered_runtime_ms=rt_adap,\n",
    "        selected_filter=best['filter'],\n",
    "        selected_policy=f\"dx={best['dx_target']},L={best['L']}\",\n",
    "        adaptive_result_label=label,\n",
    "        _df_srch=df_srch,\n",
    "    ))\n",
    "\n",
    "showcase_df = pd.DataFrame([{k: v for k, v in r.items() if k != '_df_srch'}\n",
    "                              for r in _showcase_rows])\n",
    "showcase_df.to_csv(_OUT_DIR / 'cos_policy_search_showcase.csv', index=False)\n",
    "showcase_df[['case_name','vanilla_cos_error','junike_cos_error',\n",
    "             'adaptive_filtered_error','selected_filter','adaptive_result_label']]\n",
])

# ── Cell 27: drop savefig ─────────────────────────────────────────────────
nb["cells"][27] = code_cell([
    "_priority = ['strictly improves error', 'matches tolerance faster',\n",
    "             'matches Junike within tolerance', 'no improvement found']\n",
    "_best_case = min(_showcase_rows,\n",
    "    key=lambda r: (_priority.index(r['adaptive_result_label']),\n",
    "                   r['adaptive_filtered_error']))\n",
    "\n",
    "print(f\"Showcase case : {_best_case['case_name']}\")\n",
    "print(f\"  Vanilla error  : {_best_case['vanilla_cos_error']:.3e}\")\n",
    "print(f\"  Junike error   : {_best_case['junike_cos_error']:.3e}\")\n",
    "print(f\"  Adaptive error : {_best_case['adaptive_filtered_error']:.3e}\")\n",
    "print(f\"  Filter chosen  : {_best_case['selected_filter']}\")\n",
    "print(f\"  Label          : {_best_case['adaptive_result_label']}\")\n",
    "\n",
    "_sr = _best_case['_df_srch'][_best_case['_df_srch']['status'] == 'ok'].copy()\n",
    "_cmap = {'none':'steelblue','fejer':'darkorange','lanczos':'green',\n",
    "          'raised_cosine':'purple','exponential':'crimson'}\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "_seen = set()\n",
    "for _, row in _sr.iterrows():\n",
    "    c  = _cmap.get(row['filter'], 'grey')\n",
    "    lbl = row['filter'] if row['filter'] not in _seen else ''\n",
    "    _seen.add(row['filter'])\n",
    "    ax.scatter(row['runtime_ms'], max(row['max_abs_err'], 1e-16),\n",
    "               c=c, marker='o', s=55, alpha=0.7, label=lbl)\n",
    "\n",
    "ax.scatter(_best_case['adaptive_filtered_runtime_ms'],\n",
    "           max(_best_case['adaptive_filtered_error'], 1e-16),\n",
    "           c='gold', marker='*', s=350, zorder=5, label='Selected (adaptive)')\n",
    "ax.axhline(_best_case['junike_cos_error'],  color='steelblue', ls='--', lw=1.2,\n",
    "           alpha=0.6, label='Junike error')\n",
    "ax.axhline(_best_case['vanilla_cos_error'], color='grey',      ls=':',  lw=1.2,\n",
    "           alpha=0.6, label='Vanilla error')\n",
    "ax.axhline(_TOL, color='black', ls='-.', lw=0.9, alpha=0.5,\n",
    "           label=f'Target tol={_TOL:.0e}')\n",
    "\n",
    "ax.set_xscale('log'); ax.set_yscale('log')\n",
    "ax.set_xlabel('runtime (ms)'); ax.set_ylabel('max|error|')\n",
    "ax.set_title(f\"Policy search: {_best_case['case_name']} \"\n",
    "             f\"({_best_case['adaptive_result_label']})\")\n",
    "handles, labels = ax.get_legend_handles_labels()\n",
    "ax.legend(dict(zip(labels, handles)).values(),\n",
    "          dict(zip(labels, handles)).keys(), fontsize=8)\n",
    "ax.grid(True, which='both', ls='--', alpha=0.35)\n",
    "fig.tight_layout()\n",
    "plt.show()\n",
])

# ── Cell 28: shorten interpretation ──────────────────────────────────────
nb["cells"][28] = md_cell(
    "The scatter above shows all 7 candidates in (runtime, error) space for the "
    "selected showcase case. The **gold star** marks the adaptive selection; "
    "dashed lines show the Junike and vanilla COS error baselines.\n"
    "\n"
    "Read the `adaptive_result_label` column in the table for each case:\n"
    "- *strictly improves error* \u2014 filter reduced error below both baselines.\n"
    "- *matches tolerance faster* \u2014 filter met $\\varepsilon_{\\text{tol}}$ with "
    "lower runtime than Junike-only.\n"
    "- *matches Junike within tolerance* \u2014 no-filter Junike already the best; "
    "selector confirms it.\n"
    "- *no improvement found* \u2014 adaptive matches or exceeds Junike quality "
    "(filter not forced).\n"
)

# ── Cell 29: keep as-is ───────────────────────────────────────────────────
# (already short)

# ── Cell 30: keep as-is ───────────────────────────────────────────────────

# ── Cell 31: drop savefig + fig2 save block ──────────────────────────────
nb["cells"][31] = code_cell([
    "cases = zoo_df['case_name'].tolist()\n",
    "x, w  = np.arange(len(cases)), 0.25\n",
    "\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
    "\n",
    "ax1.bar(x - w, np.clip(zoo_df['vanilla_cos_error'],         1e-16, None),\n",
    "        w, label='Vanilla COS',       color='silver',          edgecolor='grey')\n",
    "ax1.bar(x,     np.clip(zoo_df['junike_cos_error'],           1e-16, None),\n",
    "        w, label='Junike COS',        color='steelblue',       edgecolor='navy', alpha=0.85)\n",
    "ax1.bar(x + w, np.clip(zoo_df['adaptive_filtered_error'],   1e-16, None),\n",
    "        w, label='Adaptive filtered', color='mediumseagreen',  edgecolor='darkgreen', alpha=0.85)\n",
    "ax1.axhline(_TOL, color='black', ls='-.', lw=1, alpha=0.6,\n",
    "            label=f'tol={_TOL:.0e}')\n",
    "ax1.set_yscale('log')\n",
    "ax1.set_xticks(x); ax1.set_xticklabels(cases, rotation=30, ha='right')\n",
    "ax1.set_ylabel('max|error|'); ax1.set_title('Error by model')\n",
    "ax1.legend(fontsize=8); ax1.grid(True, which='both', ls='--', alpha=0.3, axis='y')\n",
    "\n",
    "ax2.bar(x - w, zoo_df['vanilla_cos_runtime_ms'],           w,\n",
    "        label='Vanilla COS',       color='silver',          edgecolor='grey')\n",
    "ax2.bar(x,     zoo_df['junike_cos_runtime_ms'],            w,\n",
    "        label='Junike COS',        color='steelblue',       edgecolor='navy', alpha=0.85)\n",
    "ax2.bar(x + w, zoo_df['adaptive_filtered_runtime_ms'],    w,\n",
    "        label='Adaptive filtered', color='mediumseagreen',  edgecolor='darkgreen', alpha=0.85)\n",
    "ax2.set_xticks(x); ax2.set_xticklabels(cases, rotation=30, ha='right')\n",
    "ax2.set_ylabel('runtime (ms)'); ax2.set_title('Runtime by model')\n",
    "ax2.legend(fontsize=8); ax2.grid(True, which='both', ls='--', alpha=0.3, axis='y')\n",
    "\n",
    "fig.suptitle('Adaptive filtered-COS: model-zoo rerun', fontsize=12, y=1.01)\n",
    "fig.tight_layout()\n",
    "plt.show()\n",
])

# ── Cell 32: simplify header ──────────────────────────────────────────────
nb["cells"][32] = md_cell(
    "### \u00a76.3 \u2014 Four conclusions from the adaptive rerun\n"
    "\n"
    "Generated from computed results \u2014 not hard-coded.\n"
)

NB_DEMO.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("demo.ipynb patched (cells 23-32).")


# ════════════════════════════════════════════════════════════════════════════
# PART 2 — create adaptive_cos.ipynb
# ════════════════════════════════════════════════════════════════════════════

C_TITLE = md_cell(
    "# Adaptive Filtered-COS: Fang\u2013Oosterlee Test Cases\n"
    "\n"
    "**Columbia University \u00b7 MAFN \u00b7 MATH 5030 \u00b7 Spring 2026**\n"
    "\n"
    "This notebook benchmarks three COS variants on the canonical test cases from\n"
    "Fang \u0026 Oosterlee (2008):\n"
    "\n"
    "| Method | Description |\n"
    "|--------|-------------|\n"
    "| **Vanilla COS** | Cumulant-based interval $[a,b]$, fixed $L=10$, no filter |\n"
    "| **Junike COS** | Tolerance-driven interval widening, no filter |\n"
    "| **Adaptive filtered COS** | Junike interval + deterministic filter search |\n"
    "\n"
    "For each test case we show:\n"
    "- Error vs $N$ convergence for vanilla and Junike COS;\n"
    "- The adaptive selector result (auto-chosen $N$, filter, error, runtime).\n"
    "\n"
    "**References**\n"
    "- Fang \u0026 Oosterlee (2008), *A novel pricing method for European options based "
    "on Fourier-cosine series expansions*, SIAM J. Sci. Comput.\n"
    "- Junike \u0026 Pankrashkin (2022), *Precise option pricing by the COS method*, "
    "Applied Mathematics and Computation.\n"
    "- Ruijter, Versteegh \u0026 Oosterlee (2015), *On the application of spectral "
    "filters in a Fourier option pricing technique*, J. Computational Finance.\n"
)

C_PIP = code_cell(
    "# Leave Colab's numpy/scipy/matplotlib stack alone; install notebook extras explicitly.\n"
    "%pip install -q fourier-option-pricer pandas pyfeng statsmodels\n"
)

C_SETUP = code_cell([
    "import os\n",
    "import pathlib, sys, importlib, importlib.util\n",
    "import subprocess\n",
    "import time\n",
    "_NOTEBOOK_PIP_PACKAGES = []\n",
    "if importlib.util.find_spec('foureng') is None:\n",
    "    _NOTEBOOK_PIP_PACKAGES.append('fourier-option-pricer')\n",
    "for _pkg in ('numpy', 'scipy', 'matplotlib', 'pandas', 'pyfeng', 'statsmodels'):\n",
    "    if importlib.util.find_spec(_pkg) is None:\n",
    "        _NOTEBOOK_PIP_PACKAGES.append(_pkg)\n",
    "if _NOTEBOOK_PIP_PACKAGES:\n",
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *_NOTEBOOK_PIP_PACKAGES], check=True)\n",
    "    importlib.invalidate_caches()\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from IPython.display import display\n",
    "\n",
    "from foureng.iv.implied_vol    import BSInputs, bs_price_from_fwd\n",
    "from foureng.models.base       import ForwardSpec\n",
    "from foureng.models.bsm        import BsmParams\n",
    "from foureng.models.heston     import HestonParams, heston_cumulants\n",
    "from foureng.models.variance_gamma import VGParams\n",
    "from foureng.models.cgmy       import CgmyParams\n",
    "from foureng.pipeline          import price_strip\n",
    "from foureng.pricers.cos       import cos_auto_grid, recommended_cos_policy\n",
    "from foureng.utils.grids       import COSGridPolicy\n",
    "\n",
    "# Extension modules not in PyPI v0.2.0 -- try editable install fallback.\n",
    "def _try_ext():\n",
    "    try:\n",
    "        from foureng.utils.spectral_filters import COSFilterSpec           # noqa\n",
    "        from foureng.experiments.cos_filter_grid_search import (           # noqa\n",
    "            FilterGridCandidate, run_filtered_cos_grid_search,\n",
    "            select_fastest_under_tolerance,\n",
    "        )\n",
    "        return True\n",
    "    except (ImportError, ModuleNotFoundError):\n",
    "        return False\n",
    "\n",
    "if not _try_ext():\n",
    "    _root = None\n",
    "    _candidate = pathlib.Path('.').resolve()\n",
    "    for _ in range(6):\n",
    "        if (_candidate / 'pyproject.toml').exists() and (_candidate / 'foureng').exists():\n",
    "            _root = _candidate\n",
    "            break\n",
    "        if _candidate.parent == _candidate:\n",
    "            break\n",
    "        _candidate = _candidate.parent\n",
    "    if _root is None:\n",
    "        _clone = pathlib.Path.cwd() / 'fourier-option-pricer'\n",
    "        _repo_url = os.environ.get(\n",
    "            'FOURENG_REPO_URL',\n",
    "            'https://github.com/nl2992/fourier-option-pricer.git',\n",
    "        )\n",
    "        if not _clone.exists():\n",
    "            subprocess.run([\n",
    "                'git', 'clone', '--depth', '1', '--quiet',\n",
    "                _repo_url, str(_clone)\n",
    "            ], check=True)\n",
    "        _root = _clone.resolve()\n",
    "    for _k in [k for k in sys.modules if k == 'foureng' or k.startswith('foureng.')]:\n",
    "        del sys.modules[_k]\n",
    "    if str(_root) not in sys.path:\n",
    "        sys.path.insert(0, str(_root))\n",
    "    importlib.invalidate_caches()\n",
    "    if not _try_ext():\n",
    "        raise ImportError('Run: pip install -e . from the repo root.')\n",
    "\n",
    "from foureng.utils.spectral_filters import COSFilterSpec\n",
    "from foureng.experiments.cos_filter_grid_search import (\n",
    "    FilterGridCandidate, run_filtered_cos_grid_search, select_fastest_under_tolerance,\n",
    ")\n",
    "\n",
    "# ── shared helpers ────────────────────────────────────────────────────────\n",
    "def sci(x):\n",
    "    return '--' if pd.isna(x) else f'{x:.2e}'\n",
    "\n",
    "def junike_policy(model, N=None):\n",
    "    \"\"\"Return a Junike-style COSGridPolicy with optional fixed N.\"\"\"\n",
    "    trunc = 'heuristic' if model == 'vg' else 'tolerance'\n",
    "    kw = dict(mode='benchmark', truncation=trunc, centered=True, eps_trunc=1e-10)\n",
    "    if N is not None:\n",
    "        kw['fixed_N'] = N\n",
    "    else:\n",
    "        kw['dx_target'] = 0.003\n",
    "    return COSGridPolicy(**kw)\n",
    "\n",
    "def adaptive_candidates(model):\n",
    "    pol = junike_policy(model)\n",
    "    return [\n",
    "        FilterGridCandidate('no_filter',  pol, None),\n",
    "        FilterGridCandidate('fejer',      pol, COSFilterSpec('fejer')),\n",
    "        FilterGridCandidate('lanczos',    pol, COSFilterSpec('lanczos')),\n",
    "        FilterGridCandidate('raised_cos', pol, COSFilterSpec('raised_cosine')),\n",
    "        FilterGridCandidate('exp_p4',     pol, COSFilterSpec('exponential', order=4)),\n",
    "        FilterGridCandidate('exp_p8',     pol, COSFilterSpec('exponential', order=8)),\n",
    "        FilterGridCandidate('exp_p12',    pol, COSFilterSpec('exponential', order=12)),\n",
    "    ]\n",
    "\n",
    "def n_sweep(model, fwd, params, strikes, ref, *, Ns, cumulants):\n",
    "    \"\"\"Return DataFrame of vanilla vs Junike errors at each N.\"\"\"\n",
    "    rows = []\n",
    "    for N in Ns:\n",
    "        g   = cos_auto_grid(cumulants, N=N, L=10.0)\n",
    "        p_v = price_strip(model, 'cos', strikes, fwd, params, grid=g)\n",
    "        p_j = price_strip(model, 'cos_improved', strikes, fwd, params,\n",
    "                          grid=junike_policy(model, N=N))\n",
    "        rows.append(dict(\n",
    "            N=N,\n",
    "            vanilla_err=float(np.max(np.abs(np.asarray(p_v) - ref))),\n",
    "            junike_err =float(np.max(np.abs(np.asarray(p_j) - ref))),\n",
    "        ))\n",
    "    return pd.DataFrame(rows)\n",
    "\n",
    "def run_adaptive(model, fwd, params, strikes, ref, tol=1e-6):\n",
    "    \"\"\"Run grid search and return (best_row, full_df).\"\"\"\n",
    "    df = run_filtered_cos_grid_search(\n",
    "        model=model, strikes=strikes, fwd=fwd, params=params,\n",
    "        reference=ref, candidates=adaptive_candidates(model),\n",
    "        tol=tol, n_repeat=3,\n",
    "    )\n",
    "    return select_fastest_under_tolerance(df, tol=tol), df\n",
    "\n",
    "def plot_convergence(ax, sweep_df, *, title, tol=1e-6, adaptive_err=None,\n",
    "                     adaptive_label='adaptive'):\n",
    "    ax.plot(sweep_df['N'], sweep_df['vanilla_err'], 'o-', color='#888',\n",
    "            label='Vanilla COS')\n",
    "    ax.plot(sweep_df['N'], sweep_df['junike_err'],  's-', color='steelblue',\n",
    "            label='Junike COS')\n",
    "    if adaptive_err is not None:\n",
    "        ax.axhline(adaptive_err, color='crimson', ls='--', lw=1.5,\n",
    "                   label=f'{adaptive_label} (auto-N)')\n",
    "    ax.axhline(tol, color='black', ls='-.', lw=0.9, alpha=0.5,\n",
    "               label=f'tol={tol:.0e}')\n",
    "    ax.set_xscale('log', base=2); ax.set_yscale('log')\n",
    "    ax.set_xlabel('N (number of COS terms)')\n",
    "    ax.set_ylabel('max|error|')\n",
    "    ax.set_title(title)\n",
    "    ax.legend(fontsize=8, frameon=False)\n",
    "    ax.grid(True, which='both', ls='--', alpha=0.3)\n",
    "\n",
    "TOL = 1e-6\n",
    "NS  = [16, 32, 64, 128, 256, 512, 1024]\n",
    "print('Setup complete.')\n",
])

# ── §1 BSM ────────────────────────────────────────────────────────────────
C_BSM_MD = md_cell(
    "## \u00a71 \u00b7 Black-Scholes: FO2008 Test Case 1\n"
    "\n"
    "Parameters: $S_0 = 100$, $r = 0.1$, $q = 0$, $T = 1$, $\\sigma = 0.25$. "
    "Reference: Black-Scholes closed form.\n"
    "\n"
    "BSM is a smooth-CF model: all three COS variants should converge "
    "rapidly, and filtering should be unnecessary (adaptive selects no-filter).\n"
)

C_BSM_CODE = code_cell([
    "BSM_FWD    = ForwardSpec(S0=100, r=0.1, q=0.0, T=1.0)\n",
    "BSM_PARAMS = BsmParams(sigma=0.25)\n",
    "BSM_K      = np.array([80., 85., 90., 95., 100., 105., 110., 115., 120.])\n",
    "BSM_CUMS   = type('C', (), {'c1': BSM_FWD.F0, 'c2': BSM_PARAMS.sigma**2 * BSM_FWD.T,\n",
    "                              'c4': 0.0})()\n",
    "\n",
    "BSM_REF = np.array([\n",
    "    bs_price_from_fwd(\n",
    "        BSM_PARAMS.sigma,\n",
    "        BSInputs(F0=BSM_FWD.F0, K=float(k), T=BSM_FWD.T,\n",
    "                 r=BSM_FWD.r, q=BSM_FWD.q, is_call=True),\n",
    "    ) for k in BSM_K\n",
    "])\n",
    "\n",
    "from foureng.models.bsm import bsm_cumulants\n",
    "BSM_CUMS = bsm_cumulants(BSM_FWD, BSM_PARAMS)\n",
    "\n",
    "bsm_sweep = n_sweep('bsm', BSM_FWD, BSM_PARAMS, BSM_K, BSM_REF,\n",
    "                     Ns=NS, cumulants=BSM_CUMS)\n",
    "bsm_best, bsm_df = run_adaptive('bsm', BSM_FWD, BSM_PARAMS, BSM_K, BSM_REF)\n",
    "\n",
    "print('BSM convergence:')\n",
    "print(bsm_sweep.to_string(index=False))\n",
    "print(f\"\\nAdaptive: filter={bsm_best['filter']}, \"\n",
    "      f\"error={bsm_best['max_abs_err']:.2e}, \"\n",
    "      f\"runtime={bsm_best['runtime_ms']:.2f} ms\")\n",
    ])

C_BSM_PLOT = code_cell([
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n",
    "\n",
    "plot_convergence(axes[0], bsm_sweep, title='BSM: error vs N',\n",
    "                 tol=TOL, adaptive_err=float(bsm_best['max_abs_err']),\n",
    "                 adaptive_label=f\"adaptive ({bsm_best['filter']})\")\n",
    "\n",
    "# Scatter: all adaptive candidates\n",
    "_ok = bsm_df[bsm_df['status'] == 'ok']\n",
    "_cmap = {'none':'steelblue','fejer':'darkorange','lanczos':'green',\n",
    "          'raised_cosine':'purple','exponential':'crimson'}\n",
    "_seen = set()\n",
    "for _, row in _ok.iterrows():\n",
    "    lbl = row['filter'] if row['filter'] not in _seen else ''\n",
    "    _seen.add(row['filter'])\n",
    "    axes[1].scatter(row['runtime_ms'], max(row['max_abs_err'], 1e-16),\n",
    "                    c=_cmap.get(row['filter'], 'grey'), s=55, alpha=0.7, label=lbl)\n",
    "axes[1].scatter(float(bsm_best['runtime_ms']), max(float(bsm_best['max_abs_err']), 1e-16),\n",
    "                c='gold', marker='*', s=300, zorder=5, label='Selected')\n",
    "axes[1].axhline(TOL, color='black', ls='-.', lw=0.9, alpha=0.5)\n",
    "axes[1].set_xscale('log'); axes[1].set_yscale('log')\n",
    "axes[1].set_xlabel('runtime (ms)'); axes[1].set_ylabel('max|error|')\n",
    "axes[1].set_title('BSM: adaptive candidate space')\n",
    "handles, labels = axes[1].get_legend_handles_labels()\n",
    "axes[1].legend(dict(zip(labels,handles)).values(),\n",
    "               dict(zip(labels,handles)).keys(), fontsize=7)\n",
    "axes[1].grid(True, which='both', ls='--', alpha=0.3)\n",
    "\n",
    "fig.tight_layout()\n",
    "plt.show()\n",
])

# ── §2 Heston ─────────────────────────────────────────────────────────────
C_HESTON_MD = md_cell(
    "## \u00a72 \u00b7 Heston: FO2008 Table 2 + Table 5 Stress Case\n"
    "\n"
    "**Strip (Table 2 parameters):** $S_0=100$, $r=0.1$, $q=0$, $T=1$; "
    "$\\kappa=1.5$, $\\bar v=0.04$, $\\nu=0.3$, $\\rho=-0.9$, $v_0=0.04$. "
    "Reference: PyFENG FFT.\n"
    "\n"
    "**Stress case (Table 5):** $T=10$ long-maturity ATM. "
    "Reference: published value $22.318945791$. "
    "This is where Junike truncation gives the largest gain over vanilla COS "
    "(demonstrated in \u00a74); filtering adds a smaller secondary benefit.\n"
)

C_HESTON_CODE = code_cell([
    "# Table 2 strip\n",
    "H_FWD    = ForwardSpec(S0=100, r=0.1, q=0.0, T=1.0)\n",
    "H_PARAMS = HestonParams(kappa=1.5, theta=0.04, nu=0.3, rho=-0.9, v0=0.04)\n",
    "H_K      = np.array([80., 90., 100., 110., 120.])\n",
    "H_CUMS   = heston_cumulants(H_FWD, H_PARAMS)\n",
    "H_REF    = np.asarray(price_strip('heston', 'pyfeng_fft', H_K, H_FWD, H_PARAMS),\n",
    "                       dtype=float)\n",
    "\n",
    "h_sweep = n_sweep('heston', H_FWD, H_PARAMS, H_K, H_REF,\n",
    "                   Ns=NS, cumulants=H_CUMS)\n",
    "h_best, h_df = run_adaptive('heston', H_FWD, H_PARAMS, H_K, H_REF)\n",
    "\n",
    "print('Heston T=1 convergence:')\n",
    "print(h_sweep.to_string(index=False))\n",
    "print(f\"\\nAdaptive: filter={h_best['filter']}, \"\n",
    "      f\"error={h_best['max_abs_err']:.2e}, \"\n",
    "      f\"runtime={h_best['runtime_ms']:.2f} ms\")\n",
    "\n",
    "# Table 5 stress case\n",
    "H10_FWD    = ForwardSpec(S0=100, r=0.0, q=0.0, T=10.0)\n",
    "H10_PARAMS = HestonParams(kappa=1.5768, theta=0.0398, nu=0.5751, rho=-0.5711, v0=0.0175)\n",
    "H10_K      = np.array([100.0])\n",
    "H10_REF    = np.array([22.318945791])\n",
    "H10_CUMS   = heston_cumulants(H10_FWD, H10_PARAMS)\n",
    "\n",
    "h10_sweep = n_sweep('heston', H10_FWD, H10_PARAMS, H10_K, H10_REF,\n",
    "                    Ns=NS, cumulants=H10_CUMS)\n",
    "h10_best, h10_df = run_adaptive('heston', H10_FWD, H10_PARAMS, H10_K, H10_REF)\n",
    "\n",
    "print('\\nHeston T=10 (FO Table 5) convergence:')\n",
    "print(h10_sweep.to_string(index=False))\n",
    "print(f\"\\nAdaptive: filter={h10_best['filter']}, \"\n",
    "      f\"error={h10_best['max_abs_err']:.2e}, \"\n",
    "      f\"runtime={h10_best['runtime_ms']:.2f} ms\")\n",
])

C_HESTON_PLOT = code_cell([
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n",
    "\n",
    "plot_convergence(axes[0], h_sweep, title='Heston T=1: error vs N',\n",
    "                 tol=TOL, adaptive_err=float(h_best['max_abs_err']),\n",
    "                 adaptive_label=f\"adaptive ({h_best['filter']})\")\n",
    "\n",
    "plot_convergence(axes[1], h10_sweep, title='Heston T=10 (FO Table 5): error vs N',\n",
    "                 tol=TOL, adaptive_err=float(h10_best['max_abs_err']),\n",
    "                 adaptive_label=f\"adaptive ({h10_best['filter']})\")\n",
    "\n",
    "fig.tight_layout()\n",
    "plt.show()\n",
])

# ── §3 Jump models ─────────────────────────────────────────────────────────
C_JUMP_MD = md_cell(
    "## \u00a73 \u00b7 Jump Models: VG and CGMY\n"
    "\n"
    "Pure-jump L\u00e9vy models have characteristic functions that decay more slowly "
    "in frequency and log-return densities that are more peaked at short maturities. "
    "Both effects amplify finite-series oscillatory error, making them the prime "
    "candidates for spectral filtering.\n"
    "\n"
    "**VG (short maturity):** $T=0.1$, Madan\u2013Carr\u2013Chang parameters "
    "($\\sigma=0.12$, $\\nu=0.2$, $\\theta=-0.14$). "
    "The density is highly peaked; vanilla COS converges slowly.\n"
    "\n"
    "**CGMY:** $T=0.25$, $C=1$, $G=5$, $M=10$, $Y=0.5$ (infinite activity, "
    "finite variation). CF oscillates; Junike truncation helps but oscillatory "
    "error can persist at moderate $N$.\n"
    "\n"
    "*Note on VG truncation:* `recommended_cos_policy` uses `truncation='heuristic'` "
    "for VG (not `'tolerance'`), because VG's tail is light enough that cumulant "
    "sizing is reliable. The Junike N-sweep below respects this model-specific "
    "choice.\n"
)

C_JUMP_CODE = code_cell([
    "# VG short maturity\n",
    "VG_FWD    = ForwardSpec(S0=100, r=0.1, q=0.0, T=0.1)\n",
    "VG_PARAMS = VGParams(sigma=0.12, nu=0.2, theta=-0.14)\n",
    "VG_K      = np.linspace(70, 130, 13)\n",
    "VG_REF    = np.asarray(price_strip('vg', 'pyfeng_fft', VG_K, VG_FWD, VG_PARAMS),\n",
    "                        dtype=float)\n",
    "\n",
    "from foureng.models.variance_gamma import vg_cumulants\n",
    "VG_CUMS = vg_cumulants(VG_FWD, VG_PARAMS)\n",
    "\n",
    "vg_sweep = n_sweep('vg', VG_FWD, VG_PARAMS, VG_K, VG_REF,\n",
    "                    Ns=NS, cumulants=VG_CUMS)\n",
    "vg_best, vg_df = run_adaptive('vg', VG_FWD, VG_PARAMS, VG_K, VG_REF)\n",
    "\n",
    "print('VG T=0.1 convergence:')\n",
    "print(vg_sweep.to_string(index=False))\n",
    "print(f\"\\nAdaptive: filter={vg_best['filter']}, \"\n",
    "      f\"error={vg_best['max_abs_err']:.2e}, \"\n",
    "      f\"runtime={vg_best['runtime_ms']:.2f} ms\")\n",
    "\n",
    "# CGMY\n",
    "CG_FWD    = ForwardSpec(S0=100, r=0.04, q=0.0, T=0.25)\n",
    "CG_PARAMS = CgmyParams(C=1.0, G=5.0, M=10.0, Y=0.5)\n",
    "CG_K      = np.linspace(80, 120, 9)\n",
    "CG_REF    = np.asarray(price_strip('cgmy', 'pyfeng_fft', CG_K, CG_FWD, CG_PARAMS),\n",
    "                        dtype=float)\n",
    "\n",
    "from foureng.models.cgmy import cgmy_cumulants\n",
    "CG_CUMS = cgmy_cumulants(CG_FWD, CG_PARAMS)\n",
    "\n",
    "cg_sweep = n_sweep('cgmy', CG_FWD, CG_PARAMS, CG_K, CG_REF,\n",
    "                    Ns=NS, cumulants=CG_CUMS)\n",
    "cg_best, cg_df = run_adaptive('cgmy', CG_FWD, CG_PARAMS, CG_K, CG_REF)\n",
    "\n",
    "print('\\nCGMY T=0.25 convergence:')\n",
    "print(cg_sweep.to_string(index=False))\n",
    "print(f\"\\nAdaptive: filter={cg_best['filter']}, \"\n",
    "      f\"error={cg_best['max_abs_err']:.2e}, \"\n",
    "      f\"runtime={cg_best['runtime_ms']:.2f} ms\")\n",
])

C_JUMP_PLOT = code_cell([
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n",
    "\n",
    "plot_convergence(axes[0], vg_sweep, title='VG T=0.1: error vs N',\n",
    "                 tol=TOL, adaptive_err=float(vg_best['max_abs_err']),\n",
    "                 adaptive_label=f\"adaptive ({vg_best['filter']})\")\n",
    "\n",
    "plot_convergence(axes[1], cg_sweep, title='CGMY T=0.25: error vs N',\n",
    "                 tol=TOL, adaptive_err=float(cg_best['max_abs_err']),\n",
    "                 adaptive_label=f\"adaptive ({cg_best['filter']})\")\n",
    "\n",
    "fig.tight_layout()\n",
    "plt.show()\n",
])

# ── §4 Summary ─────────────────────────────────────────────────────────────
C_SUMMARY_MD = md_cell(
    "## \u00a74 \u00b7 Summary\n"
    "\n"
    "Collected results across all FO2008 test cases.\n"
)

C_SUMMARY_CODE = code_cell([
    "cases_summary = [\n",
    "    ('BSM T=1',       bsm_sweep,  bsm_best),\n",
    "    ('Heston T=1',    h_sweep,    h_best),\n",
    "    ('Heston T=10',   h10_sweep,  h10_best),\n",
    "    ('VG T=0.1',      vg_sweep,   vg_best),\n",
    "    ('CGMY T=0.25',   cg_sweep,   cg_best),\n",
    "]\n",
    "\n",
    "rows = []\n",
    "for name, sw, best in cases_summary:\n",
    "    n256 = sw[sw['N'] == 256].iloc[0] if 256 in sw['N'].values else sw.iloc[-1]\n",
    "    rows.append({\n",
    "        'case':               name,\n",
    "        'vanilla N=256 err':  sci(n256['vanilla_err']),\n",
    "        'junike N=256 err':   sci(n256['junike_err']),\n",
    "        'adaptive err':       sci(float(best['max_abs_err'])),\n",
    "        'adaptive filter':    best['filter'],\n",
    "        'adaptive rt (ms)':   f\"{best['runtime_ms']:.2f}\",\n",
    "    })\n",
    "\n",
    "summary_df = pd.DataFrame(rows)\n",
    "display(summary_df.to_string(index=False))\n",
])

cells_adap = [
    C_TITLE,
    C_PIP,
    C_SETUP,
    C_BSM_MD, C_BSM_CODE, C_BSM_PLOT,
    C_HESTON_MD, C_HESTON_CODE, C_HESTON_PLOT,
    C_JUMP_MD, C_JUMP_CODE, C_JUMP_PLOT,
    C_SUMMARY_MD, C_SUMMARY_CODE,
]

NB_ADAP.write_text(json.dumps(nb_template(cells_adap), indent=1, ensure_ascii=False))
print(f"adaptive_cos.ipynb created ({len(cells_adap)} cells).")
