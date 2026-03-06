# fourier-option-pricer

Fourier pricing toolkit for European options: Carr-Madan FFT, FRFT, and COS under
characteristic-function models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nl2992/fourier-option-pricer/blob/main/notebooks/demo.ipynb)

---

## Problem solved

Pricing vanilla European options is straightforward when the model provides a closed-form
price formula — as Black-Scholes does. Most richer models (stochastic volatility,
jump-diffusion, pure-Lévy) do not. They provide a **characteristic function** of the
log-return, but not a direct formula for the call price.

The standard workaround is Monte Carlo simulation, but its error decays as O(n^{−1/2}):
reducing the error by a factor of 10 requires roughly 100× more paths. In a calibration
loop that calls the pricer thousands of times across strikes and maturities, the sampling
cost compounds badly.

Fourier methods exploit the characteristic function directly. If the characteristic function
of log-returns is available, the option price can be written as a deterministic integral —
no simulation required. This package solves that integral three different ways (Carr-Madan
FFT, FRFT, and COS), validates every method against published references and independent
benchmarks, and exposes the result as a clean, uniform Python API across twenty models.

---

## Project contribution

This project is not a re-derivation of the characteristic functions for standard models.
Its contributions are:

1. **A uniform pricing layer.** One `price_strip(model, method, strikes, fwd, params)`
   call prices any of twenty models by any of six methods without model-specific wiring.

2. **Twenty characteristic-function models.** Eight PyFENG-backed adapters and twelve
   in-house implementations (including four SVJ composites and six pure-Lévy models).
   Full model list: [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md).

3. **A structured validation harness.** 686 pytest cases classified into five evidence
   levels: published-paper tables, MathWorks software references, frozen derived references,
   cross-package parity with PyFENG, and qualitative shape checks. See
   [docs/VALIDATION_HIERARCHY.md](docs/VALIDATION_HIERARCHY.md).

4. **Improved COS truncation.** The Junike-Pankrashkin (2022) tolerance-based interval
   selection and Junike (2024) term-count policy, wired into the `cos_improved` path.
   Demonstrated to beat the naive paper-grid replay on 7 of 8 FO2008 test cases.

5. **Adaptive filtered-COS extension.** A deterministic policy-search layer that chooses
   among no-filter COS, Junike COS, and filtered Junike COS to satisfy a user tolerance
   target. Full details: [docs/FILTERED_COS_EXTENSION.md](docs/FILTERED_COS_EXTENSION.md).

---

## Course rubric map

| Rubric criterion | Where to find it |
|-----------------|-----------------|
| Mathematical background and derivations | [APPENDIX.md](APPENDIX.md) §7 (characteristic functions), §8 (pricing methods), §14 (Junike theory) |
| Implementation quality | `foureng/pricers/`, `foureng/models/`, `foureng/utils/`; `pipeline.py` dispatcher |
| Validation against published benchmarks | [docs/VALIDATION_HIERARCHY.md](docs/VALIDATION_HIERARCHY.md); [docs/paper_validation_matrix.md](docs/paper_validation_matrix.md); `tests/papers/`, `tests/models/` |
| Extension / innovation | [docs/FILTERED_COS_EXTENSION.md](docs/FILTERED_COS_EXTENSION.md); `notebooks/research/adaptive_cos.ipynb` |
| Notebook demonstrations | `notebooks/demo.ipynb`, `notebooks/demo_advanced.ipynb`, `notebooks/paper_replications/` |
| FO2008 paper replication | [docs/FO2008_REPLICATION.md](docs/FO2008_REPLICATION.md); `notebooks/fo2008_replication.ipynb` |
| Bates & 3/2 SV validation | [docs/BATES_SV32_VALIDATION.md](docs/BATES_SV32_VALIDATION.md); `notebooks/paper_replications/bates_sv32_validation_demo.ipynb` |
| Code quality / reproducibility | `pyproject.toml`, `tests/`, CI workflow; see [Reproduce results](#reproduce-results) |
| AI workflow / original contribution | [docs/AI_WORKFLOW_AND_CONTRIBUTION.md](docs/AI_WORKFLOW_AND_CONTRIBUTION.md) |

---

## AI-assisted development workflow

AI tools were used for research assistance, implementation planning, code generation, and documentation restructuring. The workflow was source-driven: Deep Research was used to identify papers, formulas, and benchmark numbers; reasoning models were used to convert those results into implementation TODOs; coding agents were used for first-pass implementation and restructuring; and final acceptance depended on human review, tests, notebooks, and CI.

Full workflow, library reuse policy, original contributions, and validation gates: [docs/AI_WORKFLOW_AND_CONTRIBUTION.md](docs/AI_WORKFLOW_AND_CONTRIBUTION.md).

---

## Installation

```bash
pip install fourier-option-pricer          # latest (v0.4.1)
pip install "fourier-option-pricer==0.4.1" # pin to this release
```

**Runtime dependencies:** `numpy>=1.24`, `scipy>=1.10`, `pyfeng>=0.4.0`.

---

## Quick start

```python
import numpy as np
import foureng as fe

fwd    = fe.ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
params = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)

phi    = lambda u: fe.heston_cf_form2(u, fwd, params)
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
grid   = fe.cos_auto_grid(fe.heston_cumulants(fwd, params), N=256, L=10.0)
result = fe.cos_prices(phi, fwd, strikes, grid)

print(result.call_prices)
```

Or use the unified dispatcher to switch methods without touching model code:

```python
prices = fe.price_strip("heston", "cos_improved", strikes, fwd, params)
prices = fe.price_strip("heston", "carr_madan",   strikes, fwd, params)
prices = fe.price_strip("heston", "lewis",        strikes, fwd, params)
```

---

## Core numerical methods

### Why Fourier methods instead of Monte Carlo?

Monte Carlo standard error is

$$
\varepsilon_{\mathrm{MC}} = O(n^{-1/2}),
$$

so reducing error by one order of magnitude needs roughly 100× more paths.
Fourier methods price a whole strike strip from a single characteristic function evaluation
and a deterministic transform, giving a more direct speed–accuracy trade-off for European
vanilla options.

### Common characteristic-function backbone

All three pricing families start from the same object:

$$
\varphi_T(u) = \mathbb{E}^{\mathbb{Q}}\!\left[e^{iuX_T}\right],
\qquad
X_T = \log\!\left(\frac{S_T}{F_0}\right).
$$

Here `i = √(−1)`, `u` is the Fourier frequency, and `X_T` is the terminal log-forward
return. This is the characteristic function **in log-forward coordinates**
(`X_T = log(S_T / F_0)`, not `log(S_T)`). Mixing log-spot and log-forward CFs introduces
a systematic pricing error.

### Carr-Madan FFT

Carr and Madan (1999) damp the call price as a function of log-strike with a factor
`exp(α k)` so that the resulting function is square-integrable. After FFT inversion on
a uniform frequency grid, prices at a uniform log-strike lattice are recovered by
interpolation.

Key parameters: damping factor `α`, frequency spacing `η`, FFT size `N`, log-strike
spacing `λ = 2π / (Nη)`. Strike and frequency resolution are coupled — finer strike
resolution requires either a larger grid or an alternative transform.

### Fractional FFT (FRFT)

FRFT (Chourdakis 2004) relaxes the strict coupling between frequency and strike spacings,
allowing both grids to be chosen more flexibly. It is useful when reporting strikes do not
align naturally with the standard Carr-Madan lattice, or when a finer strike grid is needed
without increasing `N`.

### COS method

The COS method (Fang & Oosterlee 2008) expands the log-return density on a truncated
interval `[a, b]` using a Fourier-cosine series. The density itself need not be evaluated;
the cosine coefficients are recovered directly from `φ_T(kπ / (b−a))`.

The standard cumulant-based truncation rule is

$$
[a, b] = \!\left[
c_1 - L\sqrt{c_2 + \sqrt{|c_4|}},\;
c_1 + L\sqrt{c_2 + \sqrt{|c_4|}}
\right],
$$

where `c_1`, `c_2`, `c_4` are the first, second, and fourth cumulants, and `L` is a
multiplier (typically 8–12). Accuracy depends on **both** the interval choice and the
number of terms `N` — choosing either one alone is insufficient.

### Improved COS (Junike-Pankrashkin / Junike)

A fixed cumulant multiplier is a rule of thumb. If `[a, b]` is too narrow, the method
discards tail mass before the series even starts; increasing `N` then cannot recover the
missing probability. If `[a, b]` is too wide, more terms are needed to resolve it.

Junike and Pankrashkin (2022) replace the multiplier with a tail-mass tolerance `ε_trunc`.
For a centre `m` and half-width `M`, Markov's inequality gives a sufficient condition:

$$
M \geq \left(\frac{\mathbb{E}[|X_T - m|^n]}{\varepsilon_{\text{trunc}}}\right)^{1/n}.
$$

Junike (2024) additionally specifies how many terms `N` are needed to resolve the chosen
interval to a target accuracy. The `cos_improved` path in this package implements both.

---

## Model coverage

The package supports **twenty** characteristic-function models across four families:
stochastic-volatility (Heston, OUSV, 3/2 SV, Rough Heston, GARCH), SVJ composites (Bates,
Heston-Kou, Heston-CGMY), pure-Lévy (VG, CGMY, NIG, Kou, Merton JD, Meixner, Bilateral
Gamma, GH, FMLS, VGSA), and multi-factor (Double Heston).

Full model table with parameter dataclasses, CF sources, and API notes:
[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md).

---

## Validation summary

| Model / method group | Reference type | Tolerance | Status |
|----------------------|---------------|-----------|--------|
| Carr-Madan VG (Case 4 put prices) | Published paper table — Carr & Madan (1999) | exact to 4 d.p. | ✓ done |
| Lewis Heston five-strike strip | Published paper table — Lewis (2001) | atol=1e-5 | ✓ done |
| Double Heston vanilla calls | Published paper table — Kelly (2025) | atol=1e-4 | ✓ done |
| Bates NI prices (MathWorks) | Software reference — `optByBatesNI` | atol=1e-2 | ✓ done |
| Bates FFT/FRFT surface (MathWorks) | Software reference — `optByBatesFFT` | atol=1e-2 | ✓ done |
| Bates Delta (MathWorks) | Software reference — `optSensByBatesNI` | atol=5e-3 | ✓ done |
| BSM all-pricers baseline | Frozen derived reference | COS/COS+: 1e-8; Lewis: 1e-7; CM/FRFT: 1e-4; PyFENG: 1e-5 | ✓ done |
| 3/2 SV PyFENG surface (7×4) | Frozen PyFENG adapter reference | atol=1e-3 | ✓ done |
| Merton JD | Derived reference (Poisson-BSM mixture) | atol=1e-8 | ✓ done |
| FO2008 COS Tables 1–10 | Derived reference (paper-grid replay) | Per-table; see [FO2008_REPLICATION.md](docs/FO2008_REPLICATION.md) | partial |
| Heston, VG, CGMY, NIG, OUSV, Rough Heston | PyFENG adapter parity | atol=1e-5 | partial |
| Kou, Bilateral Gamma, GH, FMLS, Meixner, VGSA | Derived reference (cross-method) | atol=1e-4 | partial |
| Filtered-COS / Junike stress tests | Numerical stability | convergence check | partial |

Full per-paper table: [docs/paper_validation_matrix.md](docs/paper_validation_matrix.md).  
Evidence-level definitions: [docs/VALIDATION_HIERARCHY.md](docs/VALIDATION_HIERARCHY.md).

---

## Innovation: adaptive filtered-COS

The main project extension goes beyond the Junike interval selection. Even with a correctly
chosen `[a, b]`, the finite COS series can carry Gibbs-like oscillations if the density has
sharp features or the characteristic function decays slowly.

The adaptive filtered-COS layer (implemented in `foureng/pricers/filtered_cos.py` and
`foureng/experiments/cos_filter_grid_search.py`) adds a second control axis:

```
price = disc · Σ_k  σ_k · A_k · V_k
```

where `σ_k` is a spectral weight near 1 for low-frequency terms and smaller toward the tail.
Four filter families are available: Fejér, Lanczos, raised-cosine, and exponential.

A deterministic policy-search selector builds a candidate set of `(COSGridPolicy,
COSFilterSpec)` pairs, prices with each, and returns the **fastest candidate that meets the
user's error tolerance** — with the no-filter Junike candidate always in the pool. The
selector therefore weakly dominates fixed Junike-COS in the joint (error, runtime) metric.

Key result: on the FO2008 test suite the adaptive selector beats the naive paper-grid replay
in 7/8 cases and beats the paper's best reported error in 6/8 cases.

Full documentation: [docs/FILTERED_COS_EXTENSION.md](docs/FILTERED_COS_EXTENSION.md).  
Demo notebook: [`notebooks/research/adaptive_cos.ipynb`](notebooks/research/adaptive_cos.ipynb).

---

## Notebooks

### Quick-start demos

| Notebook | What it covers |
|----------|---------------|
| [`notebooks/demo.ipynb`](notebooks/demo.ipynb) | Colab-ready quick-start: Carr-Madan, COS, and FRFT on a Heston strip. |
| [`notebooks/demo_advanced.ipynb`](notebooks/demo_advanced.ipynb) | Full-feature showcase (v0.4.1): all 20 models, 6 pricers, Greeks, IV surface, calibration, Monte Carlo, validation highlights. |

The advanced demo sections: all-20-model ATM table → multi-method scoreboard → Greeks → IV
smiles → volatility surface → Heston calibration → Monte Carlo → new models (Double Heston,
VGSA) → validation highlights.

### Paper replications

[`notebooks/paper_replications/`](notebooks/paper_replications/) contains five focused
validation notebooks:

| Notebook | Paper / reference | What it shows |
|----------|-------------------|---------------|
| [`cosPaper_Replication.ipynb`](notebooks/cosPaper_Replication.ipynb) | Fang & Oosterlee (2008) | Table 2 BSM baseline, Heston scalar and strip cases, VG and CGMY; extended scoreboard and error figures. |
| [`fo2008_replication.ipynb`](notebooks/fo2008_replication.ipynb) | Fang & Oosterlee (2008) full replay | Paper-faithful Tables 2, 5, 7, 8–10 (BSM, Heston, VG, CGMY) plus benchmark CSVs. |
| [`paper_replications/bates_mathworks_replication.ipynb`](notebooks/paper_replications/bates_mathworks_replication.ipynb) | MathWorks optByBatesNI / FFT | All-engine scoreboard vs frozen MathWorks reference; error plots, assertion gate, CSV. |
| [`paper_replications/three_halves_replication.ipynb`](notebooks/paper_replications/three_halves_replication.ipynb) | Lewis (2000); Baldeaux & Badran (2012) | 3/2 SV: PyFENG regression + qualitative IV smile and no-arbitrage shape checks. |
| [`paper_replications/bates_sv32_validation_demo.ipynb`](notebooks/paper_replications/bates_sv32_validation_demo.ipynb) | MathWorks Bates suite + frozen pyfeng_fft surface | 12-section validation: BATES-01–07 and SV32-01–05; assertion gates and benchmark CSVs. |

### Research notebooks

| Notebook | What it covers |
|----------|---------------|
| [`research/cos_method_improved.ipynb`](notebooks/research/cos_method_improved.ipynb) | Junike-Pankrashkin 2022 / Junike 2024 improved truncation: three pricing strategies, Heston T=10 stress case, visual diagnostics. |
| [`research/adaptive_cos.ipynb`](notebooks/research/adaptive_cos.ipynb) | Adaptive filtered-COS: BSM, Heston, VG, CGMY; summary comparison with plain COS and filtered COS. |

### Presentation notebook

[`notebooks/presentation_fourier_methods.ipynb`](notebooks/presentation_fourier_methods.ipynb) —
lecture-style walkthrough: validation-first workflow, Monte Carlo vs Carr-Madan, Lewis FFT
parameter sensitivity, plain COS, improved-truncation COS, multi-model sweep, conclusions.

---

## Reproduce results

```bash
# Install
pip install "fourier-option-pricer==0.4.1"

# Fast CI suite (excludes slow notebook tests):
pytest -q -m "not slow"

# Full suite including Monte Carlo and notebook guards:
pytest -q

# Paper-replication tests only:
pytest -q -m "paper"

# MathWorks Bates software-reference tests:
pytest -q -m "software_reference"

# FO2008 COS benchmarks:
pytest -q tests/papers/test_phase4_cos_heston_fo2008.py

# Bates + SV32 full validation notebook (requires pyfeng>=0.4.0):
jupyter nbconvert --to notebook --execute \
  notebooks/paper_replications/bates_sv32_validation_demo.ipynb
```

The repository currently collects **686 pytest cases**.

---

## Package API summary

The full API is exposed from `import foureng as fe`. Key entry points:

```python
fe.ForwardSpec(S0, r, q, T)       # market inputs
fe.HestonParams(...)               # (and 19 other parameter dataclasses)
fe.price_strip(model, method,      # unified dispatcher
               strikes, fwd, params)
fe.cos_prices(phi, fwd, strikes, grid)
fe.carr_madan_price_at_strikes(phi, fwd, grid, strikes)
fe.implied_vol_newton_safeguarded(price, inputs)
fe.calibrate_heston(...)
```

Full API tables (all 70+ public objects): [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

---

## Testing and validation layout

| Folder | Contents |
|--------|----------|
| `tests/refs/` | Frozen JSON reference files: MathWorks Bates, PyFENG 3/2 surface, Baldeaux-Badran figure parameters. |
| `tests/papers/` | Published-paper and software-reference replications: Carr-Madan, Lewis, FRFT, FO2008 COS, Kou, all six Bates pricer tests, 3/2 SV qualitative smoke test. |
| `tests/models/` | Model adapter, regression-strip, and reduction-limit tests for all 20 models; paper-backed 3-layer suites for each in-house model. |
| `tests/methods/` | Pricing-method behavior: COS policies, filters, alpha validity, cross-method agreement, robustness sweeps. |
| `tests/features/` | End-to-end features: Monte Carlo, control variates, implied vol, calibration, Greeks, public API, integration workflows. |

See [tests/README.md](tests/README.md) for the full folder map.

---

## Repository map

```text
foureng/
  models/       — 20 CF models (PyFENG-backed adapters + in-house implementations)
  pricers/      — carr_madan / frft / cos / cos_improved / filtered_cos / lewis
  utils/        — grids, cumulants, implied volatility, spectral filters, numerics
  iv/           — implied volatility routines
  mc/           — Monte Carlo baselines (BSM and Heston conditional MC)
  pipeline.py   — unified price_strip dispatcher

tests/
  refs/         — frozen JSON reference fixtures
  papers/       — paper and software-reference replication tests
  models/       — per-model validation suites
  methods/      — pricer-method behavior tests
  features/     — end-to-end feature tests

notebooks/
  demo.ipynb, demo_advanced.ipynb
  presentation_fourier_methods.ipynb
  paper_replications/
  research/
  fo2008_replication.ipynb, cosPaper_Replication.ipynb

benchmarks/
  paper_replications/fo2008_cos/
    params.py, outputs/  (CSVs, PNGs, SUMMARY.md)
  mc_vs_fourier_methods/outputs/

docs/           — detailed documentation (this tree)
.github/workflows/  — CI and test matrix
```

---

## Key papers

| Topic | Reference |
|-------|-----------|
| Carr-Madan FFT | Carr, P. & Madan, D.B. (1999), *Option Valuation Using the Fast Fourier Transform*. |
| FRFT | Chourdakis, K. (2004), *Option Pricing Using the Fractional FFT*. |
| COS method | Fang, F. & Oosterlee, C.W. (2008), *A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions*. |
| Improved COS truncation | Junike, G. & Pankrashkin, K. (2022), *Precise Option Pricing by the COS Method: How to Choose the Truncation Range*. |
| COS term-count policy | Junike, G. (2024), *On the Number of Terms in the COS Method for European Option Pricing*. |
| Spectral filtering | Ruijter, M.J., Versteegh, M. & Oosterlee, C.W. (2015), *On the Application of Spectral Filters in a Fourier Option Pricing Technique*. |
| Heston SV | Heston, S.L. (1993), *A Closed-Form Solution for Options with Stochastic Volatility*. |
| Stable Heston CF | Albrecher, H. et al. (2007), *The Little Heston Trap*. |
| Lewis benchmark | Lewis, A.L. (2001), *A Simple Option Formula for General Jump-Diffusion and Other Exponential Lévy Processes*. |
| Variance Gamma | Madan, D.B., Carr, P. & Chang, E.C. (1998), *The Variance Gamma Process and Option Pricing*. |
| Kou jump-diffusion | Kou, S.G. (2002), *A Jump-Diffusion Model for Option Pricing*. |
| Bates SVJ | Bates, D.S. (1996), *Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options*. |

Full bibliography with DOIs and free-access links: [PAPERS.md](PAPERS.md).

---

## Detailed documentation

- [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) — all 20 models
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) — full API tables
- [docs/VALIDATION_HIERARCHY.md](docs/VALIDATION_HIERARCHY.md) — evidence levels
- [docs/BATES_SV32_VALIDATION.md](docs/BATES_SV32_VALIDATION.md) — Bates & 3/2 SV validation
- [docs/FO2008_REPLICATION.md](docs/FO2008_REPLICATION.md) — FO2008 tables
- [docs/FILTERED_COS_EXTENSION.md](docs/FILTERED_COS_EXTENSION.md) — adaptive filtered-COS
- [docs/paper_validation_matrix.md](docs/paper_validation_matrix.md) — per-paper matrix
- [docs/PACKAGING.md](docs/PACKAGING.md) — PyPI release checklist
- [docs/README.md](docs/README.md) — documentation index
- [APPENDIX.md](APPENDIX.md) — methodology, derivations, references
- [PAPERS.md](PAPERS.md) — full bibliography

---

## License

MIT. See [LICENSE](LICENSE).
