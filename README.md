# fourier-option-pricer

Price European options under stochastic volatility and jump models using Fourier methods.

---

## What problem it solves

Most option pricing models beyond Black-Scholes — Heston, Bates, Variance Gamma, CGMY — do not have a closed-form price formula. Instead they give you the **characteristic function** of log-returns. This package turns that characteristic function into option prices.

**Monte Carlo** is the brute-force baseline: simulate paths and average discounted payoffs. The problem is that standard error decays as 1/√n — cutting the error by 10× costs 100× more paths. In a calibration loop that re-prices across dozens of strikes and maturities repeatedly, that compounds fast.

**Fourier methods** (Carr-Madan FFT, FRFT, COS) break that dependency. One characteristic function evaluation plus a deterministic transform prices an entire strike strip at once, giving machine-precision accuracy at sub-millisecond runtimes — orders of magnitude faster than Monte Carlo for strip pricing and calibration.

**The COS method** (Fang & Oosterlee 2008) is typically the fastest of these, but it has a hidden trap. It expands the log-return density on a finite interval [a, b]. If that interval is chosen too narrowly, tail mass is lost before the series even starts and no number of extra terms recovers it. The standard approach — scale the cumulant-based half-width by a fixed heuristic multiplier L — gets this wrong on heavy-tailed or short-maturity models, producing visible oscillation in the pricing surface:

![Adaptive filtered-COS: interval selection and spectral damping](docs/assets/adaptive_filtered_cos_schematic.png)

This project implements the **improved COS truncation** of Junike & Pankrashkin (2022) and Junike (2024), which replaces the heuristic with a rigorous tail-mass tolerance, then adds an **original adaptive filtered-COS extension**: spectral weights (Fejér, Lanczos, raised-cosine, exponential) damp the high-frequency COS modes to suppress residual oscillation from sharp density features. A deterministic policy-search selector compares `(grid policy, filter)` candidates and returns the fastest one that meets the user's tolerance — always keeping the unfiltered Junike path available as a fallback. On the Fang-Oosterlee (2008) test suite the adaptive selector matches or beats the paper's best reported error in 6 of 8 cases.

The package exposes **20 supported models** across stochastic-volatility, jump-diffusion, pure-Lévy, rough-volatility, and hybrid SVJ families, and **6 pricing engines** through one `price_strip` dispatcher.

Full methodology: [appendix.md](appendix.md) · Extension details: [docs/filtered_cos_extension.md](docs/filtered_cos_extension.md).

---

## Installation

```bash
pip install fourier-option-pricer
```

Requires Python 3.10+, `numpy>=1.26`, `scipy>=1.10`, `pyfeng>=0.4.0`.

To pin a version:

```bash
pip install "fourier-option-pricer==0.4.1"
```

---

## Quick start

```python
import numpy as np
import foureng as fe

fwd    = fe.ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
params = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)

strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
prices  = fe.price_strip("heston", "cos_improved", strikes, fwd, params)
print(prices)
```

Swap the method string to switch pricers without touching any other code:

```python
prices = fe.price_strip("heston", "carr_madan", strikes, fwd, params)
prices = fe.price_strip("heston", "lewis",      strikes, fwd, params)
prices = fe.price_strip("bates",  "frft",       strikes, fwd, params)
```

---

## API reference

Everything is importable from `import foureng as fe`.

### Market inputs

| Object | Parameters | Returns |
|--------|------------|---------|
| `ForwardSpec(S0, r, q, T)` | spot, risk-free rate, dividend yield, maturity | Market inputs container; exposes `F0` and discount factor `disc`. |

### Model parameter dataclasses

| Dataclass | Key parameters | Model family |
|-----------|----------------|-------------|
| `BsmParams` | `sigma` | Black-Scholes baseline |
| `HestonParams` | `kappa, theta, nu, rho, v0` | Stochastic volatility |
| `OusvParams` | `sigma0, kappa, theta, nu, rho` | Stochastic volatility (Schobel-Zhu) |
| `Sv32Params` | `v0, kappa, theta, nu, rho` | 3/2 stochastic volatility |
| `RoughHestonParams` | `sigma, vov, mr, rho, theta, alpha` | Rough volatility |
| `GarchWMW2012Params` | `v0, kappa, theta, nu, rho` | GARCH diffusion |
| `BatesParams` | `kappa, theta, nu, rho, v0, lam_j, mu_j, sigma_j` | Heston + log-normal jumps |
| `HestonKouParams` | `kappa, theta, nu, rho, v0, lam_j, p_j, eta1, eta2` | Heston + double-exp jumps |
| `HestonCGMYParams` | `kappa, theta, nu, rho, v0, C, G, M, Y` | Heston + CGMY jumps |
| `VGParams` | `sigma, nu, theta` | Variance Gamma |
| `CgmyParams` | `C, G, M, Y` | CGMY tempered-stable |
| `NigParams` | `sigma, nu, theta` | Normal Inverse Gaussian |
| `KouParams` | `sigma, lam, p, eta1, eta2` | Double-exponential jump-diffusion |
| `MertonJDParams` | `sigma, lam, mu_j, sigma_j` | Merton jump-diffusion |
| `MeixnerParams` | `a, b, delta` | Meixner process |
| `BilateralGammaParams` | `alpha_p, lambda_p, alpha_m, lambda_m` | Bilateral Gamma |
| `GHParams` | `lam, alpha, beta, delta` | Generalised Hyperbolic |
| `FMLSParams` | `alpha, sigma` | Finite Moment Log Stable |
| `DoubleHestonParams` | `kappa1..v01, kappa2..v02` | Two-factor Heston |
| `VGSAParams` | `C, G, M, kappa, eta, lam` | VG with stochastic activity |

Full model details: [docs/model_zoo.md](docs/model_zoo.md).

### Unified dispatcher

| Function | Parameters | Returns |
|----------|------------|---------|
| `price_strip(model, method, strikes, fwd, params, grid=None)` | model label, method label, strike array, `ForwardSpec`, model params, optional grid | `np.ndarray` of call prices |

Method labels: `"cos"`, `"cos_improved"`, `"carr_madan"`, `"frft"`, `"lewis"`, `"pyfeng_fft"`.

### Core pricing functions

| Function | Parameters | Returns |
|----------|------------|---------|
| `cos_prices(phi, fwd, strikes, grid)` | characteristic function, `ForwardSpec`, strike array, `COSGrid` | `COSResult` with `.call_prices` |
| `carr_madan_price_at_strikes(phi, fwd, grid, strikes)` | CF, `ForwardSpec`, `FFTGrid`, strike array | `np.ndarray` |
| `frft_price_at_strikes(phi, fwd, grid, strikes)` | CF, `ForwardSpec`, `FRFTGrid`, strike array | `np.ndarray` |
| `filtered_cos_prices(phi, fwd, strikes, grid, filter_spec=None)` | CF, `ForwardSpec`, strike array, `COSGrid`, optional `COSFilterSpec` | `COSResult` |

### Grid constructors

| Function / class | Parameters | Returns |
|-----------------|------------|---------|
| `cos_auto_grid(cumulants, N, L)` | cumulants, term count, truncation multiplier | `COSGrid` |
| `cos_improved_grid(cumulants, model, params)` | cumulants, model name, params | `COSGrid` via Junike truncation policy |
| `FFTGrid(N, eta, alpha)` | FFT size, frequency spacing, damping factor | Carr-Madan FFT grid |
| `FRFTGrid(N, eta, lam, alpha)` | size, freq spacing, strike step, damping | FRFT grid |

### Implied volatility and Greeks

| Function | Parameters | Returns |
|----------|------------|---------|
| `implied_vol_newton_safeguarded(price, inputs)` | option price, `BSInputs` | `float` |
| `implied_vol_brent(price, inputs)` | option price, `BSInputs` | `float` |
| `cos_price_and_greeks(phi, fwd, strikes, grid)` | CF, `ForwardSpec`, strikes, grid | `COSGreeks` with prices, delta, gamma |

Full reference for all 70+ public objects: [docs/api_reference.md](docs/api_reference.md).

---

## License

MIT. See [LICENSE](LICENSE).

---

## Demo notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nl2992/fourier-option-pricer/blob/main/notebooks/demo.ipynb)

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) is the primary entry point: Carr-Madan, COS, and FRFT on a Heston strip, runnable in Colab with no local setup.

### Supplementary notebook

[`notebooks/demo_advanced.ipynb`](notebooks/demo_advanced.ipynb) is a **supplementary reference** for readers who want a comprehensive tour after the main demo. It covers all 20 models, all 6 pricers, Greeks, IV surface, Heston calibration, Monte Carlo, new models (Double Heston, VGSA), and validation highlights. It is **not** the recommended starting point.

---

## Paper replications and research notebooks

### Paper replications

| Notebook | Paper / reference | What it shows |
|----------|-------------------|---------------|
| [`cosPaper_Replication.ipynb`](notebooks/cosPaper_Replication.ipynb) | Fang & Oosterlee (2008) | Table 2 BSM baseline, Heston scalar and strip cases, VG and CGMY; scoreboard and error figures. |
| [`fo2008_replication.ipynb`](notebooks/fo2008_replication.ipynb) | Fang & Oosterlee (2008) full replay | Paper-faithful Tables 2, 5, 7, 8–10 (BSM, Heston, VG, CGMY) plus benchmark CSVs. |
| [`paper_replications/bates_mathworks_replication.ipynb`](notebooks/paper_replications/bates_mathworks_replication.ipynb) | MathWorks optByBatesNI / FFT | All-engine scoreboard vs frozen MathWorks reference; error plots, assertion gate, CSV. |
| [`paper_replications/three_halves_replication.ipynb`](notebooks/paper_replications/three_halves_replication.ipynb) | Lewis (2000); Baldeaux & Badran (2012) | 3/2 SV: PyFENG regression and qualitative IV smile shape checks. |
| [`paper_replications/bates_sv32_validation_demo.ipynb`](notebooks/paper_replications/bates_sv32_validation_demo.ipynb) | MathWorks Bates + frozen pyfeng_fft surface | 12-section validation: BATES-01 to 07 and SV32-01 to 05; assertion gates and benchmark CSVs. |

### Research notebooks

| Notebook | What it covers |
|----------|---------------|
| [`research/cos_method_improved.ipynb`](notebooks/research/cos_method_improved.ipynb) | Junike-Pankrashkin (2022) / Junike (2024) improved truncation: three pricing strategies, Heston T=10 stress case, visual diagnostics. |
| [`research/adaptive_cos.ipynb`](notebooks/research/adaptive_cos.ipynb) | Adaptive filtered-COS: BSM, Heston, VG, CGMY; comparison with plain COS and filtered COS. |

[`notebooks/presentation_fourier_methods.ipynb`](notebooks/presentation_fourier_methods.ipynb) is a lecture-style walkthrough covering Monte Carlo vs Carr-Madan, Lewis FFT parameter sensitivity, plain COS, improved COS, multi-model sweep, and conclusions.

---

## Validation summary

| Model / method group | Reference | Tolerance | Status |
|----------------------|-----------|-----------|--------|
| Carr-Madan VG Case 4 put prices | Carr & Madan (1999) table | exact to 4 d.p. | done |
| Lewis Heston five-strike strip | Lewis (2001) table | atol=1e-5 | done |
| Double Heston vanilla calls | Kelly (2025) table | atol=1e-4 | done |
| Bates NI prices | MathWorks `optByBatesNI` | atol=1e-2 | done |
| Bates FFT/FRFT surface | MathWorks `optByBatesFFT` | atol=1e-2 | done |
| Bates Delta | MathWorks `optSensByBatesNI` | atol=5e-3 | done |
| BSM all-pricers baseline | Frozen derived reference | COS/COS+: 1e-8; Lewis: 1e-7; CM/FRFT: 1e-4 | done |
| 3/2 SV PyFENG surface (7×4) | Frozen PyFENG adapter reference | atol=1e-3 | done |
| Merton JD | Derived reference (Poisson-BSM mixture) | atol=1e-8 | done |
| FO2008 COS Tables 1–10 | Derived reference, paper-grid replay | see [fo2008_replication.md](docs/fo2008_replication.md) | partial |
| Heston, VG, CGMY, NIG, OUSV, Rough Heston | PyFENG adapter parity | atol=1e-5 | partial |
| Kou, Bilateral Gamma, GH, FMLS, Meixner, VGSA | Derived reference, cross-method | atol=1e-4 | partial |

Full per-paper matrix: [docs/paper_validation_matrix.md](docs/paper_validation_matrix.md). Evidence-level definitions: [docs/validation_hierarchy.md](docs/validation_hierarchy.md).

---

## Reproduce results

```bash
git clone https://github.com/nl2992/fourier-option-pricer.git
cd fourier-option-pricer
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

# fast CI suite
python -m pytest -q -m "not slow"

# full suite including Monte Carlo and notebook guards
pip install -e ".[test,notebook]"
python -m pytest -q

# paper-replication tests only
python -m pytest -q -m "paper"

# MathWorks Bates software-reference tests
python -m pytest -q -m "software_reference"
```

The repository has **741 pytest cases**.

For developer quality checks:

```bash
pip install -e ".[dev]"
ruff check foureng/ tests/
python -m mypy foureng
```

---

## Course rubric map

| Rubric criterion | Where to find it |
|-----------------|-----------------|
| Correct implementation and paper validation | [docs/paper_validation_matrix.md](docs/paper_validation_matrix.md); [docs/fo2008_replication.md](docs/fo2008_replication.md); [docs/bates_sv32_validation.md](docs/bates_sv32_validation.md) |
| Robustness testing | `tests/methods/`, `tests/models/`; robustness sweeps and parameter-edge tests in `test_numerical_quality.py`, `test_robustness_sweep.py` |
| Coding efficiency (vectorised NumPy) | `foureng/pricers/`, `foureng/mc/`; strip pricing via array ops, no Python path loops |
| Coding quality (class / package structure) | `foureng/` package; `ModelSpec` / `ForwardSpec` dataclasses; `pipeline.py` unified dispatcher |
| README quality | This file, structured per instructor template |
| Innovation / new idea | [What problem it solves](#what-problem-it-solves) above; [docs/filtered_cos_extension.md](docs/filtered_cos_extension.md); `notebooks/research/adaptive_cos.ipynb` |
| Mathematical background and derivations | [appendix.md](appendix.md) sections 7–8 (CF and pricing methods), 14 (Junike theory) |
| AI workflow / original contribution | [docs/ai_workflow_and_contribution.md](docs/ai_workflow_and_contribution.md) |

---

## Key papers

| Topic | Reference |
|-------|-----------|
| Carr-Madan FFT | Carr, P. and Madan, D.B. (1999), *Option Valuation Using the Fast Fourier Transform* |
| FRFT | Chourdakis, K. (2004), *Option Pricing Using the Fractional FFT* |
| COS method | Fang, F. and Oosterlee, C.W. (2008), *A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions* |
| Improved COS truncation | Junike, G. and Pankrashkin, K. (2022), *Precise Option Pricing by the COS Method: How to Choose the Truncation Range* |
| COS term-count policy | Junike, G. (2024), *On the Number of Terms in the COS Method for European Option Pricing* |
| Spectral filtering | Ruijter, M.J., Versteegh, M. and Oosterlee, C.W. (2015), *On the Application of Spectral Filters in a Fourier Option Pricing Technique* |
| Heston SV | Heston, S.L. (1993), *A Closed-Form Solution for Options with Stochastic Volatility* |
| Lewis benchmark | Lewis, A.L. (2001), *A Simple Option Formula for General Jump-Diffusion and Other Exponential Lévy Processes* |
| Kou jump-diffusion | Kou, S.G. (2002), *A Jump-Diffusion Model for Option Pricing* |
| Bates SVJ | Bates, D.S. (1996), *Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options* |

Full bibliography with DOIs and free-access links: [docs/papers.md](docs/papers.md).

---

## Documentation

All reference documentation is indexed at **[docs/README.md](docs/README.md)** — model zoo, full API, validation hierarchy, paper replication tables, filtered-COS extension, Bates/3/2 SV validation, AI workflow, and packaging checklist.

| Document | Contents |
|----------|----------|
| [appendix.md](appendix.md) | Methodology, derivations, model conventions, benchmark interpretation, and the full numbered course-project narrative (sections 1–18). |
| [docs/numerical_notes.md](docs/numerical_notes.md) | Known numerical limitations: COS truncation failure modes, Carr-Madan alpha conditions, PyFENG version caveats. |
| [docs/numerical_quality_checklist.md](docs/numerical_quality_checklist.md) | M1/M4 floating-point rubric audit: `expm1` fix, variance floor, analytic Greeks, RNG pattern, dtype guards. |
| [docs/papers.md](docs/papers.md) | Full bibliography with DOIs and free-access links, grouped by method and model family. |
