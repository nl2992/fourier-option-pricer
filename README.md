# fourier-option-pricer

Price European options under stochastic volatility and jump models using Fourier methods.

---

## What problem it solves

Most option pricing models beyond Black-Scholes (Heston, Bates, Variance Gamma, CGMY...) do not have a closed-form price formula. What they do have is an analytic **characteristic function** of log-returns, $\varphi_T(u) = \mathbb{E}[e^{iuX_T}]$. This package turns that into option prices.

**Monte Carlo** is the obvious baseline: simulate paths and average the discounted payoffs. The catch is that standard error scales as $\sigma/\sqrt{n}$, meaning 10x better accuracy requires 100x more paths. In a calibration loop repricing across a full surface, this gets expensive fast.

**Fourier methods** (Carr-Madan FFT, FRFT, COS) sidestep this entirely. A single characteristic function evaluation plus a deterministic transform prices a whole strike strip at once, reaching near-machine-precision accuracy at sub-millisecond runtimes.

**The COS method** (Fang & Oosterlee 2008) is typically the fastest of the three, but it has a subtle trap. The method approximates the log-return density as a Fourier-cosine series on a finite window $[a, b]$. The standard truncation rule places that window as

$$[a, b] = \left[c_1 - L\sqrt{c_2 + \sqrt{|c_4|}},\; c_1 + L\sqrt{c_2 + \sqrt{|c_4|}}\right]$$

where $c_1, c_2, c_4$ are the model's cumulants and $L$ is a heuristic multiplier. If the window is too narrow, tail mass is lost before the series even starts and no number of additional terms can fix that. The heuristic $L$ works fine on well-behaved models but fails on heavy-tailed or short-maturity cases, producing visible oscillation in the pricing surface:

![Adaptive filtered-COS: interval selection and spectral damping](docs/assets/adaptive_filtered_cos_schematic.png)

This project implements the **improved COS truncation** of Junike & Pankrashkin (2022) and Junike (2024), which replaces the heuristic $L$ with a rigorous tail-mass bound. On the FO2008 test suite, this truncation improvement beats the paper-grid COS in 7 of 8 cases and beats the paper's own best-N result in 6 of 8 (see [`benchmarks/cos_method_improved/`](benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv)). On top of that, we add an **original adaptive filtered-COS extension**: spectral weights $\sigma_k \in [0, 1]$ (Fejér, Lanczos, raised-cosine, or exponential) applied to the high-frequency COS coefficients to suppress residual oscillation from sharp density features. A policy-search selector automatically compares grid and filter combinations, returning the fastest configuration that meets the user's tolerance, with the plain Junike path always included as a fallback.

The package covers **20 characteristic-function models** across stochastic-volatility, jump-diffusion, pure-Lévy, rough-volatility, and hybrid SVJ families, plus a SABR approximation surface. They are priced through one `price_strip` dispatcher with COS/FFT/FRFT/CONV, first-slice Mellin and PROJ façades, BSM finite-difference/lattice baselines, and product-level exotic routes.

Full methodology: [appendix.md](appendix.md) · Extension details: [docs/filtered_cos_extension.md](docs/filtered_cos_extension.md) · Package architecture: [docs/architecture_overview.md](docs/architecture_overview.md).

---

## Installation

Choose the path that fits your goal:

| Goal | Recommended path |
|------|-----------------|
| Run notebooks, reproduce results, contribute | **Option A** — fork + venv or conda |
| Quick experiment without keeping a copy | **Option B** — direct clone + venv or conda |
| Browser-only, no local install | **Option C** — Google Colab |
| Import `foureng` in your own project | **Option D** — PyPI |

---

### Option A — fork and run everything (recommended)

Forking gives you your own copy on GitHub so you can save changes and push them back.

**1. Fork on GitHub**

Click **Fork** at the top-right of [github.com/nl2992/fourier-option-pricer](https://github.com/nl2992/fourier-option-pricer), then clone your fork:

```bash
git clone https://github.com/YOUR-USERNAME/fourier-option-pricer.git
cd fourier-option-pricer
```

**2a. Set up the environment — pip + venv**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # installs foureng + notebook + test deps
```

**2b. Set up the environment — conda**

```bash
conda env create -f environment.yml
conda activate foureng
```

**3. Run the notebooks**

```bash
jupyter lab   # navigate to notebooks/demo.ipynb to start
```

**4. Run the tests**

```bash
python -m pytest -q -m "not slow"   # fast suite (~30 s)
python -m pytest -q                  # full suite including notebook guards
```

**5. Keep your fork in sync** (optional)

```bash
git remote add upstream https://github.com/nl2992/fourier-option-pricer.git
git fetch upstream
git rebase upstream/main
```

---

### Option B — clone without forking (pip + venv or conda)

Use this if you just want to run locally and do not need your own GitHub copy.

```bash
git clone https://github.com/nl2992/fourier-option-pricer.git
cd fourier-option-pricer
```

Then follow step 2a (venv) or 2b (conda) from Option A above.

---

### Option C — Google Colab (no local setup)

Click the **Open in Colab** badge in the [Demo notebook](#demo-notebook) section. The first cell installs all dependencies automatically — no local Python needed.

> **Note:** Colab's Python 3.12 runtime ships with numpy 2.0.0, which has a known import bug. Cell 1 automatically upgrades numpy to a compatible version and clears the module cache — no manual restart needed.

---

### Option D — library only (PyPI)

Use this if you want to `import foureng` in your own code without cloning the repo.

```bash
pip install fourier-option-pricer          # latest
pip install "fourier-option-pricer==0.4.2" # pin to this release
```

Requires Python 3.10+.

---

### Dependencies at a glance

| Group | Packages |
|-------|----------|
| Runtime | `numpy>=1.26`, `scipy>=1.10`, `matplotlib>=3.7`, `statsmodels>=0.14`, `pyfeng>=0.4.0` |
| Notebooks | `pandas>=2.0`, `jupyter>=1.0`, `ipykernel>=6.0`, `nbformat>=5.10` |
| Tests | `pytest>=7.4`, `pytest-cov>=4.0`, `hypothesis>=6.112`, `nbmake>=1.5` |
| Dev tools | `ruff`, `mypy>=1.10`, `pyperf>=2.7` — install via `pip install -e ".[dev]"` |

`requirements.txt` covers runtime + notebook + test deps in one file. `environment.yml` is the conda equivalent.

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
prices = fe.price_strip("heston", "cos_improved", strikes, fwd, params)
prices = fe.price_strip("heston", "cos_filtered", strikes, fwd, params)
prices = fe.price_strip("bates",  "cos_improved", strikes, fwd, params)
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
| `SabrParams` | `alpha, beta, rho, nu` | SABR implied-vol approximation |

Full model details: [docs/model_zoo.md](docs/model_zoo.md).

### Unified dispatcher

| Function | Parameters | Returns |
|----------|------------|---------|
| `price_strip(model, method, strikes, fwd, params, grid=None)` | model label, method label, strike array, `ForwardSpec`, model params, optional grid | `np.ndarray` of call prices |

Method labels: `"cos"`, `"cos_improved"`, `"cos_filtered"`, `"carr_madan"`, `"frft"`, `"conv"`, `"cos_bermudan"`, `"mellin"`, `"proj"`, `"pyfeng_fft"`, plus product-aware `"cos_digital"` / `"digital_bsm"` / `"barrier_bsm"` / `"asian_bsm"` / `"asian_mc"` / `"double_barrier_mc"` / `"forward_start_bsm"` / `"lookback_bsm"` / `"lookback_mc"` / `"variance_analytic_bsm"` / `"variance_mc"` / `"cliquet_mc"` and SABR-only `"sabr_hagan"`.

Product-level pricing uses `price(product, model, method, fwd, params)`. It currently routes European options, cash-or-nothing and asset-or-nothing digitals via `"cos_digital"` or BSM `"digital_bsm"`, supported 1-D Levy Bermudans via `"cos_bermudan"`, BSM American options via `"lattice"` / `"pde_fd"`, continuously monitored zero-rebate BSM single barriers via `"barrier_bsm"`, BSM Asians via `"asian_bsm"` / `"asian_mc"`, BSM forward-start options via `"forward_start_bsm"`, BSM lookbacks via `"lookback_bsm"` / `"lookback_mc"`, BSM variance swaps via `"variance_analytic_bsm"` / `"variance_mc"`, integrated-variance BSM options via `"variance_analytic_bsm"` and realised/integrated BSM variance options via `"variance_mc"`, BSM cliquets via `"cliquet_mc"`, and BSM double barriers via `"double_barrier_mc"`.

### Core pricing functions

| Function | Parameters | Returns |
|----------|------------|---------|
| `cos_prices(phi, fwd, strikes, grid)` | characteristic function, `ForwardSpec`, strike array, `COSGrid` | `COSResult` with `.call_prices` |
| `carr_madan_price_at_strikes(phi, fwd, grid, strikes)` | CF, `ForwardSpec`, `FFTGrid`, strike array | `np.ndarray` |
| `frft_price_at_strikes(phi, fwd, grid, strikes)` | CF, `ForwardSpec`, `FRFTGrid`, strike array | `np.ndarray` |
| `conv_price_at_strikes(phi, fwd, grid, strikes)` | CF, `ForwardSpec`, `CONVGrid`, strike array | `np.ndarray` |
| `filtered_cos_prices(phi, fwd, strikes, grid, filter_spec=None)` | CF, `ForwardSpec`, strike array, `COSGrid`, optional `COSFilterSpec` | `COSResult` |
| `bsm_lattice_price_at_strikes(fwd, params, strikes, grid=None)` | BSM inputs, `LatticeGrid`, strike array | `np.ndarray` |
| `bsm_pde_fd_price_at_strikes(fwd, params, strikes, grid=None)` | BSM inputs, `PDEGrid`, strike array | `np.ndarray` |
| `bsm_barrier_price(S, K, H, r, q, T, sigma, barrier_type, cp=1)` | BSM single-barrier inputs | `float` |
| `bsm_discrete_geometric_asian(S, K, r, q, monitoring_times, sigma, cp=1)` | BSM geometric-Asian inputs | `float` |
| `sabr_hagan_price_at_strikes(fwd, params, strikes)` | `ForwardSpec`, `SabrParams`, strike array | `np.ndarray` |

### Grid constructors

| Function / class | Parameters | Returns |
|-----------------|------------|---------|
| `cos_auto_grid(cumulants, N, L)` | cumulants, term count, truncation multiplier | `COSGrid` |
| `cos_improved_grid(cumulants, model, params)` | cumulants, model name, params | `COSGrid` via Junike truncation policy |
| `FFTGrid(N, eta, alpha)` | FFT size, frequency spacing, damping factor | Carr-Madan FFT grid |
| `FRFTGrid(N, eta, lam, alpha)` | size, freq spacing, strike step, damping | FRFT grid |
| `CONVGrid(N, u_max)` | positive-frequency node count and cutoff | CONV-style Fourier inversion grid |
| `LatticeGrid(steps)` | tree step count | BSM CRR lattice grid |
| `PDEGrid(spot_steps, time_steps, s_max_mult)` | finite-difference grid controls | BSM implicit finite-difference grid |

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

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) is the primary entry point. It runs Carr-Madan, COS, and FRFT on a Heston strip and works in Colab with no local setup.

### Supplementary notebook

[`notebooks/supplementary/demo_advanced.ipynb`](notebooks/supplementary/demo_advanced.ipynb) is a **supplementary reference** for readers who want a comprehensive tour after the main demo. It covers all 20 models, the main Fourier pricers, Greeks, IV surface, Heston calibration, Monte Carlo, new models (Double Heston, VGSA), and validation highlights. It is **not** the recommended starting point.

---

## Paper replications and research notebooks

### Paper replications

| Notebook | Paper / reference | What it shows |
|----------|-------------------|---------------|
| [`fo2008_replication.ipynb`](notebooks/fo2008_replication.ipynb) | Fang & Oosterlee (2008) | Paper-faithful Tables 2, 5, 7, 8–10 (BSM, Heston, VG, CGMY); scoreboard, error figures, benchmark CSVs. |
| [`paper_replications/bates_mathworks_replication.ipynb`](notebooks/paper_replications/bates_mathworks_replication.ipynb) | MathWorks optByBatesNI / FFT | All-engine scoreboard vs frozen MathWorks reference; error plots, assertion gate, CSV. |
| [`paper_replications/three_halves_replication.ipynb`](notebooks/paper_replications/three_halves_replication.ipynb) | Lewis (2000); Baldeaux & Badran (2012) | 3/2 SV: PyFENG regression and qualitative IV smile shape checks. |
| [`paper_replications/bates_sv32_validation_demo.ipynb`](notebooks/paper_replications/bates_sv32_validation_demo.ipynb) | MathWorks Bates + frozen pyfeng_fft surface | 12-section validation: BATES-01 to 07 and SV32-01 to 05; assertion gates and benchmark CSVs. |

### Research notebooks

| Notebook | What it covers |
|----------|---------------|
| [`research/cos_method_improved.ipynb`](notebooks/research/cos_method_improved.ipynb) | Junike-Pankrashkin (2022) / Junike (2024) improved truncation: three pricing strategies, Heston T=10 stress case, visual diagnostics. |
| [`research/adaptive_cos.ipynb`](notebooks/research/adaptive_cos.ipynb) | Adaptive filtered-COS: BSM, Heston, VG, CGMY; comparison with plain COS and filtered COS. |

---

## Validation summary

| Model / method group | Reference | Tolerance | Status |
|----------------------|-----------|-----------|--------|
| Carr-Madan VG Case 4 put prices | Carr & Madan (1999) table | atol=1e-3 | done |
| Lewis Heston five-strike strip | Lewis (2001) table | atol=1e-4 | done |
| Double Heston vanilla calls | Kelly (2025) table | atol=5e-4 | done |
| Bates NI prices | MathWorks `optByBatesNI` | atol=1e-2 | done |
| Bates FFT/FRFT surface | MathWorks `optByBatesFFT` | atol=1e-2 | done |
| Bates Delta | MathWorks `optSensByBatesNI` | atol=5e-3 | done |
| BSM all-pricers baseline | Frozen derived reference | COS/COS+: 1e-8; CM/FRFT: 1e-4; CONV/lattice/PDE targeted tests | done |
| 3/2 SV PyFENG surface (7×4) | Frozen PyFENG adapter reference | atol=1.5e-3 | done |
| Merton JD | Derived reference (Poisson-BSM mixture) | atol=1e-4 | done |
| FO2008 COS Tables 1–10 | Derived reference, paper-grid replay | see [fo2008_replication.md](docs/fo2008_replication.md) | partial |
| Heston, CGMY | PyFENG adapter parity | atol=1e-5 | partial |
| NIG, OUSV | PyFENG adapter parity | atol=1e-4 | partial |
| VG, Rough Heston | PyFENG adapter parity | atol=1e-3 | partial |
| Kou, Bilateral Gamma, GH, FMLS, Meixner | Derived reference, cross-method | atol=1e-4 | partial |
| VGSA | Derived reference, cross-method | atol=1e-3 | partial |

Full per-paper matrix: [docs/paper_validation_matrix.md](docs/paper_validation_matrix.md). Evidence-level definitions: [docs/validation_hierarchy.md](docs/validation_hierarchy.md).

---

## Reproduce results

After cloning and installing (see [Installation](#installation) above):

```bash
# fast CI suite — skips slow Monte Carlo and notebook-execution tests
python -m pytest -q -m "not slow"

# full suite including notebook execution guards
python -m pytest -q

# paper-replication tests only
python -m pytest -q -m "paper"

# MathWorks Bates software-reference tests
python -m pytest -q -m "software_reference"
```

The repository has **741+ pytest cases**.

For linting and type checks:

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
| Coding quality (class / package structure) | `foureng/` package; `ModelSpec` / `ForwardSpec` dataclasses; `pipeline.py` unified dispatcher; [docs/architecture_overview.md](docs/architecture_overview.md) package map |
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

All reference documentation is indexed at **[docs/README.md](docs/README.md)**: model zoo, full API, validation hierarchy, paper replication tables, filtered-COS extension, Bates/3/2 SV validation, AI workflow, and packaging checklist.

| Document | Contents |
|----------|----------|
| [appendix.md](appendix.md) | Methodology, derivations, model conventions, benchmark interpretation, and the full numbered course-project narrative (sections 1–18). |
| [docs/numerical_notes.md](docs/numerical_notes.md) | Known numerical limitations: COS truncation failure modes, Carr-Madan alpha conditions, PyFENG version caveats. |
| [docs/numerical_quality_checklist.md](docs/numerical_quality_checklist.md) | M1/M4 floating-point rubric audit: `expm1` fix, variance floor, analytic Greeks, RNG pattern, dtype guards. |
| [docs/papers.md](docs/papers.md) | Full bibliography with DOIs and free-access links, grouped by method and model family. |
