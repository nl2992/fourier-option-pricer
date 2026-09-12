<div align="center">

# ⚡ fourier-option-pricer

**One characteristic function in → a whole strike strip of near-machine-precision prices out.**

*27 models · 12 Fourier engines · 23 products · calibration · 2,200+ tests*

[![CI](https://github.com/nl2992/fourier-option-pricer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nl2992/fourier-option-pricer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fourier-option-pricer.svg)](https://pypi.org/project/fourier-option-pricer/)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](pyproject.toml)
[![Typed](https://img.shields.io/badge/types-mypy-blue.svg)](pyproject.toml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nl2992/fourier-option-pricer/blob/main/notebooks/demo.ipynb)

</div>

```python
import numpy as np, foureng as fe

fwd    = fe.ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
params = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)
prices = fe.price_strip("heston", "cos_improved", np.array([80, 90, 100, 110, 120]), fwd, params)
```

Swap `"heston"` for any of 27 models, `"cos_improved"` for any of 12 engines. Same call, no rewiring.

---

## Engines at a glance

| Engine | `method=` | Idea | Convergence |
|--------|-----------|------|-------------|
| COS | `cos` / `cos_improved` / `cos_filtered` | Fourier-cosine density expansion (Fang & Oosterlee 2008) + Junike truncation + adaptive spectral filtering | exponential |
| Carr-Madan FFT | `carr_madan` | Damped-call FFT over log-strike (Carr & Madan 1999) | algebraic |
| Fractional FFT | `frft` | FFT with decoupled strike/frequency spacing (Chourdakis 2004) | algebraic |
| Hilbert transform | `hilbert` | Gil-Pelaez probabilities on the half-integer sinc grid (Feng & Linetsky 2008) | exponential |
| SINC | `sinc` | The same odd-frequency sum on a density-sized window (Baschetti et al. 2022); `sinc_smile` prices a whole smile with one FFT | exponential |
| SWIFT | `swift` | Shannon-wavelet projection of density and payoff (Ortiz-Gracia & Oosterlee 2016); scale `m` sets the accuracy | exponential in `m` |
| Optimal contour | `contour` | Lord-Kahl optimal contour + double-exponential quadrature: a high-precision reference with full *relative* accuracy deep out of the money | double-exponential |
| CONV | `conv` | Probability-transform Fourier inversion | algebraic |
| Lewis | internal | Parseval contour integral (Lewis 2001), used as adaptive fallback | spectral |
| Mellin | `mellin` | Mellin-transform façade for selected Lévy models | n/a |
| PROJ | `proj` | B-spline frame projection (Kirkby 2015/2017), European + Bermudan + single/double barrier + Asian CV | polynomial (order-tunable) |
| PyFENG FFT | `pyfeng_fft` | Third-party reference engine for 8 models | n/a |

Plus Fourier exotics engines (exact Fang-Oosterlee COS Bermudans and Richardson-extrapolated Americans, Hilbert-transform discrete barriers and lookbacks, ASCOS arithmetic Asians, PROJ recursions), non-Fourier baselines (CRR lattice, implicit PDE, CTMC generator methods, Monte Carlo with control variates and LSMC), a vectorised machine-precision implied-vol solver, and product-level pricing for 23 payoff dataclasses (barriers, Asians, cliquets, faders, step and swing options, variance products, and more).

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

The package covers **27 characteristic-function models** across stochastic-volatility, jump-diffusion, pure-Lévy, rough-volatility (including a Markovian lifted Heston), regime-switching, stochastic-rate-hybrid, time-changed Lévy, hybrid SVJ and generic affine families, plus a SABR approximation surface. They are priced through one `price_strip` dispatcher with COS/FFT/FRFT/CONV, the **Feng-Linetsky Hilbert-transform engine**, SINC, SWIFT, a high-precision optimal-contour engine, a first-slice Mellin façade, a real **PROJ frame-projection engine** (Kirkby 2015/2017: Europeans, Bermudans, single and double barriers, step and swing options, and an Asian control variate), BSM finite-difference/lattice baselines, and product-level exotic routes. For 1-D Lévy models the exotic routes include Americans, discretely monitored barriers and lookbacks, and arithmetic Asians, all priced from the same characteristic function.

Full methodology: [appendix.md](appendix.md) · Extension details: [docs/filtered_cos_extension.md](docs/filtered_cos_extension.md) · Package architecture: [docs/architecture_overview.md](docs/architecture_overview.md).

---

## 🆕 What's new in 0.22

New models, engines and exotic routes. Each one is tested against something independent of it: a closed form, a model it should reduce to, a high-precision calculation, or Monte Carlo. The full list is in the [CHANGELOG](CHANGELOG.md).

| Addition | How to use it | Checked against |
|----------|---------------|-----------------|
| Implied vol by "Let's Be Rational" (Jaeckel 2015) | `implied_vol_lets_be_rational(price, F, K, T, disc=..., cp=...)`, `black_price(...)` | 50-digit reference prices; max error 4e-15 over 200k quotes |
| American options for Levy models (Fang & Oosterlee 2009) | `price(AmericanOption(...), model, "cos_american", ...)` | Extrapolated binomial trees (BSM) and the PROJ engine (Kou, CGMY) |
| 4/2 stochastic volatility (Grasselli 2017) | `Sv42Params` with any CF engine | Heston (b = 0) and 3/2 (a = 0) limits, Monte Carlo |
| Optimal-contour engine (Lord & Kahl 2007) | `price_strip(model, "contour", ...)` | BSM closed form to 5e-14 relative, far out of the money too |
| Model-free variance and VIX-style index (Carr & Madan 1998) | `log_contract_variance_from_strip`, `vix_style_index` | Known model variances (BSM, Heston) |
| Discrete barriers and lookbacks (Feng & Linetsky 2008, 2009) | `"hilbert_barrier"`, `"hilbert_lookback"` | Fine-grid reference to 5e-7, Monte Carlo |
| Arithmetic Asians under Levy models (Zhang & Oosterlee 2013) | `price(AsianOption(...), model, "asian_cos", ...)` | Exact two-date case, Monte Carlo |
| SINC and SWIFT engines (Baschetti et al. 2022; Ortiz-Gracia & Oosterlee 2016) | `"sinc"`, `"swift"`, `sinc_smile(...)` | Contour engine to 1e-12 |
| Time-changed Levy models (Carr, Geman, Madan & Yor 2003) | `TimeChangedLevyParams(base, params, clock)` | `vgsa`, Monte Carlo |
| Lifted Heston (Abi Jaber 2019) | `LiftedHestonParams` | Heston in the one-factor case, Monte Carlo |
| BNS Gamma-OU stochastic volatility (Barndorff-Nielsen & Shephard 2001) | `BNSParams` | BSM in the no-jump case, Monte Carlo |
| Generic affine jump-diffusions (Duffie, Pan & Singleton 2000) | `AffineParams` | Heston, Bates, double Heston, Merton |

Examples for most of these are in the [Quick start](#quick-start). Also in 0.22: `bsm_lookback_floating` (and `price(..., "lookback_bsm")`) was about 10% off and is fixed, `cos_bermudan_price` is now exact and much faster, and `matplotlib` moved to the optional `[viz]` extra.

---

## Earlier releases (0.11 to 0.21)

Thirteen capabilities ported into the Fourier stack from the transform-methods literature (the territory covered by Kirkby's PROJ MATLAB toolbox), each implemented natively against `foureng`'s CF interfaces and validated against closed forms and Monte Carlo.

<details>
<summary>Show the 0.11 to 0.21 table</summary>

| Capability | Use it via | The one-line math |
|-----------|-----------|-------------------|
| **Hilbert-transform pricer** (Feng & Linetsky 2008) | `price_strip(model, "hilbert", ...)` | $\Pi = \tfrac12 + \tfrac{h}{\pi}\sum_m \mathrm{Re}\big[e^{-iu_mk}\varphi(u_m)/(iu_m)\big]$ on $u_m=(m{+}\tfrac12)h$, error decays like $e^{-c/h}$ |
| **Regime-switching jump-diffusion** (Buffington & Elliott 2002) | `RegimeSwitchingBsmParams` + any CF engine | $\varphi(u) = \pi_0^\top e^{T(Q + \mathrm{diag}\,\psi_j(u))}\mathbf{1}$, per-regime Merton blocks in $\psi_j$ |
| **Exact Lévy geometric Asians** (Fusai & Meucci 2008) | `price(product, model, "asian_cf", ...)` | $\varphi_A(u) = \prod_j \varphi_{\Delta t_j}\!\big(u\,w_j\big)$; the average's CF is a finite product, no lognormal proxy |
| **Lévy variance-swap strikes** (Carr & Wu 2009 discrete analogue) | `price(swap, model, "variance_levy_analytic", ...)` | $E[R_i^2] = \big((r{-}q)\Delta t_i + c_1\big)^2 + c_2$ per period, exact from CF cumulants |
| **Exact Lévy forward-starts** (Rubinstein 1990 homogeneity) | `price(product, model, "forward_start_cf", ...)` | $V = S_0 e^{-q t_1} \cdot \mathrm{Euro}(S_0{=}1, K{=}\alpha, \tau)$ |
| **Exact Lévy cliquets** (local collars) | `price(product, model, "cliquet_cf", ...)` | $E[\mathrm{clip}(R,\ell,c)] = \ell + \mathrm{Call}(1{+}\ell) - \mathrm{Call}(1{+}c)$ per period |
| **PROJ double barriers** (Kirkby 2015) | `price(product, model, "proj_double_barrier", ...)` | Toeplitz-FFT backward induction, absorption on both sides of $(L, U)$ |
| **Hull-White stochastic-rate hybrid** (Merton 1973; Hull & White 1990) | `HullWhiteHybridParams(base, ...)` + any CF engine | $\varphi(u) = \varphi_{\text{base}}(u)\, e^{-\frac12 V_P (u^2 + iu)}$, $V_P = \int_0^T \sigma_P^2\,ds$ |
| **Fader options** (Hakala & Wystup 2002) | `price(product, model, "fader_cf", ...)` | $V = \tfrac{D}{M}\sum_k \int_A f_{t_k}(x)\, C_k(x)\,dx$, one COS strip per date |
| **Step options** (Linetsky 1999) | `price(product, model, "proj_step", ...)` | PROJ recursion with soft killing $e^{-\rho\,\Delta t}$ beyond the barrier; $\rho{=}0$ vanilla, $\rho{\to}\infty$ knock-out |
| **Structural CDS** (Black & Cox 1976) | `levy_cds_spread(model, fwd, params, ...)` | First-passage survival via the PROJ unit-payoff recursion + O'Kane legs |
| **Swing options** (Carmona & Touzi 2008) | `price(product, model, "proj_swing", ...)` | DP over (date, rights): $V_m(x,j) = \max(C_j, g + C_{j-1})$, one convolution per level |
| **CTMC approximation** (Mijatović & Pistorius 2013) | `price_strip("bsm", "ctmc", ...)`, American via `price` | Generator $Q$ from the finite-volume stencil; $V = e^{T(Q - rI)}g$, local vol supported |

</details>

Also in this line: `cp=-1` is now honored uniformly across every Fourier engine (parity applied once at dispatch), a long-standing drift omission in `merton_jd_cumulants` is fixed, and the API reference was backfilled to cover every public symbol. Details in the [CHANGELOG](CHANGELOG.md).

---

## Installation

Choose the path that fits your goal:

| Goal | Recommended path |
|------|-----------------|
| Run notebooks, reproduce results, contribute | **Option A** (fork + venv or conda) |
| Quick experiment without keeping a copy | **Option B** (direct clone + venv or conda) |
| Browser-only, no local install | **Option C** (Google Colab) |
| Import `foureng` in your own project | **Option D** (PyPI) |

---

### Option A: fork and run everything (recommended)

Forking gives you your own copy on GitHub so you can save changes and push them back.

**1. Fork on GitHub**

Click **Fork** at the top-right of [github.com/nl2992/fourier-option-pricer](https://github.com/nl2992/fourier-option-pricer), then clone your fork:

```bash
git clone https://github.com/YOUR-USERNAME/fourier-option-pricer.git
cd fourier-option-pricer
```

**2a. Set up the environment (pip + venv)**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # installs foureng + notebook + test deps
```

**2b. Set up the environment (conda)**

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
python -m pytest -q -m "not slow"   # fast suite (about a minute)
python -m pytest -q                  # full suite including notebook guards
```

**5. Keep your fork in sync** (optional)

```bash
git remote add upstream https://github.com/nl2992/fourier-option-pricer.git
git fetch upstream
git rebase upstream/main
```

---

### Option B: clone without forking (pip + venv or conda)

Use this if you just want to run locally and do not need your own GitHub copy.

```bash
git clone https://github.com/nl2992/fourier-option-pricer.git
cd fourier-option-pricer
```

Then follow step 2a (venv) or 2b (conda) from Option A above.

---

### Option C: Google Colab (no local setup)

Click the **Open in Colab** badge in the [Demo notebook](#demo-notebook) section. The first cell installs all dependencies automatically. No local Python needed.

> **Note:** Colab's Python 3.12 runtime ships with numpy 2.0.0, which has a known import bug. Cell 1 automatically upgrades numpy to a compatible version and clears the module cache. No manual restart needed.

---

### Option D: library only (PyPI)

Use this if you want to `import foureng` in your own code without cloning the repo.

```bash
pip install fourier-option-pricer          # core: numpy, scipy, pyfeng (+ statsmodels)
pip install "fourier-option-pricer[viz]"   # + matplotlib/pandas for foureng.viz
pip install "fourier-option-pricer==0.22.1"  # pin a release
```

Requires Python 3.10 to 3.14. The package ships inline type hints (`py.typed`). Check what you have with `python -c "import foureng; print(foureng.__version__)"`.

If you are upgrading from 0.5.x (the previous PyPI release), note that `matplotlib` is now optional (install the `[viz]` extra for `foureng.viz`) and that `cos_bermudan_price` now uses the exact Fang-Oosterlee scheme, so its prices move slightly. The [CHANGELOG](CHANGELOG.md) has the rest.

---

### Dependencies at a glance

| Group | Packages |
|-------|----------|
| Runtime | `numpy>=1.26`, `scipy>=1.10`, `pyfeng>=0.4.0`, `statsmodels>=0.14` (an undeclared import of pyfeng) |
| `[viz]` | `matplotlib>=3.7`, `pandas>=2.0` (needed only for `foureng.viz`) |
| `[notebook]` | `matplotlib>=3.7`, `pandas>=2.0`, `jupyter>=1.0`, `ipykernel>=6.0`, `nbformat>=5.10` |
| `[test]` | `pytest>=7.4`, `pytest-cov>=4.0`, `hypothesis>=6.112`, `pandas>=2.0`, `mpmath>=1.3` (high-precision test oracle) |
| `[dev]` | all of the above plus `nbmake`, `build`, `twine`, `ruff`, `mypy>=1.10`, `pyperf>=2.7` (`pip install -e ".[dev]"`) |

`requirements.txt` covers runtime + notebook + test deps in one file. `environment.yml` is the conda equivalent.

---

## Quick start

### Price a strike strip

```python
import numpy as np
import foureng as fe

fwd    = fe.ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
params = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)

strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
calls   = fe.price_strip("heston", "cos_improved", strikes, fwd, params)
puts    = fe.price_strip("heston", "cos_improved", strikes, fwd, params, cp=-1)
print(calls)
```

### Switch engine or model

Only the strings and the parameter object change:

```python
for method in ["cos_filtered", "sinc", "swift", "hilbert", "contour"]:
    prices = fe.price_strip("heston", method, strikes, fwd, params)

bates = fe.BatesParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04,
                       lam_j=0.5, mu_j=-0.1, sigma_j=0.15)
prices = fe.price_strip("bates", "cos_improved", strikes, fwd, bates)
```

On this Heston strip the engines agree to about 1e-9. Some rules of thumb for picking one:

| Situation | Engine |
|-----------|--------|
| Default for European strips | `cos_improved` |
| Heavy tails, very short maturities, or visible ripples in COS prices | `cos_filtered` |
| A whole smile at once, or very few CF evaluations | `sinc` / `sinc_smile` |
| One knob for accuracy (the wavelet scale `m`) | `swift` |
| Reference prices, or deep out-of-the-money options where relative error matters | `contour` |
| Models whose CF is solved by ODEs (`lifted_heston`, `affine`) | `cos_improved` or `sinc` (the contour engine calls the CF too often) |
| Dense uniform strike grids | `carr_madan`, or `frft` with an explicit `FRFTGrid` |

### Implied vols and far wings

```python
vols = fe.implied_vol_lets_be_rational(calls, fwd.F0, strikes, fwd.T, disc=fwd.disc)

wings = fe.price_strip("heston", "contour", np.array([200.0, 300.0]), fwd, params)
```

`implied_vol_lets_be_rational` is vectorised and accurate to machine precision, including far from the money. The contour engine keeps full relative accuracy on tiny wing prices, where a fixed-grid method only guarantees absolute accuracy.

### Exotics under Lévy models

Products are dataclasses passed to `price`. For the discretely monitored routes, `grid` is the number of monitoring dates.

```python
from foureng.products.american import AmericanOption
from foureng.products.asian import AsianOption
from foureng.products.barrier import BarrierOption
from foureng.products.lookback import LookbackOption

fwd_l = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
kou   = fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)

american = fe.price(AmericanOption(strike=100.0, maturity=1.0, cp=-1),
                    "kou", "cos_american", fwd_l, kou)
barrier  = fe.price(BarrierOption(strike=100.0, barrier=85.0, maturity=1.0,
                                  barrier_type="down_out", monitoring="discrete"),
                    "kou", "hilbert_barrier", fwd_l, kou, grid=52)
lookback = fe.price(LookbackOption(maturity=1.0, cp=-1, strike_type="floating",
                                   monitoring="discrete"),
                    "kou", "hilbert_lookback", fwd_l, kou, grid=52)
asian    = fe.price(AsianOption(strike=100.0, maturity=1.0,
                                monitoring_times=np.linspace(1 / 12, 1.0, 12)),
                    "kou", "asian_cos", fwd_l, kou)
```

The same calls work for the other 1-D Lévy models (`bsm`, `vg`, `nig`, `cgmy`, `merton_jd`, `meixner` and so on).

### Newer models

```python
bns = fe.BNSParams(v0=0.04, lam=1.5, a=1.2, b=30.0, rho=-2.0)
prices = fe.price_strip("bns", "cos_improved", strikes, fwd, bns)

vg_on_cir = fe.TimeChangedLevyParams("vg", fe.VGParams(sigma=0.2, nu=0.3, theta=-0.1),
                                     fe.CirClock(y0=1.0, kappa=2.0, eta=1.0, lam=1.0))
prices = fe.price_strip("time_changed_levy", "cos_improved", strikes, fwd, vg_on_cir)

lifted = fe.LiftedHestonParams(v0=0.04, kappa=0.3, theta=0.04, nu=0.3, rho=-0.7,
                               H=0.1, n=20, r_n=2.5)
prices = fe.price_strip("lifted_heston", "sinc", strikes, fwd, lifted)
```

Every model and its parameters are listed in [docs/model_zoo.md](docs/model_zoo.md).

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
| `MertonJDParams` | `sigma, lam, muj, sigj` | Merton jump-diffusion |
| `MeixnerParams` | `a, b, delta` | Meixner process |
| `BilateralGammaParams` | `alpha_p, lambda_p, alpha_m, lambda_m` | Bilateral Gamma |
| `GHParams` | `lam, alpha, beta, delta` | Generalised Hyperbolic |
| `FMLSParams` | `alpha, sigma` | Finite Moment Log Stable |
| `DoubleHestonParams` | `kappa1..v01, kappa2..v02` | Two-factor Heston |
| `VGSAParams` | `C, G, M, kappa, eta, lam` | VG with stochastic activity |
| `RegimeSwitchingBsmParams` | `sigmas, generator, initial_probs` + optional `jump_intensities, jump_means, jump_stds` | Markov regime-switching jump-diffusion (matrix-exponential CF) |
| `HullWhiteHybridParams` | `base_model, base_params, mean_reversion, sigma_r` | Any base model + independent Hull-White stochastic rates |
| `Sv42Params` | `v0, kappa, theta, nu, rho, a, b` | 4/2 stochastic volatility (Heston + 3/2 terms) |
| `BNSParams` | `v0, lam, a, b, rho` | Barndorff-Nielsen-Shephard Γ-OU stochastic volatility |
| `LiftedHestonParams` | `v0, kappa, theta, nu, rho, H, n, r_n` | Lifted (Markovian multi-factor) rough Heston |
| `TimeChangedLevyParams` | `base_model, base_params, clock` (`CirClock` / `GammaOUClock`) | Lévy base on a stochastic business clock |
| `AffineParams` | `x0, K0, K1, H0, H1, l0, l1, jump_transform` | Generic affine jump-diffusion |
| `SabrParams` | `alpha, beta, rho, nu` | SABR implied-vol approximation |

Full model details: [docs/model_zoo.md](docs/model_zoo.md).

### Unified dispatcher

| Function | Parameters | Returns |
|----------|------------|---------|
| `price_strip(model, method, strikes, fwd, params, *, grid=None, cp=1)` | model label, method label, strike array, `ForwardSpec`, model params, optional grid, `cp=1` call / `cp=-1` put | `np.ndarray` of prices |
| `price(product, model, method, fwd, params, *, grid=None)` | product dataclass, model label, method label, `ForwardSpec`, model params, optional grid | `float` or `np.ndarray` |

Method labels: `"cos"`, `"cos_improved"`, `"cos_filtered"`, `"carr_madan"`, `"frft"`, `"conv"`, `"hilbert"`, `"sinc"`, `"swift"`, `"contour"`, `"cos_bermudan"`, `"mellin"`, `"proj"`, `"pyfeng_fft"`, plus product-aware `"asian_cf"` / `"variance_levy_analytic"` / `"forward_start_cf"` / `"cliquet_cf"` / `"fader_cf"` (exact Lévy geometric Asians, variance-swap strikes, forward-starts, and locally collared cliquets) and `"cos_digital"` / `"digital_bsm"` / `"monte_carlo"` / `"barrier_bsm"` / `"asian_bsm"` / `"asian_mc"` / `"double_barrier_mc"` / `"proj_double_barrier"` / `"forward_start_bsm"` / `"exchange_bsm"` / `"spread_bsm"` / `"multi_asset_mc"` / `"lookback_bsm"` / `"lookback_mc"` / `"variance_analytic_bsm"` / `"variance_mc"` / `"cliquet_mc"` and SABR-only `"sabr_hagan"`.

Product-level pricing uses `price(product, model, method, fwd, params)`. It currently routes European options, cash-or-nothing and asset-or-nothing digitals via `"cos_digital"` or BSM `"digital_bsm"`, supported 1-D Levy Bermudans via `"cos_bermudan"`, BSM generic Monte Carlo / Longstaff-Schwartz via `"monte_carlo"` for Europeans, Americans, Bermudans, and the GBM-simulated exotic book, continuously monitored zero-rebate BSM single barriers via `"barrier_bsm"`, BSM Asians via `"asian_bsm"` / `"asian_mc"`, BSM forward-start options via `"forward_start_bsm"`, BSM two-asset exchange options via `"exchange_bsm"` / `"multi_asset_mc"`, BSM basket and best-of options via `"multi_asset_mc"`, BSM spread options via `"spread_bsm"` / `"multi_asset_mc"`, BSM lookbacks via `"lookback_bsm"` / `"lookback_mc"`, BSM variance swaps via `"variance_analytic_bsm"` / `"variance_mc"`, integrated-variance BSM options via `"variance_analytic_bsm"` and realised/integrated BSM variance options via `"variance_mc"`, BSM cliquets via `"cliquet_mc"`, BSM double barriers via `"double_barrier_mc"` / `"proj_double_barrier"`, and, for 1-D Lévy models, Americans via `"cos_american"`, discretely monitored barriers and floating- or fixed-strike lookbacks via `"hilbert_barrier"` / `"hilbert_lookback"` (pass `grid=<int>` for the number of monitoring dates), and arithmetic Asians via `"asian_cos"`.

### Path-dependent MC engines (`foureng.mc`)

| Function | Product | Notes |
|----------|---------|-------|
| `asian_mc(S0, K, r, q, sigma, T, N_mon, cp, spec)` | Arithmetic Asian | Geometric-average control variate |
| `barrier_mc(S0, K, H, r, q, sigma, T, n_steps, barrier_type, rebate, cp, spec)` | Single barrier | BGK (1999) continuity correction |
| `lookback_mc(S0, r, q, sigma, T, n_steps, cp, spec, K=None)` | Floating/fixed lookback | K=None → floating-strike |
| `variance_swap_mc(S0, r, q, sigma, T, n_steps, spec)` | Variance swap | Returns fair rate E[RV] |
| `variance_option_mc(S0, r, q, sigma, T, K_var, n_steps, cp, spec)` | Variance option | Payoff max(cp*(RV-K_var), 0) |

All MC functions take a `GBMPathSpec(n_paths, n_steps, seed, antithetic)` configuration object.

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
| `hilbert_price_at_strikes(phi, fwd, strikes, cp=1, grid=None)` | CF, `ForwardSpec`, strike array, optional `HilbertGrid` | `np.ndarray` |
| `levy_geometric_asian_price(model, fwd, params, strikes=..., monitoring_times=..., cp=1)` | Lévy model key, market inputs, fixings | `np.ndarray` |
| `levy_variance_fair_strike(model, fwd, params, sampling_times)` | Lévy model key, market inputs, observation dates | `float` (annualized E[RV]) |
| `levy_forward_start_price(model, fwd, params, alpha=..., start_time=..., maturity=..., cp=1)` | Lévy model key, market inputs, strike ratio, reset date | `float` |
| `levy_cliquet_price(model, fwd, params, product)` | Lévy model key, market inputs, `CliquetOption` | `float` (locally collared additive/multiplicative) |
| `proj_double_barrier_price(step_cf, S0=..., L=..., U=..., M=..., knockout=True, ...)` | one-step CF, corridor, monitoring count | `float` |
| `levy_fader_price(model, fwd, params, product)` | Lévy model key, market inputs, `FaderOption` | `float` (fade-in or fade-out) |
| `proj_step_price(step_cf, S0=..., B=..., rho=..., M=..., step_type="down", ...)` | one-step CF, barrier, damping rate | `float` |
| `levy_cds_spread(model, fwd, params, default_barrier=..., recovery=..., maturity=...)` | Lévy model key, market inputs, credit terms | `float` (par spread) |
| `proj_swing_price(step_cf, S0=..., K=..., M=..., n_rights=..., cp=1, ...)` | one-step CF, dates, rights | `float` |
| `ctmc_european_price(S0, K, r, q, T, sigma, cp=1, grid=None)` | market inputs, constant or callable vol | `float` |
| `ctmc_american_price(S0, K, r, q, T, sigma, cp=1, n_steps=100, grid=None)` | market inputs, constant or callable vol | `float` |
| `sabr_hagan_price_at_strikes(fwd, params, strikes)` | `ForwardSpec`, `SabrParams`, strike array | `np.ndarray` |
| `contour_price_at_strikes(phi, fwd, strikes, cp=1, grid=None)` | CF, `ForwardSpec`, strikes, optional `ContourGrid` | `np.ndarray` (full relative precision) |
| `sinc_price_at_strikes(phi, fwd, strikes, cumulants, cp=1, grid=None)` / `sinc_smile(phi, fwd, cumulants)` | CF, `ForwardSpec`, strikes or cumulants | `np.ndarray` / `(strikes, prices)` |
| `swift_price_at_strikes(phi, fwd, strikes, cumulants, cp=1, m=None)` | CF, `ForwardSpec`, strikes, cumulants | `np.ndarray` |
| `cos_american_price(model, fwd, params, product, base_dates=64)` | Lévy model key, market inputs, `AmericanOption` | `float` |
| `hilbert_barrier_price(model, fwd, params, strike=..., barrier=..., maturity=..., barrier_type=..., n_monitor=252)` | Lévy model key, contract terms | `float` |
| `hilbert_lookback_price(model, fwd, params, maturity=..., cp=-1, strike_type="floating", strike=None, n_monitor=252)` | Lévy model key, contract terms | `float` |
| `levy_arithmetic_asian_price(model, fwd, params, strike=..., monitoring_times=..., cp=1)` | Lévy model key, fixings | `float` |
| `log_contract_variance_from_strip(strikes, fwd, calls=..., puts=...)` / `vix_style_index(...)` | option quotes | model-free variance / VIX-style index |

### Grid constructors

| Function / class | Parameters | Returns |
|-----------------|------------|---------|
| `cos_auto_grid(cumulants, N, L)` | cumulants, term count, truncation multiplier | `COSGrid` |
| `cos_improved_grid(cumulants, model, params)` | cumulants, model name, params | `COSGrid` via Junike truncation policy |
| `FFTGrid(N, eta, alpha)` | FFT size, frequency spacing, damping factor | Carr-Madan FFT grid |
| `FRFTGrid(N, eta, lam, alpha)` | size, freq spacing, strike step, damping | FRFT grid |
| `CONVGrid(N, u_max)` | positive-frequency node count and cutoff | CONV-style Fourier inversion grid |
| `HilbertGrid(h, N)` | frequency step and half-integer node count | Feng-Linetsky discrete Hilbert transform grid |
| `SincGrid(X_c, N, L)` | window half-width, odd-frequency count, cumulant multiplier | SINC window (auto by default) |
| `ContourGrid(rel_tol, c, c_bound, max_levels)` | tolerance, optional fixed contour height | optimal-contour pricer controls |
| `LatticeGrid(steps)` | tree step count | BSM CRR lattice grid |
| `PDEGrid(spot_steps, time_steps, s_max_mult)` | finite-difference grid controls | BSM implicit finite-difference grid |

### Implied volatility and Greeks

| Function | Parameters | Returns |
|----------|------------|---------|
| `implied_vol_lets_be_rational(price, F, K, T, disc=1.0, cp=1)` | arrays (broadcast) | `np.ndarray`; the recommended solver (machine precision, vectorised) |
| `black_price(F, K, T, sigma, disc=1.0, cp=1)` | arrays (broadcast) | `np.ndarray` (full relative precision far OTM) |
| `implied_vol_newton_safeguarded(price, inputs)` | option price, `BSInputs` | `float` |
| `implied_vol_brent(price, inputs)` | option price, `BSInputs` | `float` |
| `cos_price_and_greeks(phi, fwd, strikes, grid)` | CF, `ForwardSpec`, strikes, grid | `COSGreeks` with prices, delta, gamma |

Full reference for all public objects: [docs/api_reference.md](docs/api_reference.md).

---

## License

MIT. See [LICENSE](LICENSE).

---

## Demo notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nl2992/fourier-option-pricer/blob/main/notebooks/demo.ipynb)

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) is the primary entry point. It runs Carr-Madan, COS, and FRFT on a Heston strip and works in Colab with no local setup.

### Supplementary notebook

[`notebooks/supplementary/demo_advanced.ipynb`](notebooks/supplementary/demo_advanced.ipynb) is a **supplementary reference** for readers who want a comprehensive tour after the main demo. It covers the model zoo, the main Fourier pricers, Greeks, IV surface, Heston calibration, Monte Carlo, new models (Double Heston, VGSA), and validation highlights. It is **not** the recommended starting point.

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
| Implied vol (Let's Be Rational) | 50-digit mpmath reprice, 20k random OTM quotes | rel 5e-14 (vol), 1e-12 (reprice) | done |
| COS Americans (BSM) | Richardson-extrapolated 40k-step binomial trees | atol 5e-5 | done |
| COS Americans (Kou, CGMY) | Same extrapolation on the independent PROJ engine | atol 2e-5 | done |
| Hilbert discrete barriers/lookbacks (BSM) | Exact-Gaussian-kernel backward induction | atol 5e-7 / 1e-6 | done |
| Contour engine (BSM, VG) | Closed form; Gamma-mixture quadrature | rel 5e-14 / 5e-13 | done |
| 4/2, time-changed Lévy, BNS, lifted Heston, affine | Reductions to registry models + exact-simulation MC | 1e-10 to 1e-14 (reductions) | done |

Full per-paper matrix: [docs/paper_validation_matrix.md](docs/paper_validation_matrix.md). Evidence-level definitions: [docs/validation_hierarchy.md](docs/validation_hierarchy.md).

---

## Reproduce results

After cloning and installing (see [Installation](#installation) above):

```bash
# fast CI suite (skips slow Monte Carlo and notebook-execution tests)
python -m pytest -q -m "not slow"

# full suite including notebook execution guards
python -m pytest -q

# paper-replication tests only
python -m pytest -q -m "paper"

# MathWorks Bates software-reference tests
python -m pytest -q -m "software_reference"
```

The repository has 2,200+ pytest cases.

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
| Hilbert-transform pricing | Feng, L. and Linetsky, V. (2008), *Pricing Discretely Monitored Barrier Options and Defaultable Bonds in Lévy Process Models: A Fast Hilbert Transform Approach* |
| Regime switching | Buffington, J. and Elliott, R.J. (2002), *American Options with Regime Switching* |
| Lévy geometric Asians | Fusai, G. and Meucci, A. (2008), *Pricing Discretely Monitored Asian Options under Lévy Processes* |
| Variance swaps under jumps | Carr, P. and Wu, L. (2009), *Variance Risk Premiums* |
| PROJ frame projection | Kirkby, J.L. (2015), *Efficient Option Pricing by Frame Duality with the Fast Fourier Transform* |
| Implied volatility | Jäckel, P. (2015), *Let's Be Rational* |
| COS early exercise | Fang, F. and Oosterlee, C.W. (2009), *Pricing Early-Exercise and Discrete Barrier Options by Fourier-Cosine Series Expansions* |
| Optimal contours | Lord, R. and Kahl, C. (2007), *Optimal Fourier Inversion in Semi-Analytical Option Pricing* |
| Lookbacks | Feng, L. and Linetsky, V. (2009), *Computing Exponential Moments of the Discrete Maximum of a Lévy Process and Lookback Options* |
| Arithmetic Asians | Zhang, B. and Oosterlee, C.W. (2013), *Efficient Pricing of European-Style Asian Options under Exponential Lévy Processes Based on Fourier Cosine Expansions* |
| SINC | Baschetti, F., Bormetti, G., Romagnoli, S. and Rossi, P. (2022), *The SINC Way: A Fast and Accurate Approach to Fourier Pricing* |
| SWIFT | Ortiz-Gracia, L. and Oosterlee, C.W. (2016), *A Highly Efficient Shannon Wavelet Inverse Fourier Technique for Pricing European Options* |
| 4/2 model | Grasselli, M. (2017), *The 4/2 Stochastic Volatility Model: A Unified Approach for the Heston and the 3/2 Model* |
| Time-changed Lévy | Carr, P., Geman, H., Madan, D.B. and Yor, M. (2003), *Stochastic Volatility for Lévy Processes* |
| Lifted Heston | Abi Jaber, E. (2019), *Lifting the Heston Model* |
| BNS model | Barndorff-Nielsen, O.E. and Shephard, N. (2001), *Non-Gaussian Ornstein-Uhlenbeck-Based Models and Some of Their Uses in Financial Economics* |
| Affine transforms | Duffie, D., Pan, J. and Singleton, K. (2000), *Transform Analysis and Asset Pricing for Affine Jump-Diffusions* |

Full bibliography with DOIs and free-access links: [docs/papers.md](docs/papers.md).

---

## Roadmap

Transform-method territory not yet covered here, in rough priority order (the first block tracks capabilities popularized by the PROJ/CTMC MATLAB literature):

- [x] PROJ double-barrier options under Lévy models (`proj_double_barrier`, 0.14.0)
- [x] Fader options under Lévy models (`fader_cf`, 0.17.0)
- [x] Step options (occupation-time payoffs, Linetsky 1999) under Lévy models (`proj_step`, 0.18.0)
- [x] CTMC (continuous-time Markov chain) approximation: 1-D diffusions with local vol, European + American (`ctmc`, 0.21.0); stochastic-vol/SABR 2-D extension is future work
- [x] Credit default swaps via transform methods (`levy_cds_spread`, 0.19.0)
- [x] Swing options via transform methods (`proj_swing`, 0.20.0)
- [x] Regime-switching jump-diffusion regimes (per-regime Merton blocks, 0.15.0)
- [x] Stochastic-interest-rate hybrids (one-factor Hull-White composite CFs, `hw_hybrid`, 0.16.0)
- [x] Machine-precision vectorised implied volatility (Let's Be Rational) (0.22.0)
- [x] American options for Lévy models (COS Bermudans + Richardson) (0.22.0)
- [x] Discrete barriers and floating/fixed lookbacks by the Hilbert transform; arithmetic Asians (ASCOS) (0.22.0)
- [x] SINC, SWIFT and optimal-contour European engines (0.22.0)
- [x] 4/2, BNS Γ-OU, lifted Heston, time-changed Lévy and generic affine models (0.22.0)
- [x] Model-free variance and VIX-style index from option strips (0.22.0)
- [ ] Stochastic-volatility exotics: Bermudans and barriers under Heston (2-D COS or 2-D CTMC)
- [ ] Two-dimensional Fourier spread and rainbow options (Hurd & Zhou 2010)
- [ ] Registry-driven calibration with analytic CF gradients

Contributions welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Documentation

All reference documentation is indexed at **[docs/README.md](docs/README.md)**: model zoo, full API, validation hierarchy, paper replication tables, filtered-COS extension, Bates/3/2 SV validation, AI workflow, and packaging checklist.

| Document | Contents |
|----------|----------|
| [appendix.md](appendix.md) | Methodology, derivations, model conventions, benchmark interpretation, and the full numbered course-project narrative (sections 1–18). |
| [docs/numerical_notes.md](docs/numerical_notes.md) | Known numerical limitations: COS truncation failure modes, Carr-Madan alpha conditions, PyFENG version caveats. |
| [docs/numerical_quality_checklist.md](docs/numerical_quality_checklist.md) | M1/M4 floating-point rubric audit: `expm1` fix, variance floor, analytic Greeks, RNG pattern, dtype guards. |
| [docs/papers.md](docs/papers.md) | Full bibliography with DOIs and free-access links, grouped by method and model family. |
