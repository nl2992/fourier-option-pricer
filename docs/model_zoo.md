# Model Zoo

Complete catalogue of the twenty-one characteristic-function models supported by `foureng`.
All are importable from the top-level package (`import foureng as fe`).
The unified dispatcher `fe.price_strip` routes through `foureng.models.registry.MODEL_REGISTRY`.

## Model table

| Model | Public dataclass | CF source | Notes |
|-------|------------------|-----------|-------|
| Black-Scholes-Merton | `BsmParams` | PyFENG-backed adapter | Diffusion baseline and sanity-check model. |
| Heston | `HestonParams` | PyFENG-backed adapter | Main stochastic-volatility benchmark. |
| OUSV / Schobel-Zhu | `OusvParams` | PyFENG-backed adapter | Stochastic-volatility alternative to Heston. |
| Variance Gamma | `VGParams` | PyFENG-backed adapter | Pure-jump Lévy model used in the repo benchmarks. |
| CGMY | `CgmyParams` | PyFENG-backed adapter | Infinite-activity tempered-stable jump model. |
| Normal Inverse Gaussian | `NigParams` | PyFENG-backed adapter | Lévy model with heavier tails than Gaussian diffusion. |
| 3/2 Stochastic Volatility | `Sv32Params` | PyFENG-backed adapter | Mean-reverting variance process with 3/2 diffusion coefficient. |
| Rough Heston | `RoughHestonParams` | PyFENG-backed adapter | Fractional Brownian motion variance driver (Hurst index H < 1/2). |
| Kou | `KouParams` | In-house implementation | Double-exponential jump-diffusion CF and cumulants. |
| Bates | `BatesParams` | In-house composite | Heston diffusion block plus Merton jump block. |
| Heston-Kou | `HestonKouParams` | In-house composite | Heston block combined with the Kou jump CF. |
| Heston-CGMY | `HestonCGMYParams` | In-house composite | Heston block combined with a CGMY jump factor. |
| GARCH (WMW 2012) | `GarchWMW2012Params` | In-house implementation | Discrete-time GARCH model with analytic CF from Wu-Ma-Wang (2012). |
| Merton Jump-Diffusion | `MertonJDParams` | In-house implementation | Geometric Brownian motion plus compound Poisson jumps with log-normal sizes. |
| Meixner | `MeixnerParams` | In-house implementation | Lévy process with CF based on the hyperbolic cosine; fits S&P500 smile. |
| Bilateral Gamma | `BilateralGammaParams` | In-house implementation | Separate Gamma processes for upward and downward moves (Küchler & Tappe 2008). |
| Generalized Hyperbolic | `GHParams` | In-house implementation | Normal variance-mean mixture via GIG; includes NIG (λ=−½) and Hyperbolic (λ=1) as special cases. |
| Finite Moment Log Stable | `FMLSParams` | In-house implementation | Maximally negatively-skewed α-stable Lévy process; all positive moments of S_T are finite (Carr & Wu 2003). |
| Double Heston | `DoubleHestonParams` | In-house implementation | Two independent Heston variance factors; CF factorises as a product of two single-Heston CFs. |
| VGSA | `VGSAParams` | In-house implementation | Variance Gamma on a stochastic CIR activity clock; captures term-structure of skew and vol-of-vol clustering. |
| Regime-Switching BSM | `RegimeSwitchingBsmParams` | In-house implementation | Markov-modulated volatility regimes; CF via the matrix exponential of the chain generator plus per-regime Levy exponents (Buffington & Elliott 2002). Cumulants by numeric CGF differentiation. |

## PyFENG dependency

The eight PyFENG-backed models require `pyfeng>=0.4.0`.
pyfeng 0.4.0 renamed `charfunc_logprice` → `logp_cf` and changed `VarGammaFft`/`ExpNigFft` from `vov=` to `nu=`.
Rough Heston imports directly from `pyfeng.sv_fft` (not `pyfeng.ex`) to avoid a broken path that calls the removed `scipy.misc.derivative` in newer SciPy.
The `method="pyfeng_fft"` option in `price_strip` is supported only for these eight models. It refers to the PyFENG-backed Lewis-style FFT path. The remaining twelve models use the in-house `"cos"`, `"cos_improved"`, `"cos_filtered"`, `"carr_madan"`, and `"frft"` methods only. Note: `foureng/pricers/lewis.py` is an internal module used inside the COS/filtered-COS policy; `"lewis"` is not a valid `price_strip` method string.

## In-house composites

The SVJ composites (Bates, Heston-Kou, Heston-CGMY) exploit the independence factorisation

```
φ_SVJ(u) = φ_Heston(u) · φ_Jump(u)
```

At zero jump intensity each composite reduces to plain Heston  -  this is checked by `tests/models/` model-reduction gates.

## Public API

All twenty models are **first-class public API objects** importable directly from the top-level package.
Parameter dataclasses, characteristic functions, and cumulant functions are all in `foureng.__all__`:

```python
import foureng as fe

params = fe.BatesParams(kappa=1.0, theta=0.05, nu=0.2, rho=-0.7,
                        v0=0.04, lam_j=2.0, mu_j=0.017, sigma_j=0.08)
phi    = lambda u: fe.bates_cf(u, fwd, params)
cums   = fe.bates_cumulants(fwd, params)
```

The `MODEL_REGISTRY` in `foureng.models.registry` is the single source of truth for which models are supported and which have a native PyFENG FFT pricer; `price_strip` dispatches through it.

## Validation status

Each model is validated at one of five evidence levels defined in [validation_hierarchy.md](validation_hierarchy.md).

| Per-model evidence table | [paper_validation_matrix.md](paper_validation_matrix.md)  -  links every paper-backed result to its test file, reference type, and numeric target |
| Bates and 3/2 SV detail | [bates_sv32_validation.md](bates_sv32_validation.md)  -  BATES-01–07 and SV32-01–05 cases with parameters, tolerances, test files, and benchmark CSVs |
| FO2008 replication tables | [fo2008_replication.md](fo2008_replication.md)  -  Tables 1–10 paper-faithful replay and improved-COS summary |
