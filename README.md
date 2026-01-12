# fourier-option-pricer

Fourier pricing toolkit for European options using Carr-Madan FFT, FRFT, and COS under characteristic-function models.

This package solves a practical numerical-finance problem: pricing vanilla European options quickly when the model is easier to describe through its characteristic function than through a closed-form price formula. It gives one consistent workflow for pricing strips, implied volatilities, and surfaces across Heston, Variance Gamma, Kou, and related models. The PyPI package name is `fourier-option-pricer`, and the Python import name is `foureng`.

### Supported model layer

The pricing layer supports ten characteristic-function models. Some are thin adapters over [PyFENG](https://github.com/PyFE/PyFENG), while others are implemented in-house inside `foureng.models`.

| Model | Public dataclass | Characteristic-function source | Notes |
|--------|------------------|-------------------------------|-------|
| Black-Scholes-Merton | `BsmParams` | PyFENG-backed adapter | Diffusion baseline and sanity-check model. |
| Heston | `HestonParams` | PyFENG-backed adapter | Main stochastic-volatility benchmark. |
| OUSV / Schobel-Zhu | `OusvParams` | PyFENG-backed adapter | Stochastic-volatility alternative to Heston. |
| Variance Gamma | `VGParams` | PyFENG-backed adapter | Pure-jump Levy model used in the repo benchmarks. |
| CGMY | `CgmyParams` | PyFENG-backed adapter | Infinite-activity tempered-stable jump model. |
| Normal Inverse Gaussian | `NigParams` | PyFENG-backed adapter | Levy model with heavier tails than Gaussian diffusion. |
| Kou | `KouParams` | In-house implementation | Double-exponential jump-diffusion CF and cumulants are coded directly in this repo. |
| Bates | `BatesParams` | In-house composite | Heston diffusion block plus in-house Merton jump block. |
| Heston-Kou | `HestonKouParams` | In-house composite | Heston block combined with the in-house Kou jump CF. |
| Heston-CGMY | `HestonCGMYParams` | In-house composite | Heston block combined with an in-house CGMY jump factor. |

### Why use Fourier methods here instead of plain Monte Carlo?

Monte Carlo is still useful as a validation baseline, but it scales poorly for plain-vanilla European pricing once a characteristic function is available. Its standard error behaves like

$$
\text{MC error} = O(n^{-1/2}),
$$

so reducing the error by a factor of 10 usually needs about 100 times as many paths. The Fourier methods in this package reuse the same model input to price whole strike strips much more efficiently than pathwise simulation.

### Characteristic-function backbone

All three pricing families in this package start from the same object:

$$
\varphi_T(u) = \mathbb{E}^{\mathbb{Q}}\\left[e^{iuX_T}\right],
\qquad
X_T = \log\\left(\frac{S_T}{F_0}\right).
$$

Here `i = sqrt(-1)`, `u` is the Fourier frequency, and `X_T` is the terminal log-forward return. Carr-Madan FFT and FRFT recover prices through Fourier inversion of this characteristic function, while COS uses the same object to build cosine-series coefficients on a truncated interval. For PyFENG-backed models, `foureng` translates its dataclasses into the corresponding `pyfeng.*Fft` model and evaluates `charfunc_logprice`; for the in-house models, the characteristic functions are implemented directly in `foureng.models`.

## Installation

```bash
pip install fourier-option-pricer
```

## Quick start

```python
import numpy as np
import foureng as fe

fwd = fe.ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
params = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)

phi = lambda u: fe.heston_cf_form2(u, fwd, params)
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
grid = fe.cos_auto_grid(fe.heston_cumulants(fwd, params), N=256, L=10.0)
result = fe.cos_prices(phi, fwd, strikes, grid)

print(result.call_prices)
```

## API reference

The main public API is exposed from:

```python
import foureng as fe
```

The unified notebook and benchmark dispatcher is also available as:

```python
from foureng.pipeline import price_strip
```

### Market inputs and model parameters

| Object | Parameters | Returns / purpose |
|--------|------------|-------------------|
| `ForwardSpec(S0, r, q, T)` | `S0: float`, `r: float`, `q: float`, `T: float` | Market inputs. Also provides derived `F0` and discount factor `disc`. |
| `BsmParams(sigma)` | Black-Scholes volatility parameter | Diffusion baseline model dataclass. |
| `HestonParams(kappa, theta, nu, rho, v0)` | Heston stochastic-volatility parameters | Heston model parameter dataclass. |
| `OusvParams(sigma0, kappa, theta, nu, rho)` | Schobel-Zhu / OUSV stochastic-volatility parameters | OUSV model parameter dataclass. |
| `VGParams(sigma, nu, theta)` | Variance Gamma parameters | Variance Gamma parameter dataclass. |
| `CgmyParams(C, G, M, Y)` | CGMY Levy parameters | CGMY model parameter dataclass. |
| `NigParams(sigma, nu, theta)` | NIG Levy parameters | NIG model parameter dataclass. |
| `KouParams(sigma, lam, p, eta1, eta2)` | Diffusion plus jump parameters | Kou double-exponential jump-diffusion parameter dataclass. |
| `BatesParams(kappa, theta, nu, rho, v0, lam_j, mu_j, sigma_j)` | Heston block plus Merton jump parameters | Bates stochastic-volatility jump-diffusion dataclass. |
| `HestonKouParams(kappa, theta, nu, rho, v0, lam_j, p_j, eta1, eta2)` | Heston block plus Kou jump parameters | Heston-Kou composite model dataclass. |
| `HestonCGMYParams(kappa, theta, nu, rho, v0, C, G, M, Y)` | Heston block plus CGMY jump parameters | Heston-CGMY composite model dataclass. |

### Characteristic functions and cumulants

| Object | Parameters | Returns / purpose |
|--------|------------|-------------------|
| `bsm_cf(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: BsmParams` | Complex-valued Black-Scholes characteristic function. |
| `heston_cf_form2(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: HestonParams` | Complex-valued Heston characteristic function in log-forward coordinates. |
| `ousv_cf(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: OusvParams` | Complex-valued OUSV / Schobel-Zhu characteristic function. |
| `vg_cf(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: VGParams` | Complex-valued Variance Gamma characteristic function. |
| `cgmy_cf(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: CgmyParams` | Complex-valued CGMY characteristic function. |
| `nig_cf(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: NigParams` | Complex-valued NIG characteristic function. |
| `kou_cf(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: KouParams` | Complex-valued Kou characteristic function. |
| `bates_cf(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: BatesParams` | Complex-valued Bates characteristic function. |
| `heston_kou_cf(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: HestonKouParams` | Complex-valued Heston-Kou characteristic function. |
| `heston_cgmy_cf(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: HestonCGMYParams` | Complex-valued Heston-CGMY characteristic function. |
| `bsm_cumulants(fwd, params)` | `ForwardSpec`, `BsmParams` | Black-Scholes cumulants used in COS grid construction. |
| `heston_cumulants(fwd, params)` | `ForwardSpec`, `HestonParams` | Heston cumulants used to build COS truncation intervals. |
| `ousv_cumulants(fwd, params)` | `ForwardSpec`, `OusvParams` | OUSV cumulants for COS grid construction. |
| `vg_cumulants(fwd, params)` | `ForwardSpec`, `VGParams` | Variance Gamma cumulants for COS grid construction. |
| `cgmy_cumulants(fwd, params)` | `ForwardSpec`, `CgmyParams` | CGMY cumulants for COS grid construction. |
| `nig_cumulants(fwd, params)` | `ForwardSpec`, `NigParams` | NIG cumulants for COS grid construction. |
| `kou_cumulants(fwd, params)` | `ForwardSpec`, `KouParams` | Kou cumulants for COS grid construction. |
| `bates_cumulants(fwd, params)` | `ForwardSpec`, `BatesParams` | Bates cumulants for COS grid construction. |
| `heston_kou_cumulants(fwd, params)` | `ForwardSpec`, `HestonKouParams` | Heston-Kou cumulants for COS grid construction. |
| `heston_cgmy_cumulants(fwd, params)` | `ForwardSpec`, `HestonCGMYParams` | Heston-CGMY cumulants for COS grid construction. |

### Grid objects and grid builders

| Object | Parameters | Returns / purpose |
|--------|------------|-------------------|
| `COSGrid(a, b, N)` | truncation interval and number of terms | Concrete COS grid used by `cos_prices`. |
| `COSGridPolicy(...)` | policy fields such as `mode`, `truncation`, `dx_target`, `L`, `eps_trunc` | Rule-based COS grid specification used by the improved and filtered COS paths. |
| `FFTGrid(N, eta, alpha)` | FFT size, frequency spacing, damping parameter | Carr-Madan FFT grid. |
| `FRFTGrid(N, eta, lam, alpha)` | FRFT size, spacing, strike step, damping parameter | Fractional FFT grid. |
| `cos_auto_grid(cumulants, N, L)` | cumulants, term count, truncation multiplier | Returns a `COSGrid` from the standard cumulant rule. |
| `cos_improved_grid(cumulants, model=..., params=...)` | cumulants plus model context | Returns a `COSGrid` using the improved COS truncation policy. |
| `recommended_cos_policy(model, params, mode=...)` | model name and parameter dataclass | Returns a `COSGridPolicy` for the improved COS workflow. |

### Core pricing functions

| Object | Parameters | Returns / purpose |
|--------|------------|-------------------|
| `cos_prices(phi, fwd, strikes, grid)` | characteristic function, `ForwardSpec`, strike array, `COSGrid` | Returns `COSResult` with `strikes` and `call_prices`. |
| `carr_madan_price_at_strikes(phi, fwd, grid, strikes)` | characteristic function, `ForwardSpec`, `FFTGrid`, strike array | Returns NumPy array of call prices from Carr-Madan FFT. |
| `frft_price_at_strikes(phi, fwd, grid, strikes)` | characteristic function, `ForwardSpec`, `FRFTGrid`, strike array | Returns NumPy array of call prices from FRFT. |
| `filtered_cos_prices(phi, fwd, strikes, grid, filter_spec=...)` | characteristic function, `ForwardSpec`, strike array, COS grid, filter | Returns `COSResult` with spectral filtering applied to the COS coefficients. |
| `price_strip(model, method, strikes, fwd, params, grid=None, ...)` | model label, method label, strike array, market inputs, model parameters | Unified strip-pricing dispatcher used throughout the notebooks and benchmarks. |

### Filtered-COS helpers

| Object | Parameters | Returns / purpose |
|--------|------------|-------------------|
| `COSFilterSpec(name, order=..., alpha=...)` | filter family and optional shape parameters | Filter specification for the filtered COS method. |
| `cos_filter_weights(N, filter_spec)` | number of COS terms and filter spec | NumPy array of spectral weights `sigma_k`. |
| `cos_adaptive_decision(...)` | model context and COS policy inputs | Returns `COSPolicyDecision` summarizing the improved COS grid choice. |

### Implied volatility

| Object | Parameters | Returns / purpose |
|--------|------------|-------------------|
| `BSInputs(F0, K, T, r, q, is_call)` | Black-style inversion inputs | Dataclass passed into implied-vol routines. |
| `bs_price_from_fwd(sigma, inputs)` | volatility and `BSInputs` | Black-Scholes price from forward inputs. |
| `implied_vol_newton_safeguarded(price, inputs)` | option price and `BSInputs` | Returns `float` implied volatility using safeguarded Newton iterations. |
| `implied_vol_brent(price, inputs)` | option price and `BSInputs` | Returns `float` implied volatility using a bracketing solver. |

### Surfaces, calibration, and Greeks

| Object | Parameters | Returns / purpose |
|--------|------------|-------------------|
| `SurfaceSpec(S0, r, q, maturities, strikes)` | market inputs plus maturity/strike grids | Surface input container for model price or IV surfaces. |
| `model_price_surface(...)` | surface spec plus pricing callbacks | Returns a price surface over maturity and strike grids. |
| `model_iv_surface(...)` | surface spec plus pricing callbacks | Returns an implied-volatility surface. |
| `calibrate_heston(...)`, `calibrate_vg(...)`, `calibrate_kou(...)` | market targets, grid inputs, initial guesses | Return `CalibrationResult` for the chosen model. |
| `cos_price_and_greeks(phi, fwd, strikes, grid)` | characteristic function, market inputs, strike array, grid | Returns `COSGreeks` with prices and sensitivity arrays. |
| `cos_delta_gamma(phi, fwd, strikes, grid)` | characteristic function, market inputs, strike array, grid | Returns delta and gamma arrays. |
| `cos_parameter_sensitivity(...)` | model setup plus parameter perturbation inputs | Returns COS-based parameter sensitivities. |

## License

MIT. See [LICENSE](LICENSE).

## Demo notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nl2992/fourier-option-pricer/blob/main/notebooks/demo.ipynb)

The Colab-ready demo notebook lives at [notebooks/demo.ipynb](notebooks/demo.ipynb).

## Papers used

These are the main papers the package and notebook workflow are built around.

| Topic | Reference |
|-------|-----------|
| Carr-Madan FFT | Carr, P. and Madan, D.B. (1999), *Option Valuation Using the Fast Fourier Transform*. |
| FRFT for option pricing | Chourdakis, K. (2004), *Option Pricing Using the Fractional FFT*. |
| COS method | Fang, F. and Oosterlee, C.W. (2008), *A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions*. |
| Heston model | Heston, S.L. (1993), *A Closed-Form Solution for Options with Stochastic Volatility*. |
| Stable Heston CF branch handling | Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007), *The Little Heston Trap*. |
| Lewis benchmark formula | Lewis, A.L. (2001), *A Simple Option Formula for General Jump-Diffusion and Other Exponential Levy Processes*. |
| Variance Gamma model | Madan, D.B., Carr, P. and Chang, E.C. (1998), *The Variance Gamma Process and Option Pricing*. |
| Kou jump-diffusion model | Kou, S.G. (2002), *A Jump-Diffusion Model for Option Pricing*. |
| Improved COS truncation range | Junike, G. and Pankrashkin, K. (2022), *Precise Option Pricing by the COS Method: How to Choose the Truncation Range*. |
| Improved COS term-count policy | Junike, G. (2024), *On the Number of Terms in the COS Method for European Option Pricing*. |
| Spectral filtering for Fourier/COS pricing | Ruijter, M.J., Versteegh, M. and Oosterlee, C.W. (2015), *On the Application of Spectral Filters in a Fourier Option Pricing Technique*. |
