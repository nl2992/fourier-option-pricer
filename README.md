# fourier-option-pricer

Fourier pricing toolkit for European options using Carr-Madan FFT, FRFT, and COS under characteristic-function models.

This package solves a practical numerical-finance problem: pricing vanilla European options quickly when the model is easier to describe through its characteristic function than through a closed-form price formula. It gives one consistent workflow for pricing strips, implied volatilities, and surfaces across Heston, Variance Gamma, Kou, and related models. The PyPI package name is `fourier-option-pricer`, and the Python import name is `foureng`.

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
| `HestonParams(kappa, theta, nu, rho, v0)` | Heston stochastic-volatility parameters | Heston model parameter dataclass. |
| `VGParams(sigma, nu, theta)` | Variance Gamma parameters | Variance Gamma parameter dataclass. |
| `KouParams(sigma, lam, p, eta1, eta2)` | Diffusion plus jump parameters | Kou double-exponential jump-diffusion parameter dataclass. |

### Characteristic functions and cumulants

| Object | Parameters | Returns / purpose |
|--------|------------|-------------------|
| `heston_cf_form2(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: HestonParams` | Complex-valued Heston characteristic function in log-forward coordinates. |
| `vg_cf(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: VGParams` | Complex-valued Variance Gamma characteristic function. |
| `kou_cf(u, fwd, params)` | `u: np.ndarray`, `fwd: ForwardSpec`, `params: KouParams` | Complex-valued Kou characteristic function. |
| `heston_cumulants(fwd, params)` | `ForwardSpec`, `HestonParams` | Heston cumulants used to build COS truncation intervals. |
| `vg_cumulants(fwd, params)` | `ForwardSpec`, `VGParams` | Variance Gamma cumulants for COS grid construction. |
| `kou_cumulants(fwd, params)` | `ForwardSpec`, `KouParams` | Kou cumulants for COS grid construction. |

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
