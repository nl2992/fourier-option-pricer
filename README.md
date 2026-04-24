# fourier-option-pricer

Fourier pricing toolkit for fast European option pricing under characteristic-function models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nl2992/fourier-option-pricer/blob/main/notebooks/demo.ipynb)

## What problem does this solve?

Monte Carlo option pricing is flexible, but it can be slow when pricing many strikes, maturities, or implied-volatility surfaces. This package provides faster Fourier-based pricing methods for vanilla European options.

The toolkit supports Carr-Madan FFT, fractional FFT, and COS pricing under models with known characteristic functions, including Heston, Variance Gamma, Kou, and related extensions. It is designed for reproducible numerical experiments, benchmarking, implied-volatility inversion, and model comparison.

PyPI distribution name: `fourier-option-pricer`  
Python import name: `foureng`

## Installation

Install the package from PyPI:

```bash
pip install fourier-option-pricer
```

For local development:

```bash
git clone https://github.com/nl2992/fourier-option-pricer.git
cd fourier-option-pricer
pip install -e ".[test]"
```

Run the test suite:

```bash
pytest
```

## Quick start

The example below prices European call options under the Heston model using the COS method.

```python
import numpy as np
import foureng as fe

# Market inputs
fwd = fe.ForwardSpec(
    S0=100.0,
    r=0.01,
    q=0.02,
    T=1.0,
)

# Heston model parameters
params = fe.HestonParams(
    kappa=4.0,
    theta=0.25,
    nu=1.0,
    rho=-0.5,
    v0=0.04,
)

# Characteristic function
phi = lambda u: fe.heston_cf_form2(u, fwd, params)

# Strike grid
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

# COS truncation grid
cumulants = fe.heston_cumulants(fwd, params)
grid = fe.cos_auto_grid(cumulants, N=256, L=10.0)

# Price European calls
result = fe.cos_prices(phi, fwd, strikes, grid)

print(result.call_prices)

# Convert the ATM call price to Black-Scholes implied volatility
atm_iv = fe.implied_vol_newton_safeguarded(
    price=float(result.call_prices[2]),
    inputs=fe.BSInputs(
        F0=fwd.F0,
        K=100.0,
        T=fwd.T,
        r=fwd.r,
        q=fwd.q,
        is_call=True,
    ),
)

print(atm_iv)
```

## Main features

- Carr-Madan FFT pricing for European calls.
- Fractional FFT pricing for flexible strike grids.
- COS pricing with automatic and improved truncation grids.
- Characteristic-function models including Heston, Variance Gamma, and Kou.
- Black-Scholes implied-volatility inversion with safeguarded Newton iteration.
- Price and implied-volatility surface generation.
- COS-based Greeks.
- Calibration helpers for Heston, Variance Gamma, and Kou.
- Benchmarking workflow for comparing pricing accuracy and runtime.

## API reference

### Market inputs

#### `ForwardSpec(S0, r, q, T)`

Container for deterministic market inputs.

Parameters:

- `S0`: spot price.
- `r`: continuously compounded risk-free rate.
- `q`: continuously compounded dividend yield or foreign interest rate.
- `T`: time to maturity in years.

Provides:

- `F0`: forward price.
- `disc`: discount factor.

### Model parameter classes

#### `HestonParams(kappa, theta, nu, rho, v0)`

Parameter container for the Heston stochastic-volatility model.

#### `VGParams(...)`

Parameter container for the Variance Gamma model.

#### `KouParams(...)`

Parameter container for the Kou double-exponential jump-diffusion model.

### Characteristic functions

#### `heston_cf_form2(u, fwd, params)`

Heston characteristic function using the numerically stable Form 2 representation.

Parameters:

- `u`: NumPy array of Fourier frequencies.
- `fwd`: `ForwardSpec` object.
- `params`: `HestonParams` object.

Returns:

- Complex NumPy array of characteristic-function values.

#### `vg_cf(u, fwd, params)`

Variance Gamma characteristic function.

Returns:

- Complex NumPy array of characteristic-function values.

#### `kou_cf(u, fwd, params)`

Kou double-exponential jump-diffusion characteristic function.

Returns:

- Complex NumPy array of characteristic-function values.

### Cumulants and COS grids

#### `heston_cumulants(fwd, params)`

Computes Heston cumulants used to construct COS truncation intervals.

Returns:

- Model cumulants for grid construction.

#### `vg_cumulants(fwd, params)`

Computes Variance Gamma cumulants.

#### `kou_cumulants(fwd, params)`

Computes Kou model cumulants.

#### `cos_auto_grid(cumulants, N, L)`

Constructs the standard Fang-Oosterlee COS truncation grid.

Parameters:

- `cumulants`: model cumulants.
- `N`: number of COS expansion terms.
- `L`: truncation-width parameter.

Returns:

- `COSGrid`.

#### `cos_improved_grid(cumulants, model=..., params=...)`

Constructs an improved COS truncation grid for more stable numerical pricing.

Returns:

- `COSGrid`.

### Pricing methods

#### `cos_prices(phi, fwd, strikes, grid)`

Prices European calls using the COS method.

Parameters:

- `phi`: characteristic function.
- `fwd`: `ForwardSpec` object.
- `strikes`: NumPy array of strikes.
- `grid`: `COSGrid` object.

Returns:

- `COSResult`, containing `strikes` and `call_prices`.

#### `carr_madan_price_at_strikes(phi, fwd, grid, strikes)`

Prices European calls using the Carr-Madan FFT method.

Returns:

- NumPy array of call prices.

#### `frft_price_at_strikes(phi, fwd, grid, strikes)`

Prices European calls using the fractional FFT method.

Returns:

- NumPy array of call prices.

### Greeks

#### `cos_price_and_greeks(phi, fwd, strikes, grid)`

Computes COS prices and sensitivities.

Returns:

- `COSGreeks`.

#### `cos_delta_gamma(phi, fwd, strikes, grid)`

Computes COS delta and gamma.

Returns:

- Tuple of NumPy arrays: `(delta, gamma)`.

### Implied volatility

#### `implied_vol_newton_safeguarded(price, inputs)`

Computes Black-Scholes implied volatility using a safeguarded Newton method.

Parameters:

- `price`: option price.
- `inputs`: `BSInputs` object.

Returns:

- Implied volatility as a `float`.

### Higher-level helpers

#### `model_price_surface(...)`

Builds a model price surface across strikes and maturities.

Returns:

- Price-surface output.

#### `model_iv_surface(...)`

Builds an implied-volatility surface from model prices.

Returns:

- Implied-volatility surface output.

#### `calibrate_heston(...)`

Calibrates Heston parameters to market or synthetic option prices.

Returns:

- Calibration result.

#### `calibrate_vg(...)`

Calibrates Variance Gamma parameters.

Returns:

- Calibration result.

#### `calibrate_kou(...)`

Calibrates Kou model parameters.

Returns:

- Calibration result.

## Demo notebook

A Colab-ready demo notebook is available here:

[notebooks/demo.ipynb](notebooks/demo.ipynb)

The notebook demonstrates:

- installing the package in Google Colab;
- pricing European options with Fourier methods;
- comparing COS, Carr-Madan FFT, and Monte Carlo baselines;
- computing pricing errors against benchmark values;
- measuring runtime differences across methods;
- generating plots suitable for the final project report.

## Extended methodology and results

Detailed numerical experiments, replication notes, runtime benchmarks, and implementation commentary are kept outside the README to keep this page concise.

See:

```text
docs/methodology_and_results.md
```

This document records:

- the Fang-Oosterlee COS replication workflow;
- the Carr-Madan benchmark setup;
- the Monte Carlo comparison setup;
- COS truncation-interval behaviour;
- improved COS grid logic;
- runtime and error reporting rules;
- model-by-model observations;
- known numerical limitations.

## License

MIT. See [LICENSE](LICENSE).

