# fourier-option-pricer

Fourier pricing toolkit for European options using Carr-Madan FFT, FRFT, and COS under characteristic-function models.

This package solves a practical numerical-finance problem: pricing vanilla European options, computing implied volatilities, and building price or volatility surfaces without relying on slow Monte Carlo as the main engine. It wraps several characteristic-function models behind a common interface so the same workflow can be reused across Heston, Variance Gamma, Kou, and related models.

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

`ForwardSpec(S0, r, q, T)`
Market inputs for spot, rates, and maturity. Provides derived forward and discount-factor fields.

`HestonParams`, `VGParams`, `KouParams`
Model parameter dataclasses for the main supported examples.

`heston_cf_form2(u, fwd, params)`, `vg_cf(u, fwd, params)`, `kou_cf(u, fwd, params)`
Characteristic functions. Input: frequency array. Return: complex-valued NumPy array.

`heston_cumulants(fwd, params)`, `vg_cumulants(fwd, params)`, `kou_cumulants(fwd, params)`
Cumulant helpers used to construct COS truncation grids.

`cos_auto_grid(cumulants, N, L)` and `cos_improved_grid(cumulants, model=..., params=...)`
Construct COS grids. Return type: `COSGrid`.

`cos_prices(phi, fwd, strikes, grid)`
Prices a European call strip with COS. Return type: `COSResult` with `strikes` and `call_prices`.

`carr_madan_price_at_strikes(phi, fwd, grid, strikes)` and `frft_price_at_strikes(phi, fwd, grid, strikes)`
FFT-based strip pricers. Return type: NumPy array of call prices.

`implied_vol_newton_safeguarded(price, BSInputs(...))`
Inverts an option price to Black-Scholes implied volatility. Return type: `float`.

`model_price_surface(...)`, `model_iv_surface(...)`, `calibrate_heston(...)`, `calibrate_vg(...)`, `calibrate_kou(...)`
Higher-level helpers for strip generation, implied-volatility surfaces, and calibration.

## License

MIT. See [LICENSE](LICENSE).

## Demo notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nl2992/fourier-option-pricer/blob/main/notebooks/demo.ipynb)

The Colab-ready demo notebook lives at [notebooks/demo.ipynb](notebooks/demo.ipynb).

Extra project notes and methodology live in [APPENDIX.md](APPENDIX.md).
