# fourier-option-pricer

Fourier pricing toolkit for European options using Carr-Madan FFT, FRFT, and COS under characteristic-function models.

This package solves a practical numerical-finance problem: pricing vanilla European options, computing implied volatilities, and building price/IV surfaces without relying on slow Monte Carlo as the main engine. It wraps several characteristic-function models behind a common interface so the same pricing code can be reused across Heston, Variance Gamma, Kou, and related models.

PyPI distribution name: `fourier-option-pricer`  
Python import name: `foureng`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nl2992/fourier-option-pricer/blob/main/notebooks/demo.ipynb)

## Installation

```bash
pip install fourier-option-pricer
```

For local development and tests:

```bash
pip install -e ".[test]"
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

atm_iv = fe.implied_vol_newton_safeguarded(
    float(result.call_prices[2]),
    fe.BSInputs(F0=fwd.F0, K=100.0, T=fwd.T, r=fwd.r, q=fwd.q, is_call=True),
)
print(atm_iv)
```

## API reference

`ForwardSpec(S0, r, q, T)`
Deterministic market inputs. Provides `F0` and discount factor `disc`.

`HestonParams`, `VGParams`, `KouParams`
Model parameter dataclasses for the main top-level examples.

`heston_cf_form2(u, fwd, params)`, `vg_cf(u, fwd, params)`, `kou_cf(u, fwd, params)`
Characteristic functions. Input: `np.ndarray` of frequencies. Return: complex `np.ndarray`.

`heston_cumulants(fwd, params)`, `vg_cumulants(fwd, params)`, `kou_cumulants(fwd, params)`
Model cumulants used to build COS truncation intervals and grids.

`cos_auto_grid(cumulants, N, L)` and `cos_improved_grid(cumulants, model=..., params=...)`
Construct COS grids. Return type: `COSGrid`.

`cos_prices(phi, fwd, strikes, grid)`
Price European calls with COS. Return type: `COSResult` with `strikes` and `call_prices`.

`carr_madan_price_at_strikes(phi, fwd, grid, strikes)` and `frft_price_at_strikes(phi, fwd, grid, strikes)`
Price European calls with FFT-based methods. Return type: `np.ndarray`.

`cos_price_and_greeks(phi, fwd, strikes, grid)` and `cos_delta_gamma(phi, fwd, strikes, grid)`
Return COS-based price sensitivities. Return type: `COSGreeks` or `tuple[np.ndarray, np.ndarray]`.

`implied_vol_newton_safeguarded(price, BSInputs(...))`
Invert a call or put price to Black-Scholes implied volatility. Return type: `float`.

`model_price_surface(...)`, `model_iv_surface(...)`, `calibrate_heston(...)`, `calibrate_vg(...)`, `calibrate_kou(...)`
Higher-level helpers for strip generation, implied-volatility surfaces, and model calibration.

## License

MIT. See [LICENSE](LICENSE).

## Demo notebook

The Colab-ready demo notebook lives at [notebooks/demo.ipynb](notebooks/demo.ipynb).
