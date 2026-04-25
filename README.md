# fourier-option-pricer

Fast European option pricing via Fourier transform methods under **characteristic-function models**.

> Carr, P., & Madan, D. (1999). Option valuation using the fast Fourier transform.
> *Journal of Computational Finance*, 2(4), 61–73.
> https://doi.org/10.21314/JCF.1999.043

> Lewis, A. L. (2001). A simple option formula for general jump-diffusion and other
> exponential Lévy processes. *SSRN Working Paper*.
> https://ssrn.com/abstract=282110
> *(Heston and Variance Gamma characteristic functions are provided by
> [PyFENG](https://github.com/PyFENG/PyFENG), Prof. Jaehyuk Choi's library.)*

> Fang, F., & Oosterlee, C. W. (2008). A novel pricing method for European options
> based on Fourier-cosine series expansions.
> *SIAM Journal on Scientific Computing*, 31(2), 826–848.
> https://doi.org/10.1137/080718061

> Junike, G., & Pankrashkin, K. (2022). Precise option pricing by the COS method —
> how to choose the truncation range.
> *Applied Mathematics and Computation*, 421, 126935.
> https://doi.org/10.1016/j.amc.2022.126935

## Core concept

Fourier pricing exploits the fact that, for most asset models, the **characteristic function**

$$\phi(u) = \mathbb{E}\!\left[e^{iu \ln S_T}\right]$$

is known in closed form even when the option price integral has no analytic solution.
Given $\phi$, a European call can be priced by a single numerical integral.
The three methods implemented here differ in how they discretise that integral:

| Method | Key idea |
|--------|----------|
| Carr–Madan FFT | Damp the payoff, apply FFT to price a whole strike grid at once |
| Lewis single-integral | Parseval identity; avoids the dampening parameter entirely |
| COS (Fang–Oosterlee) | Expand the risk-neutral density in a cosine series on $[a, b]$ |

## Truncation

The COS method requires choosing a truncation interval $[a, b]$ for the log-price density.
Two strategies are implemented:

- **Cumulant rule** (Fang & Oosterlee 2008) — sets $[a,b]$ from the first four cumulants of $\ln S_T$.
- **Tolerance rule** (Junike & Pankrashkin 2022) — widens $[a,b]$ iteratively until the tail-mass proxy falls below a user-specified tolerance. Handles stress cases (e.g. CGMY with $Y \to 2$) where the cumulant rule diverges.

## Models

| Family | Models |
|--------|--------|
| Pure diffusion | Black–Scholes–Merton |
| Stochastic volatility | Heston, OU-SV |
| Pure jump / Lévy | Variance Gamma, NIG, CGMY |
| Jump diffusion | Kou double-exponential |
| SV + jumps | Bates, Heston–Kou, Heston–CGMY |

## Installation

```bash
pip install fourier-option-pricer
```

For local development:

```bash
git clone https://github.com/nl2992/fourier-option-pricer.git
cd fourier-option-pricer
pip install -e ".[test]"
pytest
```

## Quick start

```python
import numpy as np
import foureng as fe

# Market inputs
fwd = fe.ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)

# Heston model parameters
params = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)

# Characteristic function
phi = lambda u: fe.heston_cf_form2(u, fwd, params)

# Strike grid
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

# Price with COS (standard truncation)
cumulants = fe.heston_cumulants(fwd, params)
grid = fe.cos_auto_grid(cumulants, N=256, L=10.0)
result = fe.cos_prices(phi, fwd, strikes, grid)
print(result.call_prices)

# Price with Carr–Madan FFT
cm_grid = fe.FFTGrid(N=4096, eta=0.25, alpha=1.5)
cm_prices = fe.carr_madan_price_at_strikes(phi, fwd, cm_grid, strikes)
print(cm_prices)

# Implied volatility
atm_iv = fe.implied_vol_newton_safeguarded(
    price=float(result.call_prices[2]),
    inputs=fe.BSInputs(F0=fwd.F0, K=100.0, T=fwd.T, r=fwd.r, q=fwd.q, is_call=True),
)
print(atm_iv)
```

### Improved COS truncation (Junike rule)

```python
grid = fe.cos_improved_grid(cumulants, model="heston", params=params)
result = fe.cos_prices(phi, fwd, strikes, grid)
```

## Demo notebook

An interactive demo is available at [`notebooks/demo.ipynb`](notebooks/demo.ipynb):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nl2992/fourier-option-pricer/blob/main/notebooks/demo.ipynb)

## API reference

### `ForwardSpec(S0, r, q, T)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `S0` | `float` | Spot price |
| `r` | `float` | Continuously compounded risk-free rate |
| `q` | `float` | Dividend yield or foreign rate |
| `T` | `float` | Time to maturity in years |

Provides `F0` (forward price) and `disc` (discount factor).

### Model parameter classes

| Class | Model |
|-------|-------|
| `HestonParams(kappa, theta, nu, rho, v0)` | Heston stochastic volatility |
| `VGParams(sigma, nu, theta)` | Variance Gamma |
| `KouParams(sigma, lam, p, eta1, eta2)` | Kou double-exponential jump diffusion |
| `BatesParams(...)` | Bates (Heston + Poisson jumps) |
| `CGMYParams(C, G, M, Y)` | CGMY pure-jump Lévy |
| `NIGParams(alpha, beta, delta)` | Normal Inverse Gaussian |

### `cos_prices(phi, fwd, strikes, grid)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `phi` | callable | Characteristic function `phi(u)` |
| `fwd` | `ForwardSpec` | Market inputs |
| `strikes` | `(K,)` array | Strike prices |
| `grid` | `COSGrid` | Truncation grid from `cos_auto_grid` or `cos_improved_grid` |

Returns a `COSResult` with fields `strikes` and `call_prices`.

### `carr_madan_price_at_strikes(phi, fwd, grid, strikes)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `phi` | callable | Characteristic function |
| `fwd` | `ForwardSpec` | Market inputs |
| `grid` | `FFTGrid(N, eta, alpha)` | FFT grid |
| `strikes` | `(K,)` array | Strike prices |

Returns `(K,)` array of call prices.

### `cos_auto_grid(cumulants, N, L)` / `cos_improved_grid(cumulants, model, params)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `cumulants` | cumulant object | From `heston_cumulants`, `vg_cumulants`, etc. |
| `N` | `int` | Number of COS expansion terms |
| `L` | `float` | Truncation multiplier (standard rule only) |
| `model` | `str` | Model name, e.g. `"heston"` (improved rule only) |
| `params` | param dataclass | Model parameters (improved rule only) |

Returns a `COSGrid`.

### `implied_vol_newton_safeguarded(price, inputs)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `price` | `float` | Option price |
| `inputs` | `BSInputs` | `BSInputs(F0, K, T, r, q, is_call)` |

Returns implied volatility as `float`.

## License

MIT. See [LICENSE](LICENSE).
