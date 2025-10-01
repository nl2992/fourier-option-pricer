# fourier-option-pricer

European option pricing with Fourier-transform methods under models specified by a characteristic function.

This package focuses on pricing vanilla European calls across several model families where the transition density may be unavailable in closed form, but the characteristic function is known or easy to evaluate.

## References

> Carr, P., & Madan, D. (1999). Option valuation using the fast Fourier transform.  
> *Journal of Computational Finance*, 2(4), 61–73.  
> https://doi.org/10.21314/JCF.1999.043

> Lewis, A. L. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes.  
> *SSRN Working Paper*.  
> https://ssrn.com/abstract=282110

> Fang, F., & Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions.  
> *SIAM Journal on Scientific Computing*, 31(2), 826–848.  
> https://doi.org/10.1137/080718061

> Junike, G., & Pankrashkin, K. (2022). Precise option pricing by the COS method — how to choose the truncation range.  
> *Applied Mathematics and Computation*, 421, 126935.  
> https://doi.org/10.1016/j.amc.2022.126935

> Ruijter, M. J., Versteegh, M., & Oosterlee, C. W. (2015). On the application of spectral filters in a Fourier option pricing technique.  
> *Journal of Computational Finance*, 19(1), 75–106.  
> https://doi.org/10.21314/JCF.2015.306

PyFENG is used in parts of the project for comparison and model support, particularly for Heston and Variance Gamma characteristic-function workflows.

## Core idea

Let

$$
X_T = \log S_T, \qquad
\phi_X(u) = \mathbb{E}^{\mathbb{Q}}\left[e^{iuX_T}\right].
$$

Many option models have a tractable characteristic function even when the transition density of $S_T$ is not available in closed form. Fourier pricing methods use $\phi_X$ to recover option values through numerical inversion or through a Fourier-series approximation of the risk-neutral density.

The package implements three related approaches:

| Method | Role in the package |
| --- | --- |
| Carr–Madan FFT | Uses a damped call-price transform and the FFT to compute prices on a strike grid. |
| Lewis single-integral | Prices options through a shifted Fourier integral, without the Carr–Madan damping parameter. |
| COS method | Approximates the risk-neutral density by a Fourier-cosine expansion on a finite interval $[a,b]$. |

For Carr–Madan, the damped call transform has the standard form

$$
\psi(u)
= e^{-rT}\,
\frac{\phi_X\left(u - i(\alpha+1)\right)}
{\alpha^2 + \alpha - u^2 + i(2\alpha+1)u},
\qquad \alpha > 0,
$$

and the call price is recovered from

$$
C(K)
= \frac{e^{-\alpha k}}{\pi}
\int_0^\infty
\operatorname{Re}\left(e^{-iuk}\psi(u)\right)\,du,
\qquad k = \log K.
$$

For the COS method, the price is approximated by

$$
V_0 \approx e^{-rT}
\sum_{j=0}^{N-1}{}'
\operatorname{Re}\left[
\phi_X\left(\frac{j\pi}{b-a}\right)
\exp\left(-ij\pi\frac{a}{b-a}\right)
\right]
V_j,
$$

where the prime means the first summand is taken with weight $1/2$, and $V_j$ are the payoff cosine coefficients on $[a,b]$.

## Truncation and filtering

The COS method requires a finite interval $[a,b]$ for the log-price or log-return density. The project supports two truncation policies.

The standard cumulant rule uses the Fang–Oosterlee interval

$$
[a,b]
=
\left[
 c_1 - L\sqrt{c_2 + \sqrt{c_4}},
 c_1 + L\sqrt{c_2 + \sqrt{c_4}}
\right],
$$

where $c_1$, $c_2$, and $c_4$ are cumulants of the log variable and $L$ is a user-chosen width multiplier.

The tolerance rule follows the Junike–Pankrashkin motivation: widen the truncation interval until a tail-error or tail-mass proxy falls below the chosen tolerance. This is useful in cases where a fixed cumulant interval is either too narrow for accuracy or wider than needed for computation.

The package also includes an adaptive filtered-COS layer inspired by Ruijter, Versteegh and Oosterlee. The filter is applied in frequency space to the COS coefficients. It is used as one candidate inside a small deterministic policy search, rather than as a blanket replacement for the unfiltered method.

The current candidate set is:

- plain Junike-COS;
- Fejér-filtered Junike-COS;
- Lanczos-filtered Junike-COS;
- raised-cosine-filtered Junike-COS;
- exponential-filtered Junike-COS.

The selector returns the fastest candidate whose error is below the requested tolerance. If no filtered candidate improves the trade-off, the plain Junike-COS candidate remains available.

## Models

| Family | Models |
| --- | --- |
| Pure diffusion | Black–Scholes–Merton |
| Stochastic volatility | Heston, OU-SV |
| Pure jump / Lévy | Variance Gamma, NIG, CGMY |
| Jump diffusion | Kou double-exponential |
| Stochastic volatility with jumps | Bates, Heston–Kou, Heston–CGMY |

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

The PyPI package is installed as `fourier-option-pricer`; the Python import used in the examples is `foureng`.

## Quick start

```python
import numpy as np
import foureng as fe

# Market inputs
fwd = fe.ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)

# Heston model parameters
params = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)

# Risk-neutral characteristic function
phi = lambda u: fe.heston_cf_form2(u, fwd, params)

# Strike grid
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

# COS pricing with the standard cumulant truncation rule
cumulants = fe.heston_cumulants(fwd, params)
grid = fe.cos_auto_grid(cumulants, N=256, L=10.0)
result = fe.cos_prices(phi, fwd, strikes, grid)
print(result.call_prices)

# Carr–Madan FFT pricing on the same strikes
cm_grid = fe.FFTGrid(N=4096, eta=0.25, alpha=1.5)
cm_prices = fe.carr_madan_price_at_strikes(phi, fwd, cm_grid, strikes)
print(cm_prices)

# Safeguarded Newton implied volatility
atm_iv = fe.implied_vol_newton_safeguarded(
    price=float(result.call_prices[2]),
    inputs=fe.BSInputs(F0=fwd.F0, K=100.0, T=fwd.T, r=fwd.r, q=fwd.q, is_call=True),
)
print(atm_iv)
```

### Improved COS truncation

```python
grid = fe.cos_improved_grid(cumulants, model="heston", params=params)
result = fe.cos_prices(phi, fwd, strikes, grid)
```

### Adaptive filtered-COS extension

The Junike-style policy addresses truncation-range selection. The filtered-COS extension addresses a separate issue: finite-series oscillation caused by truncating the cosine expansion itself. This can matter near payoff kinks, at short maturities, or under jump-heavy densities.

The extension does not assume that filtering is always better. Instead, it compares filtered and unfiltered policies and selects the cheapest candidate that satisfies the target error tolerance.

```python
from foureng.pipeline import price_strip
from foureng.utils.spectral_filters import COSFilterSpec
from foureng.utils.grids import COSGridPolicy

policy = COSGridPolicy(
    mode="benchmark",
    truncation="tolerance",
    centered=True,
    dx_target=0.01,
    L=10.0,
    eps_trunc=1e-10,
    max_N=8192,
    width_fallback=0.0,
)

prices = price_strip(
    "vg",
    "cos_filtered",
    strikes,
    fwd,
    params,
    grid=(policy, COSFilterSpec("exponential", order=8)),
)
```

For the full adaptive selector:

```python
from foureng.experiments.cos_filter_grid_search import (
    default_filtered_cos_candidates,
    run_filtered_cos_grid_search,
    select_fastest_under_tolerance,
)

df = run_filtered_cos_grid_search(
    model="vg",
    strikes=strikes,
    fwd=fwd,
    params=params,
    reference=reference_prices,
    tol=1e-6,
)

best = select_fastest_under_tolerance(df, tol=1e-6)
```

Available filters: `"none"`, `"fejer"`, `"lanczos"`, `"raised_cosine"`, and `"exponential"`.

## Notebooks

| Notebook | Description |
| --- | --- |
| [`notebooks/demo.ipynb`](notebooks/demo.ipynb) | Main walkthrough covering models, pricing methods, implied-volatility surfaces, calibration, Greeks, Monte Carlo comparisons, and the adaptive filtered-COS extension. |
| [`notebooks/adaptive_cos.ipynb`](notebooks/adaptive_cos.ipynb) | Focused comparison of vanilla COS, Junike-style COS, and adaptive filtered-COS on Fang–Oosterlee-style test cases. |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nl2992/fourier-option-pricer/blob/main/notebooks/demo.ipynb)

## API reference

### `ForwardSpec(S0, r, q, T)`

| Parameter | Type | Description |
| --- | --- | --- |
| `S0` | `float` | Spot price. |
| `r` | `float` | Continuously compounded risk-free rate. |
| `q` | `float` | Continuously compounded dividend yield or foreign rate. |
| `T` | `float` | Time to maturity, in years. |

Provides `F0` for the forward price and `disc` for the discount factor.

### Model parameter classes

| Class | Model |
| --- | --- |
| `HestonParams(kappa, theta, nu, rho, v0)` | Heston stochastic volatility |
| `VGParams(sigma, nu, theta)` | Variance Gamma |
| `KouParams(sigma, lam, p, eta1, eta2)` | Kou double-exponential jump diffusion |
| `BatesParams(...)` | Bates, i.e. Heston with Poisson jumps |
| `CGMYParams(C, G, M, Y)` | CGMY pure-jump Lévy process |
| `NIGParams(alpha, beta, delta)` | Normal Inverse Gaussian |

### `cos_prices(phi, fwd, strikes, grid)`

| Parameter | Type | Description |
| --- | --- | --- |
| `phi` | callable | Risk-neutral characteristic function. |
| `fwd` | `ForwardSpec` | Market inputs. |
| `strikes` | `(K,)` array | Strike prices. |
| `grid` | `COSGrid` | Truncation grid from `cos_auto_grid` or `cos_improved_grid`. |

Returns a `COSResult` with fields `strikes` and `call_prices`.

### `carr_madan_price_at_strikes(phi, fwd, grid, strikes)`

| Parameter | Type | Description |
| --- | --- | --- |
| `phi` | callable | Risk-neutral characteristic function. |
| `fwd` | `ForwardSpec` | Market inputs. |
| `grid` | `FFTGrid(N, eta, alpha)` | Carr–Madan FFT grid. |
| `strikes` | `(K,)` array | Strike prices. |

Returns a one-dimensional array of call prices.

### `cos_auto_grid(cumulants, N, L)` and `cos_improved_grid(cumulants, model, params)`

| Parameter | Type | Description |
| --- | --- | --- |
| `cumulants` | cumulant object | Output from a model-specific cumulant function such as `heston_cumulants` or `vg_cumulants`. |
| `N` | `int` | Number of COS expansion terms. |
| `L` | `float` | Truncation multiplier for the standard cumulant rule. |
| `model` | `str` | Model name, for example `"heston"`. |
| `params` | parameter dataclass | Model parameters used by the improved grid policy. |

Returns a `COSGrid`.

### `COSGridPolicy`

Dataclass controlling truncation-interval selection and adaptive choice of the number of COS terms.

| Key parameter | Default | Description |
| --- | --- | --- |
| `truncation` | `"tolerance"` | `"heuristic"` for the Fang–Oosterlee cumulant rule, `"tolerance"` for iterative widening, or `"paper"` for paper-style settings. |
| `eps_trunc` | `1e-10` | Tail threshold for the tolerance rule. |
| `dx_target` | model default | Target spatial resolution, approximately `(b-a)/N`. |
| `fixed_N` | `None` | Optional hard override for the number of COS terms. |
| `mode` | `"benchmark"` | `"benchmark"` for tighter numerical settings; `"surface"` for faster surface generation. |
| `max_N` | `16384` | Upper cap on the adaptive number of COS terms. |

### `recommended_cos_policy(model, params, *, mode)`

Returns the recommended `COSGridPolicy` for a model string such as `"heston"`, `"vg"`, or `"kou"`. This is intended to provide stable defaults while still allowing manual overrides.

### `filtered_cos_prices(phi, fwd, strikes, grid, *, filter_spec)`

COS pricer with a spectral filter applied to the COS coefficient sequence before pricing.

| Parameter | Type | Description |
| --- | --- | --- |
| `phi` | callable | Risk-neutral characteristic function. |
| `fwd` | `ForwardSpec` | Market inputs. |
| `strikes` | `(K,)` array | Strike prices. |
| `grid` | `COSGrid` | Resolved COS grid. |
| `filter_spec` | `COSFilterSpec` | Filter specification, for example exponential order 8. |

Returns a `COSResult`.

### `COSFilterSpec(name, order, alpha)`

| `name` value | Description |
| --- | --- |
| `"none"` | Identity filter. |
| `"fejer"` | Fejér averaging filter. |
| `"lanczos"` | Lanczos sinc filter. |
| `"raised_cosine"` | Raised-cosine window. |
| `"exponential"` | Exponential filter with tunable order. |

### `implied_vol_newton_safeguarded(price, inputs)`

| Parameter | Type | Description |
| --- | --- | --- |
| `price` | `float` | Option price. |
| `inputs` | `BSInputs` | Black–Scholes inputs: `BSInputs(F0, K, T, r, q, is_call)`. |

Returns the implied volatility as a `float`.

## Extended methodology and results

Detailed numerical experiments, replication notes, benchmark tables, and implementation commentary are kept outside the README to keep this page readable.

See [`methodology_and_results.md`](methodology_and_results.md) for:

- Fang–Oosterlee COS replication notes;
- Carr–Madan benchmark setup;
- Monte Carlo comparison setup;
- truncation-interval diagnostics;
- improved COS grid logic;
- runtime and error reporting rules;
- model-by-model observations;
- known numerical limitations;
- adaptive filtered-COS experiments.

## License

MIT. See [LICENSE](LICENSE).
