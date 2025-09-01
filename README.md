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

> Ruijter, M. J., Versteegh, M., & Oosterlee, C. W. (2015). On the application of
> spectral filters in a Fourier option pricing technique.
> *Journal of Computational Finance*, 19(1), 75–106.
> https://doi.org/10.21314/JCF.2015.306
> *(Used as inspiration for the spectral-filtering layer in the adaptive
> filtered-COS extension. The project adapts this idea into a tolerance-driven
> selector over no-filter and filtered COS policies, rather than forcing a single
> fixed filter.)*



## Core concept

Fourier pricing exploits the fact that, for most asset models, the **characteristic function**

$$\phi(u) = \mathbb{E}\\left[e^{iu \ln S_T}\right]$$

is known in closed form even when the option price integral has no analytic solution.
Given $\phi$, a European call can be priced by a single numerical integral.
The three methods implemented here differ in how they discretise that integral:

| Method | Key idea |
|--------|----------|
| Carr–Madan FFT | Damp the payoff, apply FFT to price a whole strike grid at once |
| Lewis single-integral | Parseval identity; avoids the dampening parameter entirely |
| COS (Fang–Oosterlee) | Expand the risk-neutral density in a cosine series on $[a, b]$ |

## Truncation and filtering

The COS method requires choosing a truncation interval $[a,b]$ for the log-price
density. Two truncation strategies are implemented:

- **Cumulant rule** (Fang & Oosterlee 2008) — sets $[a,b]$ from the first four
  cumulants of $\ln S_T$.
- **Tolerance rule** (Junike & Pankrashkin 2022) — widens $[a,b]$ iteratively
  until the tail-mass proxy falls below a user-specified tolerance. This is used
  for stress cases where the cumulant rule can become unreliable or overly wide.

In addition, the project implements an **adaptive filtered-COS extension** inspired by
Ruijter, Versteegh and Oosterlee (2015). Rather than applying one fixed spectral filter,
the idea is adapted into a tolerance-driven policy selector: five candidates
(plain Junike-COS and four filtered variants) are evaluated; the fastest that
meets a target error tolerance is returned.

- **Why filtering is added:** truncation fixes the interval, but COS can still
  show finite-series ringing near payoff kinks, short maturities, or jump-heavy
  densities.
- **Candidate filters:** Fejér, Lanczos, raised-cosine, and exponential; plain
  Junike-COS is always kept in the pool so the selector can fall back when
  filtering does not help.
- **Selection rule:** use the cheapest candidate that meets the tolerance
  (default `ε = 1 × 10⁻⁶`). Filter selection is entirely deterministic —
  no tuning required.


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

### Extension: adaptive filtered-COS policy layer

The Junike-style COS policy improves truncation-range selection, but COS accuracy
is controlled by both the truncation interval and the finite cosine expansion.
As an extension, this repo adds an **adaptive filtered-COS overlay** inspired by
Ruijter, Versteegh and Oosterlee's spectral-filtering work for Fourier option
pricing.

The filter is applied to the COS expansion coefficients and is tested as one
candidate inside a **deterministic numerical-policy search**.  The adaptive
selector compares vanilla COS, Junike-COS, and filtered Junike-COS, then selects
the fastest candidate satisfying a target error tolerance.

> **This extension does not claim that filtered-COS universally dominates Junike-COS.**
> The intended contribution is the adaptive selector, which can choose no filter where
> filtering is unnecessary.

**Usage:**

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

For the full adaptive grid-search selector (comparing vanilla, Junike, and
filtered candidates):

```python
from foureng.experiments.cos_filter_grid_search import (
    default_filtered_cos_candidates,
    run_filtered_cos_grid_search,
    select_fastest_under_tolerance,
)

df = run_filtered_cos_grid_search(
    model="vg", strikes=strikes, fwd=fwd, params=params,
    reference=reference_prices, tol=1e-6,
)
best = select_fastest_under_tolerance(df, tol=1e-6)
```

**References:**
- Junike, G. and Pankrashkin, K. (2022), "Precise option pricing by the COS
  method — How to choose the truncation range," *Applied Mathematics and
  Computation*, 421, 126935.
- Ruijter, M. J., Versteegh, M. and Oosterlee, C. W. (2015), "On the application
  of spectral filters in a Fourier option pricing technique," *Journal of
  Computational Finance*.
- Junike, G. (2024), "On the number of terms in the COS method for European
  option pricing," *Numerische Mathematik*.

**Available spectral filters:** `"none"` (identity), `"fejer"`, `"lanczos"`,
`"raised_cosine"`, `"exponential"` (order-*p* tunable).

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`notebooks/demo.ipynb`](notebooks/demo.ipynb) | Full walkthrough: models, pricers, IV surface, calibration, Greeks, MC, and the adaptive filtered-COS extension |
| [`notebooks/adaptive_cos.ipynb`](notebooks/adaptive_cos.ipynb) | Standalone comparison of vanilla COS, Junike-COS, and adaptive filtered-COS on the canonical Fang–Oosterlee (2008) test cases |

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

### `COSGridPolicy`

Dataclass that controls the adaptive truncation-interval and N-selection strategy
used by `cos_improved_grid` and `filtered_cos_prices`.

| Key parameter | Default | Description |
|---------------|---------|-------------|
| `truncation` | `"tolerance"` | `"heuristic"` (Fang–Oosterlee L rule), `"tolerance"` (Junike iterative widening), or `"paper"` |
| `eps_trunc` | `1e-10` | Tail-mass threshold for the tolerance rule |
| `dx_target` | model default | Target spatial resolution `(b−a)/N`; drives adaptive N selection |
| `fixed_N` | `None` | Hard override for N (bypasses adaptive selection) |
| `mode` | `"benchmark"` | `"benchmark"` for tighter accuracy, `"surface"` for speed |
| `max_N` | `16384` | Upper cap on adaptively chosen N |

### `recommended_cos_policy(model, params, *, mode)`

Returns the recommended `COSGridPolicy` for a given model string (e.g. `"heston"`,
`"vg"`, `"kou"`) and parameter object. Provides sensible defaults without
manual tuning.

### `filtered_cos_prices(phi, fwd, strikes, grid, *, filter_spec)`

COS pricer with a spectral filter applied to the characteristic-function samples
before cosine inversion. Reduces finite-series oscillations (Gibbs-like ringing)
near payoff kinks, short maturities, or jump-heavy densities.

| Parameter | Type | Description |
|-----------|------|-------------|
| `phi` | callable | Characteristic function `phi(u)` |
| `fwd` | `ForwardSpec` | Market inputs |
| `strikes` | `(K,)` array | Strike prices |
| `grid` | `COSGrid` | Resolved grid from `cos_improved_grid` |
| `filter_spec` | `COSFilterSpec` | Filter to apply (default: exponential, order 8) |

Returns a `COSResult`.

### `COSFilterSpec(name, order, alpha)`

Spectral filter specification passed to `filtered_cos_prices`.

| `name` value | Description |
|--------------|-------------|
| `"none"` | Identity — no filtering (plain COS) |
| `"fejer"` | Fejér averaging kernel |
| `"lanczos"` | Lanczos (sinc) filter |
| `"raised_cosine"` | Raised-cosine (Hann) window |
| `"exponential"` | Order-*p* exponential; `order=8` recommended |

### `implied_vol_newton_safeguarded(price, inputs)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `price` | `float` | Option price |
| `inputs` | `BSInputs` | `BSInputs(F0, K, T, r, q, is_call)` |

Returns implied volatility as `float`.

## Extended methodology and results

Detailed numerical experiments, replication notes, runtime benchmarks, and implementation commentary are kept outside the README to keep this page concise.

See [`methodology_and_results.md`](methodology_and_results.md).

This document records:

- the Fang-Oosterlee COS replication workflow;
- the Carr-Madan benchmark setup;
- the Monte Carlo comparison setup;
- COS truncation-interval behaviour;
- improved COS grid logic;
- runtime and error reporting rules;
- model-by-model observations;
- known numerical limitations;
- adaptive filtered-COS extension (spectral filters + policy grid-search selector).

## License

MIT. See [LICENSE](LICENSE).

