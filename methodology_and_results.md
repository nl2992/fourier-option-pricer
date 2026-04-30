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
Given $\phi$, a European option can be priced by numerical inversion or by a
series approximation of the risk-neutral density. The pricing layer is organized
around the following methods:

| Method | Main idea | Best use |
|--------|-----------|----------|
| Carr–Madan FFT | Damped call transform plus FFT | Many strikes quickly on a regular log-strike grid |
| FRFT | Carr–Madan with flexible log-strike spacing | Many strikes when the FFT grid is too restrictive |
| Lewis single-integral | Shifted Fourier integral without a damping parameter | Few strikes or robust fallback pricing |
| COS (Fang–Oosterlee) | Cosine expansion of the density on $[a,b]$ | High accuracy when the truncation interval is good |
| Improved COS | Adaptive interval and $N$ selection | Safer COS automation across model families |
| Filtered COS | COS with high-frequency damping | Jump-heavy, short-maturity, or oscillatory cases |
| PyFENG FFT | External PyFENG pricer | Benchmarking and model support for known cases |

This separation is useful computationally: FFT/FRFT methods amortize work over
many strikes, Lewis is a dependable sparse-strike fallback, and the COS family
is the main high-accuracy engine when the interval and series resolution are
well controlled.

## COS workflow, truncation, and filtering

The COS method approximates the risk-neutral density with a finite cosine
series. In implementation terms, the workflow is:

```text
model parameters
      ↓
build characteristic function φ(u)
      ↓
compute cumulants c1, c2, c4
      ↓
choose truncation interval [a,b]
      ↓
choose number of COS terms N
      ↓
evaluate φ at u_j = jπ/(b-a)
      ↓
compute model-side and payoff-side coefficients
      ↓
optionally apply spectral filter weights
      ↓
discounted dot product gives the option price
```

The key numerical choices are the truncation interval, the number of terms, and
whether to filter high-frequency terms. Two truncation strategies are
implemented:

- **Cumulant rule** (Fang & Oosterlee 2008) — sets $[a,b]$ from the first four
  cumulants of $\ln S_T$.
- **Tolerance rule** (Junike & Pankrashkin 2022) — widens $[a,b]$ iteratively
  until the tail-mass proxy falls below a user-specified tolerance. This is used
  for stress cases where the cumulant rule can become unreliable or overly wide.

For vanilla calls, the implementation often prices the put coefficients first
and recovers the call by put-call parity,

$$
C = P + e^{-rT}(F_0 - K).
$$

This avoids direct-call coefficient cancellation on wide intervals, where
exponential payoff terms can become numerically large.

Improved COS keeps the COS pricing formula but automates the interval and
resolution choices. It classifies the model tail behavior, widens the interval
until a tail proxy is acceptable, selects $N$ from the interval width, and can
fall back to Lewis or Carr–Madan when the interval becomes too wide for COS to
be efficient.

| Tail family | Interpretation | Example models |
|-------------|----------------|----------------|
| Gaussian-like | Thin tails | BSM, Heston, OU-SV, NIG |
| Semi-heavy | Jump or exponential-type tails | Kou, Bates, Heston-Kou |
| Heavy | Slower-decaying tails | VG, some CGMY regimes |

The project also implements an **adaptive filtered-COS extension** inspired by
Ruijter, Versteegh and Oosterlee (2015). Rather than applying one fixed spectral
filter, the repo treats filters as candidates inside a tolerance-driven selector
around COS pricing.

- **Why filtering is added:** truncation fixes the interval, but COS can still
  show finite-series ringing near payoff kinks, short maturities, or jump-heavy
  densities.
- **Candidate filters:** the selector keeps plain Junike-COS in the pool and
  also tests Fejér, Lanczos, raised-cosine, and exponential filters.
- **Selection rule:** use the cheapest candidate that meets the tolerance. If a
  filter does not help, the method falls back to the no-filter Junike choice.

The implemented filter weights $\sigma_j$ satisfy $0 \leq \sigma_j \leq 1$ and
multiply the COS terms before the final dot product:

$$
V_0 \approx e^{-rT}\sum_{j=0}^{N-1}\sigma_j A_j U_j.
$$

| Filter | Main idea | Practical effect |
|--------|-----------|------------------|
| `none` | No damping | Baseline COS |
| `fejer` | Linear high-frequency reduction | Simple smoothing |
| `lanczos` | Sinc-shaped damping | Gentle smoothing |
| `raised_cosine` | Smooth cosine taper | Smooth fade-out |
| `exponential` | Strong damping near the highest frequencies | Useful default for jump-heavy cases |

The adaptive filtered-COS selector is therefore not a claim that filtering
always dominates. It is a policy layer that can choose the unfiltered Junike
setup when that is faster or more accurate.


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

> **Important:** filtered-COS is not assumed to universally dominate Junike-COS.
> The contribution is the adaptive selector, which can choose no filter when
> filtering is unnecessary or slower.

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
- known numerical limitations;
- adaptive filtered-COS extension (spectral filters + policy grid-search selector).

## License

MIT. See [LICENSE](LICENSE).
