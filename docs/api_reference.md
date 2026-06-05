# API Reference

Complete public API for `foureng`. All objects are importable as `import foureng as fe`.

```python
import foureng as fe
# or for the unified dispatcher only:
from foureng.pipeline import price, price_strip
```

---

## Market inputs and model parameters

| Object | Parameters | Purpose |
|--------|------------|---------|
| `ForwardSpec(S0, r, q, T)` | `S0: float`, `r: float`, `q: float`, `T: float` | Market inputs. Provides derived `F0` and discount factor `disc`. |
| `BsmParams(sigma)` | Black-Scholes volatility | Diffusion baseline model dataclass. |
| `HestonParams(kappa, theta, nu, rho, v0)` | Heston stochastic-volatility parameters | Heston model parameter dataclass. |
| `OusvParams(sigma0, kappa, theta, nu, rho)` | Schobel-Zhu / OUSV parameters | OUSV model parameter dataclass. |
| `VGParams(sigma, nu, theta)` | Variance Gamma parameters | VG parameter dataclass. |
| `CgmyParams(C, G, M, Y)` | CGMY Lévy parameters | CGMY model parameter dataclass. |
| `NigParams(sigma, nu, theta)` | NIG Lévy parameters | NIG model parameter dataclass. |
| `KouParams(sigma, lam, p, eta1, eta2)` | Diffusion plus jump parameters | Kou double-exponential jump-diffusion dataclass. |
| `BatesParams(kappa, theta, nu, rho, v0, lam_j, mu_j, sigma_j)` | Heston block plus Merton jump parameters | Bates SVJ dataclass. |
| `HestonKouParams(kappa, theta, nu, rho, v0, lam_j, p_j, eta1, eta2)` | Heston block plus Kou jump parameters | Heston-Kou composite dataclass. |
| `HestonCGMYParams(kappa, theta, nu, rho, v0, C, G, M, Y)` | Heston block plus CGMY jump parameters | Heston-CGMY composite dataclass. |
| `Sv32Params(v0, kappa, theta, nu, rho)` | 3/2 model parameters | 3/2 SV parameter dataclass. |
| `GarchWMW2012Params(v0, kappa, theta, nu, rho)` | GARCH diffusion parameters | Wu-Ma-Wang (2012) GARCH option pricing dataclass. |
| `RoughHestonParams(sigma, vov, mr, rho, theta, alpha)` | Rough Heston parameters | `alpha` in `(0, 1)` (fractional exponent). |
| `MertonJDParams(sigma, lam, mu_j, sigma_j)` | Diffusion volatility plus jump parameters | Merton jump-diffusion dataclass. |
| `MeixnerParams(a, b, delta)` | Meixner Lévy parameters | Meixner process dataclass. |
| `BilateralGammaParams(alpha_p, lambda_p, alpha_m, lambda_m)` | Bilateral Gamma parameters | Separate up/down Gamma processes. |
| `GHParams(lam, alpha, beta, delta)` | Generalised Hyperbolic parameters | `lam=-0.5` gives NIG, `lam=1` gives Hyperbolic. |
| `FMLSParams(alpha, sigma)` | Stability index and scale | `alpha` in `(1, 2]`, recovers BSM at `alpha=2`. |
| `DoubleHestonParams(kappa1, theta1, nu1, rho1, v01, kappa2, theta2, nu2, rho2, v02)` | Two independent Heston variance-factor sets | CF factorises as product of two single-Heston CFs. |
| `VGSAParams(C, G, M, kappa, eta, lam)` | VG tempering rates plus CIR activity-clock parameters | `lam=0` reduces to standard VG. |
| `SabrParams(alpha, beta, rho, nu)` | SABR approximation parameters | Used by Hagan lognormal implied-vol pricing. |

---

## Characteristic functions and cumulants

### Characteristic functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `bsm_cf` | `(u, fwd, params)` | Black-Scholes CF in log-forward coordinates. |
| `heston_cf_form2` | `(u, fwd, params)` | Heston CF, stable Formulation 2 (Little Heston Trap). |
| `ousv_cf` | `(u, fwd, params)` | OUSV / Schobel-Zhu CF. |
| `vg_cf` | `(u, fwd, params)` | Variance Gamma CF. |
| `cgmy_cf` | `(u, fwd, params)` | CGMY CF. |
| `nig_cf` | `(u, fwd, params)` | Normal Inverse Gaussian CF. |
| `kou_cf` | `(u, fwd, params)` | Kou double-exponential jump-diffusion CF. |
| `bates_cf` | `(u, fwd, params)` | Bates CF (Heston × Merton-jump block). |
| `heston_kou_cf` | `(u, fwd, params)` | Heston-Kou CF. |
| `heston_cgmy_cf` | `(u, fwd, params)` | Heston-CGMY CF. |
| `sv32_cf` | `(u, fwd, params)` | 3/2 SV CF. |
| `garch_wmw2012_cf` | `(u, fwd, params)` | GARCH CF (Wu-Ma-Wang 2012). |
| `rough_heston_cf` | `(u, fwd, params)` | Rough Heston CF via Adams scheme. |
| `merton_jd_cf` | `(u, fwd, params)` | Merton jump-diffusion CF. |
| `meixner_cf` | `(u, fwd, params)` | Meixner CF. |
| `bilateral_gamma_cf` | `(u, fwd, params)` | Bilateral Gamma CF. |
| `gh_cf` | `(u, fwd, params)` | Generalised Hyperbolic CF (uses Bessel K). |
| `fmls_cf` | `(u, fwd, params)` | FMLS CF via principal branch of `(iu)^alpha`. |
| `double_heston_cf` | `(u, fwd, params)` | Double Heston CF; product of two single-Heston CFs. |
| `vgsa_cf` | `(u, fwd, params)` | VGSA CF via CIR Laplace transform of the VG Lévy exponent. |

### Cumulants

| Function | Parameters | Notes |
|----------|------------|-------|
| `bsm_cumulants(fwd, params)` | `ForwardSpec`, `BsmParams` | Black-Scholes cumulants for COS grid construction. |
| `heston_cumulants(fwd, params)` | `ForwardSpec`, `HestonParams` | Heston cumulants for COS truncation intervals. |
| `ousv_cumulants(fwd, params)` | `ForwardSpec`, `OusvParams` | OUSV cumulants. |
| `vg_cumulants(fwd, params)` | `ForwardSpec`, `VGParams` | VG cumulants. |
| `cgmy_cumulants(fwd, params)` | `ForwardSpec`, `CgmyParams` | CGMY cumulants. |
| `nig_cumulants(fwd, params)` | `ForwardSpec`, `NigParams` | NIG cumulants. |
| `kou_cumulants(fwd, params)` | `ForwardSpec`, `KouParams` | Kou cumulants. |
| `bates_cumulants(fwd, params)` | `ForwardSpec`, `BatesParams` | Bates cumulants. |
| `heston_kou_cumulants(fwd, params)` | `ForwardSpec`, `HestonKouParams` | Heston-Kou cumulants. |
| `heston_cgmy_cumulants(fwd, params)` | `ForwardSpec`, `HestonCGMYParams` | Heston-CGMY cumulants. |
| `sv32_cumulants(fwd, params)` | `ForwardSpec`, `Sv32Params` | 3/2 model cumulants. |
| `garch_wmw2012_cumulants(fwd, params)` | `ForwardSpec`, `GarchWMW2012Params` | GARCH cumulants. |
| `rough_heston_cumulants(fwd, params)` | `ForwardSpec`, `RoughHestonParams` | Rough Heston cumulants. |
| `merton_jd_cumulants(fwd, params)` | `ForwardSpec`, `MertonJDParams` | Merton JD cumulants. |
| `meixner_cumulants(fwd, params)` | `ForwardSpec`, `MeixnerParams` | Meixner cumulants. |
| `bilateral_gamma_cumulants(fwd, params)` | `ForwardSpec`, `BilateralGammaParams` | Bilateral Gamma cumulants (closed form). |
| `gh_cumulants(fwd, params)` | `ForwardSpec`, `GHParams` | GH cumulants. |
| `fmls_cumulants(fwd, params)` | `ForwardSpec`, `FMLSParams` | FMLS cumulants via numerical Cauchy integration. **Note:** COS is not recommended for α<2 (power-law tails); prefer Carr-Madan or FRFT. |
| `double_heston_cumulants(fwd, params)` | `ForwardSpec`, `DoubleHestonParams` | Sum of the two single-factor Heston cumulants. |
| `vgsa_cumulants(fwd, params)` | `ForwardSpec`, `VGSAParams` | VGSA cumulants via CIR moment formulas. |

---

## Grid objects and grid builders

| Object | Parameters | Purpose |
|--------|------------|---------|
| `COSGrid(a, b, N)` | truncation interval and term count | Concrete COS grid used by `cos_prices`. |
| `COSGridPolicy(...)` | `mode`, `truncation`, `dx_target`, `L`, `eps_trunc` | Rule-based COS grid specification for improved / filtered COS paths. |
| `FFTGrid(N, eta, alpha)` | FFT size, frequency spacing, damping parameter | Carr-Madan FFT grid. |
| `FRFTGrid(N, eta, lam, alpha)` | FRFT size, spacing, strike step, damping parameter | Fractional FFT grid. |
| `CONVGrid(N, u_max)` | positive-frequency count and integration cutoff | CONV-style Fourier inversion grid. |
| `LatticeGrid(steps, scheme="crr")` | binomial step count | BSM Cox-Ross-Rubinstein tree grid. |
| `PDEGrid(spot_steps, time_steps, s_max_mult)` | finite-difference controls | BSM implicit finite-difference grid. |
| `cos_auto_grid(cumulants, N, L)` | cumulants, term count, truncation multiplier | Returns a `COSGrid` from the standard cumulant rule. |
| `cos_improved_grid(cumulants, model=..., params=...)` | cumulants plus model context | Returns a `COSGrid` using the improved COS truncation policy. |
| `recommended_cos_policy(model, params, mode=...)` | model name and parameter dataclass | Returns a `COSGridPolicy` for the improved COS workflow. |

---

## Core pricing functions

| Function | Signature | Returns |
|----------|-----------|---------|
| `cos_prices(phi, fwd, strikes, grid)` | CF, `ForwardSpec`, strike array, `COSGrid` | `COSResult` with `strikes` and `call_prices`. |
| `carr_madan_price_at_strikes(phi, fwd, grid, strikes)` | CF, `ForwardSpec`, `FFTGrid`, strike array | NumPy array of call prices. |
| `frft_price_at_strikes(phi, fwd, grid, strikes)` | CF, `ForwardSpec`, `FRFTGrid`, strike array | NumPy array of call prices. |
| `conv_price_at_strikes(phi, fwd, grid, strikes, cp=...)` | CF, `ForwardSpec`, `CONVGrid`, strike array | NumPy array of call or put prices. |
| `cos_bermudan_price(model, fwd, params, product, grid=...)` | model key, market inputs, params, `BermudanOption` | Scalar Bermudan COS price for supported 1-D Levy models. |
| `cos_bermudan_price_strip(model, fwd, params, strikes, maturity, exercise_times, cp=...)` | model key, market inputs, params, strike array | Bermudan COS strip for supported 1-D Levy models. |
| `filtered_cos_prices(phi, fwd, strikes, grid, filter_spec=...)` | CF, `ForwardSpec`, strike array, grid, filter | `COSResult` with spectral filtering applied. |
| `bsm_lattice_price_at_strikes(fwd, params, strikes, cp=..., exercise=...)` | `ForwardSpec`, `BsmParams`, strike array | BSM European/American CRR tree prices. |
| `bsm_pde_fd_price_at_strikes(fwd, params, strikes, cp=..., exercise=...)` | `ForwardSpec`, `BsmParams`, strike array | BSM European/American finite-difference prices. |
| `bsm_barrier_price(S, K, H, r, q, T, sigma, barrier_type, cp=...)` | BSM single-barrier inputs | Closed-form continuous zero-rebate barrier price. |
| `bsm_discrete_geometric_asian(S, K, r, q, monitoring_times, sigma, cp=...)` | BSM geometric-average inputs | Closed-form discretely monitored geometric Asian price. |
| `bsm_forward_start(S, alpha, t_start, T, r, q, sigma, cp=...)` | BSM forward-start inputs | Closed-form forward-start price. |
| `bsm_lookback_floating(S, S_min, S_max, r, q, T, sigma, cp=...)` | BSM floating-lookback inputs | Closed-form continuous floating-strike lookback price. |
| `proj_european_price_at_strikes(phi, fwd, cumulants, strikes, cp=...)` | CF, market inputs, cumulants, strikes | First-slice European PROJ façade using COS projection coefficients. |
| `mellin_price_at_strikes(phi, fwd, strikes, cp=...)` | CF, market inputs, strikes | First-slice European Mellin façade for selected Lévy models. |
| `sabr_hagan_price_at_strikes(fwd, params, strikes, cp=...)` | `ForwardSpec`, `SabrParams`, strikes | Hagan SABR implied-vol prices. |
| `price_strip(model, method, strikes, fwd, params, grid=None, ...)` | model label, method label, strike array, market inputs, params | Unified dispatcher  -  returns NumPy array of call prices. |
| `price(product, model, method, fwd, params, grid=None)` | product dataclass, model label, method label, market inputs, params | Product-aware dispatcher for scalar contracts. |

### `price_strip` method labels

| Label | Pricer |
|-------|--------|
| `"cos"` | Plain COS |
| `"cos_improved"` | COS with Junike-style truncation policy |
| `"carr_madan"` | Carr-Madan FFT |
| `"frft"` | Fractional FFT |
| `"conv"` | CONV-style Fourier probability inversion |
| `"cos_bermudan"` | COS Bermudan backward induction for supported 1-D Levy models |
| `"mellin"` | First-slice Mellin European façade for VG, NIG, FMLS, bilateral gamma |
| `"proj"` | First-slice European PROJ façade using COS projection coefficients |
| `"cos_filtered"` | COS with spectral damping (Fejér, Lanczos, raised-cosine, or exponential filter) |
| `"pyfeng_fft"` | PyFENG-backed Lewis-style FFT path (PyFENG-backed models only) |
| `"lattice"` | BSM-only CRR lattice for European strips |
| `"pde_fd"` | BSM-only implicit finite-difference solver for European strips |
| `"barrier_bsm"` | BSM-only analytic single-barrier pricing through `price()` |
| `"forward_start_bsm"` | BSM-only analytic forward-start pricing through `price()` |
| `"lookback_bsm"` | BSM-only analytic continuous floating-strike lookback pricing through `price()` |
| `"lookback_mc"` | BSM-only Monte Carlo lookback pricing through `price()` |
| `"variance_mc"` | BSM-only Monte Carlo variance swap / variance option pricing through `price()` |
| `"cliquet_mc"` | BSM-only Monte Carlo cliquet pricing through `price()` |
| `"sabr_hagan"` | SABR-only Hagan implied-vol approximation for European strips |

### Product-level dispatcher

| Product | Supported method(s) | Scope |
|---------|---------------------|-------|
| `EuropeanOption` | all compatible `price_strip` methods | Vanilla call/put. |
| `AmericanOption` | `"lattice"`, `"pde_fd"` | BSM call/put. |
| `BermudanOption` | `"cos_bermudan"` | Supported 1-D Levy call/put with discrete exercise dates. |
| `BarrierOption` | `"barrier_bsm"` | Continuously monitored, zero-rebate, single-barrier BSM knock-in/knock-out call/put. |
| `AsianOption` | `"asian_bsm"`, `"asian_mc"` | BSM fixed-strike geometric closed form or BSM arithmetic/geometric Monte Carlo. |
| `ForwardStartOption` | `"forward_start_bsm"` | BSM forward-start call/put with strike set at `alpha * S_{t_start}`. |
| `LookbackOption` | `"lookback_bsm"`, `"lookback_mc"` | BSM continuous floating-strike closed form or BSM Monte Carlo fixed/floating lookback. |
| `VarianceSwap` | `"variance_mc"` | BSM Monte Carlo realized-variance swap. |
| `VarianceOption` | `"variance_mc"` | BSM Monte Carlo realized/integrated variance option. |
| `CliquetOption` | `"cliquet_mc"` | BSM Monte Carlo additive or multiplicative cliquet. |
| `DoubleBarrierOption` | `"double_barrier_mc"` | BSM zero-rebate double knock-in/knock-out Monte Carlo. |

---

## Filtered-COS helpers

| Object | Parameters | Purpose |
|--------|------------|---------|
| `COSFilterSpec(name, order=..., alpha=...)` | filter family and optional shape parameters | Filter specification for the filtered COS method. |
| `cos_filter_weights(N, filter_spec)` | term count and filter spec | NumPy array of spectral weights `sigma_k`. |
| `cos_adaptive_decision(...)` | model context and COS policy inputs | Returns `COSPolicyDecision` summarizing the improved COS grid choice. |

---

## Implied volatility

| Object | Parameters | Purpose |
|--------|------------|---------|
| `BSInputs(F0, K, T, r, q, is_call)` | Black-style inversion inputs | Dataclass for implied-vol routines. |
| `bs_price_from_fwd(sigma, inputs)` | volatility and `BSInputs` | Black-Scholes price from forward inputs. |
| `implied_vol_newton_safeguarded(price, inputs)` | option price and `BSInputs` | Implied vol via safeguarded Newton. |
| `implied_vol_brent(price, inputs)` | option price and `BSInputs` | Implied vol via bracketing solver. |

---

## Surfaces, calibration, and Greeks

| Object | Parameters | Purpose |
|--------|------------|---------|
| `SurfaceSpec(S0, r, q, maturities, strikes)` | market inputs plus maturity/strike grids | Surface input container. |
| `model_price_surface(...)` | surface spec plus pricing callbacks | Price surface over maturity and strike grids. |
| `model_iv_surface(...)` | surface spec plus pricing callbacks | Implied-volatility surface. |
| `calibrate_heston(...)`, `calibrate_vg(...)`, `calibrate_kou(...)` | market targets, grid inputs, initial guesses | Return `CalibrationResult` for the chosen model. |
| `cos_price_and_greeks(phi, fwd, strikes, grid)` | CF, market inputs, strike array, grid | `COSGreeks` with prices and sensitivity arrays. |
| `cos_delta_gamma(phi, fwd, strikes, grid)` | CF, market inputs, strike array, grid | Delta and gamma arrays. |
| `cos_parameter_sensitivity(...)` | model setup plus parameter perturbation inputs | COS-based parameter sensitivities. |
