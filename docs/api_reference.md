# API Reference

All public names are accessible from the top-level `foureng` namespace after `import foureng`.

---

## Version

| Name | Type | Description |
|------|------|-------------|
| `__version__` | str | Package version string, e.g. `"0.5.0"`. |

---

## Base types

| Class | Purpose |
|-------|---------|
| `BSInputs` | Black-Scholes forward inputs: `S0`, `r`, `q`, `T`, `K`, `sigma`. |
| `CharFunc` | Protocol for characteristic-function callables. |
| `ForwardSpec` | Forward-measure specification: `S0`, `r`, `q`, `T`. |
| `ModelSpec` | Tagged union of all model parameter dataclasses. |

---

## Model parameter dataclasses

Each dataclass holds the calibrated parameters for one stochastic-volatility or jump model.

| Dataclass | Model |
|-----------|-------|
| `BsmParams` | Black-Scholes-Merton |
| `HestonParams` | Heston (1993) |
| `OusvParams` | Ornstein-Uhlenbeck stochastic volatility |
| `VGParams` | Variance Gamma |
| `CgmyParams` | CGMY / KoBoL |
| `NigParams` | Normal Inverse Gaussian |
| `Sv32Params` | 3/2 stochastic-volatility model |
| `RoughHestonParams` | Rough Heston (El Euch and Rosenbaum) |
| `KouParams` | Kou double-exponential jump diffusion |
| `BatesParams` | Bates (Heston + Merton jumps) |
| `HestonKouParams` | Heston with Kou jumps |
| `HestonCGMYParams` | Heston with CGMY jumps |
| `GarchWMW2012Params` | GARCH option model (Whaley-Mrozek-Weiss 2012) |
| `MertonJDParams` | Merton jump-diffusion |
| `MeixnerParams` | Meixner process |
| `BilateralGammaParams` | Bilateral Gamma |
| `GHParams` | Generalised Hyperbolic |
| `FMLSParams` | Finite Moment Log-Stable |
| `DoubleHestonParams` | Double Heston |
| `VGSAParams` | Variance Gamma with stochastic arrival |
| `RegimeSwitchingBsmParams` | Markov regime-switching BSM: per-regime vols, chain generator, initial distribution |
| `SabrParams` | SABR: `alpha`, `beta`, `rho`, `nu`, `F`, `T`. |

---

## Characteristic functions

All characteristic functions accept a model parameter dataclass and a complex-valued frequency array.

| Function | Model |
|----------|-------|
| `bsm_cf` | BSM |
| `heston_cf_form2` | Heston form-2 |
| `ousv_cf` | OUSV |
| `vg_cf` | Variance Gamma |
| `cgmy_cf` | CGMY |
| `nig_cf` | NIG |
| `sv32_cf` | 3/2 |
| `rough_heston_cf` | Rough Heston |
| `kou_cf` | Kou |
| `bates_cf` | Bates |
| `heston_kou_cf` | Heston-Kou |
| `heston_cgmy_cf` | Heston-CGMY |
| `garch_wmw2012_cf` | GARCH WMW2012 |
| `merton_jd_cf` | Merton JD |
| `meixner_cf` | Meixner |
| `bilateral_gamma_cf` | Bilateral Gamma |
| `gh_cf` | Generalised Hyperbolic |
| `fmls_cf` | FMLS |
| `double_heston_cf` | Double Heston |
| `vgsa_cf` | VGSA |

### SABR implied volatility

| Function | Signature | Purpose |
|----------|-----------|---------|
| `sabr_hagan_implied_vol(F, K, T, params)` | forward, strike, maturity, `SabrParams` | Hagan et al. (2002) lognormal SABR implied-vol approximation. Returns a float. |

### Cumulants

Each model exposes a cumulant function that returns the first four log-return cumulants. Used internally by grid builders.

| Function | Model |
|----------|-------|
| `bsm_cumulants` | BSM |
| `heston_cumulants` | Heston |
| `ousv_cumulants` | OUSV (returns four cumulants by numerical integration) |
| `vg_cumulants` | VG |
| `cgmy_cumulants` | CGMY |
| `nig_cumulants` | NIG |
| `sv32_cumulants` | 3/2 |
| `rough_heston_cumulants` | Rough Heston |
| `kou_cumulants` | Kou |
| `bates_cumulants` | Bates |
| `heston_kou_cumulants` | Heston-Kou |
| `heston_cgmy_cumulants` | Heston-CGMY |
| `garch_wmw2012_cumulants` | GARCH WMW2012 |
| `merton_jd_cumulants` | Merton JD |
| `meixner_cumulants` | Meixner |
| `bilateral_gamma_cumulants` | Bilateral Gamma |
| `gh_cumulants` | GH |
| `fmls_cumulants` | FMLS |
| `double_heston_cumulants` | Double Heston |
| `vg_cumulants` | VG |
| `vgsa_cumulants` | VGSA |

### Miscellaneous model helpers

| Function | Purpose |
|----------|---------|
| `heston_riccati_cd` | Riccati ODE coefficients used in the Heston CF derivation. |

---

## Pipeline dispatchers

| Function | Signature | Purpose |
|----------|-----------|---------|
| `price(product, model_params, fwd_spec, **kwargs)` | product spec, model dataclass, forward spec | Product-aware dispatcher: routes to the correct pricer based on product type and model. Returns a scalar price. |
| `price_strip(strikes, option_type, model_params, fwd_spec, method=..., **kwargs)` | strike array, "call"/"put", model dataclass, forward spec | Prices a strip of European options at multiple strikes. Returns an array. |

---

## Grid objects and builders

### Grid dataclasses

| Class | Fields | Purpose |
|-------|--------|---------|
| `COSGrid` | `N`, `L` | COS grid: number of terms and truncation half-width. |
| `COSGridPolicy` | `N_base`, `L_base`, `overrides` | Policy-based COS grid selector. |
| `FFTGrid` | `N`, `eta`, `alpha` | Carr-Madan FFT grid. |
| `FRFTGrid` | `N`, `eta`, `alpha`, `lambda_` | FRFT grid. |
| `CONVGrid` | `N`, `L` | CONV method grid. |
| `LatticeGrid` | `n_steps` | Binomial/trinomial lattice grid. |
| `PDEGrid` | `n_steps`, `n_spot`, `theta` | Finite-difference PDE grid. |
| `ProjGrid` | `N`, `alph`, `order` | PROJ frame-projection grid (Haar/linear/quadratic/cubic B-spline). |
| `HilbertGrid` | `h`, `N` | Feng-Linetsky half-integer sinc grid for the discrete Hilbert transform. |

### Grid builders

| Function | Inputs | Output |
|----------|--------|--------|
| `cos_auto_grid(cumulants, N, L)` | cumulants | Builds a `COSGrid` from forward cumulants. |
| `cos_improved_grid(cumulants, ...)` | cumulants | Builds a `COSGrid` with improved truncation bounds. |
| `proj_auto_grid(cumulants, N=..., L=..., order=...)` | cumulants | Builds a `ProjGrid` from forward cumulants. |
| `recommended_cos_policy(model, params, mode=...)` | model name, params dataclass | Returns a `COSGridPolicy` for the given model and accuracy mode. |

---

## COS pricing

| Name | Type | Description |
|------|------|-------------|
| `COSResult` | dataclass | Holds prices, Greeks, and diagnostics from a COS computation. |
| `COSPolicyDecision` | dataclass | Stores the resolved `COSGrid` and the policy that produced it. |
| `cos_adaptive_decision(model, params, T, ...)` | function | Returns a `COSPolicyDecision` for a given model, adapting to accuracy requirements. |
| `cos_prices(cf, grid, fwd_spec, strikes, ...)` | function | Core COS pricer. Returns an array of call/put prices. |
| `cos_bermudan_price(cf, grid, fwd_spec, ...)` | function | COS Bermudan pricing via backward induction. |
| `cos_bermudan_price_strip(...)` | function | COS Bermudan pricer over a strike strip. |
| `cos_digital_price(cf, grid, fwd_spec, ...)` | function | COS pricing for cash-or-nothing and asset-or-nothing digitals. |
| `cos_digital_price_strip(...)` | function | COS digital pricer over a strike strip. |

---

## Fourier pricers

| Function | Signature | Purpose |
|----------|-----------|---------|
| `carr_madan_fft_prices(cf, grid, fwd_spec)` | CF, `FFTGrid`, forward spec | Carr-Madan (1999) FFT pricer. Returns prices on the FFT log-strike grid. |
| `carr_madan_price_at_strikes(cf, grid, fwd_spec, strikes)` | CF, `FFTGrid`, forward spec, strikes | FFT pricer interpolated to specific strikes. |
| `frft_prices(cf, grid, fwd_spec)` | CF, `FRFTGrid`, forward spec | Fractional FFT pricer. Returns prices on the FRFT log-strike grid. |
| `frft_price_at_strikes(cf, grid, fwd_spec, strikes)` | CF, `FRFTGrid`, forward spec, strikes | FRFT pricer interpolated to specific strikes. |
| `lewis_prices(cf, grid, fwd_spec, strikes)` | CF, `COSGrid`, forward spec, strikes | Lewis (2001) call integral formula. |
| `lewis_call_prices(cf, grid, fwd_spec, strikes)` | CF, `COSGrid`, forward spec, strikes | Lewis pricer returning call prices. |
| `conv_price_at_strikes(cf, grid, fwd_spec, strikes)` | CF, `CONVGrid`, forward spec, strikes | CONV method (Lord et al. 2008) evaluated at specific strikes. |
| `mellin_price_at_strikes(cf, grid, fwd_spec, strikes)` | CF, forward spec, strikes | Mellin-transform pricer evaluated at specific strikes. |
| `hilbert_price_at_strikes(phi, fwd, strikes, cp=1, grid=None)` | CF, forward spec, strikes | Feng-Linetsky (2008) Hilbert-transform pricer; exponentially convergent Gil-Pelaez probabilities. |
| `hilbert_itm_probabilities(phi, fwd, strikes, grid=None)` | CF, forward spec, strikes | Share- and cash-measure ITM probabilities (Pi_1, Pi_2); N(d1)/N(d2) under BSM. |
| `levy_geometric_asian_price(model, fwd, params, strikes=..., monitoring_times=..., cp=1)` | Levy model key, market inputs, fixings | Exact discrete geometric-Asian prices via the per-increment CF product (Fusai-Meucci 2008). |
| `levy_forward_start_price(model, fwd, params, alpha=..., start_time=..., maturity=..., cp=1)` | Levy model key, market inputs, strike ratio, reset date | Exact Levy forward-start price via homogeneity factorization; COS European leg. |
| `filtered_cos_prices(cf, grid, fwd_spec, strikes, spec)` | CF, `COSGrid`, forward spec, strikes, `COSFilterSpec` | COS with Conze-Viswanathan or exponential filters to suppress Gibbs oscillations. |

---

## PROJ frame-projection pricers

| Function | Signature | Purpose |
|----------|-----------|---------|
| `proj_european_price_at_strikes(cf, grid, fwd_spec, strikes)` | CF, `ProjGrid`, forward spec, strikes | PROJ European pricer (Kirkby 2015). Matches `carr_madan_price_at_strikes` to ~1e-7 across the Levy family. |
| `proj_price_at_strikes(cf, grid, fwd_spec, strikes, ...)` | CF, `ProjGrid`, forward spec, strikes | General PROJ European dispatch (entry point used by `price_strip`). |
| `proj_bermudan_put(cf, grid, fwd_spec, strikes, n_ex)` | CF, `ProjGrid`, forward spec, strikes, exercise count | PROJ Bermudan put via Toeplitz-FFT backward recursion (Kirkby 2017). |

---

## BSM lattice and PDE

| Function | Signature | Purpose |
|----------|-----------|---------|
| `bsm_lattice_price(product, bsm_params, fwd_spec, grid)` | product spec, `BsmParams`, forward spec, `LatticeGrid` | European or American option via binomial lattice. |
| `bsm_lattice_price_at_strikes(strikes, ..., grid)` | strikes, model params, forward spec, `LatticeGrid` | Lattice pricer over a strike strip. |
| `bsm_pde_fd_price(product, bsm_params, fwd_spec, grid)` | product spec, `BsmParams`, forward spec, `PDEGrid` | European or American option via Crank-Nicolson finite differences. |
| `bsm_pde_fd_price_at_strikes(strikes, ..., grid)` | strikes, model params, forward spec, `PDEGrid` | PDE pricer over a strike strip. |

---

## SABR pricer

| Function | Signature | Purpose |
|----------|-----------|---------|
| `sabr_hagan_price_at_strikes(strikes, params, fwd_spec)` | strikes, `SabrParams`, forward spec | Prices a strip of calls via SABR Hagan implied vol then BSM. |

---

## Filtered COS helpers

| Name | Type | Description |
|------|------|-------------|
| `COSFilterSpec` | dataclass | Filter type and strength parameters for filtered COS. |
| `FilteredCOSDecision` | dataclass | Holds the filter weights alongside the resolved `COSGrid`. |
| `cos_filter_weights(N, spec)` | function | Returns a weight array of length `N` for the given filter spec. |

---

## Implied volatility

| Function | Signature | Purpose |
|----------|-----------|---------|
| `bs_price_from_fwd(fwd_spec, K, sigma, option_type)` | forward spec, strike, vol, type | BSM call or put price from forward inputs. |
| `implied_vol_brent(fwd_spec, K, price, option_type)` | forward spec, strike, price, type | Implied vol via Brent root-finding. Safe for deep-ITM/OTM. |
| `implied_vol_newton_safeguarded(fwd_spec, K, price, option_type)` | forward spec, strike, price, type | Newton-Raphson with Brent fallback. Faster for near-ATM. |

---

## Surface and calibration

| Name | Type | Description |
|------|------|-------------|
| `SurfaceSpec` | dataclass | Defines the moneyness-tenor grid used in calibration targets. |
| `CalibrationResult` | dataclass | Holds calibrated parameter dataclass, residuals, and optimizer diagnostics. |
| `model_iv_surface(model_params, fwd_spec, surface_spec)` | function | Evaluates a model implied-vol surface on a `SurfaceSpec` grid. |
| `model_price_surface(model_params, fwd_spec, surface_spec)` | function | Evaluates a model price surface on a `SurfaceSpec` grid. |
| `calibrate_heston(...)` | function | Calibrates Heston parameters to market targets. Returns `CalibrationResult`. |
| `calibrate_vg(...)` | function | Calibrates VG parameters to market targets. Returns `CalibrationResult`. |
| `calibrate_kou(...)` | function | Calibrates Kou parameters to market targets. Returns `CalibrationResult`. |
| `calibrate_cgmy(...)` | function | Calibrates CGMY parameters to market targets. Returns `CalibrationResult`. |
| `calibrate_nig(...)` | function | Calibrates NIG parameters to market targets. Returns `CalibrationResult`. |

---

## Greeks

| Name | Type | Description |
|------|------|-------------|
| `COSGreeks` | dataclass | Delta, gamma, and vega computed via the COS method. |
| `cos_price_and_greeks(cf, grid, fwd_spec, strikes)` | function | Returns prices and `COSGreeks` in a single pass. |
| `cos_delta_gamma(cf, grid, fwd_spec, strikes)` | function | Returns delta and gamma arrays from the COS expansion. |
| `cos_parameter_sensitivity(cf, grid, fwd_spec, strikes, params, ...)` | function | Finite-difference parameter sensitivities (model-parameter Greeks). |

---

## Monte Carlo

| Name | Type | Description |
|------|------|-------------|
| `MCSpec` | dataclass | Monte Carlo specification: product, model, paths, time steps, random seed, antithetics, control-variate flag. |
| `MCResult` | dataclass | Output of `mc_price`: estimated price, standard error, confidence interval, paths used. |
| `mc_price(spec, model_params, fwd_spec)` | function | Generic MC dispatcher. Routes to the correct simulation engine based on `spec.model` and `spec.product`. |
| `european_call_mc(bsm_params, fwd_spec, K, n_paths, n_steps, seed)` | function | Vectorised BSM European call MC. |
| `heston_conditional_mc_calls(heston_params, fwd_spec, strikes, n_paths, seed)` | function | Heston exact conditional simulation (Broadie-Kaya scheme). |
| `HestonMCScheme` | enum | Discretisation scheme selector: `EULER`, `MILSTEIN`, `QE`. |
| `bs_call_cv` | function | BSM control-variate correction for a Monte Carlo estimate. |
| `heston_call_bs_control` | function | Heston MC pricer with a BSM control variate. |
| `CVResult` | dataclass | Control-variate result: raw estimate, corrected estimate, variance reduction ratio. |

---

## BSM closed-form analytics

Exact closed-form prices for non-vanilla payoffs under BSM dynamics.

### Exotic single-asset options

| Function | Signature | Purpose |
|----------|-----------|---------|
| `bsm_barrier_price(fwd_spec, K, H, barrier_type, option_type)` | forward spec, strike, barrier level, barrier type, option type | Analytical up/down in/out barrier option. |
| `bsm_forward_start(fwd_spec, alpha, T_start, T_end)` | forward spec, moneyness, start date, end date | Rubinstein (1991) forward-start call. |
| `bsm_lookback_floating(fwd_spec, option_type)` | forward spec, option type | Floating-strike lookback call or put (Goldman-Sosin-Gatto). |
| `bsm_geometric_asian(fwd_spec, K, n)` | forward spec, strike, number of fixings | Geometric-average Asian option closed form. |
| `bsm_geometric_asian_parity(fwd_spec, K, n)` | forward spec, strike, number of fixings | Put-call parity check for geometric Asian. |
| `bsm_variance_swap(fwd_spec)` | forward spec | Fair variance swap strike under BSM (equals `sigma^2`). |
| `levy_variance_fair_strike(model, fwd, params, sampling_times, maturity=None)` | Levy model key, market inputs, dates | Exact annualized E[RV] from per-increment CF cumulants; prices jump risk. |
| `levy_variance_swap(model, fwd, params, product)` | Levy model key, market inputs, `VarianceSwap` | Discounted variance-swap value, `disc * notional * E[RV]`. |
| `bsm_variance_option_integrated(fwd_spec, K_var, option_type)` | forward spec, variance strike, option type | Variance call or put price via integrated BSM formula. |
| `bsm_gap_call(fwd_spec, K1, K2)` | forward spec, trigger strike, payoff strike | Gap call: pays `S - K2` if `S > K1`. |
| `bsm_discrete_geometric_asian(fwd_spec, K, fixing_times)` | forward spec, strike, fixing schedule | Geometric Asian with an explicit fixing schedule. |

### Digital options

| Function | Signature | Purpose |
|----------|-----------|---------|
| `bsm_cash_or_nothing(fwd_spec, K, option_type)` | forward spec, strike, option type | Cash-or-nothing digital: pays 1 unit if `S_T > K`. |
| `bsm_asset_or_nothing(fwd_spec, K, option_type)` | forward spec, strike, option type | Asset-or-nothing digital: pays `S_T` if `S_T > K`. |

### Multi-asset analytics

| Function | Signature | Purpose |
|----------|-----------|---------|
| `margrabe_exchange(fwd_spec_1, fwd_spec_2, sigma_1, sigma_2, rho)` | two forward specs, vols, correlation | Margrabe (1978) exchange option: pays `max(S2 - S1, 0)`. |
| `kirk_spread(fwd_spec_1, fwd_spec_2, K, sigma_1, sigma_2, rho)` | two forward specs, strike, vols, correlation | Kirk (1995) spread option approximation: pays `max(S2 - S1 - K, 0)`. |