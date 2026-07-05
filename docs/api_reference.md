# API Reference

All public names are accessible from the top-level `foureng` namespace after `import foureng`.

---

## Version

| Name | Type | Description |
|------|------|-------------|
| `__version__` | str | Package version string (PEP 440); matches the latest entry in [CHANGELOG.md](../CHANGELOG.md). |

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
| `RegimeSwitchingBsmParams` | Markov regime-switching jump-diffusion: per-regime vols, chain generator, initial distribution, optional per-regime Merton jump blocks |
| `HullWhiteHybridParams` | Stochastic-rate hybrid: any base registry model plus an independent one-factor Hull-White short rate |
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
| `regime_switching_cf` | Markov regime-switching BSM (matrix-exponential CF) |
| `hw_hybrid_cf` | Equity + Hull-White hybrid (base CF times the bond-variance Gaussian) |

### SABR implied volatility

| Function | Signature | Purpose |
|----------|-----------|---------|
| `sabr_hagan_implied_vol(F, K, T, alpha, beta, rho, nu)` | forward, strike, maturity, `SabrParams` | Hagan et al. (2002) lognormal SABR implied-vol approximation. Returns a float. |

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
| `regime_switching_cumulants` | Markov regime-switching BSM (numeric CGF differentiation) |
| `hw_hybrid_cumulants` | Equity + Hull-White hybrid (base cumulants shifted by the bond variance V_P) |
| `hw_bond_variance` | Integrated Hull-White bond-price variance V_P(a, sigma_r, T) with the Ho-Lee small-a limit |

### Miscellaneous model helpers

| Function | Purpose |
|----------|---------|
| `heston_riccati_cd` | Riccati ODE coefficients used in the Heston CF derivation. |

---

## Pipeline dispatchers

| Function | Signature | Purpose |
|----------|-----------|---------|
| `price(product, model, method, fwd, params, grid=...)` | product spec, model dataclass, forward spec | Product-aware dispatcher: routes to the correct pricer based on product type and model. Returns a scalar price. |
| `price_strip(model, method, strikes, fwd, params, grid=..., cp=...)` | strike array, "call"/"put", model dataclass, forward spec | Prices a strip of European options at multiple strikes. Returns an array. |

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
| `cos_adaptive_decision(cumulants, model=..., params=..., policy=..., strike_count=...)` | function | Returns a `COSPolicyDecision` for a given model, adapting to accuracy requirements. |
| `cos_prices(phi, fwd, strikes, grid, payoff_mode=..., call_direct_width_max=..., ...)` | function | Core COS pricer. Returns an array of call/put prices. |
| `cos_bermudan_price(model, fwd, params, product, grid=..., n_spatial=..., ...)` | function | COS Bermudan pricing via backward induction. |
| `cos_bermudan_price_strip(...)` | function | COS Bermudan pricer over a strike strip. |
| `cos_digital_price(model, fwd, params, product, grid=..., N=..., L=...)` | function | COS pricing for cash-or-nothing and asset-or-nothing digitals. |
| `cos_digital_price_strip(...)` | function | COS digital pricer over a strike strip. |

---

## Fourier pricers

| Function | Signature | Purpose |
|----------|-----------|---------|
| `carr_madan_fft_prices(phi, fwd, grid, k0=...)` | CF, `FFTGrid`, forward spec | Carr-Madan (1999) FFT pricer. Returns prices on the FFT log-strike grid. |
| `carr_madan_price_at_strikes(phi, fwd, grid, strikes, window_factor=...)` | CF, `FFTGrid`, forward spec, strikes | FFT pricer interpolated to specific strikes. |
| `frft_prices(phi, fwd, grid, k0=...)` | CF, `FRFTGrid`, forward spec | Fractional FFT pricer. Returns prices on the FRFT log-strike grid. |
| `frft_price_at_strikes(phi, fwd, grid, strikes, window_factor=...)` | CF, `FRFTGrid`, forward spec, strikes | FRFT pricer interpolated to specific strikes. |
| `lewis_prices(cf, strikes, spot, texp, cp=..., intr=..., ...)` | CF, `COSGrid`, forward spec, strikes | Lewis (2001) call integral formula. |
| `lewis_call_prices(cf, strikes, spot, texp, intr=..., divr=..., ...)` | CF, `COSGrid`, forward spec, strikes | Lewis pricer returning call prices. |
| `conv_price_at_strikes(phi, fwd, grid, strikes, cp=...)` | CF, `CONVGrid`, forward spec, strikes | CONV method (Lord et al. 2008) evaluated at specific strikes. |
| `mellin_price_at_strikes(phi, fwd, strikes, cp=..., grid=...)` | CF, forward spec, strikes | Mellin-transform pricer evaluated at specific strikes. |
| `hilbert_price_at_strikes(phi, fwd, strikes, cp=1, grid=None)` | CF, forward spec, strikes | Feng-Linetsky (2008) Hilbert-transform pricer; exponentially convergent Gil-Pelaez probabilities. |
| `hilbert_itm_probabilities(phi, fwd, strikes, grid=None)` | CF, forward spec, strikes | Share- and cash-measure ITM probabilities (Pi_1, Pi_2); N(d1)/N(d2) under BSM. |
| `levy_geometric_asian_price(model, fwd, params, strikes=..., monitoring_times=..., cp=1)` | Levy model key, market inputs, fixings | Exact discrete geometric-Asian prices via the per-increment CF product (Fusai-Meucci 2008). |
| `levy_forward_start_price(model, fwd, params, alpha=..., start_time=..., maturity=..., cp=1)` | Levy model key, market inputs, strike ratio, reset date | Exact Levy forward-start price via homogeneity factorization; COS European leg. |
| `levy_cliquet_price(model, fwd, params, product)` | Levy model key, market inputs, `CliquetOption` | Exact locally collared cliquet: per-period COS call spreads, additive or multiplicative. |
| `levy_fader_price(model, fwd, params, product)` | Levy model key, market inputs, `FaderOption` | Fade-in/fade-out via per-date COS density times the remaining-life European strip. |
| `filtered_cos_prices(phi, fwd, strikes, grid, filter_spec=..., payoff_mode=...)` | CF, `COSGrid`, forward spec, strikes, `COSFilterSpec` | COS with Conze-Viswanathan or exponential filters to suppress Gibbs oscillations. |

---

## PROJ frame-projection pricers

| Function | Signature | Purpose |
|----------|-----------|---------|
| `proj_european_price_at_strikes(phi, fwd, cumulants, strikes, cp=..., N=..., L=...)` | CF, `ProjGrid`, forward spec, strikes | PROJ European pricer (Kirkby 2015). Matches `carr_madan_price_at_strikes` to ~1e-7 across the Levy family. |
| `proj_price_at_strikes(phi, fwd, grid, strikes, cp=..., c1=...)` | CF, `ProjGrid`, forward spec, strikes | General PROJ European dispatch (entry point used by `price_strip`). |
| `proj_bermudan_put(step_cf, S0, r, T, W, M, ...)` | CF, `ProjGrid`, forward spec, strikes, exercise count | PROJ Bermudan put via Toeplitz-FFT backward recursion (Kirkby 2017). |
| `proj_barrier_price(...)` | CF, forward spec, barrier contract terms | PROJ discretely monitored single-barrier pricer (down-out / up-out, knock-in via parity). |
| `proj_asian_price_cv(...)` | CF, forward spec, Asian contract terms | Arithmetic Asian via Monte Carlo with a PROJ/analytic geometric control variate. |
| `proj_double_barrier_price(step_cf, S0=..., K=..., L=..., U=..., M=..., knockout=True, ...)` | one-step CF, corridor, monitoring count | PROJ double-barrier knock-out/knock-in via two-sided absorption in the backward induction. |
| `proj_step_price(step_cf, S0=..., K=..., B=..., rho=..., M=..., step_type="down", ...)` | one-step CF, barrier, damping rate, monitoring count | PROJ step option: occupation-time soft killing exp(-rho dt) beyond the barrier (Linetsky 1999). |
| `proj_survival_probability(step_cf, S0=..., B=..., M=...)` | one-step CF, barrier, monitoring count | First-passage survival probability via the undiscounted down-and-out unit-payoff recursion. |
| `proj_swing_price(step_cf, S0=..., K=..., M=..., n_rights=..., cp=1, ...)` | one-step CF, exercise dates, rights count | Swing option DP over (date, rights); n_rights=1 Bermudan, n_rights>=M sum of Europeans. |

---

## BSM lattice and PDE

| Function | Signature | Purpose |
|----------|-----------|---------|
| `bsm_lattice_price(fwd, params, strike, cp=..., exercise=..., grid=...)` | product spec, `BsmParams`, forward spec, `LatticeGrid` | European or American option via binomial lattice. |
| `bsm_lattice_price_at_strikes(strikes, ..., grid)` | strikes, model params, forward spec, `LatticeGrid` | Lattice pricer over a strike strip. |
| `bsm_pde_fd_price(fwd, params, strike, cp=..., exercise=..., grid=...)` | product spec, `BsmParams`, forward spec, `PDEGrid` | European or American option via Crank-Nicolson finite differences. |
| `bsm_pde_fd_price_at_strikes(strikes, ..., grid)` | strikes, model params, forward spec, `PDEGrid` | PDE pricer over a strike strip. |

---

## SABR pricer

| Function | Signature | Purpose |
|----------|-----------|---------|
| `sabr_hagan_price_at_strikes(fwd, params, strikes, cp=...)` | strikes, `SabrParams`, forward spec | Prices a strip of calls via SABR Hagan implied vol then BSM. |

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
| `bs_price_from_fwd(vol, inp)` | forward spec, strike, vol, type | BSM call or put price from forward inputs. |
| `implied_vol_brent(price, inp, lo=..., hi=...)` | forward spec, strike, price, type | Implied vol via Brent root-finding. Safe for deep-ITM/OTM. |
| `implied_vol_newton_safeguarded(price, inp, vol0=..., iters=..., tol=..., lo=..., hi=...)` | forward spec, strike, price, type | Newton-Raphson with Brent fallback. Faster for near-ATM. |

---

## Surface and calibration

| Name | Type | Description |
|------|------|-------------|
| `SurfaceSpec` | dataclass | Defines the moneyness-tenor grid used in calibration targets. |
| `CalibrationResult` | dataclass | Holds calibrated parameter dataclass, residuals, and optimizer diagnostics. |
| `model_iv_surface(spec, cf_factory, cumulant_factory, N=..., L=...)` | function | Evaluates a model implied-vol surface on a `SurfaceSpec` grid. |
| `model_price_surface(spec, cf_factory, cumulant_factory, N=..., L=...)` | function | Evaluates a model price surface on a `SurfaceSpec` grid. |
| `calibrate_heston(...)` | function | Calibrates Heston parameters to market targets. Returns `CalibrationResult`. |
| `calibrate_vg(...)` | function | Calibrates VG parameters to market targets. Returns `CalibrationResult`. |
| `calibrate_kou(...)` | function | Calibrates Kou parameters to market targets. Returns `CalibrationResult`. |
| `calibrate_cgmy(...)` | function | Calibrates CGMY parameters to market targets. Returns `CalibrationResult`. |
| `calibrate_nig(...)` | function | Calibrates NIG parameters to market targets. Returns `CalibrationResult`. |
| `calibrate_sabr_smile(...)` | function | Fits SABR (alpha, nu, rho; optional beta) to a single-maturity smile. Returns `SabrSmileCalibResult`. |
| `SabrSmileCalibResult` | dataclass | SABR smile calibration output: parameters, residuals, diagnostics. |
| `SVIParams` / `SVIFitResult` | dataclasses | Gatheral (2004) SVI smile parameters and fit output. |
| `svi_total_variance` / `svi_implied_vol` | functions | SVI total variance and implied vol at given log-moneyness. |
| `svi_butterfly_density` / `svi_check_butterfly_arbitrage` | functions | Gatheral-Jacquier g(k) density and butterfly-arbitrage check. |
| `fit_svi_smile(...)` | function | Calibrates SVI to a smile (L-BFGS-B with optional differential-evolution pre-fit). |
| `SSVIParams` / `SSVIFitResult` | dataclasses | Gatheral-Jacquier (2014) Surface-SVI parameters and fit output. |
| `ssvi_phi_power_law` / `ssvi_phi_heston` | functions | SSVI phi parameterizations. |
| `ssvi_total_variance` / `ssvi_implied_vol` | functions | SSVI total variance and implied vol. |
| `ssvi_check_butterfly_free` / `ssvi_check_calendar_free` | functions | SSVI static-arbitrage sufficient conditions. |
| `fit_ssvi_surface(...)` | function | Two-stage joint SSVI surface calibration. |
| `LocalVolSurface` | dataclass | Dupire local-volatility surface container. |
| `dupire_local_vol_from_svi(...)` | function | Local vol from SVI slices (analytic k-derivatives, FD in T). |
| `dupire_local_vol_grid(...)` | function | Local vol from a raw IV grid via finite differences with optional smoothing. |

---

## Greeks

| Name | Type | Description |
|------|------|-------------|
| `COSGreeks` | dataclass | Delta, gamma, and vega computed via the COS method. |
| `cos_price_and_greeks(phi, fwd, strikes, grid)` | function | Returns prices and `COSGreeks` in a single pass. |
| `cos_delta_gamma(phi, fwd, strikes, grid)` | function | Returns delta and gamma arrays from the COS expansion. |
| `cos_parameter_sensitivity(dphi_dparam, fwd, strikes, grid)` | function | Finite-difference parameter sensitivities (model-parameter Greeks). |
| `bsm_delta` / `bsm_gamma` / `bsm_vega` / `bsm_theta` / `bsm_rho` | functions | Analytic first-order BSM Greeks (theta is dV/dt; rho scaled by 1/100). |
| `bsm_vanna` / `bsm_volga` | functions | Analytic second-order/cross BSM Greeks. |
| `bsm_all_greeks(...)` | function | All analytic BSM Greeks in one call. |

---

## Monte Carlo

| Name | Type | Description |
|------|------|-------------|
| `MCSpec` | dataclass | Monte Carlo specification: product, model, paths, time steps, random seed, antithetics, control-variate flag. |
| `MCResult` | dataclass | Output of `mc_price`: estimated price, standard error, confidence interval, paths used. |
| `mc_price(fwd, sigma, product, mc=...)` | function | Generic MC dispatcher. Routes to the correct simulation engine based on `spec.model` and `spec.product`. |
| `european_call_mc(S0, K, T, r, q, vol, mc)` | function | Vectorised BSM European call MC. |
| `heston_conditional_mc_calls(S0, strikes, T, r, q, p, mc)` | function | Heston exact conditional simulation (Broadie-Kaya scheme). |
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
| `bsm_barrier_price(S, K, H, r, q, T, ...)` | forward spec, strike, barrier level, barrier type, option type | Analytical up/down in/out barrier option. |
| `bsm_forward_start(S, alpha, t_start, T, r, q, ...)` | forward spec, moneyness, start date, end date | Rubinstein (1991) forward-start call. |
| `bsm_lookback_floating(S, S_min, S_max, r, q, T, ...)` | forward spec, option type | Floating-strike lookback call or put (Goldman-Sosin-Gatto). |
| `bsm_geometric_asian(S, K, r, q, T, sigma, cp=...)` | forward spec, strike, number of fixings | Geometric-average Asian option closed form. |
| `bsm_geometric_asian_parity(S, K, r, q, T, sigma)` | forward spec, strike, number of fixings | Put-call parity check for geometric Asian. |
| `bsm_variance_swap(fwd, params, product)` | forward spec | Fair variance swap strike under BSM (equals `sigma^2`). |
| `levy_variance_fair_strike(model, fwd, params, sampling_times, maturity=None)` | Levy model key, market inputs, dates | Exact annualized E[RV] from per-increment CF cumulants; prices jump risk. |
| `levy_variance_swap(model, fwd, params, product)` | Levy model key, market inputs, `VarianceSwap` | Discounted variance-swap value, `disc * notional * E[RV]`. |
| `levy_survival_curve(model, fwd, params, default_barrier=..., horizons=...)` | Levy model key, market inputs, barrier, dates | Discretely monitored first-passage survival probabilities (Black-Cox structural default). |
| `levy_cds_spread(model, fwd, params, default_barrier=..., recovery=..., maturity=...)` | Levy model key, market inputs, credit terms | Structural CDS par spread from the PROJ survival curve and O'Kane running-spread legs. |
| `cds_par_spread_from_survival(survival, payment_times, r, recovery)` | survival curve, premium dates | Leg assembly only; credit-triangle consistent. |
| `bsm_variance_option_integrated(fwd, params, product)` | forward spec, variance strike, option type | Variance call or put price via integrated BSM formula. |
| `bsm_gap_call(S, K1, K2, r, q, T, sigma)` | forward spec, trigger strike, payoff strike | Gap call: pays `S - K2` if `S > K1`. |
| `geske_compound_price(...)` | compound contract terms | Geske (1979) compound option (all four call/put-on-call/put types). |
| `bsm_chooser_price(...)` | chooser contract terms | Rubinstein (1991) chooser via the call+put decomposition. |
| `bsm_quanto_forward(...)` / `bsm_quanto_option(...)` | quanto contract terms | Reiner (1992) quanto-adjusted forward and option prices. |
| `bsm_discrete_geometric_asian(S, K, r, q, monitoring_times, sigma, cp=...)` | forward spec, strike, fixing schedule | Geometric Asian with an explicit fixing schedule. |

### Digital options

| Function | Signature | Purpose |
|----------|-----------|---------|
| `bsm_cash_or_nothing(fwd, p, K, cp=..., cash_amount=...)` | forward spec, strike, option type | Cash-or-nothing digital: pays 1 unit if `S_T > K`. |
| `bsm_asset_or_nothing(fwd, p, K, cp=...)` | forward spec, strike, option type | Asset-or-nothing digital: pays `S_T` if `S_T > K`. |

### Product dataclasses (`foureng.products`)

Every product is a frozen dataclass inheriting from `ProductSpec` (the root
base class carrying the `product_type` tag used by the capability registry).
Priced through `price(product, model, method, fwd, params)`.

| Class | Contract | Typical methods |
|-------|----------|-----------------|
| `EuropeanOption` | Plain-vanilla European call/put | any CF engine via `price_strip` |
| `AmericanOption` | American-exercise put/call | `monte_carlo` (LSMC), `lattice` |
| `BermudanOption` | Finite exercise-date put/call | `cos_bermudan`, `proj`, `monte_carlo` |
| `DigitalOption` | Cash-/asset-or-nothing digital | `cos_digital`, `digital_bsm` |
| `BarrierOption` | Single knock-in/knock-out | `barrier_bsm`, `proj_barrier`, `monte_carlo` |
| `DoubleBarrierOption` | Corridor knock-out/knock-in | `double_barrier_bsm`, `proj_double_barrier`, MC |
| `StepOption` | Occupation-time-damped vanilla (Linetsky 1999) | `proj_step` |
| `SwingOption` | Multiple vanilla exercise rights, one per date (Carmona-Touzi 2008) | `proj_swing` |
| `FaderOption` | Range-monitored faded notional | `fader_cf` |
| `AsianOption` | Discretely monitored average-rate | `asian_cf`, `asian_bsm`, `proj_asian`, MC |
| `LookbackOption` | Fixed-/floating-strike lookback | `lookback_bsm`, `lookback_mc` |
| `ParisianOption` | Excursion-triggered barrier | `parisian_mc` |
| `ForwardStartOption` | Strike set as `alpha * S_{t_start}` | `forward_start_cf`, `forward_start_bsm` |
| `CliquetOption` | Capped/floored period-return sum/product | `cliquet_cf`, `cliquet_mc` |
| `VarianceSwap` | Realised variance vs fair strike | `variance_levy_analytic`, `variance_analytic_bsm`, MC |
| `VarianceOption` | Option on realised variance | `variance_analytic_bsm`, `variance_mc` |
| `CompoundOption` | Option on an option (Geske 1979) | `geske` |
| `ChooserOption` | Choose call/put at a fixed date | `analytic` |
| `QuantoOption` | Foreign underlying, domestic payout | BSM quanto analytics |
| `ExchangeOption` | Margrabe `max(S1 - S2, 0)` | `exchange_bsm`, `multi_asset_mc` |
| `SpreadOption` | Call on `S1 - S2` with strike K | `spread_bsm`, `multi_asset_mc` |
| `BasketOption` | Weighted-sum basket | `multi_asset_mc` |
| `BestOfOption` | Call on `max(S1, S2, ...)` | `multi_asset_mc` |

### Multi-asset analytics

| Function | Signature | Purpose |
|----------|-----------|---------|
| `margrabe_exchange(S1, S2, q1, q2, T, sigma1, ...)` | two forward specs, vols, correlation | Margrabe (1978) exchange option: pays `max(S2 - S1, 0)`. |
| `kirk_spread(S1, S2, K, r, q1, q2, ...)` | two forward specs, strike, vols, correlation | Kirk (1995) spread option approximation: pays `max(S2 - S1 - K, 0)`. |