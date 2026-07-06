# Changelog

## 0.21.0 - 2026-07-06

- Implemented the CTMC (continuous-time Markov chain) approximation layer, completing the PROJ-toolbox gap roadmap. New `foureng/pricers/ctmc.py` with `CTMCGrid`, `ctmc_european_price`, `ctmc_european_price_at_strikes` (one matrix exponential shared across a strike strip), and `ctmc_american_price` (Bermudan time-stepping with a single one-step transition matrix). Generator built from the Mijatovic-Pistorius / Lo-Skindilias finite-volume stencil with automatic upwind fallback, on a spot-centered log grid. Volatility may be a constant (BSM) or a callable `sigma(S)` for local-vol/CEV-type diffusions -- the CTMC's purpose: state-dependent coefficients with no characteristic function. Wired as `method="ctmc"` in `price_strip` (European, BSM) and `price` (American), filling the long-standing "planned" registry entry. Tests: BSM closed form at 5e-4 with verified second-order grid convergence, American put vs the CRR lattice at 2e-3, American call with q=0 equals European, constant-callable identity, CEV local-vol skew direction, dispatch, parity, validation.

## 0.20.0 - 2026-07-05

- Added swing options (multiple exercise rights, Carmona-Touzi 2008) for 1-D Levy models: new `SwingOption` product and `method="proj_swing"` (`proj_swing_price` exported). Dynamic programming over (date, rights remaining) with one Toeplitz-FFT convolution per rights level: `V_m(x,j) = max(C_m(x,j), g(x) + C_m(x,j-1))` on the strike-aligned PROJ grid. Two exact degeneracies anchor it: `n_rights = 1` is the Bermudan option (agrees with the independent COS-Bermudan engine to 5e-3 and with proj_bermudan_put to 1e-3 at matched grid width) and `n_rights = n_dates` values as the sum of the per-date Europeans (2e-3, calls and puts, BSM and Kou). Also verified: value increasing and concave in rights, subadditivity, excess rights worthless, validation errors.

## 0.19.0 - 2026-07-05

- Added structural CDS pricing under Levy models (Black-Cox first passage, discretely monitored): `proj_survival_probability` (down-and-out unit payoff through the undiscounted PROJ recursion), `levy_survival_curve` (one run per premium date), `cds_par_spread_from_survival` (O'Kane running-spread legs with half-period accrual on default), and `levy_cds_spread` (end-to-end par spread). Tests: single-date survival equals the exact lognormal tail probability, survival curves vs 200k-path first-passage Monte Carlo on the same monthly grid for BSM and Kou, the credit-triangle identity spread = (1-R) lambda on a synthetic exponential curve (leg assembly validated independently of the survival engine), horizon/barrier monotonicity, spread monotonicity in barrier and exact linearity in (1-R), far-barrier degeneracy, jumps-widen-spreads, validation errors.

## 0.18.0 - 2026-07-05

- Added step options (Linetsky 1999 occupation-time damping, discretely monitored) for 1-D Levy models: new `StepOption` product and `method="proj_step"` (`proj_step_price` exported). The PROJ barrier backward induction generalizes from hard knock-out to *soft killing*: value mass beyond the barrier is multiplied by `exp(-rho dt)` at each monitoring date, so `rho = 0` recovers the vanilla and `rho -> infinity` recovers the knock-out barrier, with a continuous delta in between. Tests: both limits against the same-engine vanilla/knock-out (2e-3), monotone decay in rho, BSM and Kou full-path Monte Carlo with the discrete occupation-time payoff at 200k paths, up-step direction check, pipeline dispatch, validation errors.
- Docs audit: `AsianOption`, `DoubleBarrierOption`, and `ForwardStartOption` were missing from the API-reference product table; backfilled along with a pointer to the remaining product dataclasses.

## 0.17.0 - 2026-07-05

- Added fader (faded-notional) options for Levy models: new `FaderOption` product (fade-in/fade-out over a range (L, U) monitored on discrete dates) and `method="fader_cf"` (`levy_fader_price` exported). Linearity splits the payoff date by date; independence factorizes each term into the COS density of the date-k marginal times the remaining-life European value, and spot-shift homogeneity turns the conditional values into a single COS strike strip per date (2M COS runs total). Fade-out via the exact notional-split parity. Tests: all-encompassing range == vanilla, single-date-at-maturity vs the exact BSM call-spread + digital decomposition (5e-6), BSM and Kou full-path Monte Carlo at 200k paths, fade-in + fade-out == vanilla, range monotonicity, put route through the pipeline, validation errors.

## 0.16.0 - 2026-07-05

- Added the equity + one-factor Hull-White stochastic-rate hybrid as registry model `"hw_hybrid"` (`HullWhiteHybridParams(base_model, base_params, mean_reversion, sigma_r)`): under independent rates, T-forward-measure pricing multiplies the base CF by `exp(-0.5 V_P (u^2 + iu))` where `V_P` is the integrated Hull-White bond-price variance (`hw_bond_variance`, with the Ho-Lee `sigma_r^2 T^3/3` limit at small mean reversion). Any registry model can be the base -- BSM, Levy jump models, Heston -- and `phi(-i) = 1` is preserved. Exported: `HullWhiteHybridParams`, `hw_hybrid_cf`, `hw_hybrid_cumulants`, `hw_bond_variance`. Tests: bond variance vs numerical quadrature and the Ho-Lee limit, BSM base equals the Merton (1973) effective-vol closed form to 1e-8, sigma_r = 0 collapse, martingale normalization, cumulant additivity, monotonicity in rate vol, COS/Hilbert cross-engine and parity, Kou base vs exact 250k-path Monte Carlo (independent Gaussian rate leg), Heston base through the pipeline, parameter validation. Model count: 22.

## 0.15.0 - 2026-07-05

- Upgraded the regime-switching model to a full regime-switching jump-diffusion: `RegimeSwitchingBsmParams` gains optional per-regime Merton jump blocks (`jump_intensities`, `jump_means`, `jump_stds`, all defaulting to no jumps, fully backward compatible). Each regime's Levy exponent becomes `-0.5 sigma_j^2 (u^2+iu) + lambda_j (phi_Y(u) - 1 - iu zeta_j)`, compensated regime-wise so `phi(-i) = 1` for any regime path; the matrix-exponential CF and the numeric CGF cumulants pick the jump blocks up automatically. Tests: zero-intensity equivalence with the pure-diffusion CF, single-regime reduction to the Merton CF and cumulants, zero-generator Merton mixture identity, martingale normalization, cross-engine (COS vs Hilbert) and parity checks, an exact occupation-time Monte Carlo (chain drawn via exponential holding times -- no time-discretization bias) at 200k paths, and jump-parameter validation.

## 0.14.0 - 2026-07-05

- Added the PROJ double-barrier pricer for 1-D Levy models as `method="proj_double_barrier"` (`proj_double_barrier_price` exported): the Kirkby (2015) Toeplitz-FFT backward induction with absorption on *both* sides of the corridor (L, U) at each of the M monitoring dates; double knock-ins via same-engine in-out parity so the discrete-monitoring bias cancels. Tests: far-barrier reduction to the COS vanilla and to the single-barrier PROJ price, BSM agreement with the eigenfunction closed form at Broadie-Glasserman-Kou continuity-corrected barriers (rel 1e-2 at M=126/252) with monotone approach to the continuous limit, Kou vs a discretely monitored 200k-path Monte Carlo with the same monitoring grid, corridor monotonicity and knock-out bounds, in-out parity to 1e-8, pipeline dispatch, validation errors.

## 0.13.0 - 2026-07-05

- Added the exact locally collared cliquet pricer for Levy models as `method="cliquet_cf"`: each period's collared return `clip(R_k, lf, lc)` is a static call spread `lf + (R-lf)^+ - (R-lc)^+` priced by COS on a unit-spot asset over the period; additive payoffs sum and multiplicative payoffs factorize by independence of Levy increments. Non-positive strikes (no floor) handled analytically via `E[(R-a)^+] = E[R] - a`. Global floors/caps couple the periods and are rejected in favor of `cliquet_mc`. `levy_cliquet_price` exported. Tests: no-collar closed forms, one-period floor-at-zero == ATM-forward COS call identity, BSM vs the existing `cliquet_mc` engine (400k paths), Kou vs in-test jump Monte Carlo (200k paths), cp sign flip, collar bounds, pipeline dispatch, global-collar/model rejection.

## 0.12.0 - 2026-07-04

- Docs audit: `docs/api_reference.md` backfilled with every public symbol shipped in 0.7.0-0.12.0 (analytic BSM Greeks, compound/chooser/quanto, SVI/SSVI/local-vol/SABR-smile calibration, regime-switching CF pair, PROJ barrier/Asian entries) and its version example refreshed; capability snapshot now lists the full method registry; stale twenty-model counts fixed in the model zoo, appendix, and README.
- Added the exact Levy forward-start pricer as `method="forward_start_cf"`: for stationary independent increments the strike-reset payoff factorizes as `V = S0 e^{-q t1} * EuropeanPrice(S0=1, K=alpha, tau)` (Rubinstein 1990 homogeneity, exact across the Levy class), with the European leg priced by COS on the model CF. `levy_forward_start_price` exported. Tests: BSM vs the Rubinstein closed form to 1e-8, zero-start reduction to the vanilla European for Kou/VG/Merton, 200k-path Kou Monte Carlo band, alpha-parity identity, pipeline dispatch, validation errors.

## 0.11.0 - 2026-07-04

Transform-methods expansion: four new pricing capabilities, inspired by the coverage of the PROJ MATLAB option-pricing toolboxes and implemented natively on the `foureng` CF stack, plus two correctness fixes surfaced by the new cross-checks.

**New engines and models**

- Added the Hilbert-transform European pricer (Feng & Linetsky 2008) as `method="hilbert"` in `price_strip`: Gil-Pelaez tail probabilities on the half-integer sinc grid `u_m = (m + 1/2)h`, exponentially convergent for strip-analytic CFs. New `HilbertGrid` dataclass; `hilbert_price_at_strikes` and `hilbert_itm_probabilities` exported. Under BSM the probabilities reproduce `N(d1)`/`N(d2)` to 1e-10.
- Added the Markov regime-switching BSM model (Buffington & Elliott 2002) as registry model `"regime_switching"`: CF via the matrix exponential `pi0' expm(T(Q + diag(psi_j(u)))) 1` with per-regime martingale drift; cumulants by 4th-order finite differences of the CGF. Prices through every CF engine. Tests include the fast-switching homogenization limit and single-regime/zero-generator degeneracies.
- Added the exact discrete geometric-Asian pricer for Levy models (Fusai & Meucci 2008) as `method="asian_cf"`: the average's CF is the finite product of per-increment CFs at scaled frequencies — no lognormal approximation. Matches the discrete Kemna-Vorst BSM closed form to 1e-8 and Kou Monte Carlo within the confidence band. `levy_geometric_asian_price` exported.
- Added exact discrete variance-swap fair strikes for Levy models as `method="variance_levy_analytic"`: per-period squared-return expectations from CF cumulants, `E[R^2] = ((r-q)dt + c1)^2 + c2`; prices jump risk (Carr-Wu 2009 discrete analogue) and collapses to the BSM closed form at zero intensity. `levy_variance_fair_strike` and `levy_variance_swap` exported.

**Fixes**

- `price_strip` now honors `cp=-1` for the call-only Fourier engines (`cos`, `cos_improved`, `cos_filtered`, `frft`, `carr_madan`) via a single parity conversion at dispatch level; previously these silently returned call prices.
- `merton_jd_cumulants` was missing the `-sigma^2/2` diffusion drift in `c1` (the CF already had it), which off-centered COS/PROJ truncation windows for Merton.
- CI restored to green: repo-wide ruff lint/format cleanup, SABR implied-vol signature widened to strike arrays, and the mypy analysis target bumped to 3.12 so it can parse numpy >= 2.5 PEP 695 stubs.

## 0.10.0 - 2026-07-01

- Added SSVI (Surface SVI) joint surface parameterization (Gatheral & Jacquier 2014) in `foureng/surface/ssvi.py`: `SSVIParams` dataclass (rho, eta, gamma), `ssvi_phi_power_law` and `ssvi_phi_heston` phi functions, `ssvi_total_variance`, `ssvi_implied_vol`, `ssvi_check_butterfly_free` (sufficient condition eta*(1+|rho|) <= 4), `ssvi_check_calendar_free` (phi non-increasing check), `fit_ssvi_surface` (two-stage joint calibration: per-slice ATM total variance then global L-BFGS-B for rho/eta/gamma).
- Added 39 tests in `tests/surface/test_ssvi.py`: param validation, phi shape/monotonicity, total variance formula, ATM recovery, arbitrage conditions, round-trip fit on synthetic SSVI data, public API.
- Added demo notebook `notebooks/supplementary/ssvi_surface.ipynb` (6 sections): phi sensitivity, smile slices, parameter sensitivity, arbitrage-free region visualization, joint calibration to Heston, SSVI vs. SVI comparison.
- Exported all SSVI symbols from `foureng/__init__.py` and `foureng/surface/__init__.py`; added to `_BASELINE_API` in `tests/meta/test_api_snapshot.py`.

## 0.9.0 - 2026-07-01

- Added quanto option pricing (Reiner 1992) in `foureng/analytics/bsm_quanto.py`: `bsm_quanto_forward` computes the domestic risk-neutral adjusted forward `F_adj = S * exp((r_dom - q_for - rho*sigma_S*sigma_X)*T)`; `bsm_quanto_option` prices calls and puts using the standard BSM formula with F_adj. FX volatility enters only through the drift, not the BSM variance. Put-call parity holds to 1e-12.
- Added `QuantoOption` product dataclass in `foureng/products/quanto.py`; wired into `foureng/products/__init__.py`.
- Added 29 tests in `tests/analytics/test_bsm_quanto.py`: forward formula checks, put-call parity, zero-vol/zero-maturity degeneracy, non-negativity sweep, rho monotonicity, input validation, public API.
- Added 18 tests in `tests/products/test_quanto_product.py` for `QuantoOption` construction and validation.
- Added demo notebook `notebooks/supplementary/quanto_options.ipynb` (5 sections): quanto-adjusted forward with rho/sigma_X sensitivity, put-call parity verification, correlation effect on prices, quanto vs. standard BSM comparison, spot sweep.
- Exported `bsm_quanto_forward`, `bsm_quanto_option`, `QuantoOption` from `foureng/__init__.py`; added to `_BASELINE_API` in `tests/meta/test_api_snapshot.py`.

## 0.8.0 - 2026-07-01

- Added Dupire (1994) local volatility surface extraction in `foureng/surface/local_vol.py`: `LocalVolSurface` dataclass, `dupire_local_vol_from_svi` (analytical k-derivatives from SVI, FD T-derivative between adjacent maturities), `dupire_local_vol_grid` (full numerical FD on an IV grid with optional Gaussian smoothing). Gatheral-Jacquier (2014) denominator `g(k,w)` used throughout; negative local variance clipped to zero.
- Added 21 tests in `tests/surface/test_local_vol.py`: shape, midpoint maturities, flat-IV -> flat-LV identity, non-negativity, finiteness, two-maturity minimum, error handling (too few maturities, mismatched SVI params, non-increasing grids), grid route shape/flat-IV/raises, SVI-vs-grid consistency, public API.
- Added demo notebook `notebooks/supplementary/local_vol.ipynb` (5 sections): Dupire formula intuition, flat-IV sanity check, SVI-calibrated LV from Heston, analytical vs. numerical route comparison, 3D surface and heatmap visualizations.
- Exported `LocalVolSurface`, `dupire_local_vol_from_svi`, `dupire_local_vol_grid` from `foureng/__init__.py` and `foureng/surface/__init__.py`; added all three to `_BASELINE_API` in `tests/meta/test_api_snapshot.py`.

## 0.7.0 - 2026-07-01

- Added BSM Greeks: `bsm_delta`, `bsm_gamma`, `bsm_vega`, `bsm_theta`, `bsm_rho`, `bsm_vanna`, `bsm_volga`, `bsm_all_greeks` in `foureng/analytics/bsm_greeks.py`. All Greeks are validated against finite-difference benchmarks; `bsm_theta` returns dV/dt (time, not calendar), `bsm_rho` is scaled by 1/100. BSM PDE identity verified at machine precision across a spot/maturity/vol grid.
- Added compound option pricing: `geske_compound_price` in `foureng/analytics/bsm_compound.py`, implementing the Geske (1979) / Haug (2007) formula for all four types (call-on-call, put-on-call, call-on-put, put-on-put). Critical stock price solved via Brent's method; bivariate normal CDF via `scipy.stats.multivariate_normal.cdf`. Put-call parity and lower-bound identities pass to 1e-6.
- Added `CompoundOption` product dataclass in `foureng/products/compound.py`, wired via `method="geske"` in `price()`.
- Added chooser option pricing: `bsm_chooser_price` in `foureng/analytics/bsm_chooser.py` via the Rubinstein (1991) decomposition `chooser = call(S, K*, T_choice) + put(S, K, T_exp)` where `K* = K * exp(-(r-q)*(T_exp-T_choice))`. Rubinstein decomposition holds to 1e-12.
- Added `ChooserOption` product dataclass in `foureng/products/chooser.py`, wired via `method="analytic"` in `price()`.
- Added SABR smile calibration: `calibrate_sabr_smile` and `SabrSmileCalibResult` in `foureng/surface/calibration.py`, fitting SABR (alpha, nu, rho) to a single-maturity implied-vol smile with optional beta specification.
- Added demo notebooks: `notebooks/supplementary/bsm_greeks.ipynb` (6-section Greeks explorer with BSM PDE verification and delta-hedging simulation), `notebooks/supplementary/compound_and_chooser.ipynb` (Geske compound pricing, put-call parity, spot sweep, chooser vs straddle), `notebooks/supplementary/calibration.ipynb` (Heston, VG, SABR, and multi-model calibration to IV surface).
- Exported all new public symbols from `foureng/__init__.py`; updated `tests/meta/test_api_snapshot.py` baseline to include all new names.
- Added SVI (Stochastic Volatility Inspired) smile parameterization (Gatheral 2004) in `foureng/surface/svi.py`: `SVIParams` dataclass, `svi_total_variance`, `svi_implied_vol`, `svi_butterfly_density`, `svi_check_butterfly_arbitrage`, `fit_svi_smile`. Butterfly arbitrage check via Gatheral-Jacquier (2014) $g(k) \geq 0$ criterion. Calibration via L-BFGS-B with optional global differential-evolution pre-fit. 36 tests in `tests/surface/test_svi.py`. Demo notebook `notebooks/supplementary/svi_calibration.ipynb` with 6 sections: raw surface, parameter sensitivity, butterfly arbitrage $g(k)$, Heston smile calibration, term structure, and SVI vs SABR comparison.
- Added tests: `tests/analytics/test_bsm_greeks.py` (37 tests), `tests/analytics/test_bsm_compound.py` (33 tests), `tests/analytics/test_bsm_chooser.py` (25 tests).
- Added `CompoundOption` and `ChooserOption` to `foureng/products/__init__.py` and to `tests/products/test_product_dataclasses.py`.

## 0.6.0 - 2026-06-14

- Added `proj_barrier_price`: PROJ single-barrier European option pricer for all 1-D Lévy models (Kirkby 2015 backward induction with barrier absorption). Supports all four barrier types (down-out, up-out, down-in via in-out parity, up-in) for calls and puts. Wired as `method="proj_barrier"` in `price()`.
- Added `proj_asian_price_cv`: arithmetic Asian MC pricer with PROJ-computed geometric control variate. Uses BSM analytic geometric formula for BSM model, PROJ European with adjusted cumulants for other Lévy models. Wired as `method="proj_asian"` in `price()`.
- Fixed PROJ barrier backward induction: removed spurious B-spline re-projection step in the loop (only needed in Bermudan for early exercise re-projection; for European barrier options it caused monotone value decay with M).
- Added Sprint 3 BSM closed-form exotics: `analytic_bsm.py` (digital, geometric Asian, forward-start, single-barrier Reiner-Rubinstein 1991, floating/fixed-strike lookback Conze-Viswanathan 1991).
- Added Sprint 4 path-dependent MC engines: `GBMPathSpec`/`simulate_gbm_paths`, arithmetic Asian MC (geometric-average CV), barrier MC (BGK 1999 continuity correction), lookback MC, variance swap/option MC. `mc_gbm` registered in `METHOD_REGISTRY`.
- Registered `proj_barrier` and `proj_asian` in `capabilities.py` METHOD_REGISTRY.

## 0.5.0 - 2026-06-10

- Replaced the COS-backed PROJ facade with a real PROJ frame-projection engine (Kirkby 2015/2017). `proj_price_at_strikes` ports `PROJ_European.m` with Haar/linear/quadratic/cubic B-spline orders, `proj_auto_grid` builds a cumulant-driven `ProjGrid`, and `proj_bermudan_put` ports the `PROJ_Bermudan_Put.m` Toeplitz-FFT backward recursion. European PROJ matches COS to ~1e-7 across the Levy family; Bermudan PROJ matches `cos_bermudan` to 1e-5 to 1e-3. See [docs/proj_parity_roadmap.md](docs/proj_parity_roadmap.md).
- Added generic Monte Carlo dispatch via `mc_price` and the `MCSpec`/`MCResult` dataclasses. Covers European, American (LSMC), Bermudan, Asian, barrier, double-barrier, lookback, variance, cliquet, exchange, basket, spread, and best-of products under BSM.
- Added multi-asset pricing routes: `ExchangeOption` via Margrabe closed form and correlated MC, `BasketOption`, `SpreadOption` via Kirk approximation and correlated MC, and `BestOfOption`.
- Added `margrabe_exchange` and `kirk_spread` as standalone public functions.
- Added `LookbackOption` pricing via closed-form floating-strike BSM and Monte Carlo.
- Added `VarianceSwap` and `VarianceOption` pricing via analytic BSM and Monte Carlo.
- Added `CliquetOption` pricing via Monte Carlo.
- Added `DoubleBarrierOption` pricing via Monte Carlo.
- Added `ForwardStartOption` pricing via closed-form BSM.
- Added `calibrate_cgmy` and `calibrate_nig` surface calibration functions.
- Added `bsm_geometric_asian` and `bsm_geometric_asian_parity` to the public analytics API.
- Added `bsm_gap_call`, `bsm_cash_or_nothing`, and `bsm_asset_or_nothing` to the public analytics API.
- Added Sprint 3 BSM closed-form exotics: `analytic_bsm.py` (digital, geometric Asian,
  forward-start, single-barrier Reiner-Rubinstein 1991, lookback Conze-Viswanathan 1991);
  lookback floating-strike formula corrected and MC-verified; 6 product test files.
- Added Sprint 4 path-dependent MC engines: `GBMPathSpec`/`simulate_gbm_paths`, arithmetic
  Asian MC (geometric-average CV), barrier MC (BGK 1999 continuity correction), lookback MC,
  variance swap/option MC; `mc_gbm` registered in `METHOD_REGISTRY`; 4 product test files.
- Added repository-wide quality gates for `tests/` linting with documented test-specific exceptions.
- Added `mypy` type-checking support for the `foureng/` package.
- Added Hypothesis-backed property tests for numerical invariants and model reductions.
- Added a `pyperf` benchmark harness for canonical pricing cases.
- Added contributor and citation metadata for research-project hygiene.
- Updated API snapshot test to reflect the full current public API surface.
- Removed all em-dashes from documentation prose.

## 0.4.1 - 2026-05-13

- Final submission release with package publication, notebook reproducibility fixes, and validation/reporting polish.
