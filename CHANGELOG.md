# Changelog

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
