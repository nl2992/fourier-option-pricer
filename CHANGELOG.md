# Changelog

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
- Added repository-wide quality gates for `tests/` linting with documented test-specific exceptions.
- Added `mypy` type-checking support for the `foureng/` package.
- Added Hypothesis-backed property tests for numerical invariants and model reductions.
- Added a `pyperf` benchmark harness for canonical pricing cases.
- Added contributor and citation metadata for research-project hygiene.
- Updated API snapshot test to reflect the full current public API surface.
- Removed all em-dashes from documentation prose.

## 0.4.1 - 2026-05-13

- Final submission release with package publication, notebook reproducibility fixes, and validation/reporting polish.