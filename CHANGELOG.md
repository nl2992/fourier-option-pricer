# Changelog

## Unreleased

### Sprint 4 — Path-dependent MC engines

- **`foureng/mc/path_engine.py`**: Batched GBM log-Euler path simulator (`simulate_gbm_paths`)
  with antithetic variates and configurable `GBMPathSpec`.
- **`foureng/mc/asian_mc.py`**: Arithmetic Asian call/put MC with geometric-average control
  variate (`bsm_geometric_asian` as known-mean control).
- **`foureng/mc/barrier_mc.py`**: Single-barrier MC with Broadie-Glasserman-Kou (1999)
  continuity correction; knock-in via in-out parity.
- **`foureng/mc/lookback_mc.py`**: Floating- and fixed-strike lookback MC (S₀ included in
  running min/max).
- **`foureng/mc/variance_mc.py`**: Variance swap fair rate and variance option MC under BSM.
- Added `"mc_gbm"` to the `METHOD_REGISTRY` capability registry.
- 4 new product test files covering Asian, barrier, lookback, and variance MC engines.

### Sprint 3 — Closed-form exotic references

- **`foureng/pricers/analytic_bsm.py`**: BSM closed forms for European, cash/asset-or-nothing
  digitals, discrete geometric Asian, forward-starting, single-barrier (Reiner-Rubinstein 1991),
  and floating/fixed-strike lookback (Conze-Viswanathan 1991).
- **`foureng/pricers/cos_digital.py`**: COS payoff coefficients for digital options.
- Pipeline and capability registry updated for `DigitalOption` dispatch and `bsm_analytic` method.
- 6 new product test files; lookback floating-strike formula corrected and MC-verified.

### Earlier

- Added repository-wide quality gates for `tests/` linting with documented test-specific exceptions.
- Added `mypy` type-checking support for the `foureng/` package.
- Added Hypothesis-backed property tests for numerical invariants and model reductions.
- Added a `pyperf` benchmark harness for canonical pricing cases.
- Added contributor and citation metadata for research-project hygiene.

## 0.4.1 - 2026-05-13

- Final submission release with package publication, notebook reproducibility fixes, and validation/reporting polish.
