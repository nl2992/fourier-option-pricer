# Changelog

## Unreleased

- Replaced the COS-backed PROJ façade with a real **PROJ frame-projection engine** (Kirkby 2015/2017): `proj_price_at_strikes` ports `PROJ_European.m` with Haar/linear/quadratic/cubic B-spline orders, `proj_auto_grid` builds a cumulant-driven `ProjGrid`, and `proj_bermudan_put` ports the `PROJ_Bermudan_Put.m` Toeplitz-FFT backward recursion. The existing `proj_european_price_at_strikes` entry point and `method="proj"` dispatch are unchanged (now routed through the real engine). European PROJ matches COS to ~1e-7 across the Lévy family; Bermudan PROJ matches `cos_bermudan` to 1e-5–1e-3. See [docs/proj_parity_roadmap.md](docs/proj_parity_roadmap.md).
- Added repository-wide quality gates for `tests/` linting with documented test-specific exceptions.
- Added `mypy` type-checking support for the `foureng/` package.
- Added Hypothesis-backed property tests for numerical invariants and model reductions.
- Added a `pyperf` benchmark harness for canonical pricing cases.
- Added contributor and citation metadata for research-project hygiene.

## 0.4.1 - 2026-05-13

- Final submission release with package publication, notebook reproducibility fixes, and validation/reporting polish.
