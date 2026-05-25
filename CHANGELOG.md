# Changelog

## Unreleased

- Added 6 new characteristic-function models, bringing the catalogue to 26 total:
  - `heston_nig`  — Heston SV + NIG jumps (Cont-Tankov 2004)
  - `heston_vg`   — Heston SV + VG jumps (Cont-Tankov 2004)
  - `svjj`        — SV with simultaneous correlated price and variance jumps (Duffie-Pan-Singleton 2000)
  - `bns_gamma_ou`— BNS Gamma-OU stochastic variance (Barndorff-Nielsen & Shephard 2001; CF via Nicolato-Venardos 2003)
  - `nts`         — Normal Tempered Stable, Kim-Rachev-Rüschendorf (2008) tempered stable parametrisation
  - `cgmysa`      — CGMY on CIR stochastic arrival clock (Carr-Geman-Madan-Yor 2003)
- Added `tests/models/test_new_models_batch.py` covering all 6 new models with 49 tests (CF structure, COS/CM cross-engine, no-arbitrage, model-specific structural checks).
- Test suite grows to 820 cases.
- Updated README, `docs/model_zoo.md`, `docs/api_reference.md`, and `docs/paper_validation_matrix.md` to reflect the 26-model catalogue.
- Added repository-wide quality gates for `tests/` linting with documented test-specific exceptions.
- Added `mypy` type-checking support for the `foureng/` package.
- Added Hypothesis-backed property tests for numerical invariants and model reductions.
- Added a `pyperf` benchmark harness for canonical pricing cases.
- Added contributor and citation metadata for research-project hygiene.

## 0.4.1 - 2026-05-13

- Final submission release with package publication, notebook reproducibility fixes, and validation/reporting polish.
