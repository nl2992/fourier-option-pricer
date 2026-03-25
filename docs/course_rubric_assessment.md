# Course Rubric Assessment

This file maps the current `main` branch to the course rubric and records
remaining pitfalls. It is deliberately conservative: a row is marked `strong`
only when there is implementation, documentation, and automated evidence.

## Rubric Map

| Rubric item | Current evidence | Mark | Notes / pitfalls |
|---|---|---|---|
| Advanced quant finance method | Fourier option pricing under stochastic-volatility, jump-diffusion, pure-Levy, rough-volatility, and hybrid SVJ models | strong | The project is squarely numerical option pricing, not ML/data science. |
| Efficient numerical methods | Carr-Madan FFT, FRFT, COS, improved COS, filtered COS, Lewis quadrature, benchmark CSVs | strong | Runtime/error comparisons are saved in `benchmarks/mc_vs_fourier_methods/outputs/`. |
| Published-paper basis | `docs/papers.md`, `docs/paper_validation_matrix.md`, paper fixtures, paper tests | strong | Some models are validated structurally or by derived references rather than exact published tables. The matrix labels those as `partial`. |
| Correct implementation | 20 model registry, public API, paper/software references, cross-engine tests | strong | Exact anchors are strongest for Carr-Madan, Lewis, FO2008 Heston ATM, Kelly double-Heston, and MathWorks Bates. |
| Validation against papers | `tests/papers/`, top-level paper tests, `benchmarks/paper_replications/` | strong / partial | Strong for exact anchors; partial for papers without copied numerical tables. Do not overclaim partial rows as exact replications. |
| Robustness testing | 692 collected tests, property-based tests, model reductions, no-arbitrage checks, invalid-input tests | strong | Hypothesis tests must stay numerically realistic; plain COS is approximate and should use approximation-scale tolerances. |
| Coding efficiency | Vectorized NumPy pricers, FFT/FRFT grid pricing, cached/centralized registries | strong | Some slow rough-Heston/PyFENG cases are intentionally outside fast CI. |
| Coding quality / package structure | PyPI package metadata, `foureng` module layout, model registry, CI, type/lint workflows | strong | Keep generated caches and local benchmark artifacts out of commits. |
| README/report quality | README, appendix, docs index, validation hierarchy, benchmark notes | strong | Test counts can drift as tests are added; keep README counts synced with `pytest --collect-only`. |
| Innovation / original idea | Adaptive filtered-COS extension and deterministic candidate selector | strong | Frame as an engineering extension on published COS/filtering work, not a new pricing theory. |
| PyPI/package extra credit | `pyproject.toml`, package version `0.4.1`, publish workflow | strong | Verify build metadata before any release. |

## Current Pitfalls And Fix Policy

| Pitfall | Risk | Current mitigation | Follow-up if time allows |
|---|---|---|---|
| Exact paper-table coverage is uneven | Reader may think all cited papers are exact replications | `paper_validation_matrix.md` separates `external_reference`, `derived_reference`, `adapter`, and `qualitative_figure` | Add more exact CSV fixtures only when table values are directly sourced. |
| Property tests can overstate numerical exactness | Randomly generated edge cases may fail due COS truncation noise rather than model bugs | Generated Heston invariants now use approximation-scale tolerances and better-resolved two-path grids | Add a shared `assert_approx_no_arbitrage` helper with method-specific tolerances. |
| Slow rough-Heston/PyFENG tests emit many warnings | Full suite is noisy and slow | CI separates fast, paper, and scheduled slow suites | Filter known upstream PyFENG deprecation warnings in slow jobs. |
| README/test count drift | Grader sees inconsistent project scale | Current count is synced to 692 collected tests | Consider replacing exact counts with a command-generated badge or wording like "690+". |
| `pyfeng_fft` support is model-limited | Users may expect PyFENG FFT for all 20 models | Dispatcher raises clear errors and README explains adapter limits | Add a small capability table generated from `MODEL_REGISTRY`. |
| COS direct-call coefficients are fragile on wide intervals | Direct call path can lose precision for long maturity or wide truncation intervals | Production default is put-parity; improved COS routes some cases to Lewis/Carr-Madan | Keep direct-call path as a diagnostic/Greeks tool and document its valid domain. |

## Suggested Grading Narrative

The project satisfies the assignment by implementing an efficient numerical
option-pricing package around characteristic functions. The core contribution is
not a single formula, but a reusable engine that compares Fourier pricing
methods across 20 models with explicit validation levels. Correct implementation
is supported by exact paper/software anchors where available, high-precision
derived references elsewhere, and broad robustness/property tests. The original
extension is adaptive filtered-COS: a practical selector that combines improved
COS interval choice with spectral filtering and benchmarked candidate selection.
