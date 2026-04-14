# Documentation Index

Reference documentation for the `foureng` / `fourier-option-pricer` package.
Start at [README.md](../README.md) for the project overview; come here for depth.

---

## Package reference

| Document | Contents |
|----------|----------|
| [model_zoo.md](model_zoo.md) | Complete catalogue of all 20 supported models  -  parameter dataclasses, CF sources, PyFENG dependency note, and `MODEL_REGISTRY` usage. |
| [api_reference.md](api_reference.md) | Full public API tables: market inputs, parameter dataclasses, characteristic functions, cumulants, grid objects, core pricing functions, filtered-COS helpers, implied vol, surfaces/calibration/Greeks. |
| [packaging.md](packaging.md) | PyPI package identity, install instructions, runtime dependencies, CI setup, and build/release checklist. |
| [papers.md](papers.md) | Comprehensive bibliography: all papers cited in the codebase, grouped by category, with DOIs and free-access links. |
| [numerical_notes.md](numerical_notes.md) | Known numerical limitations: COS truncation failure modes, Carr-Madan alpha conditions, PyFENG version caveats, parameter edge cases, IV inversion guidance. |

## Validation and results

| Document | Contents |
|----------|----------|
| [validation_hierarchy.md](validation_hierarchy.md) | The five evidence levels (`external_reference` → `qualitative_figure`), current status summary, and instructions for adding a new validation case. |
| [paper_validation_matrix.md](paper_validation_matrix.md) | Per-paper validation matrix: every claim linked to its test file, reference type, and numeric target. |
| [fo2008_replication.md](fo2008_replication.md) | Paper-faithful Fang & Oosterlee (2008) replication tables (Tables 1–10), interpretation notes, and improved-COS summary with benchmark CSV links. |
| [bates_sv32_validation.md](bates_sv32_validation.md) | Detailed validation record for Bates (BATES-01–07) and 3/2 SV (SV32-01–05): parameters, mu_j formula, reference types, tolerances, and test-file links. |

## Extensions and workflow

| Document | Contents |
|----------|----------|
| [filtered_cos_extension.md](filtered_cos_extension.md) | Adaptive filtered-COS extension: motivation, spectral filter formulas, policy-search selector, conservative framing, output files, and test coverage. |
| [ai_workflow_and_contribution.md](ai_workflow_and_contribution.md) | AI-assisted development workflow, library reuse policy, original contributions, and validation gate log. |

---

## Related top-level files

| File | Contents |
|------|----------|
| [README.md](../README.md) | Project overview, installation, quick start, API reference, notebooks, and key papers. Follows the instructor-designated 7-section structure. |
| [appendix.md](../appendix.md) | Methodology derivations, model conventions, benchmark interpretation, and the full numbered course-project narrative (sections 1–18). |
| [tests/README.md](../tests/README.md) | Test-folder map and `pytest` mark guide. |
