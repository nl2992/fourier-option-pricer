# Documentation Index

Reference documentation for the `foureng` / `fourier-option-pricer` package.

---

| Document | Contents |
|----------|----------|
| [MODEL_ZOO.md](MODEL_ZOO.md) | Complete catalogue of all 20 supported models — parameter dataclasses, CF sources, PyFENG dependency note, and `MODEL_REGISTRY` usage. |
| [API_REFERENCE.md](API_REFERENCE.md) | Full public API tables: market inputs, parameter dataclasses, characteristic functions, cumulants, grid objects, core pricing functions, filtered-COS helpers, implied vol, surfaces/calibration/Greeks. |
| [VALIDATION_HIERARCHY.md](VALIDATION_HIERARCHY.md) | The five evidence levels (`external_reference` → `qualitative_figure`), current status summary by level, and instructions for adding a new validation case. |
| [BATES_SV32_VALIDATION.md](BATES_SV32_VALIDATION.md) | Detailed validation record for Bates (BATES-01–07) and 3/2 SV (SV32-01–05): parameters, reference types, tolerances, and test-file links. |
| [FO2008_REPLICATION.md](FO2008_REPLICATION.md) | Full paper-faithful Fang & Oosterlee (2008) replication tables (Tables 1–10), interpretation notes, and improved-COS summary table. |
| [FILTERED_COS_EXTENSION.md](FILTERED_COS_EXTENSION.md) | The adaptive filtered-COS extension: motivation, spectral filter formulas, adaptive policy selector, conservative framing, output files, and test coverage. |
| [paper_validation_matrix.md](paper_validation_matrix.md) | Per-paper validation matrix linking each claim to its test file, reference type, and numeric target. |
| [PACKAGING.md](PACKAGING.md) | PyPI package identity, install instructions, runtime dependencies, CI setup, and release checklist. |

---

## Related top-level files

| File | Contents |
|------|----------|
| [README.md](../README.md) | Project overview, installation, quick start, core methods, validation summary, innovation section, notebooks, and key papers. |
| [APPENDIX.md](../APPENDIX.md) | Methodology notes, derivations, FO2008 pointer, filtered-COS pointer, and full references. |
| [PAPERS.md](../PAPERS.md) | Comprehensive bibliography: all papers cited in the codebase, grouped by category, with DOIs and free-access links. |
| [tests/README.md](../tests/README.md) | Test-folder map and `pytest` mark guide. |
| [methodology_and_results.md](../methodology_and_results.md) | Jump-convention derivation, reference values, and rationale for qualitative vs exact 3/2 validation. |
