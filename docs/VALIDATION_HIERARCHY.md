# Validation Hierarchy

This document explains the five evidence levels used to classify test cases in this project.
Each test in `tests/papers/`, `tests/models/`, and `tests/methods/` is tagged with one of
these levels via the `reference_type` field in the corresponding JSON fixture or via the
`pytest.mark` on the test module.

---

## The five levels

| Level | Tag | What it means | Numeric target? | Example |
|-------|-----|---------------|-----------------|---------|
| 1 (strongest) | `external_reference` | Exact price copied from a published paper table | Yes — paper | Carr & Madan (1999) VG Case 4 put prices |
| 2 | `software_reference` | Exact price from official software documentation | Yes — software | MathWorks `optByBatesNI` / `optByBatesFFT` |
| 3 | `derived_reference` | High-precision in-house computation frozen at generation time; reproducible but not from a paper | Yes — derived | Double Heston vanilla price table; pyfeng_fft BSM baseline |
| 4 | `adapter` | Cross-package parity: our CF agrees with PyFENG's `logp_cf` to tight tolerance | Yes — derived | Heston, VG, BSM adapter tests |
| 5 (weakest) | `qualitative_figure` / `numerical_stability` | Shape checks, convergence rate checks, stress-regime robustness — no single exact numeric target | No — figure only | Baldeaux-Badran 3/2 IV smile shape; Junike stress tests |

---

## Validation matrix

See [paper_validation_matrix.md](paper_validation_matrix.md) for the complete per-model table
linking each paper claim to its test file, reference type, and numeric target.

---

## Current status summary

| Reference type | Models / methods covered | Count | All passing? |
|----------------|--------------------------|-------|-------------|
| `external_reference` | Carr-Madan VG, Lewis Heston, Double Heston Kelly (2025) | 3 papers | Yes (done) |
| `software_reference` | MathWorks Bates NI surface, FFT subset, FRFT surface, Delta | 4 test files | Yes (done) |
| `derived_reference` | BSM all-pricers, Merton JD, Kou, Meixner, Bilateral Gamma, GH, FMLS, Double Heston, VGSA, GARCH | 10+ test files | Partial |
| `adapter` | BSM, Heston, VG, CGMY, NIG, OUSV, 3/2 SV, Rough Heston | 8 models | Partial |
| `qualitative_figure` | Baldeaux-Badran 3/2 SV figure params | 1 test file | xfail-if-unstable |
| `numerical_stability` | Junike improved COS, filtered COS, Albrecher branch | 5+ test files | Partial |

---

## Why this matters for the course project

The rubric asks for validation against published references. This hierarchy makes explicit
what "validated" means at each level:

- **Levels 1–2** satisfy the strictest interpretation: prices match a published table or
  official reference implementation.
- **Level 3** satisfies reproducibility: a frozen snapshot ensures the same output on every
  CI run, even without a published table.
- **Level 4** satisfies cross-package consistency: if PyFENG agrees with our CF, we have
  independent algebraic verification.
- **Level 5** satisfies qualitative correctness: no-arbitrage bounds hold, error decays with
  N, the model produces a plausible smile shape.

Most models in this repo are at Level 3–4. Levels 1–2 cover the key benchmark models
(Carr-Madan VG, Lewis Heston, Bates MathWorks, Double Heston Kelly).

---

## Adding a new validation case

1. Choose the reference level (1–5) for the new case.
2. For levels 1–2: copy the exact numeric value(s) from the published source into a JSON
   fixture under `tests/refs/`.
3. For levels 3–4: generate a high-precision reference using the `pyfeng_fft` oracle or a
   fine Carr-Madan grid, freeze it in a JSON fixture, and document the generation parameters.
4. Add a test in the appropriate `tests/papers/` or `tests/models/` subfolder with the
   correct `pytest.mark` (`@pytest.mark.paper`, `@pytest.mark.software_reference`, etc.).
5. Add a row to [paper_validation_matrix.md](paper_validation_matrix.md).
