# Bates and 3/2 SV Validation Cases

Detailed validation record for the Bates stochastic-volatility jump model and the 3/2
stochastic-volatility model.

**Validation notebook:** [`notebooks/paper_replications/bates_sv32_validation_demo.ipynb`](../notebooks/paper_replications/bates_sv32_validation_demo.ipynb)  
**Bates test files:** `tests/papers/test_phase5_bates_mathworks.py`, `tests/papers/test_bates_mathworks_ni_surface.py`, `tests/papers/test_bates_mathworks_fft_frft.py`, `tests/features/test_bates_mathworks_delta.py`  
**SV32 test files:** `tests/models/test_sv32_pyfeng_paper.py`, `tests/models/test_sv32_pyfeng_reference.py`, `tests/models/test_sv32_pyfeng_surface_reference.py`

---

## Bates model parameters (MathWorks reference)

| Parameter | Value |
|-----------|-------|
| Spot S0 | 80.0 |
| Risk-free rate r | 0.03 |
| Dividend yield q | 0.02 |
| Initial variance v0 | 0.04 |
| Long-run variance θ | 0.05 |
| Mean-reversion speed κ | 1.0 |
| Vol-of-vol ν | 0.2 |
| Correlation ρ | −0.7 |
| Jump intensity λ_J | 2.0 |
| Mean jump % | 0.02 |
| Jump vol σ_J | 0.08 |
| Converted log-jump mean μ_J | 0.01660262729617973 |

Reference: MathWorks Financial Toolbox `optByBatesNI`, `optByBatesFFT`, `optSensByBatesNI`.
Frozen values stored in `tests/refs/bates_mathworks_ni.json` and `tests/refs/bates_mathworks_fft_frft.json`.

---

## Bates validation cases (BATES-01 through BATES-07)

### BATES-01 — NI scalar price

- **What:** Single ATM call (K=80, T=0.5) via `cos_improved` vs MathWorks `optByBatesNI` value (5.3484).
- **Reference type:** `software_reference`
- **Tolerance:** atol=1e-2 (MathWorks publishes 4 decimal places; grid-convention gap ~7.6e-3).
- **Test:** `tests/papers/test_phase5_bates_mathworks.py`

### BATES-02 — NI five-strike strip

- **What:** Five strikes [76, 78, 80, 82, 84] at T=0.5 via all six pricers vs MathWorks NI strip.
- **Reference type:** `software_reference`
- **Tolerance:** atol=1e-2
- **Test:** `tests/papers/test_phase5_bates_mathworks.py`

### BATES-03 — NI 5×6 surface

- **What:** Five strikes × six maturities (T=0.5, 1.0, 1.5, 2.0, 2.5, 3.0) via `cos_improved` vs MathWorks `optByBatesNI` surface.
- **Reference type:** `software_reference`
- **Tolerance:** atol=1e-2
- **Test:** `tests/papers/test_bates_mathworks_ni_surface.py`

### BATES-04 — COS N-convergence

- **What:** COS-improved ATM price at [16, 32, 64, 128, 256, 512, 1024] terms; monotone error decrease.
- **Reference type:** `numerical_stability`
- **Tolerance:** qualitative (convergence shape check)
- **Notebook section:** BATES-04 in `bates_sv32_validation_demo.ipynb`

### BATES-05 — IV smile

- **What:** Black-Scholes implied volatility smile from Bates prices across strikes at T=0.5 and T=1.0; no-arbitrage shape check (positive vol, left-skewed smile).
- **Reference type:** `qualitative_figure`
- **Tolerance:** IV > 0 and decreasing from left to right
- **Notebook section:** BATES-05

### BATES-06 — FRFT 5×6 surface

- **What:** FRFT-based 5×6 surface vs internal `cos_improved` reference (not MathWorks truncated values).
- **Reference type:** `derived_reference` (cross-method parity)
- **Tolerance:** atol=1e-2 vs MathWorks; atol=1e-3 vs internal `cos_improved`
- **Test:** `tests/papers/test_bates_mathworks_fft_frft.py`

### BATES-07 — Delta vector

- **What:** Five-strike delta vector at T=0.5 vs MathWorks `optSensByBatesNI`.
- **Reference type:** `software_reference`
- **Tolerance:** atol=5e-3 (finite-difference approximation; MathWorks publishes 4 d.p.)
- **Test:** `tests/features/test_bates_mathworks_delta.py`

---

## 3/2 SV model parameters (PyFENG regression)

| Parameter | Value |
|-----------|-------|
| Initial variance v0 | 0.09 |
| Mean-reversion speed κ | 22.84 |
| Long-run variance θ | 0.92 |
| Vol-of-vol ν | 8.56 |
| Correlation ρ | −0.99 |

Frozen pyfeng_fft reference surface in `tests/refs/sv32_pyfeng_surface.json` (7×4 grid:
T = 0.25, 0.5, 1.0, 2.0; K = 80, 90, 100, 110, 120, 130, 140).

---

## SV32 validation cases (SV32-01 through SV32-05)

### SV32-01 — PyFENG surface regression

- **What:** 7×4 surface via `cos_improved` vs frozen pyfeng_fft reference surface.
- **Reference type:** `adapter` / `derived_reference`
- **Tolerance:** atol=5e-4 (pyfeng_fft) and atol=1e-3 (cos_improved vs pyfeng_fft)
- **Test:** `tests/models/test_sv32_pyfeng_surface_reference.py`

### SV32-02 — COS vs PyFENG across maturities

- **What:** Seven-strike COS-improved prices vs pyfeng_fft for T = 0.25, 0.5, 1.0, 2.0.
- **Reference type:** `adapter`
- **Tolerance:** max error ≤ 1.5e-3
- **Test:** `tests/models/test_sv32_pyfeng_paper.py`

### SV32-03 — Lewis T ≥ 0.5

- **What:** Lewis Fourier inversion vs pyfeng_fft for T ≥ 0.5 (Lewis is unstable at very
  short maturities for the 3/2 model).
- **Reference type:** `derived_reference`
- **Tolerance:** max error ≤ 1.5e-3
- **Test:** `tests/models/test_sv32_pyfeng_paper.py`

### SV32-04 — IV surface shape

- **What:** Black-Scholes implied volatility surface for the 3/2 model; no-arbitrage bounds
  and hump-shaped term structure check.
- **Reference type:** `qualitative_figure`
- **Tolerance:** IV ∈ (0, 2), term structure humped
- **Notebook section:** SV32-04 in `bates_sv32_validation_demo.ipynb`

### SV32-05 — N-convergence

- **What:** COS-improved ATM price at [16, 32, 64, 128, 256, 512, 1024] terms; monotone
  error decrease to machine-precision plateau.
- **Reference type:** `numerical_stability`
- **Tolerance:** qualitative
- **Notebook section:** SV32-05

---

## Baldeaux-Badran qualitative smoke test

The paper Baldeaux & Badran (2012) does not publish a price table; it publishes a VIX and
equity IV figure.
The corresponding test (`tests/papers/test_sv32_baldeaux_badran_original_smoke.py`) is
therefore marked `qualitative_figure` and `xfail-if-unstable`: it checks smile shape
(positive IV, left skew) but not exact numeric values.
Parameters are stored in `tests/refs/sv32_baldeaux_badran_figure_params.json`.

---

## Tolerance notes

The gap between this repo's prices and MathWorks prices is structural, not a model error:

- MathWorks publishes values truncated to 4 decimal places.
- MathWorks centres the FFT grid differently from this repo's log(F0)-centred convention.
- The resulting convention gap reaches approximately 7.6e-3 in practice.

The achievable tolerance for a `software_reference` comparison is therefore atol=1e-2.
For internal cross-method parity (e.g. FRFT vs `cos_improved`) the achievable tolerance
is atol=1e-3.
