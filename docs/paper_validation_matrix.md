# Paper Validation Matrix

This matrix is the single source of truth for what is validated, at what level, and where the evidence lives.
Every reported result in the README and APPENDIX can be traced back to a row here.

Evidence-level definitions are in [validation_hierarchy.md](validation_hierarchy.md).
Quick key:

## Validation snapshot

| Summary item | Count / note |
| --- | --- |
| Total tracked validation rows | 27 |
| `done` | 13 |
| `partial` | 13 |
| `xfail-if-unstable` | 1 |
| Exact published-paper anchors | Carr-Madan (1999), Lewis (2001), FO2008 Heston ATM, Kelly (2025) |
| Exact software anchors | MathWorks Bates price, surface, FFT/FRFT, and delta references |
| Derived / internal anchors | Frozen regression strips, high-resolution Fourier oracles, improved-COS comparison outputs |

## Status guide

- `done`: the repo has strong evidence for the stated result and the linked tests or artifacts are current.
- `partial`: the implementation is validated, but the evidence is more structural, derived, or incomplete than a direct paper-table replay.
- `xfail-if-unstable`: the case is retained intentionally as a documented known-instability or figure-only regime.

| Tag | Meaning |
|-----|---------|
| `external_reference` | Exact price from a published paper table |
| `software_reference` | Exact price from official software docs (e.g. MathWorks) |
| `derived_reference` | High-precision internal reference frozen at generation time |
| `adapter` | Cross-package parity with PyFENG's `logp_cf` |
| `numerical_stability` | Stress/convergence test; no single exact numeric target |
| `qualitative_figure` | Shape checks only; paper has no price table |
| Paper | Repo model/method | Exact paper table/equation used | Test file | Reference type | Exact numeric target? | Status |
|---|---|---|---|---|---|---|
| Carr & Madan (1999) | `carr_madan`, `VGParams` | VG Case 4 put prices | `tests/test_paper_carr_madan_1999.py`; `tests/papers/test_phase2_carr_madan_vg.py` | `external_reference` | yes-paper | done |
| Chourdakis (2004) | `frft` | FRFT grid property and Lewis-Heston strip agreement | `tests/test_paper_chourdakis_2004.py`; `tests/papers/test_phase3_frft.py` | `derived_reference` / `numerical_stability` | yes-derived | done |
| Fang & Oosterlee (2008) | `cos`, `BsmParams`, `HestonParams`, `VGParams`, `CgmyParams` | Table 1 Heston ATM; FO2008 BSM/Heston/VG/CGMY replay rows | `tests/test_paper_fang_oosterlee_2008.py`; `tests/papers/test_phase4_cos_heston_fo2008.py`; `benchmarks/paper_replications/fo2008_cos/params.py` | `external_reference` / `derived_reference` | yes-paper | partial |
| Heston (1993) | `HestonParams`, `heston_cf_form2`, `lewis_call_prices` | Semi-closed Fourier/Lewis Heston integration benchmark; no original Heston table copied into fixtures | `tests/test_paper_lewis_2001.py`; `tests/methods/test_cos_vs_pyfeng_fft_heston_vg.py` | `external_reference` / `adapter` | yes-paper | partial |
| Albrecher et al. (2007), Little Heston Trap | `heston_cf_form2` | Stable characteristic-function branch behavior under difficult Heston regimes | `tests/methods/test_phase4_alpha_validity.py`; `tests/methods/test_robustness_parametrize.py` | `numerical_stability` | no-figure-only | partial |
| Lewis (2001) | `lewis_call_prices` | Five-strike Heston strip used as independent Fourier integration benchmark | `tests/test_paper_lewis_2001.py`; `tests/test_paper_chourdakis_2004.py` | `external_reference` | yes-paper | done |
| Madan, Carr & Chang (1998) | `VGParams`, `vg_cf`, `cos`, `carr_madan`, `frft` | VG model parameter/regression checks; Carr-Madan VG Case 4 is the exact price anchor | `tests/test_paper_carr_madan_1999.py`; `tests/methods/test_cos_vs_pyfeng_fft_heston_vg.py` | `external_reference` / `adapter` | yes-paper | partial |
| Kou (2002) | `KouParams`, `kou_cf`, `cos` | Double-exponential jump-diffusion CF/cumulants; high-resolution Carr-Madan/COS references | `tests/papers/test_kou_2002_derived_reference.py`; `tests/papers/test_phase4_cos_kou.py` | `derived_reference` | yes-derived | done |
| Junike & Pankrashkin (2022) | `cos_improved`, `cos_adaptive_decision` | Truncation-range comparison outputs and failure-case policy checks | `tests/test_paper_junike_2022_2024.py`; `tests/methods/test_cos_improved_policy.py`; `benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv` | `derived_reference` / `numerical_stability` | yes-derived | done |
| Junike (2024) | `recommended_cos_policy`, `COSGridPolicy` | Term-count policy and grid-size behavior | `tests/test_paper_junike_2022_2024.py`; `tests/methods/test_cos_improved_policy.py` | `derived_reference` / `numerical_stability` | yes-derived | done |
| Ruijter, Versteegh & Oosterlee (2015) | `filtered_cos_prices`, `cos_filter_weights` | Spectral-filter behavior and fixed-grid error improvement checks | `tests/methods/test_cos_spectral_filters.py`; `tests/methods/test_filtered_cos_pricing.py`; `tests/methods/test_filtered_cos_outperforms_baselines.py` | `numerical_stability` | no-figure-only | partial |
| Merton (1976) | `MertonJDParams`, `merton_jd_cf` | Poisson-mixture-of-BSM closed-form and lambda-to-zero limit | `tests/models/test_merton_jd_paper.py` | `derived_reference` | yes-derived | done |
| Schoutens (2002) | `MeixnerParams`, `meixner_cf` | Meixner martingale, cumulant, and option-pricing structural checks | `tests/models/test_meixner_paper.py` | `derived_reference` | yes-derived | partial |
| Kuchler & Tappe (2008) | `BilateralGammaParams`, `bilateral_gamma_cf` | Moment, cumulant, martingale, and DAX-style parameter sanity checks | `tests/models/test_bilateral_gamma_paper.py` | `derived_reference` | yes-derived | partial |
| Barndorff-Nielsen (1977); Eberlein & Keller (1995) | `GHParams`, `gh_cf` | GH structural and special-case behavior including NIG/hyperbolic families | `tests/models/test_generalized_hyperbolic_paper.py` | `derived_reference` | yes-derived | partial |
| Carr & Wu (2003) | `FMLSParams`, `fmls_cf` | Alpha equals 2 BSM limit and FMLS structural checks | `tests/models/test_fmls_paper.py` | `derived_reference` | yes-derived | partial |
| Lewis (2000); Carr & Sun (2007) | `Sv32Params`, `sv32_cf` | PyFENG-backed 3/2 stochastic-volatility parity and structural checks | `tests/models/test_sv32_pyfeng_paper.py`; `tests/models/test_sv32_pyfeng_reference.py` | `adapter` | yes-derived | partial |
| El Euch & Rosenbaum (2019); Callegaro et al. (2021) | `RoughHestonParams`, `rough_heston_cf` | PyFENG-backed rough-Heston parity and structural checks | `tests/models/test_rough_heston_pyfeng_paper.py` | `adapter` | yes-derived | partial |
| Wu, Ma & Wang (2012) | `GarchWMW2012Params`, `garch_wmw2012_cf` | GARCH diffusion CF, martingale, and Lewis-style pricing checks | `tests/models/test_garch_wmw2012_paper.py` | `derived_reference` | yes-derived | partial |
| Kelly (2025) | `DoubleHestonParams`, `double_heston_cf` | Vanilla call/put price table for two-factor Heston | `tests/papers/test_double_heston_kelly2025_vanilla.py`; `tests/models/test_double_heston_model.py` | `external_reference` | yes-paper | done |
| CGMY/VGSA parameter sets | `VGSAParams`, `vgsa_cf` | CGMY/VGSA-style parameter-set sanity and cross-method pricing checks | `tests/papers/test_vgsa_cgmy_2003_parameter_sets.py`; `tests/models/test_vgsa_model.py` | `derived_reference` | yes-derived | partial |
| MathWorks optByBatesNI | `BatesParams`, `bates_cf` | 5-strike T=0.5 vector; 5x6 NI surface | `tests/papers/test_phase5_bates_mathworks.py`; `tests/papers/test_bates_mathworks_ni_surface.py` | `software_reference` | yes-software | done |
| MathWorks optByBatesFFT | `BatesParams`, `bates_cf` | 7-strike default FFT subset; tuned FRFT 5x6 surface | `tests/papers/test_bates_mathworks_fft_frft.py` | `software_reference` | yes-software | done |
| MathWorks optSensByBatesNI | `BatesParams`, `bates_cf` | Delta vector for 5 strikes at T=0.5 | `tests/features/test_bates_mathworks_delta.py` | `software_reference` / `numerical_stability` | yes-software | done |
| Lewis (2000); Baldeaux & Badran (2012) | `Sv32Params` | Original BB figure params, shape check only; no printed price table | `tests/papers/test_sv32_baldeaux_badran_original_smoke.py` | `qualitative_figure` | no-figure-only | xfail-if-unstable |
| Sv32 PyFENG surface | `Sv32Params`, `sv32_cf` | 7x4 surface from pyfeng_fft (frozen at generation time) | `tests/models/test_sv32_pyfeng_surface_reference.py` | `adapter` / `derived_reference` | yes-derived | done |
| BSM all-pricer baseline | `BsmParams` | Closed-form Black-Scholes call prices across all six pricing methods (cos/cos_improved 1e-8; lewis 1e-7; carr_madan/frft 1e-4; pyfeng_fft 1e-5) | `tests/models/test_bsm_closed_form_all_pricers.py` | `derived_reference` | yes-derived | done |
