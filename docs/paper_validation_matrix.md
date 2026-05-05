# Paper Validation Matrix

This matrix links each paper claim in the public README/PAPERS list to the
repo model or method, the exact reference used, and the test file that backs
the claim. `external_reference` means a published table or independent source;
`derived_reference` means a high-precision internal reference; `adapter` means
third-party wrapper parity; `numerical_stability` means a stress/convergence
test rather than a direct paper-table replication.

| Paper | Repo model/method | Exact paper table/equation used | Test file | Reference type | Status |
|---|---|---|---|---|---|
| Carr & Madan (1999) | `carr_madan`, `VGParams` | VG Case 4 put prices | `tests/test_paper_carr_madan_1999.py`; `tests/papers/test_phase2_carr_madan_vg.py` | `external_reference` | done |
| Chourdakis (2004) | `frft` | FRFT grid property and Lewis-Heston strip agreement | `tests/test_paper_chourdakis_2004.py`; `tests/papers/test_phase3_frft.py` | `derived_reference` / `numerical_stability` | partial |
| Fang & Oosterlee (2008) | `cos`, `BsmParams`, `HestonParams`, `VGParams`, `CgmyParams` | Table 1 Heston ATM; FO2008 BSM/Heston/VG/CGMY replay rows | `tests/test_paper_fang_oosterlee_2008.py`; `tests/papers/test_phase4_cos_heston_fo2008.py`; `benchmarks/paper_replications/fo2008_cos/params.py` | `external_reference` / `derived_reference` | partial |
| Heston (1993) | `HestonParams`, `heston_cf_form2`, `lewis_call_prices` | Semi-closed Fourier/Lewis Heston integration benchmark; no original Heston table copied into fixtures | `tests/test_paper_lewis_2001.py`; `tests/methods/test_cos_vs_pyfeng_fft_heston_vg.py` | `external_reference` / `adapter` | partial |
| Albrecher et al. (2007), Little Heston Trap | `heston_cf_form2` | Stable characteristic-function branch behavior under difficult Heston regimes | `tests/methods/test_phase4_alpha_validity.py`; `tests/methods/test_robustness_parametrize.py` | `numerical_stability` | partial |
| Lewis (2001) | `lewis_call_prices` | Five-strike Heston strip used as independent Fourier integration benchmark | `tests/test_paper_lewis_2001.py`; `tests/test_paper_chourdakis_2004.py` | `external_reference` | done |
| Madan, Carr & Chang (1998) | `VGParams`, `vg_cf`, `cos`, `carr_madan`, `frft` | VG model parameter/regression checks; Carr-Madan VG Case 4 is the exact price anchor | `tests/test_paper_carr_madan_1999.py`; `tests/methods/test_cos_vs_pyfeng_fft_heston_vg.py` | `external_reference` / `adapter` | partial |
| Kou (2002) | `KouParams`, `kou_cf`, `cos` | Double-exponential jump-diffusion CF/cumulants; high-resolution Carr-Madan/COS references | `tests/test_paper_kou_2002.py`; `tests/papers/test_phase4_cos_kou.py` | `derived_reference` | partial |
| Junike & Pankrashkin (2022) | `cos_improved`, `cos_adaptive_decision` | Truncation-range comparison outputs and failure-case policy checks | `tests/test_paper_junike_2022_2024.py`; `tests/methods/test_cos_improved_policy.py`; `benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv` | `derived_reference` / `numerical_stability` | partial |
| Junike (2024) | `recommended_cos_policy`, `COSGridPolicy` | Term-count policy and grid-size behavior | `tests/test_paper_junike_2022_2024.py`; `tests/methods/test_cos_improved_policy.py` | `derived_reference` / `numerical_stability` | partial |
| Ruijter, Versteegh & Oosterlee (2015) | `filtered_cos_prices`, `cos_filter_weights` | Spectral-filter behavior and fixed-grid error improvement checks | `tests/methods/test_cos_spectral_filters.py`; `tests/methods/test_filtered_cos_pricing.py`; `tests/methods/test_filtered_cos_outperforms_baselines.py` | `numerical_stability` | partial |
| Merton (1976) | `MertonJDParams`, `merton_jd_cf` | Poisson-mixture-of-BSM closed-form and lambda-to-zero limit | `tests/models/test_merton_jd_paper.py` | `external_reference` / `derived_reference` | done |
| Schoutens (2002) | `MeixnerParams`, `meixner_cf` | Meixner martingale, cumulant, and option-pricing structural checks | `tests/models/test_meixner_paper.py` | `derived_reference` | partial |
| Kuchler & Tappe (2008) | `BilateralGammaParams`, `bilateral_gamma_cf` | Moment, cumulant, martingale, and DAX-style parameter sanity checks | `tests/models/test_bilateral_gamma_paper.py` | `derived_reference` | partial |
| Barndorff-Nielsen (1977); Eberlein & Keller (1995) | `GHParams`, `gh_cf` | GH structural and special-case behavior including NIG/hyperbolic families | `tests/models/test_generalized_hyperbolic_paper.py` | `derived_reference` | partial |
| Carr & Wu (2003) | `FMLSParams`, `fmls_cf` | Alpha equals 2 BSM limit and FMLS structural checks | `tests/models/test_fmls_paper.py` | `derived_reference` | partial |
| Lewis (2000); Carr & Sun (2007) | `Sv32Params`, `sv32_cf` | PyFENG-backed 3/2 stochastic-volatility parity and structural checks | `tests/models/test_sv32_pyfeng_paper.py` | `adapter` | partial |
| El Euch & Rosenbaum (2019); Callegaro et al. (2021) | `RoughHestonParams`, `rough_heston_cf` | PyFENG-backed rough-Heston parity and structural checks | `tests/models/test_rough_heston_pyfeng_paper.py` | `adapter` | partial |
| Wu, Ma & Wang (2012) | `GarchWMW2012Params`, `garch_wmw2012_cf` | GARCH diffusion CF, martingale, and Lewis-style pricing checks | `tests/models/test_garch_wmw2012_paper.py` | `derived_reference` | partial |
| Kelly (2025) | `DoubleHestonParams`, `double_heston_cf` | Vanilla call/put price table for two-factor Heston | `tests/papers/test_double_heston_kelly2025_vanilla.py`; `tests/models/test_double_heston_model.py` | `external_reference` | done |
| CGMY/VGSA parameter sets | `VGSAParams`, `vgsa_cf` | CGMY/VGSA-style parameter-set sanity and cross-method pricing checks | `tests/papers/test_vgsa_cgmy_2003_parameter_sets.py`; `tests/models/test_vgsa_model.py` | `derived_reference` | partial |
