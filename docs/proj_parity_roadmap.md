# PROJ-parity roadmap

Goal: encompass the functionality of Justin Kirkby's
[`PROJ_Option_Pricing_Matlab`](https://github.com/jaehyukchoi/PROJ_Option_Pricing_Matlab)
(the jaehyukchoi fork) inside `foureng`. The defining feature of that library is
the **PROJ (frame-projection)** method, which prices path-dependent exotics at
near-Fourier speed, plus a **CTMC** engine for SV/SLV exotics. We are pursuing
**full parity** across all phases below.

## Gap summary (at kickoff)

| Dimension | foureng had | PROJ has | Gap |
|---|---|---|---|
| Fourier (European) | COS(+improved/filtered), Carr-Madan, FRFT, CONV, Lewis, pyfeng-FFT | + Hilbert, Mellin, PROJ | Hilbert, Mellin, PROJ |
| PROJ method | none | core method | **largest** |
| CTMC | none | SV/SLV exotics | major |
| Exotics via Fourier | European + digital only (rest MC-only) | Asian, barrier, lookback, Bermudan, cliquet, Parisian, step, swing, var/vol swaps | major |
| Models | 20+ | + 4/2, SABR, regime-switching, Hull-White rates, time-changed | several |
| Exotic contracts | european, digital, asian, barrier, american, bermudan, cliquet, lookback, forward-start, multi-asset, variance | + Parisian, swing, step, fader/range-accrual, CDS, EIA | several |

## Phases

- **Phase 1: PROJ European core.** ✅ **DONE.**
  `foureng/pricers/proj.py` ports Kirkby's `PROJ_European.m` (Haar / linear /
  quadratic / cubic B-spline orders). `proj_auto_grid` sizes the half-width from
  cumulants. Wired as `method="proj"` in `price_strip`. Validated against COS to
  ~1e-7 across BSM/VG/CGMY/Kou/Merton-JD/NIG, calls+puts, T∈{0.5,1,2}; parity to
  machine precision. Tests: `tests/methods/test_proj_pricing.py` (29 cases).
- **Phase 2: PROJ exotics (1-D Lévy).** *In progress.*
  - **Bermudan put** ✅ **DONE.** `proj_bermudan_put` in `foureng/pricers/proj.py`
    ports `PROJ_Bermudan_Put.m` (Toeplitz-FFT backward recursion, linear-spline
    projection, Gaussian-quadrature early-exercise stencils). Cross-validated vs
    `cos_bermudan` to 1e-5 (BSM/Kou) and ~1e-3 (VG/CGMY), M∈{10,50}. Tests in
    `tests/methods/test_proj_pricing.py`.
  - **Bermudan call (non-dividend case)** ✅ **DONE.** The product-level
    `price(..., method="proj")` route now supports `cp=+1` when `q<=0`, using
    the standard result that Bermudan call value equals European call value in
    the absence of dividends/carry drag. Dividend-sensitive calls still fall
    back to the remaining backlog below.
  - **Single barrier** ✅ **DONE.** `proj_barrier_price` ports
    `PROJ_Barrier.m` for the 1-D Lévy family with all four single-barrier types
    (knock-in via in-out parity), and is wired into `price(..., method="proj_barrier")`.
    Coverage lives in `tests/methods/test_proj_barrier_asian.py`.
  - **Arithmetic Asian (PROJ-assisted CV)** ✅ **DONE.**
    `proj_asian_price_cv` uses a PROJ geometric-Asian control variate to reduce
    variance in arithmetic-Asian Monte Carlo, and is wired into
    `price(..., method="proj_asian")` for fixed-strike arithmetic Asians on the
    supported 1-D Lévy models.
  - **TODO:** Dividend-sensitive Bermudan/American call (general `cp` case),
    double barrier, lookback,
    step, cliquet. Port from `PROJ/LEVY/*_Options`.
- **Phase 3: CTMC.** `foureng/pricers/ctmc.py`: generator + matrix-exponential
  pricer for 1-D diffusion European/barrier/Bermudan; then 2-D SV/SLV (Heston,
  SABR) for barrier + Bermudan.
- **Phase 4: New models.** 4/2 SV, SABR, regime-switching; finish wiring the
  already-present `models/nts.py` and `models/stein_stein.py`.
- **Phase 5: Remaining Fourier pricers.** Hilbert-transform (barrier-friendly),
  Mellin-transform.
- **Phase 6: Long-tail exotics.** Parisian, swing, fader/range-accrual,
  variance/vol swap via PROJ, CDS.

## Cross-cutting

- Keep `foureng/core/capabilities.py` honest: declare each new
  (method × product × model) triple as it lands.
- Add paper-replication tests benchmarking PROJ output against Kirkby's
  published tables, matching the existing validation-matrix style.

## Reference source paths (in the MATLAB repo)

`PROJ/LEVY/` subfolders: `European_Options` (done), `American_Options`,
`Asian_Options`, `Geometric_Asian_Options`, `Barrier_Options`, `Cliquets`,
`Lookback_Options`, `Step_Options`, `Parisian_Options`, `Swing_Options`,
`Variance_Swaps_Options`, `Forward_Starting_Options`, `Fader_Options`,
`Credit_Default_Swaps`, plus `RN_CHF` and `Helper_Functions`.
