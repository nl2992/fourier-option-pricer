# PROJ-parity roadmap

Goal: encompass the functionality of Justin Kirkby's
[`PROJ_Option_Pricing_Matlab`](https://github.com/jaehyukchoi/PROJ_Option_Pricing_Matlab)
(the jaehyukchoi fork) inside `foureng`. The defining feature of that library is
the **PROJ (frame-projection)** method, which prices path-dependent exotics at
near-Fourier speed, plus a **CTMC** engine for SV/SLV exotics. We are pursuing
**full parity** across all phases below.

> **Status (2026-09).** Beyond the PROJ ports listed below, the remaining Phase 2
> exotics are now covered by other transform engines: discretely monitored
> barriers and floating/fixed-strike lookbacks by the Feng-Linetsky Hilbert
> transform (`hilbert_barrier`, `hilbert_lookback`), arithmetic Asians by the
> ASCOS recursion (`asian_cos`), and Americans by Richardson-extrapolated COS
> Bermudans (`cos_american`). Still open: Parisian options via Fourier, and the
> 2-D CTMC for stochastic-volatility exotics. `proj_barrier` and
> `proj_double_barrier` were fixed (A3): the grid half-width was silently
> halved, the barrier was snapped to the nearest node instead of weighted by
> its sub-cell position, and the discretized transition operator could alias
> past unit magnitude for slowly decaying short-step CFs (VG at small dt).
> Against the fast Hilbert-transform reference, `proj_barrier` now agrees to
> about 1e-5 at 12 monitoring dates and 1e-4 (BSM/Kou) to 1e-3 (VG, the
> hardest case) at 252, versus ~1e-3 at any frequency before the fix. The same
> grid-domain bug was shared by `proj_bermudan_put`, `proj_step`,
> `proj_swing` and `proj_survival_probability` (A5); the fix (domain sized
> from the live nodes, partial-cell boundary weighting where there is a
> barrier/damping level to snap, transfer-function clipping, Richardson
> extrapolation in the grid size) brings `proj_bermudan_put` to ~1e-12 (BSM),
> ~1e-6 (VG) and ~1e-11 (Kou) against the exact FO2009 COS-Bermudan engine;
> `proj_step` to ~1e-4 against the European vanilla at rho=0 and ~1e-3
> against the fixed `proj_barrier_price` knock-out at rho→∞; `proj_swing` to
> ~1e-4 (one right vs. COS-Bermudan) and ~1e-3 (full rights vs. the sum of
> per-date Europeans); and `proj_survival_probability` to ~1e-3-4e-3 against
> the Broadie-Glasserman-Kou (1997) continuity-corrected BSM first-passage
> probability. See `tests/methods/test_proj_recursions_accuracy.py`.

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
    projection, Gaussian-quadrature early-exercise stencils), with the A5
    grid-domain fix and Richardson extrapolation in the grid size.
    Cross-validated vs `cos_bermudan` to ~1e-12 (BSM), ~1e-6 (VG) and ~1e-11
    (Kou), M∈{12,52}. Tests in `tests/methods/test_proj_pricing.py` and
    `tests/methods/test_proj_recursions_accuracy.py`.
  - **TODO:** Bermudan/American call (or general cp), single/double barrier,
    arithmetic Asian, lookback, step, cliquet. Port from `PROJ/LEVY/*_Options`.
    Then wire Bermudan into the product-level `price()` dispatcher.
- **Phase 3: CTMC.** `foureng/pricers/ctmc.py`: generator + matrix-exponential
  pricer for 1-D diffusion European/barrier/Bermudan; then 2-D SV/SLV (Heston,
  SABR) for barrier + Bermudan.
- **Phase 4: New models.** 4/2 SV, SABR, regime-switching (done); NTS and
  Stein-Stein are not implemented yet (`models/nts.py` and
  `models/stein_stein.py` do not exist).
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
