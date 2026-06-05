# Current Capability Snapshot — v0.5.1-baseline

Frozen 2026-05-29 before the Sprint 2–10 expansion.
This file is the reference point for all capability-gate tests.

## Model registry (20 models)

| Key | Family | PyFENG FFT | Status |
|-----|--------|-----------|--------|
| `bsm` | Diffusion | yes | stable |
| `heston` | Stochastic volatility | yes | stable |
| `ousv` | Stochastic volatility (OU) | yes | stable |
| `vg` | Lévy (VG) | yes | stable |
| `cgmy` | Lévy (CGMY) | yes | stable |
| `nig` | Lévy (NIG) | yes | stable |
| `sv32` | Stochastic volatility (3/2) | yes | stable |
| `rough_heston` | Rough volatility | yes | stable |
| `kou` | Jump diffusion | no | stable |
| `bates` | Jump diffusion + SV | no | stable |
| `heston_kou` | SV + Kou jumps | no | stable |
| `heston_cgmy` | SV + CGMY jumps | no | stable |
| `garch_wmw2012` | GARCH diffusion | no | stable |
| `merton_jd` | Jump diffusion (Merton) | no | stable |
| `meixner` | Lévy (Meixner) | no | stable |
| `bilateral_gamma` | Lévy (Bilateral Gamma) | no | stable |
| `generalized_hyperbolic` | Lévy (GH) | no | stable |
| `fmls` | Lévy (FMLS) | no | stable |
| `double_heston` | Two-factor SV | no | stable |
| `vgsa` | VG with stochastic arrival | no | stable |

## Pricing methods (Sprint 1 expansion: 9 engines)

| Method key | Engine | Reference |
|------------|--------|-----------|
| `cos` | COS / Fang-Oosterlee 2008 | Fang & Oosterlee (2008) |
| `cos_improved` | Adaptive COS with Junike-2024 policy | Junike & Pankrashkin (2022, 2024) |
| `cos_filtered` | Filtered COS with spectral windowing | Ruijter, Versteegh & Oosterlee (2015) |
| `carr_madan` | FFT / Carr-Madan 1999 | Carr & Madan (1999) |
| `frft` | FRFT / Chourdakis 2004 | Chourdakis (2004) |
| `pyfeng_fft` | PyFENG native FFT | pyfeng package |
| `conv` | CONV-style Fourier probability inversion | Choi/Kirkby MATLAB comparison target |
| `lattice` | BSM Cox-Ross-Rubinstein tree | Cox, Ross & Rubinstein (1979) |
| `pde_fd` | BSM implicit finite difference | Black-Scholes PDE |
| `barrier_bsm` | BSM closed-form single-barrier option | Reiner-Rubinstein / Haug |
| `asian_bsm` | BSM discrete geometric Asian closed form | Kemna-Vorst style lognormal average |
| `asian_mc` | BSM Asian Monte Carlo | GBM path simulation |
| `double_barrier_mc` | BSM double-barrier Monte Carlo | GBM path simulation |
| `forward_start_bsm` | BSM analytic forward-start option | Rubinstein-style forward-start closed form |
| `lookback_bsm` | BSM analytic floating-strike lookback | Goldman-Sosin-Gatto closed form |
| `lookback_mc` | BSM Monte Carlo lookback | GBM path simulation |
| `variance_mc` | BSM Monte Carlo variance products | GBM path simulation |
| `cliquet_mc` | BSM Monte Carlo cliquet | GBM path simulation |
| `proj` | First-slice European PROJ façade | COS-backed projection baseline |
| `mellin` | First-slice European Mellin façade | Mellin-transform expansion target |
| `sabr_hagan` | SABR Hagan approximation | Hagan et al. |

## Products supported

- European call / put (via `price_strip`)
- American BSM call / put (via `price(..., method="lattice"|"pde_fd")`)
- Continuous zero-rebate BSM single-barrier call / put (via `price(..., method="barrier_bsm")`)
- BSM Asian options (geometric closed form via `asian_bsm`; arithmetic/geometric MC via `asian_mc`)
- BSM forward-start call / put (via `price(..., method="forward_start_bsm")`)
- BSM lookback call / put (continuous floating closed form via `lookback_bsm`; Monte Carlo via `lookback_mc`)
- BSM variance swaps and variance options (via `variance_mc`)
- BSM cliquets (via `cliquet_mc`)
- BSM zero-rebate double-barrier options (via `double_barrier_mc`)
- SABR European call / put strips (via `price_strip("sabr", "sabr_hagan", ...)`)

## Public API object count

117 public names exported from `foureng` at freeze time.

## Test suite (pre-expansion)

- **Collected**: 832 tests before Sprint 1 expansion (857 total, 25 deselected/skipped)
- **Test directories**: features/, methods/, models/, papers/, refs/

## Validation matrix summary

| Status | Count |
|--------|-------|
| `done` | 13 |
| `partial` | 19 |
| `xfail-if-unstable` | 1 |
| **Total rows** | **33** |

Exact paper anchors: Carr-Madan (1999), Lewis (2001), FO2008 Heston ATM,
Kelly (2025 Double Heston), MathWorks Bates NI/FFT/delta.

## Benchmark CSVs (pre-expansion)

Located in `benchmarks/`:
- `benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv`
- `benchmarks/paper_replications/fo2008_cos/`
