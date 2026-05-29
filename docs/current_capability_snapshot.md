# Current Capability Snapshot — v0.5.1-baseline

Frozen 2026-05-29 before the Sprint 2–10 expansion.
This file is the reference point for all capability-gate tests.

## Model registry (26 models)

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
| `heston_nig` | SV + NIG | no | stable |
| `heston_vg` | SV + VG | no | stable |
| `svjj` | SV with correlated jumps | no | stable |
| `bns_gamma_ou` | BNS Gamma-OU | no | stable |
| `nts` | Normal Tempered Stable | no | stable |
| `cgmysa` | CGMY with stochastic arrival | no | stable |

## Pricing methods (6 engines)

| Method key | Engine | Reference |
|------------|--------|-----------|
| `cos` | COS / Fang-Oosterlee 2008 | Fang & Oosterlee (2008) |
| `cos_improved` | Adaptive COS with Junike-2024 policy | Junike & Pankrashkin (2022, 2024) |
| `cos_filtered` | Filtered COS with spectral windowing | Ruijter, Versteegh & Oosterlee (2015) |
| `carr_madan` | FFT / Carr-Madan 1999 | Carr & Madan (1999) |
| `frft` | FRFT / Chourdakis 2004 | Chourdakis (2004) |
| `pyfeng_fft` | PyFENG native FFT | pyfeng package |

## Products supported (pre-expansion)

- European call / put (via `price_strip`)

## Public API object count

117 public names exported from `foureng` at freeze time.

## Test suite (pre-expansion)

- **Collected**: 832 tests (857 total, 25 deselected/skipped)
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
