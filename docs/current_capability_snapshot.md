# Current Capability Snapshot (expanded baseline)

Updated 2026-07-04 after the transform-methods expansion (Hilbert transform,
regime switching, exact Levy geometric Asians and variance swaps).
This file tracks the capability surface used by the registry and dispatcher tests.

## Model registry (21 models)

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
| `regime_switching` | Markov regime-switching BSM | no | stable |

## Pricing methods

| Method key | Engine | Reference |
|------------|--------|-----------|
| `cos` | COS / Fang-Oosterlee 2008 | Fang & Oosterlee (2008) |
| `cos_improved` | Adaptive COS with Junike-2024 policy | Junike & Pankrashkin (2022, 2024) |
| `cos_filtered` | Filtered COS with spectral windowing | Ruijter, Versteegh & Oosterlee (2015) |
| `carr_madan` | FFT / Carr-Madan 1999 | Carr & Madan (1999) |
| `frft` | FRFT / Chourdakis 2004 | Chourdakis (2004) |
| `pyfeng_fft` | PyFENG native FFT | pyfeng package |
| `hilbert` | Discrete Hilbert transform on the half-integer sinc grid | Feng & Linetsky (2008) |
| `asian_cf` | Exact Levy geometric-Asian via per-increment CF product | Fusai & Meucci (2008) |
| `variance_levy_analytic` | Exact discrete variance-swap fair strike from CF cumulants | Carr & Wu (2009), discrete analogue |
| `conv` | CONV-style Fourier probability inversion | Choi/Kirkby MATLAB comparison target |
| `lattice` | BSM Cox-Ross-Rubinstein tree | Cox, Ross & Rubinstein (1979) |
| `pde_fd` | BSM implicit finite difference | Black-Scholes PDE |
| `digital_bsm` | BSM closed-form digital option | Black-Scholes closed form |
| `cos_digital` | COS digital option pricing | Fang-Oosterlee payoff extension |
| `monte_carlo` | BSM Monte Carlo / Longstaff-Schwartz | GBM simulation plus early-exercise regression |
| `barrier_bsm` | BSM closed-form single-barrier option | Reiner-Rubinstein / Haug |
| `asian_bsm` | BSM discrete geometric Asian closed form | Kemna-Vorst style lognormal average |
| `asian_mc` | BSM Asian Monte Carlo | GBM path simulation |
| `double_barrier_mc` | BSM double-barrier Monte Carlo | GBM path simulation |
| `forward_start_bsm` | BSM analytic forward-start option | Rubinstein-style forward-start closed form |
| `exchange_bsm` | BSM analytic exchange option | Margrabe closed form |
| `spread_bsm` | BSM analytic spread option | Kirk approximation |
| `multi_asset_mc` | BSM correlated multi-asset Monte Carlo | Terminal GBM simulation |
| `lookback_bsm` | BSM analytic floating-strike lookback | Goldman-Sosin-Gatto closed form |
| `lookback_mc` | BSM Monte Carlo lookback | GBM path simulation |
| `variance_analytic_bsm` | BSM analytic variance products | Exact realised-variance expectation / deterministic integrated variance |
| `variance_mc` | BSM Monte Carlo variance products | GBM path simulation |
| `cliquet_mc` | BSM Monte Carlo cliquet | GBM path simulation |
| `proj` | Real PROJ frame projection (European) | B-spline (Haar/linear/quad/cubic) frame duality, Kirkby 2015/2017 |
| `mellin` | First-slice European Mellin façade | Mellin-transform expansion target |
| `sabr_hagan` | SABR Hagan approximation | Hagan et al. |

## Products supported

- European call / put (via `price_strip`)
- Digital cash-or-nothing and asset-or-nothing options (via `price(..., method="cos_digital"|"digital_bsm")`)
- American BSM call / put (via `price(..., method="lattice"|"pde_fd"|"monte_carlo")`)
- Bermudan BSM call / put (via `price(..., method="monte_carlo")`) alongside the existing 1-D Levy `cos_bermudan` route
- Continuous zero-rebate BSM single-barrier call / put (via `price(..., method="barrier_bsm"|"monte_carlo")`)
- BSM Asian options (geometric closed form via `asian_bsm`; arithmetic/geometric MC via `asian_mc` or `monte_carlo`)
- BSM forward-start call / put (via `price(..., method="forward_start_bsm")`)
- BSM exchange options on two correlated assets (via `price(..., method="exchange_bsm"|"multi_asset_mc"|"monte_carlo")`)
- BSM basket options on correlated assets (via `price(..., method="multi_asset_mc"|"monte_carlo")`)
- BSM spread options on two correlated assets (via `price(..., method="spread_bsm"|"multi_asset_mc"|"monte_carlo")`)
- BSM best-of options on correlated assets (via `price(..., method="multi_asset_mc"|"monte_carlo")`)
- BSM lookback call / put (continuous floating closed form via `lookback_bsm`; Monte Carlo via `lookback_mc` or `monte_carlo`)
- BSM variance swaps (via `variance_analytic_bsm` or `variance_mc` / `monte_carlo`) and integrated-variance options (via `variance_analytic_bsm`; Monte Carlo via `variance_mc` / `monte_carlo`)
- BSM cliquets (via `cliquet_mc` or `monte_carlo`)
- BSM zero-rebate double-barrier options (via `double_barrier_mc` or `monte_carlo`)
- SABR European call / put strips (via `price_strip("sabr", "sabr_hagan", ...)`)

## Public API object count

Public exports include the multi-asset analytic helper `kirk_spread` alongside the prior dispatcher and analytics surface.

## Test suite

- Capability, product, and dispatcher tests now include multi-asset coverage plus generic Monte Carlo / LSMC dispatch for Americans and Bermudans.
- Broad CI continues to validate lint, typing, package build, paper/reference checks, and the Python version matrix.

## Notes

- `proj` is now a real B-spline frame-projection engine for European vanillas (validated against COS to ~1e-7), with a standalone Bermudan-put recursion (`proj_bermudan_put`) cross-validated against `cos_bermudan`. The broader exotic PROJ recursion family (barrier, Asian, lookback, step, cliquet) is still planned; see [proj_parity_roadmap.md](proj_parity_roadmap.md).
- `mellin` remains a validated European façade rather than the full model-specific contour implementation set.
