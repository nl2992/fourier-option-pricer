# Fourier coverage plan

Goal: make every transform method in `foureng` do what its name says, add the Fourier methods that are still missing, and extend transform pricing to the products that today fall back to BSM or Monte Carlo.

Status key: `todo`, `in progress`, `done`. Each item lands on `dev` as its own commit with tests, and CI must pass before the next wave starts.

## Part A: make existing methods honest

| ID | Item | Status |
|----|------|--------|
| A1 | Capability registry and docs agree with the code: drop claims no route serves (COS digitals, `cos_bermudan` Europeans, `pde_fd`/`lattice`/`ctmc` barriers and Bermudans), mark the Levy-only methods in `_model_restriction`, fix hints that name missing methods, register dispatched methods that have no entry, correct model lists in notes, remove the stale "planned" header, fix the stale error message in `price_strip`, refile the snapshot tables, remove references to `nts.py` and `stein_stein.py`, retire the `pricers/base.py` placeholder, and call the Carr-Madan damping check. | done |
| A2 | `proj_asian` is Monte Carlo on Gaussian paths with a bare `except`. Route arithmetic Asians to the exact `asian_cos` engine, keep the key as a deprecated alias, and remove the silent zero. | done |
| A3 | `proj_barrier` is about 1e-3 off at 252 monitoring dates. Find the cause and fix it; respect `product.monitoring`. | done |
| A4 | `cos_improved` default accuracy is about 1e-9 on Heston and 4.6e-6 on a jump-heavy affine case. Tighten the default grid so it matches the contour reference to 1e-10. | done |
| A5 | The PROJ Bermudan, step and swing recursions share the grid-domain bug fixed in A3 (the value grid spans half the intended domain, and boundaries snap to nodes). Apply the same fix and tighten the tolerance that `test_proj_step.py` had to loosen. | done |

## Part B: missing Fourier methods

| ID | Item | Reference | Status |
|----|------|-----------|--------|
| B1 | Public `method="lewis"` (already implemented, only reachable as a fallback). | Lewis (2001) | done |
| B2 | True CONV engine (FFT convolution) for Europeans and Bermudans. The current `conv` (a Gil-Pelaez probability inversion) moves to `method="gil_pelaez"`. | Lord, Fang, Bervoets & Oosterlee (2008) | in progress |
| B3 | Real Mellin-transform pricer behind `method="mellin"` (today it calls `conv`). | Panini & Srivastav (2004) | todo |
| B4 | Fourier space time stepping for European, Bermudan, American and barrier options under Levy and regime-switching models. | Jackson, Jaimungal & Surkov (2008) | todo |
| B5 | Wiener-Hopf / Spitzer methods: continuously monitored barriers and lookbacks, and fast discrete monitoring. | Fusai, Germano & Marazzina (2016) | todo |
| B6 | Saddlepoint approximation for far-wing prices and tail probabilities. | Rogers & Zane (1999); Carr & Madan (2009) | todo |
| B7 | Density, quantile and moment recovery from the CF. | Fang & Oosterlee (2008) | todo |

## Part C: product coverage

| ID | Item | Reference | Status |
|----|------|-----------|--------|
| C1 | Greeks through `price_strip` and `price`: delta, gamma and parameter sensitivities, reusing `cf_and_gradient`. | | todo |
| C2 | Forward-start and cliquet options under Heston and Bates from the forward CF. | Kruse & Nogel (2005) | in progress |
| C3 | Asians, lookbacks, double barriers and faders under Heston and Bates through the variance-chain engine. | Cui, Kirkby & Nguyen (2018) | todo |
| C4 | Variance options and VIX options under Heston. | Sepp (2008) | todo |
| C5 | SABR and stochastic local volatility through the variance-chain engine. | Cui, Kirkby & Nguyen (2018) | todo |
| C6 | Multi-asset COS: baskets on more than two assets, and quantos as two-asset products. | Ruijter & Oosterlee (2012) | todo |
| C7 | Floating-strike Asians, Parisian options and globally collared cliquets by Fourier methods. | | todo |

## Out of scope

Neural pricers, models without a characteristic function (for example rough Bergomi), and interest-rate products.

## Waves

1. A1, A2, A3, A4, A5, B1
2. B2, B6, B7, C1, C2
3. B3, B5, C4
4. B4, C3, C5
5. C6, C7
