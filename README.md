# fourier-option-pricer

If your model has a characteristic function, you usually do not want to price or calibrate a European implied-vol surface by simulating huge Monte Carlo path sets. This repo implements three Fourier-based European pricers — Carr–Madan FFT, FRFT, and COS — behind a single `CharFunc` protocol. It wires those pricers to Heston, Variance Gamma, and Kou, and it only treats a method as "working" once it matches published reference values. Monte Carlo remains in the repo, but as a baseline for timing and error comparisons.

---

## What is in scope

This project builds a small pricing stack with four parts.

- **Models**
  - Heston
  - Variance Gamma
  - Kou

- **European pricers**
  - Carr–Madan FFT
  - FRFT
  - COS

- **Numerical utilities**
  - implied-vol inversion
  - interpolation
  - cumulant and truncation helpers
  - benchmark harnesses

- **Validation**
  - tests and notebooks that reproduce published tables before any performance claims are made

---

## End-to-end workflow

1. Implement the model characteristic function

$$
\varphi_T(u) = \mathbb{E}^{\mathbb{Q}}\!\left[e^{iuX_T}\right]
$$

where

$$
X_T = \log\!\left(\frac{S_T}{F_0}\right), \qquad F_0 = S_0 \, e^{(r-q)T}.
$$

2. Price a strip of strikes with one of the three Fourier pricers.
3. Convert prices to implied vols with a robust root finder.
4. Check prices against published reference tables.
5. Benchmark runtime and error only after the validation step passes.

---

## Why Monte Carlo is only the baseline

Monte Carlo is useful, but for European calibration it is usually the wrong default.

- The standard error decays like

$$
\varepsilon_{\mathrm{MC}} = O\!\left(n^{-1/2}\right).
$$

Cutting error by a factor of 10 usually needs about 100 times as many paths.

- Calibration means repeated strip pricing across strikes, maturities, and optimizer iterations. That makes Monte Carlo noise and runtime expensive.
- Even variance-reduced or conditional Monte Carlo still pays the path-count cost.

For European options under characteristic-function models, Fourier inversion gives deterministic prices and a much better speed versus accuracy trade-off.

---

## The `CharFunc` protocol

All models satisfy one interface:

```python
from typing import Protocol
import numpy as np

class CharFunc(Protocol):
    def __call__(self, u: np.ndarray) -> np.ndarray:
        """Return phi_T(u) = E^Q[exp(i u X_T)] for X_T = log(S_T / F0)."""
        ...
```

Once a model exposes $\varphi_T(u)$, the same model can be passed into FFT, FRFT, or COS without rewriting the pricer.

---

## Pricers

### 1. Carr–Madan FFT

Carr–Madan prices a damped call transform on a uniform frequency grid and then uses the FFT to recover prices across a log-strike grid.

Key parameters:

- damping parameter $\alpha$
- grid size $N$
- frequency spacing $\eta$
- strike spacing

$$
\lambda = \frac{2\pi}{N\,\eta}.
$$

Main practical issue:

- the FFT grid ties together frequency resolution and strike resolution, so you cannot choose them independently.

### 2. FRFT

FRFT relaxes that grid coupling. It lets you keep the Fourier acceleration while giving more flexibility over the strike grid and integration grid. In practice, that often means you can hit the same tolerance with a smaller grid than plain FFT.

### 3. COS

COS expands the density on a finite interval $[a,b]$ using a cosine series. The density itself never needs to be written down explicitly because the coefficients are recovered from the characteristic function.

A standard truncation rule uses cumulants:

$$
[a,b] = \left[\, c_1 - L\sqrt{c_2 + \sqrt{c_4}}, \; c_1 + L\sqrt{c_2 + \sqrt{c_4}} \,\right].
$$

For Kou, COS is not ruled out in principle. If it misbehaves, the usual cause is the truncation interval or the implementation details, not the fact that Kou is a jump model.

---

## Model conventions

We work in **log-forward space**:

$$
X_T = \log\!\left(\frac{S_T}{F_0}\right), \qquad F_0 = S_0 \, e^{(r-q)T}.
$$

So every characteristic function below is the characteristic function of $X_T$, not of $\log S_T$ itself.

If you want the characteristic function of $\log S_T$, multiply by

$$
e^{iu\log F_0}.
$$

A note on notation: $\nu$ is reused across sections — in Heston it is the vol-of-vol, in Variance Gamma it is the gamma-time-change variance rate. The meaning is always local to the model's own parameter list.

---

## Characteristic functions

### Heston

Parameters: $\kappa, \theta, \nu, \rho, v_0$, where $\nu$ is the vol-of-vol.

Define

$$
b(u) = \kappa - \rho\,\nu\, i u,
$$

$$
d(u) = \sqrt{\, b(u)^2 + \nu^2\,(u^2 + iu)\,},
$$

$$
g(u) = \frac{b(u) - d(u)}{b(u) + d(u)}.
$$

Using the stable "Formulation 2" / "Little Heston Trap" form, with $e^{-d(u)T}$:

$$
D(u,T) = \frac{b(u) - d(u)}{\nu^2} \cdot \frac{1 - e^{-d(u)T}}{1 - g(u)\, e^{-d(u)T}},
$$

$$
C(u,T) = \frac{\kappa\,\theta}{\nu^2} \left[\, (b(u) - d(u))\,T \; - \; 2\log\!\left(\frac{1 - g(u)\, e^{-d(u)T}}{1 - g(u)}\right) \right].
$$

Then the log-forward characteristic function is

$$
\varphi_H(u) = \exp\!\left(\, C(u,T) + D(u,T)\, v_0 \,\right).
$$

This is the version you want in code. The original "Form 1" representation is mathematically equivalent, but numerically it can cross the wrong complex-log branch and quietly return bad prices.

---

### Variance Gamma

Parameters: $\sigma, \nu, \theta$, where $\nu$ is the variance rate of the gamma time change (unrelated to Heston's vol-of-vol).

The martingale correction is

$$
\omega = \frac{1}{\nu}\log\!\left(1 - \theta\,\nu - \tfrac{1}{2}\sigma^2\,\nu\right),
$$

which requires

$$
1 - \theta\,\nu - \tfrac{1}{2}\sigma^2\,\nu \;>\; 0.
$$

Under the log-forward convention,

$$
\varphi_{VG}(u) = \exp(iu\,\omega\, T)\,\left(1 - i\,\theta\,\nu\, u + \tfrac{1}{2}\sigma^2\,\nu\, u^2\right)^{-T/\nu}.
$$

---

### Kou

Parameters: $\sigma, \lambda, p, \eta_1, \eta_2$, with jump-size density

$$
f_Y(y) \;=\; p\,\eta_1\, e^{-\eta_1 y}\,\mathbf{1}_{\{y \ge 0\}} \;+\; (1-p)\,\eta_2\, e^{\eta_2 y}\,\mathbf{1}_{\{y < 0\}}.
$$

The jump characteristic function is

$$
\varphi_Y(u) \;=\; \frac{p\,\eta_1}{\eta_1 - iu} \;+\; \frac{(1-p)\,\eta_2}{\eta_2 + iu}.
$$

The exponential-jump compensator is

$$
\zeta \;=\; \mathbb{E}[e^Y] - 1 \;=\; \frac{p\,\eta_1}{\eta_1 - 1} \;+\; \frac{(1-p)\,\eta_2}{\eta_2 + 1} \;-\; 1,
$$

which requires $\eta_1 > 1$.

Under the log-forward convention,

$$
X_T \;=\; \left(-\tfrac{1}{2}\sigma^2 - \lambda\,\zeta\right)T \;+\; \sigma\, W_T \;+\; \sum_{j=1}^{N_T} Y_j,
$$

so the characteristic function is

$$
\varphi_{Kou}(u) \;=\; \exp\!\left(\, iu\left(-\tfrac{1}{2}\sigma^2 - \lambda\,\zeta\right)T \;-\; \tfrac{1}{2}\sigma^2\, u^2\, T \;+\; \lambda\, T\,(\varphi_Y(u) - 1) \,\right).
$$

That is the correct log-forward version. If you see an extra factor $e^{iu\log F_0}$, you are no longer in normalized forward coordinates.

---

## Validation gate

Nothing should be called correct until it reproduces published reference results within a stated tolerance.

Suggested validation order:

1. **Carr–Madan FFT** — replicate benchmark prices from Carr–Madan style test cases, especially Variance Gamma examples.
2. **Heston** — compare against high-precision Heston benchmarks and include at least one branch-cut stress test.
3. **COS** — replicate published Heston COS tables first.
4. **Kou** — replicate Kou reference prices only after the Fourier engine is already stable.

This keeps the debugging order sensible. First get one method working on one model. Then widen coverage.

---

## Benchmarks to report

Only publish benchmark numbers after the validation tests are passing.

Measure:

- runtime versus number of strikes
- runtime versus grid size
- error versus reference prices
- FFT versus FRFT versus COS
- Monte Carlo baseline runtime and error

Do not put placeholder speedups in the README.

---

## Repository layout

```text
src/foureng/
  char_func/        # heston / vg / kou
  pricers/          # carr_madan / frft / cos
  iv/               # implied-vol inversion
  mc/               # Monte Carlo baselines
  utils/            # grids, interpolation, cumulants, numerics

tests/              # paper-table replication tests
notebooks/          # validation and benchmark notebooks
```

---

## Extensions after validation

Only add these once the core pricing stack is already validated.

- Fourier Greeks
- FFT price as a control variate for Monte Carlo
- calibration routines
- packaged API or external-library adapter

---

## PyFENG integration

PyFENG already has useful option-pricing components in pure Python. The goal here is not to duplicate Heston pricing for its own sake. The useful angle is:

- one common characteristic-function interface
- one validation harness
- multiple Fourier pricers behind the same API

That makes the repo cleaner and makes cross-method comparisons straightforward.

---

## Roadmap

1. **Phase 1**
   - Monte Carlo baseline
   - timing versus strike count
   - error decay checks
2. **Phase 2**
   - Carr–Madan FFT for Variance Gamma and Heston
   - validation against published benchmarks
3. **Phase 3**
   - FRFT implementation
   - FFT versus FRFT benchmark study
4. **Phase 4**
   - COS implementation
   - validation on Heston, then extension to Kou if the truncation setup is stable
5. **Phase 5**
   - Kou replication tests
6. **Phase 6**
   - optional extensions such as Greeks or control variates
7. **Phase 7**
   - packaging and library integration

---

## References

Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*, January, 83–92. [[PDF](https://perswww.kuleuven.be/~u0009713/HestonTrap.pdf)]

Benhamou, E. (2002). Fast Fourier transform for discrete Asian options. *Journal of Computational Finance*, 6(1), 49–68. [[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=269491)]

Carr, P. and Madan, D.B. (1999). Option valuation using the fast Fourier transform. *Journal of Computational Finance*, 2(4), 61–73. [[PDF](https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf)]

Chourdakis, K. (2004). Option pricing using the fractional FFT. *Journal of Computational Finance*, 8(2), 1–18. [[CiteSeer](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=6bdf4696312d37427eda2740137650c09deacda7)]

Choi, J. and Wu, L. (2021). The equivalent constant-elasticity-of-variance (CEV) volatility of the stochastic-alpha-beta-rho (SABR) model. *Journal of Economic Dynamics and Control*, 128, 104143.

Fang, F. and Oosterlee, C.W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM Journal on Scientific Computing*, 31(2), 826–848. [[Preprint](http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf)] [[SIAM](https://epubs.siam.org/doi/10.1137/080718061)]

Hagan, P.S., Kumar, D., Lesniewski, A.S. and Woodward, D.E. (2002). Managing smile risk. *Wilmott Magazine*, September, 84–108. [[PDF](http://www.deriscope.com/docs/Hagan_2002.pdf)]

Heston, S.L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*, 6(2), 327–343. [[PDF](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf)]

Kahl, C. and Jäckel, P. (2005). Not-so-complex logarithms in the Heston model. *Wilmott Magazine*, September, 94–103. [[PDF](http://www2.math.uni-wuppertal.de/~kahl/publications/NotSoComplexLogarithmsInTheHestonModel.pdf)]

Kou, S.G. (2002). A jump-diffusion model for option pricing. *Management Science*, 48(8), 1086–1101.

Lewis, A.L. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes. *Envision Financial Systems working paper*. [[SSRN](https://www.researchgate.net/publication/2499800_A_Simple_Option_Formula_for_General_Jump-Diffusion_and_Other_Exponential_Levy_Processes)]

Lord, R. and Kahl, C. (2010). Complex logarithms in Heston-like models. *Mathematical Finance*, 20(4), 671–694. [[Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9965.2010.00416.x)]

Madan, D.B., Carr, P. and Chang, E.C. (1998). The Variance Gamma process and option pricing. *European Finance Review*, 2(1), 79–105.

---

*MATH5030 Numerical Methods in Finance — Columbia University MAFN, Spring 2026. Instructor: Prof. Jaehyuk Choi.*