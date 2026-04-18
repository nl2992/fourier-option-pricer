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

## Practical notes

A few details matter enough that they should be stated explicitly.

- **Heston branch handling matters.** Use the stable logarithm formulation in code.
- **Interpolation matters.** FFT prices live on a grid, but your benchmark strikes usually do not.
- **Convention mistakes are common.** Always check whether a paper is using log-spot or log-forward variables.
- **Validation comes before speed claims.** A fast wrong price is still wrong.

---

## Math rendering

All formulas use KaTeX/MathJax-compatible LaTeX inside `$…$` (inline) and `$$…$$` (display) delimiters. This renders correctly on GitHub, GitLab, Obsidian, and most Markdown previewers. It does **not** render on PyPI — if you publish this package, either strip the math section from the long description or link to the GitHub-hosted README.
