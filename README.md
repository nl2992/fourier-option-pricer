# fourier-option-pricer

If your model gives you a characteristic function, you shouldn’t be simulating tens of thousands of Monte Carlo paths to price (or calibrate to) an implied-vol surface. This repo implements three Fourier-based European pricers—Carr–Madan FFT, FRFT, and COS—behind a single `CharFunc` protocol, wires them to Heston / Variance Gamma / Kou, and validates everything against published reference tables before claiming anything works. Monte Carlo stays in the repo, but only as the baseline you benchmark against.

---

## What’s in scope

We’re building a small, validation-first pricing stack:

- Models (characteristic functions):
  - Heston (stochastic volatility)
  - Variance Gamma (pure-jump Lévy)
  - Kou (double-exponential jump diffusion)

- European pricers (three ways):
  - Carr–Madan FFT (1999)
  - Fractional FFT / FRFT (2004)
  - COS (2008)

- Glue:
  - robust implied-vol inversion (price → IV)
  - a benchmark harness (speed + accuracy)
  - tests/notebooks that replicate paper tables before we trust anything

---

## The workflow (end-to-end)

1) Implement the model characteristic function  
   \(\varphi_T(u) = \mathbb{E}^\mathbb{Q}[e^{iuX_T}]\) for \(X_T = \log(S_T/F_0)\) (we work in log-forward space).

2) Price a *strip of strikes* using one of:
- Carr–Madan FFT (fixed log-strike grid, fast strip pricing)
- FRFT (decouples frequency/strike grid choices)
- COS (cosine-series expansion on a truncation range from cumulants)

3) Convert prices to implied vols (robust root-finding).

4) Validate against published reference tables (this is the gate).

5) Benchmark scaling (runtime vs strikes, runtime vs tolerance, accuracy vs references).

---

## Why Monte Carlo is only the baseline here

Monte Carlo is fine for one-off pricing; it’s a bad default for calibration loops.

- Convergence is slow: \(\varepsilon_{MC} \propto 1/\sqrt{n}\). Reducing error by 10× costs ~100× more paths.
- Calibration is repeated strip pricing (50–200 strikes across maturities, many optimizer iterations). Sampling noise becomes a bottleneck.
- Even conditional MC for Heston (where you reduce variance by conditioning) still pays the path-count tax and still carries sampling noise.

For European options under CF models, Fourier inversion replaces sampling with deterministic numerics and gives a clean speed/accuracy frontier.

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

Once a model implements `phi(u)`, it plugs into FFT / FRFT / COS without changing the pricer.

---

## Methods (pricers)

### Carr–Madan FFT (1999)
Paper (PDF): https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf

We implement the damped call transform and invert it on a uniform frequency grid with Simpson weights. Core knobs:
- damping \(\alpha\)
- grid size \(N\)
- frequency step \(\eta\)
- log-strike step \(\lambda = 2\pi/(N\eta)\)
- interpolation to off-grid strikes (important for table replication)

Limitation:
- the FFT grid couples \(\eta\) and \(\lambda\) (you trade strike resolution for integration resolution).

### FRFT (Chourdakis, 2004)
Reference page: https://www.semanticscholar.org/paper/Option-Pricing-Using-the-Fractional-FFT-Chourdakis/6bdf4696312d37427eda2740137650c09deacda7

FRFT breaks the FFT grid coupling by allowing an arbitrary fractional parameter. Implemented via a Bluestein / chirp-z style decomposition (three FFTs). The point is practical: it often reaches the same tolerance with much smaller \(N\).

### COS (Fang & Oosterlee, 2008)
Preprint (PDF): http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf

COS expands the density on \([a,b]\) as a cosine series. The coefficients come from the CF; you never need the density explicitly. The truncation interval is set from cumulants \(c_1,c_2,c_4\) and a scale \(L\).

Scope note for Kou:
- COS is always conceptually applicable to CF models, but for Kou you must ensure the required moments exist and choose \([a,b]\) carefully (jump models tighten the strip of analyticity). In the current plan, COS is validated first on Heston (FO2008 table) and then extended to other models once the truncation/cumulant machinery is stable.

---

## Models + characteristic functions (log-forward conventions)

We define the forward \(F_0 = S_0 e^{(r-q)T}\) and work with  
\(X_T = \log(S_T/F_0)\). Under this convention, the CFs below are for \(X_T\).

### Heston (1993) — stochastic volatility
Paper (PDF): https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf

Implementation note (non-negotiable):
- the Heston CF contains a complex logarithm; naive evaluation (“Form 1”) can silently break due to branch cuts.
- we implement the numerically stable “Formulation 2” (Little Heston Trap) and keep a small test demonstrating the failure case.

References:
- Albrecher et al. (2007) “The little Heston trap” (PDF): https://perswww.kuleuven.be/~u0009713/HestonTrap.pdf
- Kahl & Jäckel (2005) (PDF): http://www2.math.uni-wuppertal.de/~kahl/publications/NotSoComplexLogarithmsInTheHestonModel.pdf

**Characteristic function (stable Formulation 2)**

Let \(i = \sqrt{-1}\). Parameters: \(\kappa,\theta,\nu,\rho,v_0\). Define
\[
b(u) = \kappa - \rho\nu i u
\]
\[
d(u) = \sqrt{b(u)^2 + \nu^2\,(u^2 + i u)}
\]
\[
g(u) = \frac{b(u) - d(u)}{b(u) + d(u)}
\]

Formulation 2 uses \(e^{-dT}\) (not \(e^{+dT}\)):

\[
D(u,T) = \frac{b(u) - d(u)}{\nu^2}\,\frac{1 - e^{-d(u)T}}{1 - g(u)e^{-d(u)T}}
\]
\[
C(u,T) = \frac{\kappa\theta}{\nu^2}\left((b(u) - d(u))T - 2\ln\left(\frac{1 - g(u)e^{-d(u)T}}{1 - g(u)}\right)\right)
\]
Then
\[
\varphi_{H}(u) = \exp\left(C(u,T) + D(u,T)v_0\right)
\]
If you prefer the log-price CF instead of log-forward CF, multiply by \(e^{iu\log F_0}\).

### Variance Gamma (Madan–Carr–Chang, 1998) — pure jump Lévy
Paper PDF (common lecture copy): https://engineering.nyu.edu/sites/default/files/2018-09/CarrEuropeanFinReview1998.pdf

Parameters: \(\sigma,\nu,\theta\). The VG “martingale correction” is
\[
\omega = \frac{1}{\nu}\ln\left(1 - \theta\nu - \tfrac{1}{2}\sigma^2\nu\right)
\]
Under the log-forward convention,
\[
\varphi_{VG}(u) = \exp(iu\omega T)\left(1 - i\theta\nu u + \tfrac{1}{2}\sigma^2\nu u^2\right)^{-T/\nu}
\]

### Kou (2002) — double-exponential jump diffusion
Paper (PDF): https://www.columbia.edu/~sk75/MagSci02.pdf

Jump size CF (double-exponential):
\[
\varphi_Y(u) = \frac{p\eta_1}{\eta_1 - i u} + \frac{(1-p)\eta_2}{\eta_2 + i u}
\]
Let \(\lambda\) be jump intensity and \(\sigma\) diffusion vol. The compensator uses
\[
\zeta = \mathbb{E}[e^Y] - 1 = \frac{p\eta_1}{\eta_1 - 1} + \frac{(1-p)\eta_2}{\eta_2 + 1} - 1
\]
(Requires \(\eta_1 > 1\).)

Under the log-forward convention, the CF of \(X_T\) is
\[
\varphi_{Kou}(u) = \exp\left(iu(-\tfrac{1}{2}\sigma^2 - \lambda\zeta)T - \tfrac{1}{2}\sigma^2 u^2 T + \lambda T(\varphi_Y(u) - 1)\right)
\]

Implementation note:
- \(\varphi_Y(u)\) has poles at \(u = -i\eta_1\) and \(u = +i\eta_2\). Carr–Madan damping/contours must stay inside the strip of analyticity for the parameter sets you use.

---

## Validation gates (what “working” means)

We do not claim correctness until we replicate published tables within stated tolerances. These replications live as notebooks + `pytest` tests.

- Carr–Madan (1999) VG cases (reference tables): https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf
- Fang–Oosterlee (2008) COS table (Heston): http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf
- Kou (2002) Table 1: https://www.columbia.edu/~sk75/MagSci02.pdf
- Lewis (2001) high-precision benchmark (independent cross-check):
  - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=282110
  - PDF mirror often used in implementations: https://www.maths.univ-evry.fr/pages_perso/crepey/Finance/ExpLevy.pdf

---

## Benchmarks (what we measure)

- runtime vs number of strikes (strip pricing)
- runtime vs tolerance (error target)
- accuracy vs reference prices
- scaling comparisons: FFT vs FRFT vs COS
- MC baseline runtime/error (including conditional MC where applicable)

We do not publish placeholder speedups here. Measured results live in notebooks and get copied into the README only after they’re reproducible.

---

## Repository layout (high level)

```text
src/foureng/
  char_func/        # base protocol + heston/vg/kou
  pricers/          # carr_madan / frft / cos
  iv/               # implied-vol inversion (robust solvers)
  mc/               # MC baselines (incl conditional MC where relevant)
  utils/            # grids, interpolation, cumulants, numerics

tests/              # paper-table replication tests
notebooks/          # narrative: MC baseline -> FFT -> FRFT -> COS -> full benchmark
```

---

## Extensions (only after validations are green)

- Fourier Greeks (differentiate under the integral sign; avoid bump-and-reprice)
- FFT price as a control variate for MC (variance reduction)
- Discrete Asian pricing via FFT convolution (Benhamou, 2002)

Benhamou (2002) SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=269491

---

## PyFENG integration

PyFENG: https://github.com/PyFE/PyFENG (docs: https://pyfeng.readthedocs.io/en/latest/)

PyFENG already includes Heston FFT pricing in pure Python. The integration goal here is not “rewrite Heston”; it’s:
- unify FFT/FRFT/COS behind a model-agnostic CF interface
- provide an adapter (or PR) so PyFENG models can plug into FRFT/COS and our validation/benchmark harness

---

## Roadmap

- Phase 1 — MC baseline: conditional MC timing vs strike count (10 / 100 / 1000)
- Phase 2 — Carr–Madan FFT: VG + Heston, validate vs CM1999 + Lewis
- Phase 3 — FRFT: implement + speed/accuracy frontier vs FFT
- Phase 4 — COS: validate vs FO2008; extend as appropriate
- Phase 5 — Kou replication: validate vs Kou Table 1
- Phase 6 — Extensions (Greeks / control variate / Asian convolution)
- Phase 7 — Packaging + adapter (PyFENG integration or PyPI-ready standalone)
