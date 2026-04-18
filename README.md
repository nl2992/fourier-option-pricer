# fourier-option-pricer

If your model gives you a characteristic function, you shouldn’t be simulating tens of thousands of Monte Carlo paths to price (or calibrate to) an implied-vol surface. This repo implements three Fourier-based European pricers Carr–Madan FFT, FRFT, and COS behind a single CharFunc protocol, wires them to Heston / Variance Gamma / Kou, and validates everything against published reference tables before claiming anything works. Monte Carlo stays in the repo, but only as the baseline you benchmark against.

## What we are building

- A model-agnostic pricing engine driven by characteristic functions (CFs).
- Three deterministic European pricers:
  - Carr–Madan FFT (Carr & Madan, 1999)
  - Fractional FFT / FRFT (Chourdakis, 2004)
  - COS method (Fang & Oosterlee, 2008)
- Three CF models that represent distinct smile mechanisms:
  - Heston (stochastic volatility; continuous paths)
  - Variance Gamma (pure-jump Lévy)
  - Kou (jump diffusion with asymmetric double-exponential jumps)
- A validation-first workflow: replicate published tables (and keep them as tests/notebooks).
- A benchmark harness: compare speed/accuracy vs Monte Carlo and across the three Fourier methods.

## The workflow (end-to-end)

1) Implement the model characteristic function  
   φ_T(u) = E^Q[ exp(i u X_T) ] for X_T = log(S_T/F_0) (or a consistent log-forward variant).

2) Price a strike strip in one call using a Fourier pricer:
- Carr–Madan FFT: inverse transform on a fixed log-strike grid (fast strip pricing, but grid-coupled).
- FRFT: decouple strike spacing from frequency spacing (same accuracy with fewer points, in practice).
- COS: cosine-series density expansion on a truncation range set by cumulants.

3) Convert price → implied vol (robust IV inversion).

4) Validate against published reference tables and record the achieved error tolerance.

5) Benchmark scaling:
- Runtime vs number of strikes
- Runtime vs target tolerance
- Accuracy vs reference prices
- Calibration-style repeated strip pricing

## Why Monte Carlo is only the baseline here

Monte Carlo is a reasonable tool for one-off pricing. It is not a good default tool for calibrating a full surface.

- Convergence is slow: ε_MC ∝ 1/√n. Cutting error by 10× costs ~100× paths.
- Calibration is repeated strip pricing (50–200 strikes across maturities, many iterations). Sampling noise becomes a practical bottleneck.
- Even with conditional Monte Carlo (where available), you still pay for path count and still carry sampling noise.

For European options under CF models, deterministic Fourier inversion avoids sampling entirely and gives a cleaner speed/accuracy frontier.

## The CharFunc protocol

Everything shares a single model interface:

```python
from typing import Protocol
import numpy as np

class CharFunc(Protocol):
    def __call__(self, u: np.ndarray) -> np.ndarray:
        """Return phi(u) = E^Q[exp(i u X_T)] for log-return X_T."""
        ...
```

Any model implementing this works with all pricers.

## Methods

### Carr–Madan FFT (1999)

Paper (PDF): https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf

We implement the damped call transform and invert it on a uniform grid using FFT (Simpson weights). Key parameters:
- damping α
- grid size N
- frequency step η
- strike grid spacing λ = 2π/(Nη)
- interpolation to off-grid strikes for validation tables

Known limitation:
- grid coupling: strike spacing and frequency spacing are linked by the FFT grid.

### FRFT (Chourdakis, 2004)

Reference page (PDF links): https://www.semanticscholar.org/paper/Option-Pricing-Using-the-Fractional-FFT-Chourdakis/6bdf4696312d37427eda2740137650c09deacda7

FRFT breaks the FFT grid coupling by allowing an arbitrary fractional parameter. Implemented via a Bluestein/chirp-z style decomposition (three FFTs). The point is not that FRFT is always faster; the point is that it often hits the same tolerance at much smaller N.

### COS (Fang & Oosterlee, 2008)

Preprint (PDF): http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf

COS expands the density on [a,b] using cosine series. Coefficients are obtained directly from the CF. Truncation range is set using cumulants c1,c2,c4 and a scale L. We validate against FO2008 tables before using it elsewhere.

## Models

### Heston (1993)

Paper (PDF): https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf

Implementation note:
- The standard Heston CF contains a complex logarithm. Naive Form 1 evaluation can silently break (branch cut issues) at large u and/or longer maturities.
- We implement the numerically stable Formulation 2 and keep a small demonstration test showing when Form 1 fails and Form 2 remains stable.

References:
- Albrecher et al. (2007) “The little Heston trap” (PDF): https://perswww.kuleuven.be/~u0009713/HestonTrap.pdf
- Kahl & Jäckel (2005) (PDF): http://www2.math.uni-wuppertal.de/~kahl/publications/NotSoComplexLogarithmsInTheHestonModel.pdf

### Variance Gamma (Madan–Carr–Chang, 1998)

VG is a clean early validation target because its CF is simple and stable.

PDF: https://engineering.nyu.edu/sites/default/files/2018-09/CarrEuropeanFinReview1998.pdf

### Kou (2002) double-exponential jump diffusion

Paper (PDF): https://www.columbia.edu/~sk75/MagSci02.pdf

Kou adds compound Poisson jumps with asymmetric double-exponential jump sizes, making crash asymmetry explicit. Implementation note:
- Kou’s CF has poles off the real axis; Carr–Madan damping/contour choices must respect the strip of analyticity. We document the constraints on α for the parameter sets we replicate.

## Why these three (quick table)

| | Heston | Variance Gamma | Kou |
|---|---|---|---|
| Path type | Continuous, stochastic vol | Pure jump | Diffusion + discrete jumps |
| Smile driver | ρ (skew), ν (curvature) | θ (skew), ν (tails) | λ, p (crash asymmetry) |
| CF difficulty | Medium (log + branch issues) | Simple | Simple but watch poles |
| Free parameters | 5 | 3 | 5 |
| Validation | Lewis (2001) benchmark | Carr–Madan (1999) cases | Kou (2002) Table 1 |

## Validation gates (what “working” means)

We do not treat the implementation as correct until it reproduces published reference tables within stated tolerances. These validations are also kept as tests and notebooks.

- Carr–Madan (1999) VG cases: https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf
- Fang–Oosterlee (2008) COS tables: http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf
- Kou (2002) Table 1: https://www.columbia.edu/~sk75/MagSci02.pdf
- Lewis (2001) high-precision benchmark (independent cross-check):
  - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=282110
  - PDF mirror used in some implementations: https://www.maths.univ-evry.fr/pages_perso/crepey/Finance/ExpLevy.pdf

## Benchmarks (what we measure)

We benchmark:
- runtime vs number of strikes (strip pricing)
- runtime vs tolerance (error target)
- accuracy vs reference prices
- scaling comparisons: FFT vs FRFT vs COS
- MC baseline runtime/error (including conditional MC where applicable)

We will not publish made-up speedups in the README; measured results live in notebooks and are copied here once verified.

## Repository layout (high level)

```text
src/foureng/
  char_func/        # heston / vg / kou + base protocol
  pricers/          # carr_madan / frft / cos
  iv/               # implied-vol inversion (robust solvers)
  mc/               # MC baselines (incl conditional MC where relevant)
  utils/            # grids, interpolation, cumulants, numerics

tests/              # paper-table replication tests
notebooks/          # narrative: MC baseline -> FFT -> FRFT -> COS -> benchmark tables
```

## Extensions (only after validations are green)

We only attempt extensions after replication is stable:
- Fourier Greeks (differentiate under the integral sign; avoid bump-and-reprice)
- FFT price as a control variate for MC (variance reduction)
- Discrete Asian pricing via FFT convolution (separate notebook/module)

Benhamou (2002) SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=269491

## PyFENG integration

PyFENG: repo https://github.com/PyFE/PyFENG · docs https://pyfeng.readthedocs.io/en/latest/

PyFENG already includes Heston FFT pricing in pure Python. The integration goal here is not rewrite Heston; it’s:
- unify FFT/FRFT/COS behind a model-agnostic CharFunc interface
- provide an adapter layer (or PR) so PyFENG models can plug into FRFT/COS and our validation/benchmark harness

## Roadmap

- Phase 1   MC baseline: conditional MC timing vs strike count (10 / 100 / 1000)
- Phase 2   Carr–Madan FFT: VG + Heston, validate against CM1999 + Lewis
- Phase 3   FRFT: implement + speed/accuracy frontier vs FFT
- Phase 4   COS: validate vs FO2008; extend as appropriate
- Phase 5   Kou replication: validate vs Kou Table 1
- Phase 6   Extensions (Greeks / control variate / Asian convolution)
- Phase 7   Packaging + adapter (PyFENG integration or PyPI-ready standalone)
