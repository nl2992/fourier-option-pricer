# fourier-option-pricer

> Fast European and Asian option pricing via FFT, Fractional FFT, and COS — model-agnostic, benchmarked against Monte Carlo, and designed to integrate with [PyFENG](https://github.com/PyFE/PyFENG).

---

If your model gives you a characteristic function, you shouldn't need to run tens of thousands of Monte Carlo paths every time you want to price a smile. This project implements three Fourier-based pricing methods, plugs them into three models (Heston, Variance Gamma, and Kou), and packages everything cleanly so it can work alongside PyFENG or stand alone on PyPI.

The short version: $O(N \cdot n)$ for pricing $N$ strikes one-by-one becomes $O(n \log n)$ for the whole strip at once. That difference matters a lot when you're running a calibration loop.

---

## Why Monte Carlo isn't enough for calibration

On a trading desk, you're not pricing a single option in isolation — you're fitting a model to a strip of 50–200 strikes across multiple maturities, and you're doing that fit hundreds of times as the calibration iterates. MC works fine for a one-off price, but its convergence rate creates a compounding problem.

The fundamental result from MATH5030 Module 5: the MC estimator satisfies

$$\varepsilon_{\text{MC}} \propto \frac{1}{\sqrt{n}}$$

Ten times more paths gets you only ~3× better accuracy. To cut error by a factor of 10, you need 100× more paths. A single MC price at 10k paths might take 1 second; getting it to 3 decimal places takes 100 seconds. Multiply by 150 strikes, 6 maturities, and 500 calibration iterations, and the numbers get uncomfortable fast.

The Heston model (covered in Module 8) makes this worse. Simulating a single path requires discretizing the variance process at each step — either via the Milstein scheme or exact noncentral chi-squared sampling. The exact approach draws from

$$v_{t+h} = \frac{\nu^2(1 - e^{-\kappa h})}{4\kappa} \chi^2(\delta, \lambda)$$

which is accurate but expensive when you're calling `np.random.noncentral_chisquare` at scale. The conditional MC trick from the lectures (simulate $\sigma_t$ only, apply Black-Scholes analytically given $V_T$) helps at the margin, but you're still bottlenecked by path count.

Deterministic lattice methods converge faster at $\varepsilon \propto n^{-2/D}$ for $D < 4$ dimensions, but the dimensionality blowup for path-dependent or multi-asset problems sends you back to MC. For a vanilla European on a single underlying, though, there's a cleaner route.

---

## The characteristic function shortcut

The characteristic function of a model is

$$\varphi(u) = \mathbb{E}^{\mathbb{Q}}\left[e^{iu \log(S_T/F_0)}\right]$$

the Fourier transform of the risk-neutral log-return distribution. For Heston, Variance Gamma, Kou, and several other models used in practice, this is available in closed form. Once you have it, you can write the call price as a Fourier integral and evaluate it without simulating a single path.

Pricing $N$ strikes one at a time via numerical integration costs $O(N \cdot n)$ — $n$ quadrature points each time, repeated $N$ times. But the integral has the structure of a Fourier transform evaluated on an evenly-spaced grid, so you can compute all $N$ prices simultaneously via FFT in $O(n \log n)$, regardless of $N$.

That's the entire idea. The implementation follows.

---

## Methods

### Carr-Madan FFT (1999)

> Carr, P. and Madan, D.B. *Option valuation using the fast Fourier transform.* Journal of Computational Finance, 2(4), 61–73. [[PDF](https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf)]

The call price $C_T(k)$ as a function of log-strike $k$ isn't square-integrable — it converges to the spot price as $k \to -\infty$. Carr and Madan's fix is to multiply by a damping factor $e^{\alpha k}$, after which the Fourier transform exists and equals

$$\psi_T(v) = \frac{e^{-rT} \varphi\left(v - (\alpha+1)i\right)}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}$$

Prices come back via inverse Fourier transform:

$$C_T(k) = \frac{e^{-\alpha k}}{\pi}\int_0^\infty \text{Re}\left[e^{-ivk}\psi_T(v)\right] dv$$

Discretize on a uniform frequency grid, apply Simpson's rule weights, one FFT — and you have option prices across an entire log-strike grid. With $N = 4096$, $\eta = 0.25$, and $\alpha = 1.5$, the log-strike spacing works out to $\lambda = 2\pi/(N\eta) \approx 0.006$.

The limitation worth knowing: the Nyquist constraint $\eta \cdot \lambda = 2\pi/N$ ties frequency spacing to strike spacing. Finer strikes means coarser integration or more points. That's what motivates FRFT.

---

### Fractional FFT — Chourdakis (2004)

> Chourdakis, K. *Option pricing using the fractional FFT.* Journal of Computational Finance, 8(2), 1–18. [[CiteSeer](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=6bdf4696312d37427eda2740137650c09deacda7)]

FRFT breaks the Nyquist coupling by introducing a fractional parameter $\zeta = \eta \cdot \lambda$ that's no longer pinned to $2\pi/N$. You can choose a fine frequency grid for accurate quadrature and a fine strike grid centred exactly at-the-money, independently.

The algorithm uses Bluestein's chirp-z decomposition: the identity $jk = \frac{1}{2}[j^2 + k^2 - (k-j)^2]$ turns the arbitrary-$\zeta$ transform into a convolution, solvable with three standard FFTs of size $2N$. That's roughly 4× the cost of one FFT — but because FRFT converges with far fewer points ($N = 16$–$128$ instead of $4096$), the practical speedup is large.

Chourdakis showed that a 16-point FRFT matches a 4096-point Carr-Madan FFT in accuracy while running about 45× faster. No existing Python package implements FRFT. That's one of the things this project adds.

---

### COS method — Fang and Oosterlee (2008)

> Fang, F. and Oosterlee, C.W. *A novel pricing method for European options based on Fourier-cosine series expansions.* SIAM Journal on Scientific Computing, 31(2), 826–848. [[Preprint](http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf)] [[SIAM](https://epubs.siam.org/doi/10.1137/080718061)]

Rather than transforming in strike-space, COS expands the risk-neutral density directly in a Fourier-cosine series on a truncated interval $[a, b]$. The cosine coefficients come from the characteristic function, so the density never needs to appear explicitly. The price becomes a finite series:

$$V(x, t_0) \approx e^{-r\Delta t} \sum_{k=0}^{N-1}{}' \text{Re}\left[\varphi\left(\frac{k\pi}{b-a}\right) e^{-ik\pi a/(b-a)}\right] V_k$$

where the payoff coefficients $V_k$ have closed-form expressions. The truncation interval uses cumulants of $\log(S_T/S_0)$:

$$[a, b] = \left[c_1 - L\sqrt{c_2 + \sqrt{c_4}},\; c_1 + L\sqrt{c_2 + \sqrt{c_4}}\right], \qquad L = 10$$

For smooth densities, COS converges exponentially in $N$ and has $O(N)$ complexity per strike (implemented as a single matrix multiply in NumPy). On the Heston model, $N = 160$ achieves error around $3 \times 10^{-6}$ in about 1.2 ms. Carr-Madan needs 38 ms for comparable accuracy. COS is also simpler to implement — no damping parameter to tune, no Nyquist constraint to worry about.

---

## Models

### Heston (1993)

> Heston, S.L. *A closed-form solution for options with stochastic volatility.* Review of Financial Studies, 6(2), 327–343. [[PDF](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf)]

The standard stochastic volatility model. Asset price follows geometric Brownian motion with variance $v_t$ driven by a CIR mean-reverting process:

$$\frac{dS_t}{S_t} = \sqrt{v_t} \, dW_1, \qquad dv_t = \kappa(\theta - v_t) \, dt + \nu\sqrt{v_t} \, dW_2, \qquad \langle dW_1, dW_2 \rangle = \rho \, dt$$

The characteristic function has a log-affine form $\varphi(u) = \exp(C(u,T) + D(u,T)v_0 + iu\log F_0)$ where $C$ and $D$ come from Riccati ODEs solved analytically. There's a well-known numerical issue: the original Heston formulation contains a complex logarithm whose principal branch cuts incorrectly for large $u$ or long maturities, silently producing wrong prices. We use the numerically stable "Formulation 2" described in:

> Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. *The little Heston trap.* Wilmott Magazine, January 2007. [[PDF](https://perswww.kuleuven.be/~u0009713/HestonTrap.pdf)]

> Kahl, C. and Jäckel, P. *Not-so-complex logarithms in the Heston model.* Wilmott Magazine, September 2005. [[PDF](http://www2.math.uni-wuppertal.de/~kahl/publications/NotSoComplexLogarithmsInTheHestonModel.pdf)]

The Fourier price is the exact analogue to the conditional MC approach from Module 8 — same model, same distribution, just integrated directly rather than sampled.

---

### Variance Gamma — Madan, Carr, Chang (1998)

> Madan, D.B., Carr, P. and Chang, E.C. *The Variance Gamma process and option pricing.* European Finance Review, 2(1), 79–105.

VG replaces Brownian motion in the log-price with a Brownian motion time-changed by a Gamma process:

$$X_t = \theta G_t + \sigma W_{G_t}, \qquad G_t \sim \text{Gamma}(t/\nu, \nu)$$

The result is a pure-jump process with infinite activity and finite variation. Tail thickness and skew are controlled separately through $\sigma$, $\nu$, and $\theta$.

The characteristic function is:

$$\varphi_{\text{VG}}(u) = \left(1 - iu\theta\nu + \tfrac{1}{2}\sigma^2\nu u^2\right)^{-T/\nu}$$

No ODEs, no branch cuts — a few lines of NumPy. This is one of the two models Carr and Madan use in their original paper to validate the FFT method, so we have exact reference prices to check against from day one.

---

### Kou double-exponential jump diffusion (2002)

> Kou, S.G. *A jump-diffusion model for option pricing.* Management Science, 48(8), 1086–1101.

Kou keeps the Black-Scholes diffusion and adds a compound Poisson jump process where jump sizes follow an asymmetric double-exponential: upward jumps have rate $\eta_1$, downward jumps have rate $\eta_2$, and $p$ controls the fraction of up-jumps.

$$\frac{dS}{S} = (r - \lambda\zeta) \, dt + \sigma \, dW + dJ$$

The characteristic function is:

$$\varphi_{\text{Kou}}(u) = \exp\left(T\left[iur - \tfrac{1}{2}u^2\sigma^2 + \lambda\left(\frac{p\eta_1}{\eta_1 - iu} + \frac{(1-p)\eta_2}{\eta_2 + iu} - 1\right)\right]\right)$$

The poles at $u = -i\eta_1$ and $u = i\eta_2$ sit off the real axis for physical parameters. Kou sits between Heston and VG: it has a diffusion component, but with discrete jumps. The asymmetric jump structure directly encodes the observation that equity crashes are sharper and faster than rallies.

---

## Why these three

| | Heston | Variance Gamma | Kou |
|---|---|---|---|
| Path type | Continuous, stoch. vol | Pure jump | Diffusion + discrete jumps |
| Smile driver | $\rho$ (skew), $\nu$ (curvature) | $\theta$ (skew), $\nu$ (tails) | $\lambda, p$ (crash asymmetry) |
| CF difficulty | Medium — Riccati + branch cut | Trivial — power function | Easy — poles, no cuts |
| Free parameters | 5 | 3 | 5 |
| Validation source | Lewis (2001) 15-digit tables | Carr-Madan (1999) Cases 1–4 | Kou (2002) Table 1 |
| Course connection | Modules 7 and 8 directly | Carr-Madan paper itself | Jump-diffusion contrast |

Every model implements a single `CharFunc` protocol. Adding a fourth model is adding one file.

---

## Connection to the course

**Module 5** establishes the $1/\sqrt{n}$ MC convergence rate. The Fourier methods replace MC entirely for European options on models with a known CF — exact integration, no probabilistic noise.

**Module 6** covers spread and basket options, where the sum of lognormals has no closed-form distribution and MC is the standard approach. The same FFT convolution idea applies to arithmetic Asian options: build the distribution of the running average as an $N$-fold convolution of per-step return densities, each step an FFT. We implement this using the approach from Benhamou (2002) and validate it against the course MC pricer from Module 6.

**Module 7** motivates the project from the SABR side. Hagan's approximation formula produces negative implied densities at low strikes for large $\nu\sqrt{T}$ — documented in Choi and Wu (2021). Our engine gives exact Heston prices as ground truth for parameter regions where Hagan's formula is known to break.

**Module 8** is the most direct connection. The conditional MC method from the lectures — simulate $\sigma_t$, apply Black-Scholes given $V_T$ — is sampling from exactly the same distribution that the characteristic function describes. Same model, different route to the answer, no variance from sampling.

---

## Validation

We replicate three sets of published reference prices before calling anything working. Each is a `pytest` test; the suite won't pass until the numbers match.

**Carr-Madan (1999) — Variance Gamma.** Four parameter sets with $S_0 = 100$, $r = 0.05$, $q = 0.03$. Case 4 ($\sigma = 0.25$, $\nu = 2.0$, $\theta = -0.10$, $T = 0.25$) has exact VGP reference prices at strikes 77–79 of **0.6356, 0.6787, 0.7244**. Target: max absolute error below $10^{-4}$.

**Fang-Oosterlee (2008) — Heston COS.** $\kappa = 1.5768$, $\theta = 0.0398$, $\nu = 0.5751$, $v_0 = 0.0175$, $\rho = -0.5711$, $S_0 = K = 100$, $T = 1$. Reference call price: **5.785155450**. The Feller condition $2\kappa\theta = 0.1255 < \nu^2 = 0.3307$ is violated — intentional. Target: COS with $N = 160$ within $10^{-5}$.

**Lewis (2001) — Heston high-precision.** Computed to 15+ digits. $r = 0.01$, $q = 0.02$, $S_0 = 100$, $T = 1$, $v_0 = 0.04$, $\kappa = 4.0$, $\theta = 0.25$, $\nu = 1.0$, $\rho = -0.5$.

| Strike | Call price |
|--------|------------|
| 80 | 26.774758743998854 |
| 90 | 20.933349000596710 |
| 100 | 16.070154917028834 |
| 110 | 12.132211516709845 |
| 120 | 9.024913483457836 |

Target: COS within $10^{-6}$, Carr-Madan within $10^{-4}$.

---

## Architecture

> `[paper]` direct replication &nbsp;&nbsp;·&nbsp;&nbsp; `[core]` engine scaffolding &nbsp;&nbsp;·&nbsp;&nbsp; `[new]` beyond the papers &nbsp;&nbsp;·&nbsp;&nbsp; `[orig]` original contribution

```
fourier-option-pricer/
│
├── src/
│   └── foureng/
│       │
│       ├── char_func/                          [paper]
│       │   ├── base.py                         # CharFunc protocol — the only public contract
│       │   ├── heston.py                       # Albrecher Formulation 2, branch-cut safe
│       │   ├── variance_gamma.py               # 3 params, no ODEs, no branch cuts
│       │   └── kou.py                          # double-exponential jump diffusion
│       │
│       ├── pricers/                            [paper]
│       │   ├── carr_madan.py                   # FFT + Simpson weights, Nyquist constraint
│       │   ├── frft.py                         # Bluestein chirp-z, decouples η and λ
│       │   └── cos.py                          # Fourier-cosine series, exponential convergence
│       │
│       ├── greeks/                             [orig]
│       │   └── fourier_greeks.py               # Delta/Gamma/Vega via dCF/dS — one FFT call
│       │
│       ├── surface/                            [new]
│       │   ├── implied_vol.py                  # Newton/Brent IV inversion (Module 2)
│       │   ├── vol_surface.py                  # FFT prices -> IV grid -> surface plot
│       │   └── calibration.py                  # scipy minimize, fit model to market strikes
│       │
│       ├── exotic/                             [new]
│       │   ├── asian_fft.py                    # Benhamou 2002 convolution — O(N·n log n)
│       │   └── variance_swap.py                # fair strike via CF second moment (Module 8)
│       │
│       ├── control_variate/                    [orig]
│       │   └── fft_cv.py                       # FFT price as MC control variate (Module 5)
│       │
│       ├── market/                             [new]
│       │   ├── loader.py                       # yfinance SPX options pull + cleaning
│       │   └── pyfeng_adapter.py               # wrap PyFENG model -> CharFunc protocol
│       │
│       └── utils/                              [core]
│           ├── cumulants.py                    # COS truncation range [a,b] from cumulants
│           └── grid.py                         # FFT/FRFT grid construction helpers
│
├── tests/
│   ├── conftest.py                             # shared fixtures, reference price constants
│   ├── test_carr_madan_vg.py                   # [paper] CM1999 Cases 1-4 exact prices
│   ├── test_cos_heston.py                      # [paper] FO2008 Table 1, violated Feller
│   ├── test_lewis_benchmark.py                 # [paper] 15-digit Heston prices
│   ├── test_greeks.py                          # [orig]  analytical vs finite-difference greeks
│   ├── test_put_call_parity.py                 # [core]  robustness: extreme strikes, short T
│   ├── test_asian.py                           # [new]   Benhamou vs Module 6 MC pricer
│   └── test_variance_swap.py                   # [new]   CF fair strike vs Module 8 formula
│
├── notebooks/
│   ├── 01_mc_baseline.ipynb                    # [core]  course MC timing curves — the baseline
│   ├── 02_carr_madan.ipynb                     # [paper] FFT demo + CM1999 replication
│   ├── 03_frft_vs_fft.ipynb                    # [paper] Chourdakis accuracy/speed comparison
│   ├── 04_cos_method.ipynb                     # [paper] FO2008 Table 1 replication
│   ├── 05_fourier_greeks.ipynb                 # [orig]  Delta/Vega vs bump-and-reprice
│   ├── 06_fft_control_variate.ipynb            # [orig]  MC + FFT as control variate
│   ├── 07_spx_calibration.ipynb                # [new]   real market data, Heston + VG fit
│   ├── 08_asian_options.ipynb                  # [new]   Benhamou convolution vs course MC
│   └── 09_full_benchmark.ipynb                 # [core]  final comparison table
│
├── benchmarks/
│   ├── time_strikes.py                         # wall-clock vs N strikes for all methods
│   └── error_vs_n.py                           # max abs error vs N grid points
│
├── data/
│   └── spx_sample.csv                          # cached SPX option chain for reproducibility
│
├── pyproject.toml
├── CONTRIBUTING.md                             # how to add a model + PyFENG PR guidelines
└── README.md
```

The protocol that holds it together:

```python
from typing import Protocol
import numpy as np

class CharFunc(Protocol):
    def __call__(self, u: np.ndarray) -> np.ndarray:
        """Return phi(u) = E^Q[exp(iu * log(S_T / F_0))] for each u."""
        ...
```

Any model that satisfies this gets the full engine for free. The two `[orig]` modules — `greeks/` and `control_variate/` — are where the project goes beyond replication. Fourier Greeks give you Delta, Gamma, and Vega from a single FFT call by differentiating the characteristic function analytically; the control variate module uses the FFT price as the exact control variate for the MC pricer, directly connecting Module 5 theory to the Fourier engine.

---

## Benchmarks (target)

Single CPU core, 1,000 strikes at one maturity, Heston model.

| Method | Points | Max abs error | Time (ms) | Speedup vs MC |
|--------|--------|---------------|-----------|----------------|
| MC (conditional, 100k paths) | — | ~$10^{-3}$ | ~800 | 1× |
| Carr-Madan FFT | 4,096 | $< 10^{-4}$ | ~2 | ~400× |
| Fractional FFT | 128 | $< 10^{-4}$ | ~0.3 | ~2,600× |
| COS | 160 | $< 10^{-5}$ | ~1.2 | ~650× |

*These are targets from the literature. We'll update with measured numbers as implementation completes.*

---

## PyFENG integration

[PyFENG](https://github.com/PyFE/PyFENG) implements Heston, SABR, OUSV, GARCH diffusion, and the 3/2 model, with Fourier-transform pricing available internally ([docs](https://pyfeng.readthedocs.io/en/latest/)). What it doesn't expose is a model-agnostic engine with COS or FRFT as pricing backends.

The plan is to make our `CharFunc` protocol compatible with PyFENG's model structure — so a PyFENG model instance can be passed to our pricers directly — and open a PR to [PyFE/PyFENG](https://github.com/PyFE/PyFENG) with COS and FRFT as optional backends. If that's out of scope for the timeline, `foureng` ships as a standalone package on [PyPI](https://pypi.org/) with a documented adapter for PyFENG objects.

---

## Roadmap

- [x] Literature survey, paper replication targets identified
- [ ] **Phase 1** — MC baseline: time Heston conditional MC at $N = 10, 100, 1000$ strikes using Module 8 code
- [ ] **Phase 2** — Carr-Madan FFT: Heston + VG, validate against CM1999 and Lewis tables
- [ ] **Phase 3** — Fractional FFT: Bluestein decomposition, accuracy/speed comparison
- [ ] **Phase 4** — COS method: Heston + VG + Kou, validate against FO2008 Table 1
- [ ] **Phase 5** — Kou CF and calibration to market smiles
- [ ] **Phase 6** — Asian option via FFT convolution, validate against Module 6 MC
- [ ] **Phase 7** — Package structure, `pyproject.toml`, type hints, full test suite green
- [ ] **Phase 8** — PyFENG PR or standalone PyPI publish
- [ ] **Presentation** — April 28/30

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