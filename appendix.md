# Appendix

This appendix collects the extra project material that does not belong in the package-facing `README.md`: repository context, methodology notes, benchmark setup, and interpretation. The package README stays focused on install, quick start, API surface, and the demo notebook; the longer project narrative lives here instead.

## Repository guide

- `foureng/`: packaged pricing library and public API
- `notebooks/demo.ipynb`: Colab-friendly quick-start walkthrough
- `notebooks/supplementary/demo_advanced.ipynb`: **supplementary** full-feature showcase  -  all 20 models, 6 pricers, Greeks, IV surface, calibration, MC, new models, validation highlights (v0.4.1); not the recommended starting point
- `notebooks/supplementary/presentation_fourier_methods.ipynb`: presentation notebook version
- `notebooks/fo2008_replication.ipynb`: full Fang-Oosterlee (2008) paper-faithful replication
- `notebooks/cosPaper_Replication.ipynb`: COS paper replication with extended scoreboard
- `notebooks/paper_replications/bates_mathworks_replication.ipynb`: Bates all-engine scoreboard vs MathWorks frozen reference
- `notebooks/paper_replications/three_halves_replication.ipynb`: 3/2 SV PyFENG regression + Baldeaux-Badran qualitative IV smile
- `notebooks/paper_replications/bates_sv32_validation_demo.ipynb`: instructor-requested 12-section Bates + 3/2 SV validation demo
- `notebooks/research/cos_method_improved.ipynb`: COS truncation and policy notebook
- `notebooks/research/adaptive_cos.ipynb`: adaptive filtered-COS extension notebook
- `benchmarks/`: generated tables and figures used by the notebooks; paper-replication CSVs written here
- `tests/`: regression and public API tests; `tests/refs/` holds frozen JSON reference data used by no-network paper-replication tests
- `docs/`: detailed documentation  -  model zoo, API reference, validation hierarchy, FO2008 tables, filtered-COS extension, Bates/SV32 validation cases, bibliography, and numerical notes

## Methods at a glance

The project compares several ways to price European options once the model's characteristic function is available.

| Method | Core idea | Typical role in this repo |
|--------|-----------|----------------------------|
| Monte Carlo | Simulate paths and average discounted payoffs | Baseline only; useful to show the sampling-cost benchmark |
| Carr-Madan FFT | Dampen the payoff and invert the transform on a uniform frequency grid | Classical strip pricer and validation reference |
| FRFT | Carr-Madan-style transform with more flexible strike-frequency coupling | Alternative transform method when grid tuning matters |
| COS classic | Expand the density in a cosine series on a truncated interval | Main spectral pricer for smooth, well-behaved cases |
| COS improved | Keep COS, but choose the truncation interval and grid more carefully | Better default COS variant for serious use |
| Adaptive filtered-COS | Search across no-filter and filtered COS policies under a tolerance target | Extension for short-maturity or jump-heavy cases where plain COS can ring |
| PyFENG FFT reference | External FFT implementation from PyFENG | Benchmark oracle for supported models, not the main in-house production path |

## Which pricer is good for what

| Pricer | Best use cases | Why it helps | Main tradeoff |
|--------|----------------|--------------|---------------|
| Monte Carlo | Sanity checks, payoff generality, non-Fourier extensions | Minimal model-specific numerical tuning | Slow convergence for dense vanilla strips |
| Carr-Madan FFT | Dense strike strips under models with a stable characteristic function | Prices many strikes at once and is easy to benchmark | Sensitive to `alpha`, `eta`, and interpolation choices |
| FRFT | Cases where the Carr-Madan strike lattice is too rigid | Decouples frequency and strike resolution more flexibly | More setup complexity than standard FFT |
| COS classic | Smooth models, moderate maturities, well-behaved cumulants | Very fast and often spectrally accurate | Static truncation can break down in stress cases |
| COS improved | General-purpose default for vanilla pricing in this repo | Better interval selection and grid policy than plain COS | Still a finite series, so ringing can remain |
| Adaptive filtered-COS | Short maturities, jump-heavy models, kink-driven oscillation | Adds a second control knob when interval selection alone is not enough | More machinery and more candidate evaluations |
| PyFENG FFT reference | Benchmarking Heston, VG, CGMY, OUSV, and other supported models | Independent external check against the in-house layer | Coverage depends on what PyFENG implements |

## Spectral filters in the adaptive COS layer

Junike-style truncation fixes the interval-selection problem, but it does not automatically remove finite-series oscillation. The filtered-COS extension damps the highest cosine modes before the final sum, which acts like a low-pass filter on the truncated expansion.

If the unfiltered COS series is

`price = disc * sum_k A_k V_k`

then the filtered version is

`price = disc * sum_k sigma_k A_k V_k`

where `sigma_k` is near one for the low modes and smaller for the tail modes.

| Filter | Shape | Practical effect |
|--------|-------|------------------|
| Fejer | Linear taper | Simple, robust damping of the tail modes |
| Lanczos | Sinc taper | Usually preserves mid-frequency detail better than Fejer |
| Raised cosine | Smooth cosine cutoff | Cleaner tail suppression with a softer transition |
| Exponential | `exp(-alpha (k/N)^p)` | Most flexible family; strongest control through the order `p` |

In the project extension, filtering is never forced. The selector keeps the no-filter improved COS candidate in the pool, then checks filtered alternatives and picks the cheapest candidate that still meets the target tolerance. The full implementation details and formulas appear later in [Section 17](#17-adaptive-filtered-cos-extension).

## 1. Project objective

This project implements deterministic Fourier pricing methods for European options under models with tractable characteristic functions. The implemented pricers are:

- Carr--Madan FFT;
- fractional FFT / FRFT;
- COS;
- an improved COS policy for truncation and series-resolution selection.

Monte Carlo is retained only as a baseline for accuracy and runtime comparison. The argument is not that Monte Carlo is invalid. The argument is narrower: for dense vanilla option strips and repeated calibration loops, deterministic Fourier methods are usually better suited when the model gives a usable characteristic function.

The analytical thread of the project is:

1. build a common characteristic-function interface;
2. price the same model through Carr--Madan FFT, FRFT, and COS;
3. validate the results against published references or independent numerical anchors;
4. extend the framework to stochastic-volatility plus jump composites implemented in-house;
5. compare accuracy, runtime, robustness, and model coverage.

## 2. End-to-end workflow

The workflow is:

1. define deterministic market inputs using `ForwardSpec`;
2. select a model and parameter set;
3. construct the model characteristic function;
4. choose a pricing method;
5. price one option, a strike strip, or an implied-volatility surface;
6. compare against benchmark prices;
7. convert prices to Black--Scholes implied volatilities where needed;
8. measure error and runtime;
9. plot the results for the notebook and report.

In code, the model-specific component reduces to a characteristic function:

```python
phi = lambda u: model_cf(u, fwd, params)
```

The pricer layer does not need to know whether the characteristic function comes from PyFENG or from an in-house model file.

## 3. Why Fourier methods are the focus

Monte Carlo has standard error of order

```math
\varepsilon_{MC} = O(n^{-1/2}).
```

Reducing Monte Carlo error by one order of magnitude therefore requires roughly two orders of magnitude more paths. In a calibration setting, this cost compounds across strikes, maturities, and optimizer iterations.

Fourier methods use the characteristic function instead. If a model gives the characteristic function of log-returns, pricing can be expressed as deterministic integration or as a Fourier series. That removes sampling noise and gives a more direct speed--accuracy trade-off for European vanilla options.

## 4. Model convention: log-forward coordinates

The repository works in log-forward coordinates:

```math
X_T = \log\left(\frac{S_T}{F_0}\right), \qquad
F_0 = S_0 e^{(r-q)T}.
```

All characteristic functions in the project are characteristic functions of `X_T`, not of `log(S_T)`. If the characteristic function of `log(S_T)` is needed instead, multiply by

```math
e^{iu\log F_0}.
```

This convention matters. Mixing log-spot and log-forward characteristic functions can create an apparently small but systematic pricing error.

A notation warning: the symbol `nu` is model-specific. In Heston, `nu` denotes vol-of-vol. In Variance Gamma, `nu` denotes the variance rate of the gamma time change.

## 5. Common characteristic-function interface

All model wrappers conform to one interface:

```python
from typing import Protocol
import numpy as np

class CharFunc(Protocol):
    def __call__(self, u: np.ndarray) -> np.ndarray:
        """Return phi_T(u) = E^Q[exp(i u X_T)] for X_T = log(S_T / F0)."""
        ...
```

Once a model exposes `phi(u)`, it can be priced by Carr--Madan FFT, FRFT, or COS without any model-specific changes to the pricer code.

## 6. Model coverage

The full model catalogue  -  all twenty supported models with parameter dataclasses,
CF sources, and API notes  -  is in [docs/model_zoo.md](docs/model_zoo.md).

The twenty models split into two groups.

### 6.1 PyFENG-backed characteristic functions

For models where PyFENG already provides a production-quality FFT model, the repository uses PyFENG as the characteristic-function backend rather than re-implementing it. These adapters call `pyfeng.*Fft.logp_cf` (renamed from `charfunc_logprice` in pyfeng 0.4.0):

- Black--Scholes--Merton (`pyfeng.BsmFft`);
- Heston (`pyfeng.HestonFft`);
- Schobel--Zhu / OUSV (`pyfeng.OusvFft`);
- Variance Gamma (`pyfeng.VarGammaFft`);
- CGMY (`pyfeng.CgmyFft`);
- Normal Inverse Gaussian (`pyfeng.ExpNigFft`);
- 3/2 Stochastic Volatility (`pyfeng.sv_fft`);
- Rough Heston (`pyfeng.sv_fft`).

The project contribution is not the re-derivation of these characteristic functions. The contribution is the common wrapper, the unified Fourier pricing layer, the validation harness, and the benchmark scoreboard.

### 6.2 In-house characteristic functions

The following twelve models are implemented directly:

- Kou double-exponential jump diffusion;
- Bates: Heston plus Merton lognormal jumps (SVJ composite);
- Heston--Kou: Heston plus Kou double-exponential jumps (SVJ composite);
- Heston--CGMY: Heston plus CGMY tempered-stable jumps (SVJ composite);
- GARCH diffusion (Wu, Ma & Wang 2012 analytic CF);
- Merton jump-diffusion (compound Poisson with log-normal jump sizes);
- Meixner process (hyperbolic-cosine CF);
- Bilateral Gamma (separate up/down Gamma processes, Küchler & Tappe 2008);
- Generalised Hyperbolic (normal variance-mean mixture via GIG);
- Finite Moment Log Stable (α-stable, Carr & Wu 2003);
- Double Heston (two independent Heston variance factors);
- VGSA (Variance Gamma on a stochastic CIR activity clock).

These are validated by:

1. independence factorisation against the PyFENG-backed Heston characteristic function;
2. model-reduction gates, such as zero jump intensity reducing Bates and Heston--Kou to Heston;
3. frozen 41-strike regression strips cross-verified between high-grid Carr--Madan FFT and FRFT;
4. COS convergence checks in both `N` and truncation-width settings.

## 7. Characteristic functions

### 7.1 Heston

Parameters: `kappa`, `theta`, `nu`, `rho`, `v0`, where `nu` is vol-of-vol.

Define

```math
b(u) = \kappa - \rho\nu i u,
```

```math
d(u) = \sqrt{b(u)^2 + \nu^2(u^2 + iu)},
```

```math
g(u) = \frac{b(u)-d(u)}{b(u)+d(u)}.
```

Using the stable Formulation 2 / Little Heston Trap representation,

```math
D(u,T)
=
\frac{b(u)-d(u)}{\nu^2}
\cdot
\frac{1-e^{-d(u)T}}{1-g(u)e^{-d(u)T}},
```

```math
C(u,T)
=
\frac{\kappa\theta}{\nu^2}
\left[
(b(u)-d(u))T
-
2\log\left(
\frac{1-g(u)e^{-d(u)T}}{1-g(u)}
\right)
\right].
```

The log-forward characteristic function is

```math
\varphi_H(u) = \exp\left(C(u,T) + D(u,T)v_0\right).
```

The stable representation is used because the original algebraically equivalent formulation can encounter complex-log branch issues in some parameter regimes.

### 7.2 Variance Gamma

Parameters: `sigma`, `nu`, `theta`, where `nu` is the variance rate of the gamma time change.

The martingale correction is

```math
\omega
=
\frac{1}{\nu}
\log\left(1-\theta\nu-\frac{1}{2}\sigma^2\nu\right),
```

which requires

```math
1-\theta\nu-\frac{1}{2}\sigma^2\nu > 0.
```

Under the log-forward convention,

```math
\varphi_{VG}(u)
=
\exp(iu\omega T)
\left(
1-i\theta\nu u+\frac{1}{2}\sigma^2\nu u^2
\right)^{-T/\nu}.
```

### 7.3 Kou double-exponential jump diffusion

Parameters: `sigma`, `lambda`, `p`, `eta_1`, `eta_2`.

The jump-size density is

```math
f_Y(y)
=
p\eta_1 e^{-\eta_1 y}\mathbf{1}_{\{y\ge 0\}}
+
(1-p)\eta_2 e^{\eta_2 y}\mathbf{1}_{\{y<0\}}.
```

The jump characteristic function is

```math
\varphi_Y(u)
=
\frac{p\eta_1}{\eta_1-iu}
+
\frac{(1-p)\eta_2}{\eta_2+iu}.
```

The exponential-jump compensator is

```math
\zeta
=
E[e^Y]-1
=
\frac{p\eta_1}{\eta_1-1}
+
\frac{(1-p)\eta_2}{\eta_2+1}
-
1,
```

which requires `eta_1 > 1`.

Under log-forward coordinates,

```math
X_T
=
\left(-\frac{1}{2}\sigma^2-\lambda\zeta\right)T
+
\sigma W_T
+
\sum_{j=1}^{N_T}Y_j.
```

Therefore,

```math
\varphi_{Kou}(u)
=
\exp\left(
iu\left(-\frac{1}{2}\sigma^2-\lambda\zeta\right)T
-\frac{1}{2}\sigma^2u^2T
+\lambda T(\varphi_Y(u)-1)
\right).
```

### 7.4 Stochastic-volatility plus jump composites

Under independence of the Heston diffusion block and the jump block, the log-forward characteristic function factorises:

```math
\varphi_{SVJ}(u) = \varphi_H(u)\varphi_J(u).
```

For a pure-jump block with Levy exponent `psi(u)`, use

```math
\varphi_J(u) = \exp\left(T\psi(u) - iuT\psi(-i)\right),
```

where the second term is the martingale compensator.

#### Bates

Bates combines Heston with Merton lognormal compound-Poisson jumps. For jump-log-mean `mu_J`, jump-log-vol `sigma_J`, and intensity `lambda_J`,

```math
\varphi_Y(u)
=
\exp\left(iu\mu_J-\frac{1}{2}\sigma_J^2u^2\right),
```

```math
\zeta
=
\exp\left(\mu_J+\frac{1}{2}\sigma_J^2\right)-1.
```

Then

```math
\varphi_J^{Bates}(u)
=
\exp\left(\lambda_JT(\varphi_Y(u)-1-iu\zeta)\right).
```

At `lambda_J = 0`, the jump block is one and Bates reduces to Heston.

#### Heston--Kou

Heston--Kou uses the same Heston block and Kou double-exponential jump block. The jump CF and compensator are the Kou formulas above. At zero jump intensity, Heston--Kou reduces to Heston.

#### Heston--CGMY

Heston--CGMY uses the CGMY tempered-stable Levy exponent

```math
\psi(u)
=
C\Gamma(-Y)
\left[
(M-iu)^Y-M^Y
+
(G+iu)^Y-G^Y
\right].
```

The martingale-compensated jump block is

```math
\varphi_J^{CGMY}(u)
=
\exp\left(T\psi(u)-iuT\psi(-i)\right).
```

At `C = 0`, the jump block is one and Heston--CGMY reduces to Heston.

## 8. Pricing methods

### 8.1 Carr--Madan FFT

Carr--Madan applies a damping factor to the call price as a function of log-strike so that the Fourier transform is integrable. The damped transform is evaluated on a uniform frequency grid, then inverted with the FFT to recover prices on a log-strike grid.

Important parameters:

- damping parameter `alpha`;
- FFT grid size `N`;
- frequency spacing `eta`;
- log-strike spacing `lambda`.

The grid relation is

```math
\lambda = \frac{2\pi}{N\eta}.
```

This means strike resolution and frequency resolution are coupled. Finer strike resolution requires changing the integration grid or increasing the grid size. The project uses interpolation to recover prices at requested strikes when those strikes do not lie exactly on the FFT grid.

### 8.2 Fractional FFT / FRFT

FRFT relaxes the strict grid coupling in the standard FFT construction. It allows the frequency and strike spacings to be chosen more flexibly. This makes it useful when the benchmark strikes or reporting strikes do not align naturally with the standard Carr--Madan FFT grid.

In the project narrative, FRFT should be described as a grid-flexibility improvement, not as a different pricing theory. It is still Fourier inversion of the same characteristic function.

### 8.3 COS

COS prices by expanding the density on a finite interval `[a,b]` using a Fourier-cosine series. The density itself does not need to be evaluated directly; the expansion coefficients are recovered from the characteristic function.

A standard cumulant-based truncation rule is

```math
[a,b]
=
\left[
c_1 - L\sqrt{c_2+\sqrt{|c_4|}},
\;
c_1 + L\sqrt{c_2+\sqrt{|c_4|}}
\right].
```

Here:

- `c1` is the first cumulant;
- `c2` is the second cumulant;
- `c4` is the fourth cumulant;
- `L` is a truncation-width multiplier;
- `N` is the number of cosine terms.

The absolute value around `c4` is a numerical safeguard in the standard heuristic. It prevents the square-root expression from becoming ill-conditioned when a numerical cumulant estimator returns a negative value.

### 8.4 Put-plus-parity implementation

For wide intervals, direct call payoff coefficients can contain large exponential terms. A more stable implementation prices the put and then recovers the call using put-call parity:

```math
C = P + S_0e^{-qT} - Ke^{-rT}.
```

This is only an implementation choice. It does not change the pricing model.

## 9. Repository structure

```text
src/foureng/
  models/           # PyFENG-backed and in-house characteristic functions
  pricers/          # carr_madan / frft / cos
  refs/             # paper anchors and frozen regression strips
  utils/            # grids, cumulants, implied volatility, numerics
  mc/               # Monte Carlo baselines
  pipeline.py       # unified price_strip dispatcher

tests/              # replication tests, PyFENG identity gates,
                    # model-reduction gates, frozen regression strips

notebooks/          # validation, benchmarking, demo, FO2008 replication

benchmarks/
  paper_replications/   # per-paper CSVs, params.py, summary.md files
  cos_method_improved/  # COS policy comparison CSVs
  mc_vs_fourier_methods/# cross-model and adaptive-selector CSVs
.github/workflows/  # CI and test matrix
```

The intended top-level narrative is:

1. MC baseline;
2. Carr--Madan FFT;
3. FRFT;
4. COS;
5. FO2008 replication;
6. improved COS truncation;
7. full benchmark scoreboard.

## 10. Validation gates

The project should not call a model or pricer correct until it passes explicit validation gates.

A sensible validation sequence is:

1. validate Carr--Madan FFT on published Variance Gamma benchmarks from Carr--Madan-style cases;
2. validate Heston prices against high-precision references, including at least one branch-cut stress case;
3. validate COS on Fang--Oosterlee Heston tables;
4. validate Kou by cross-checking Carr--Madan FFT, FRFT, and COS using the same Kou characteristic function;
5. for PyFENG-backed models, require characteristic-function identity against `pyfeng.*Fft.logp_cf`;
6. for SVJ composites, require model-reduction gates and high-resolution cross-method agreement.

This ordering reduces debugging ambiguity. First establish a reliable method--model pair, then widen the supported model/method matrix.

## 11. Accuracy metrics

For one option, use absolute error:

```math
\text{absolute error}
=
|P_{\text{method}}-P_{\text{benchmark}}|.
```

Relative error is

```math
\text{relative error}
=
\frac{|P_{\text{method}}-P_{\text{benchmark}}|}
{\max(|P_{\text{benchmark}}|,\varepsilon)}.
```

For a strip of strikes, use maximum absolute error:

```math
\max_i |P_i^{\text{method}}-P_i^{\text{benchmark}}|.
```

Optionally report RMSE:

```math
\text{RMSE}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\left(P_i^{\text{method}}-P_i^{\text{benchmark}}\right)^2
}.
```

## 12. FO2008 full-paper replication

The repository carries a paper-faithful replication of Fang and Oosterlee (2008) covering
BSM Table 2, Heston Tables 4–6, Variance Gamma Table 7, and CGMY Tables 8–10.

Full tables, interpretation, and the improved-COS summary are in
**[docs/fo2008_replication.md](docs/fo2008_replication.md)**.

Canonical notebook: [`notebooks/fo2008_replication.ipynb`](notebooks/fo2008_replication.ipynb)  
Parameter registry: `benchmarks/paper_replications/fo2008_cos/params.py`  
Generated CSVs: `benchmarks/paper_replications/fo2008_cos/outputs/`

## 13. What the FO2008 replication actually shows

The "ugly rows" are not a single COS failure.

- BSM Table 2: flat local error floor under paper-grid replay reflects truncation /
  reference-rounding rather than series-resolution failure.
- Heston Table 5: the long-maturity / wide-interval case needs interval and term count
  chosen jointly  -  increasing N alone cannot recover mass discarded by a too-narrow interval.
- Paper timings are 2008 hardware measurements and are not directly portable runtime comparisons.

This motivates the improved COS policy described in section 14.

## 14. Junike / Junike--Pankrashkin improved COS policy

The baseline COS implementation follows Fang--Oosterlee's cumulant interval rule. That gives exponential convergence in many smooth cases, but the replication shows that this behaviour is not automatic under a naive static grid.

A fixed cumulant interval is a rule of thumb. If `[a,b]` is too narrow, the method discards tail mass before the cosine expansion starts. Increasing `N` then improves the approximation only on the truncated interval; it cannot recover the missing mass. If `[a,b]` is too wide, the series requires more terms to resolve the interval, and payoff coefficients can become harder to evaluate stably.

The Junike--Pankrashkin idea is to choose the interval from a target tail tolerance rather than from a fixed multiplier. Let `m` be a center and `M` a half-width. For any `n >= 1`, Markov's inequality gives

```math
P(|X-m|\ge M)
\le
\frac{E[|X-m|^n]}{M^n}.
```

To make the tail probability at most `epsilon`, it is sufficient to choose

```math
M
\ge
\left(
\frac{E[|X-m|^n]}{\epsilon}
\right)^{1/n}.
```

Then set

```math
[a,b] = [m-M,\;m+M].
```

Junike's number-of-terms result reinforces the same message: COS has two numerical knobs, not one. The interval and `N` should be chosen jointly.

The practical policy is:

1. choose the support interval from the target truncation tolerance;
2. center the state variable where possible;
3. choose enough cosine terms to resolve the interval;
4. price the bounded payoff side where stable;
5. recover calls by put-call parity if direct call coefficients are unstable;
6. route to another Fourier method when the COS geometry is unfavourable.

This is why the improved method should be described as a robustness and policy improvement, not as a new pricing formula.

## 15. Results for the Junike-style fix

The improved COS truncation (Junike & Pankrashkin 2022; Junike 2024) replaces the heuristic `L` multiplier in the truncation interval with a rigorous tail-mass bound. On the FO2008 test suite, the improved path beats the strict paper-grid replay in 7 of 8 cases and beats the paper's own best-N result in 6 of 8 cases.

Per-case numbers with error and method columns: [`benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv`](benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv).
Full tables with paper comparison: [`docs/fo2008_replication.md`](docs/fo2008_replication.md).

The Heston T=10 case illustrates the core issue most clearly: naive paper-grid COS reaches an error of 4.68e-03 because the truncation window is too narrow for the long maturity, while the improved policy selects a wider interval and larger N automatically, recovering sub-1e-10 accuracy.

Conclusion: the Junike-style fix is not a different pricing formula. It is a better approximation policy for choosing the support and series resolution in COS.


## 16. Final project narrative

The final narrative should be:

1. Monte Carlo is flexible but inefficient for repeated vanilla pricing.
2. Characteristic-function models allow deterministic Fourier pricing.
3. Carr--Madan FFT is efficient but grid-coupled.
4. FRFT improves strike-grid flexibility.
5. COS can be very accurate, but its realised performance depends on the truncation interval and number of terms.
6. The full FO2008 replication exposes where naive COS settings work and where they do not.
7. Junike-style interval and term selection improves robustness and makes the COS behaviour easier to diagnose.
8. PyFENG integration lets the project focus on the numerical-methods layer rather than re-implementing every characteristic function.
9. In-house jump composites demonstrate that the common characteristic-function interface extends naturally beyond the PyFENG model set.
10. The adaptive filtered-COS layer adds a second control dimension (spectral filtering of the finite series) and a deterministic policy-search selector, giving the pricing pipeline a principled response to slow convergence without overriding the Junike interval selection.

## 17. Adaptive filtered-COS extension

Full implementation details, spectral filter formulas, the adaptive policy selector
algorithm, conservative framing, output files, and test coverage are in
**[docs/filtered_cos_extension.md](docs/filtered_cos_extension.md)**.

Demo notebook: [`notebooks/research/adaptive_cos.ipynb`](notebooks/research/adaptive_cos.ipynb)  
Improved-truncation notebook: [`notebooks/research/cos_method_improved.ipynb`](notebooks/research/cos_method_improved.ipynb)

**Summary:** Junike-style truncation fixes the interval-selection problem. Spectral
filtering damps the high-frequency COS coefficients before the payoff dot product,
addressing residual finite-series oscillation and nonsmoothness at the truncation boundary.
The adaptive selector builds a candidate set of `(COSGridPolicy, COSFilterSpec)` pairs and
returns the fastest candidate satisfying the user's error tolerance  -  with the no-filter
Junike candidate always in the pool, so the selector weakly dominates fixed Junike-COS.

The key interpretation is: *"Junike helps truncation. Filtering helps residual finite-series /
nonsmoothness cases. The adaptive selector chooses among vanilla COS, Junike-COS, and
filtered Junike-COS."*

## 18. References

Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). *The Little Heston Trap*. Wilmott Magazine, January, 83--92.  
https://perswww.kuleuven.be/~u0009713/HestonTrap.pdf

Benhamou, E. (2002). *Fast Fourier Transform for Discrete Asian Options*. Journal of Computational Finance, 6(1), 49--68.  
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=269491

Carr, P. and Madan, D.B. (1999). *Option Valuation Using the Fast Fourier Transform*. Journal of Computational Finance, 2(4), 61--73.  
https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf

Chourdakis, K. (2004). *Option Pricing Using the Fractional FFT*. Journal of Computational Finance, 8(2), 1--18.  
https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=6bdf4696312d37427eda2740137650c09deacda7

Fang, F. and Oosterlee, C.W. (2008). *A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions*. SIAM Journal on Scientific Computing, 31(2), 826--848.  
http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf  
https://epubs.siam.org/doi/10.1137/080718061

Hagan, P.S., Kumar, D., Lesniewski, A.S. and Woodward, D.E. (2002). *Managing Smile Risk*. Wilmott Magazine, September, 84--108.  
http://www.deriscope.com/docs/Hagan_2002.pdf

Heston, S.L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility*. Review of Financial Studies, 6(2), 327--343.  
https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf

Junike, G. (2024). *On the Number of Terms in the COS Method for European Option Pricing*. arXiv preprint arXiv:2303.16012.  
https://arxiv.org/abs/2303.16012

Junike, G. and Pankrashkin, K. (2022). *Precise Option Pricing by the COS Method: How to Choose the Truncation Range*. Applied Mathematics and Computation, 421, 126935.  
https://arxiv.org/abs/2109.01030  
https://doi.org/10.1016/j.amc.2022.126935

Ruijter, M. J., Versteegh, M. and Oosterlee, C. W. (2015). *On the Application of Spectral Filters in a Fourier Option Pricing Technique*. Journal of Computational Finance.  
https://doi.org/10.21314/JCF.2015.314

Kahl, C. and Jackel, P. (2005). *Not-so-complex Logarithms in the Heston Model*. Wilmott Magazine, September, 94--103.  
http://www2.math.uni-wuppertal.de/~kahl/publications/NotSoComplexLogarithmsInTheHestonModel.pdf

Kou, S.G. (2002). *A Jump-Diffusion Model for Option Pricing*. Management Science, 48(8), 1086--1101.  
https://www.columbia.edu/~sk75/MagSci02.pdf

Lewis, A.L. (2001). *A Simple Option Formula for General Jump-Diffusion and Other Exponential Levy Processes*. Envision Financial Systems working paper.  
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=282110

Lord, R. and Kahl, C. (2010). *Complex Logarithms in Heston-like Models*. Mathematical Finance, 20(4), 671--694.  
https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9965.2010.00416.x

Madan, D.B., Carr, P. and Chang, E.C. (1998). *The Variance Gamma Process and Option Pricing*. European Finance Review, 2(1), 79--105.

---

*MATH5030 Numerical Methods in Finance -- Columbia University MAFN, Spring 2026.*
