# Methodology and Results

This document records the numerical setup, characteristic-function conventions, validation workflow, benchmark results, and interpretation for `fourier-option-pricer`.

The README is the clean entry point. This file is the technical write-up: it explains where the models come from, how the pricing methods are connected, what was benchmarked, what the tables show, and how the results should be interpreted.

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

The ten supported models split into two groups.

### 6.1 PyFENG-backed characteristic functions

For models where PyFENG already provides a production-quality FFT model, the repository uses PyFENG as the characteristic-function backend rather than re-implementing it. These adapters route through `pyfeng.*Fft.charfunc_logprice`:

- Black--Scholes--Merton (`pyfeng.BsmFft`);
- Heston (`pyfeng.HestonFft`);
- Schobel--Zhu / OUSV (`pyfeng.OusvFft`);
- Variance Gamma (`pyfeng.VarGammaFft`);
- CGMY (`pyfeng.CgmyFft`);
- Normal Inverse Gaussian (`pyfeng.ExpNigFft`).

The project contribution is not the re-derivation of these characteristic functions. The contribution is the common wrapper, the unified Fourier pricing layer, the validation harness, and the benchmark scoreboard.

### 6.2 In-house characteristic functions

The following models are implemented directly:

- Kou double-exponential jump diffusion;
- Bates: Heston plus Merton lognormal jumps;
- Heston--Kou: Heston plus Kou double-exponential jumps;
- Heston--CGMY: Heston plus CGMY tempered-stable jumps.

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
  paper_replications/fo2008_cos/
    params.py
    outputs/
      SUMMARY.md
      *.csv
      *.png

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
5. for PyFENG-backed models, require characteristic-function identity against `pyfeng.*Fft.charfunc_logprice`;
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

The repository carries a paper-faithful replication report for Fang and Oosterlee (2008). The replication covers:

- BSM Table 2;
- Heston Tables 4, 5, and 6;
- Variance Gamma Table 7 at both maturities;
- CGMY Tables 8, 9, and 10.

The canonical notebook is:

```text
notebooks/fo2008_replication.ipynb
```

The paper registry is:

```text
benchmarks/paper_replications/fo2008_cos/params.py
```

Generated CSVs, figures, and summaries live under:

```text
benchmarks/paper_replications/fo2008_cos/outputs/
```

The tables below use the same horizontal format as the original paper, with `N` across the columns and error / runtime down the rows.

### 12.1 Table 1: GBM density recovery warm-up

This experiment reconstructs the standard normal density from its characteristic function using the COS density expansion on `[-10,10]`. The reported error is the maximum absolute error evaluated at `x=-5` and `x=5`.

|  | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---:|---:|---:|---:|---:|
| max error | 4.9999e-02 | 3.2088e-02 | 3.6067e-03 | 3.1511e-07 | 5.5040e-17 |
| cpu time (sec) | ~0.0000 | ~0.0000 | ~0.0000 | ~0.0000 | ~0.0000 |

Interpretation: the error decays rapidly and reaches machine precision by `N=64`. This validates the core COS identity: the density coefficients can be recovered from the characteristic function.

### 12.2 Table 2: GBM calls, COS versus Carr--Madan

Parameters: GBM volatility `sigma=0.25`, interest rate `r=0.1`, dividend yield `q=0`, maturity `T=0.1`, spot `S0=100`, strikes `K=80,100,120`.

|  | N=32 | N=64 | N=128 | N=256 | N=512 |
|---|---:|---:|---:|---:|---:|
| paper COS msec | 0.0303 | 0.0327 | 0.0349 | 0.0434 | 0.0588 |
| paper COS max error | 2.43e-07 | 3.55e-15 | 3.55e-15 | 3.55e-15 | 3.55e-15 |
| our COS msec | 0.0832 | 0.0841 | 0.1111 | 0.1211 | 0.1695 |
| our COS max error | 3.15e-05 | 3.15e-05 | 3.15e-05 | 3.15e-05 | 3.15e-05 |
| paper Carr--Madan msec | 0.0857 | 0.0791 | 0.0853 | 0.0907 | 0.1111 |
| paper Carr--Madan max error | 9.77e-01 | 1.23e+00 | 7.84e-02 | 6.04e-04 | 4.12e-04 |
| our Carr--Madan msec | 0.3763 | 0.1569 | 0.1730 | 0.1923 | 0.2651 |
| our Carr--Madan max error | 1.34e+00 | 1.34e+00 | 4.58e-02 | 1.32e-02 | 4.85e-04 |

Interpretation: our Carr--Madan replay converges toward the paper's final accuracy as `N` increases. The COS row has a flat observed error floor in this paper-grid replay. This is not a general failure of COS; it is evidence that the local setup is dominated by either truncation / reference rounding / implementation-grid choices rather than by the number of series terms.

### 12.3 Table 3: Cash-or-nothing digital option under GBM

Parameters: `sigma=0.2`, `r=0.05`, `q=0`, `T=0.1`, `S0=100`, `K=120`. The paper's quoted reference is `0.273306496497`, corresponding to a unit-cash digital `e^{-rT}N(d_2)`.

|  | N=40 | N=60 | N=80 | N=100 | N=120 | N=140 |
|---|---:|---:|---:|---:|---:|---:|
| error | 4.40e-09 | 2.86e-14 | 2.86e-14 | 2.86e-14 | 2.86e-14 | 2.86e-14 |
| cpu time (msec) | 0.0165 | 0.0169 | 0.0178 | 0.0182 | 0.0190 | 0.0202 |

Interpretation: despite the discontinuous payoff, the COS approximation reaches roundoff-level error quickly when analytic payoff coefficients are used.

### 12.4 Table 4: Heston, `T=1`, ATM

|  | N=40 | N=80 | N=120 | N=160 | N=200 |
|---|---:|---:|---:|---:|---:|
| paper max error | 4.69e-02 | 3.81e-04 | 1.17e-05 | 6.18e-07 | 3.70e-09 |
| our max error | 2.68e-02 | 3.33e-03 | 8.25e-05 | 1.31e-05 | 6.41e-07 |
| paper msec | 0.0607 | 0.0805 | 0.1078 | 0.1300 | 0.1539 |
| our msec | 0.3811 | 0.1281 | 0.1138 | 0.1134 | 0.1374 |

Interpretation: the local Heston implementation converges clearly with `N`, but remains less accurate than the paper's reported final row in the strict paper-grid replay. This motivates the improved COS policy below.

### 12.5 Table 5: Heston, `T=10`, ATM

|  | N=40 | N=65 | N=90 | N=115 | N=140 |
|---|---:|---:|---:|---:|---:|
| paper max error | 4.96e-01 | 4.63e-03 | 1.35e-05 | 1.08e-07 | 9.88e-10 |
| our max error | 3.24e+00 | 7.65e-01 | 1.54e-01 | 1.97e-02 | 4.68e-03 |
| paper msec | 0.0598 | 0.0747 | 0.0916 | 0.1038 | 0.1230 |
| our msec | 0.1386 | 0.1040 | 0.1190 | 0.1935 | 0.1109 |

Interpretation: this is the most important diagnostic table. The long maturity and wide interval make the naive paper-grid replay converge much more slowly. The issue is the joint choice of interval width and number of terms, not the Heston model itself.

### 12.6 Table 6: Heston, `T=1`, 21-strike strip

|  | N=40 | N=80 | N=160 | N=200 |
|---|---:|---:|---:|---:|
| paper max error | 5.19e-02 | 7.18e-04 | 6.18e-07 | 2.05e-08 |
| our max error | 1.15e-01 | 5.46e-03 | 2.00e-05 | 2.63e-06 |
| paper msec | 0.1015 | 0.1766 | 0.3383 | 0.4214 |
| our msec | 0.1337 | 0.1395 | 0.2018 | 0.2347 |

Interpretation: the strip is harder than the ATM single-strike case because one shared interval must serve a wider range of strikes.

### 12.7 Table 7: Variance Gamma

For `T=0.1`:

|  | N=128 | N=256 | N=512 | N=1024 | N=2048 |
|---|---:|---:|---:|---:|---:|
| paper max error | 6.97e-04 | 4.19e-06 | 6.80e-06 | 5.70e-07 | 7.98e-08 |
| our max error | 4.28e-04 | 4.44e-05 | 8.97e-07 | 1.49e-08 | 4.94e-08 |
| our msec | 0.1412 | 0.1358 | 0.1346 | 0.1734 | 0.2687 |

For `T=1.0`:

|  | N=30 | N=60 | N=90 | N=120 | N=150 |
|---|---:|---:|---:|---:|---:|
| paper max error | 7.06e-03 | 1.29e-05 | 2.81e-07 | 3.16e-08 | 1.51e-09 |
| our max error | 4.57e-04 | 9.34e-06 | 1.71e-07 | 5.47e-09 | 4.39e-10 |
| our msec | 0.1116 | 0.0779 | 0.0811 | 0.0817 | 0.0876 |

Interpretation: the Variance Gamma replication is strong. The one-year case beats the paper's reported error by the final row. The shorter maturity requires larger `N`, consistent with slower characteristic-function decay and sharper density features.

### 12.8 Tables 8--10: CGMY

For `Y=0.5`:

|  | N=40 | N=60 | N=80 | N=100 | N=120 | N=140 |
|---|---:|---:|---:|---:|---:|---:|
| paper max error | 3.82e-02 | 6.87e-04 | 2.11e-05 | 9.45e-07 | 5.56e-08 | 4.04e-09 |
| our max error | 9.01e-04 | 1.68e-05 | 5.74e-07 | 2.81e-08 | 1.73e-09 | 2.16e-10 |
| paper msec | 0.0560 | 0.0645 | 0.0844 | 0.1280 | 0.1051 | 0.1216 |
| our msec | 0.1086 | 0.1194 | 0.1881 | 0.1084 | 0.1346 | 0.1074 |

For `Y=1.5`:

|  | N=40 | N=45 | N=50 | N=55 | N=60 | N=65 |
|---|---:|---:|---:|---:|---:|---:|
| paper max error | 1.38e+00 | 1.98e-02 | 4.52e-04 | 9.59e-06 | 1.22e-09 | 7.53e-10 |
| our max error | 6.57e-07 | 8.72e-09 | 6.62e-10 | 4.79e-10 | 4.77e-10 | 4.77e-10 |
| paper msec | 0.0545 | 0.0589 | 0.0689 | 0.0690 | 0.0732 | 0.0748 |
| our msec | 0.1090 | 0.1559 | 0.0939 | 0.1228 | 0.0977 | 0.1340 |

For `Y=1.98`:

|  | N=20 | N=25 | N=30 | N=35 | N=40 |
|---|---:|---:|---:|---:|---:|
| paper max error | 4.17e-02 | 5.15e-01 | 6.54e-05 | 1.10e-09 | 1.94e-15 |
| our max error | 1.81e-06 | 1.71e-09 | 1.48e-11 | 1.47e-11 | 1.47e-11 |
| paper msec | 0.0463 | 0.0438 | 0.0485 | 0.0511 | 0.0538 |
| our msec | 0.0874 | 0.0813 | 0.0871 | 0.0828 | 0.0837 |

Interpretation: the CGMY cases show that the COS method can remain effective for infinite-activity Levy models when the interval and cumulants are handled correctly. The `Y=1.98` case is numerically delicate, so the correct report language should focus on the error curve and the truncation-policy diagnostic rather than claiming generic dominance.

## 13. What the FO2008 replication actually shows

The ugly rows are not one single "COS failure."

The correct reading is:

- BSM Table 2 shows a flat local error floor under the paper-grid replay. This is consistent with truncation or reference-rounding effects rather than series-resolution error.
- Heston Table 5 is a long-maturity / wide-interval resolution problem. A wide interval needs enough cosine terms to resolve it.
- Heston Table 6 is harder because a single interval is used across a 21-strike strip.
- Paper timings are historical 2008 hardware measurements and should not be treated as directly portable runtime claims.
- COS accuracy depends on two choices at once: support interval and number of cosine terms.

This motivates the improved COS policy.

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

The improved notebook does not claim that COS dominates every case. It shows where the adaptive policy helps, where it merely matches, and where another method is the honest fallback.

| Case | Paper best N | Paper best max error | Old default error | Our paper-grid replay | Improved method | Improved N | Improved error | Vs default | Vs paper | Vs paper-grid |
|---|---:|---:|---:|---:|---|---:|---:|---|---|---|
| BSM Table 2 | 64 | 3.55e-15 | 1.60e-14 | 1.60e-14 | COS | 64 | 1.54e-14 | better | worse | better |
| Heston Table 4 | 200 | 3.70e-09 | 6.10e-08 | 6.57e-07 | COS | 512 | 2.22e-11 | better | better | better |
| Heston Table 5 | 140 | 9.88e-10 | 5.07e-12 | 4.68e-03 | COS | 1024 | 9.68e-11 | worse | better | better |
| Heston Table 6 strip | 200 | 2.05e-08 | 6.98e-08 | 2.62e-06 | COS | 512 | 2.92e-10 | better | better | better |
| VG Table 7, T=0.1 | 2048 | 7.98e-08 | 4.44e-05 | 4.94e-08 | COS | 1024 | 1.49e-08 | better | better | better |
| VG Table 7, T=1.0 | 150 | 1.51e-09 | 1.99e-10 | 4.39e-10 | COS | 2048 | 2.00e-10 | worse | better | better |
| CGMY Table 8, Y=0.5 | 140 | 4.04e-09 | 1.19e-10 | 2.16e-10 | COS | 1024 | 1.19e-10 | worse | better | better |
| CGMY Table 10, Y=1.98 | 40 | 1.94e-15 | 2.46e-11 | 1.47e-11 | Lewis | 8192 | 6.41e-11 | worse | worse | worse |

Headline interpretation:

- the adaptive path beats the old default in `4/8` summary cases;
- it beats the strict paper-grid replay in `7/8` cases;
- it beats the paper's best reported error in `6/8` cases.

The Heston `T=10` diagnostics are especially useful:

- support-truncation error falls from about `7.00e-07` at `L=6` to `1.84e-09` at `L=8` and `5.06e-12` at `L=10`;
- the paper-wide `L=32` interval only becomes highly accurate when `N` is also increased substantially;
- direct call coefficients become unstable on wide intervals, while put-plus-parity remains better behaved.

Conclusion: the Junike-style fix is not a different pricing formula. It is a better approximation policy for choosing the support and series resolution in COS.

## 16. PyFENG integration

PyFENG provides useful FFT-style pricing and characteristic-function components. This repository uses PyFENG in two ways:

1. as a characteristic-function backend for models where PyFENG already has production-quality implementations;
2. as an independent oracle for pricing comparisons where PyFENG's own price method is available.

The value of the repository is therefore:

- a common characteristic-function abstraction across PyFENG-backed and in-house models;
- a single pricer interface for Carr--Madan FFT, FRFT, and COS;
- validation gates that catch adapter errors before they become price errors;
- model-reduction tests for jump composites;
- benchmark tables comparing method accuracy and runtime.

Implementation note: implied-volatility inversion is routed through a project-side Brent / safeguarded Newton implementation rather than relying blindly on any external wrapper. This keeps price-to-IV conversion consistent across methods.

## 17. Demo notebook workflow

The demo notebook should be results-led rather than method-led. The recommended sequence is:

1. install `fourier-option-pricer`;
2. import NumPy, pandas, plotting utilities, and `foureng`;
3. set a clean market configuration;
4. show one simple Heston or BSM option strip;
5. price the same strip using Carr--Madan FFT, FRFT, and COS;
6. compute absolute errors against a benchmark;
7. compute Black--Scholes implied volatilities;
8. time each method;
9. display a runtime table and an error table;
10. plot price error against runtime;
11. show one implied-volatility smile or surface;
12. close with the numerical lesson: Fourier methods are fast, but grid design determines reliability.

The notebook should not only explain methods. It should print the actual tables used in the report.

## 18. How to report the results

Use careful wording.

Correct:

- "COS reaches the best accuracy in this case once the interval and `N` are chosen jointly."
- "The naive paper-grid replay is truncation- or resolution-dominated."
- "The improved policy is a robustness improvement, not a universal dominance claim."
- "Paper timings are shown for historical comparison; local timings should be interpreted relative to the local machine."

Avoid:

- "COS failed."
- "Junike fixes COS everywhere."
- "Our runtime is faster than the paper" without specifying machine, implementation, and whether timings are comparable.
- "The method is validated" unless the relevant test gate passes.

## 19. Reproducibility checklist

For every reported result, include:

- repository commit hash;
- Python version;
- package version;
- notebook or script path;
- random seed if Monte Carlo is used;
- benchmark source;
- exact model parameters;
- pricing grid parameters;
- whether the price is one strike or a full strip;
- whether setup time is included;
- generated output path.

Recommended commands:

```bash
pytest
python benchmarks/run_all.py
```

Recommended report outputs:

```text
benchmarks/paper_replications/fo2008_cos/outputs/SUMMARY.md
benchmarks/paper_replications/fo2008_cos/outputs/*.csv
benchmarks/paper_replications/fo2008_cos/outputs/*.png
```

## 20. Final project narrative

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

## 21. References

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
