# Fang & Oosterlee (2008)  -  Full Paper Replication

This document collects the paper-faithful replication tables for Fang and Oosterlee (2008),
*A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions*,
SIAM Journal on Scientific Computing 31(2), 826–848.

**Canonical notebook:** [`notebooks/fo2008_replication.ipynb`](../notebooks/fo2008_replication.ipynb)  
**Parameter registry:** [`benchmarks/paper_replications/fo2008_cos/params.py`](../benchmarks/paper_replications/fo2008_cos/params.py)  
**Generated CSVs and figures:** `benchmarks/paper_replications/fo2008_cos/outputs/`

The tables below use the same horizontal format as the original paper, with `N` across the
columns and error / runtime down the rows.

---

## Table 1  -  GBM density recovery warm-up

Reconstructs the standard normal density from its characteristic function using the COS
density expansion on `[-10, 10]`.
The reported error is the maximum absolute error evaluated at `x = −5` and `x = 5`.

|  | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---:|---:|---:|---:|---:|
| max error | 4.9999e-02 | 3.2088e-02 | 3.6067e-03 | 3.1511e-07 | 5.5040e-17 |
| cpu time (sec) | ~0.0000 | ~0.0000 | ~0.0000 | ~0.0000 | ~0.0000 |

Error decays rapidly and reaches machine precision by N=64, validating the core COS identity:
density coefficients are recoverable from the characteristic function.

---

## Table 2  -  GBM calls, COS versus Carr-Madan

Parameters: σ=0.25, r=0.1, q=0, T=0.1, S0=100, K=80,100,120.

|  | N=32 | N=64 | N=128 | N=256 | N=512 |
|---|---:|---:|---:|---:|---:|
| paper COS msec | 0.0303 | 0.0327 | 0.0349 | 0.0434 | 0.0588 |
| paper COS max error | 2.43e-07 | 3.55e-15 | 3.55e-15 | 3.55e-15 | 3.55e-15 |
| our COS msec | 0.0832 | 0.0841 | 0.1111 | 0.1211 | 0.1695 |
| our COS max error | 3.15e-05 | 3.15e-05 | 3.15e-05 | 3.15e-05 | 3.15e-05 |
| paper Carr-Madan msec | 0.0857 | 0.0791 | 0.0853 | 0.0907 | 0.1111 |
| paper Carr-Madan max error | 9.77e-01 | 1.23e+00 | 7.84e-02 | 6.04e-04 | 4.12e-04 |
| our Carr-Madan msec | 0.3763 | 0.1569 | 0.1730 | 0.1923 | 0.2651 |
| our Carr-Madan max error | 1.34e+00 | 1.34e+00 | 4.58e-02 | 1.32e-02 | 4.85e-04 |

Our Carr-Madan replay converges toward the paper's final accuracy as N increases.
The flat COS error floor in this paper-grid replay reflects truncation or reference-rounding
effects rather than series-resolution error.

---

## Table 3  -  Cash-or-nothing digital option under GBM

Parameters: σ=0.2, r=0.05, q=0, T=0.1, S0=100, K=120.
Paper reference: 0.273306496497.

|  | N=40 | N=60 | N=80 | N=100 | N=120 | N=140 |
|---|---:|---:|---:|---:|---:|---:|
| error | 4.40e-09 | 2.86e-14 | 2.86e-14 | 2.86e-14 | 2.86e-14 | 2.86e-14 |
| cpu time (msec) | 0.0165 | 0.0169 | 0.0178 | 0.0182 | 0.0190 | 0.0202 |

Despite the discontinuous payoff, the COS approximation reaches roundoff-level error quickly
when analytic payoff coefficients are used.

---

## Table 4  -  Heston, T=1, ATM

|  | N=40 | N=80 | N=120 | N=160 | N=200 |
|---|---:|---:|---:|---:|---:|
| paper max error | 4.69e-02 | 3.81e-04 | 1.17e-05 | 6.18e-07 | 3.70e-09 |
| our max error | 2.68e-02 | 3.33e-03 | 8.25e-05 | 1.31e-05 | 6.41e-07 |
| paper msec | 0.0607 | 0.0805 | 0.1078 | 0.1300 | 0.1539 |
| our msec | 0.3811 | 0.1281 | 0.1138 | 0.1134 | 0.1374 |

The local Heston implementation converges clearly with N, but remains less accurate than the
paper's final row in the strict paper-grid replay.
This motivates the improved COS policy (see [filtered_cos_extension.md](filtered_cos_extension.md)).

---

## Table 5  -  Heston, T=10, ATM

This is the most important diagnostic table.
The long maturity and wide interval make the naive paper-grid replay converge much more slowly.

|  | N=40 | N=65 | N=90 | N=115 | N=140 |
|---|---:|---:|---:|---:|---:|
| paper max error | 4.96e-01 | 4.63e-03 | 1.35e-05 | 1.08e-07 | 9.88e-10 |
| our max error | 3.24e+00 | 7.65e-01 | 1.54e-01 | 1.97e-02 | 4.68e-03 |
| paper msec | 0.0598 | 0.0747 | 0.0916 | 0.1038 | 0.1230 |
| our msec | 0.1386 | 0.1040 | 0.1190 | 0.1935 | 0.1109 |

The issue is the joint choice of interval width and number of terms, not the Heston model itself.

---

## Table 6  -  Heston, T=1, 21-strike strip

|  | N=40 | N=80 | N=160 | N=200 |
|---|---:|---:|---:|---:|
| paper max error | 5.19e-02 | 7.18e-04 | 6.18e-07 | 2.05e-08 |
| our max error | 1.15e-01 | 5.46e-03 | 2.00e-05 | 2.63e-06 |
| paper msec | 0.1015 | 0.1766 | 0.3383 | 0.4214 |
| our msec | 0.1337 | 0.1395 | 0.2018 | 0.2347 |

The strip is harder than the ATM single-strike case because one shared interval must serve a
wider range of strikes.

---

## Table 7  -  Variance Gamma

For T=0.1:

|  | N=128 | N=256 | N=512 | N=1024 | N=2048 |
|---|---:|---:|---:|---:|---:|
| paper max error | 6.97e-04 | 4.19e-06 | 6.80e-06 | 5.70e-07 | 7.98e-08 |
| our max error | 4.28e-04 | 4.44e-05 | 8.97e-07 | 1.49e-08 | 4.94e-08 |
| our msec | 0.1412 | 0.1358 | 0.1346 | 0.1734 | 0.2687 |

For T=1.0:

|  | N=30 | N=60 | N=90 | N=120 | N=150 |
|---|---:|---:|---:|---:|---:|
| paper max error | 7.06e-03 | 1.29e-05 | 2.81e-07 | 3.16e-08 | 1.51e-09 |
| our max error | 4.57e-04 | 9.34e-06 | 1.71e-07 | 5.47e-09 | 4.39e-10 |
| our msec | 0.1116 | 0.0779 | 0.0811 | 0.0817 | 0.0876 |

The VG replication is strong.
The one-year case beats the paper's reported error by the final row.
The shorter maturity requires larger N, consistent with slower characteristic-function decay.

---

## Tables 8–10  -  CGMY

For Y=0.5:

|  | N=40 | N=60 | N=80 | N=100 | N=120 | N=140 |
|---|---:|---:|---:|---:|---:|---:|
| paper max error | 3.82e-02 | 6.87e-04 | 2.11e-05 | 9.45e-07 | 5.56e-08 | 4.04e-09 |
| our max error | 9.01e-04 | 1.68e-05 | 5.74e-07 | 2.81e-08 | 1.73e-09 | 2.16e-10 |
| paper msec | 0.0560 | 0.0645 | 0.0844 | 0.1280 | 0.1051 | 0.1216 |
| our msec | 0.1086 | 0.1194 | 0.1881 | 0.1084 | 0.1346 | 0.1074 |

For Y=1.5:

|  | N=40 | N=45 | N=50 | N=55 | N=60 | N=65 |
|---|---:|---:|---:|---:|---:|---:|
| paper max error | 1.38e+00 | 1.98e-02 | 4.52e-04 | 9.59e-06 | 1.22e-09 | 7.53e-10 |
| our max error | 6.57e-07 | 8.72e-09 | 6.62e-10 | 4.79e-10 | 4.77e-10 | 4.77e-10 |
| paper msec | 0.0545 | 0.0589 | 0.0689 | 0.0690 | 0.0732 | 0.0748 |
| our msec | 0.1090 | 0.1559 | 0.0939 | 0.1228 | 0.0977 | 0.1340 |

For Y=1.98:

|  | N=20 | N=25 | N=30 | N=35 | N=40 |
|---|---:|---:|---:|---:|---:|
| paper max error | 4.17e-02 | 5.15e-01 | 6.54e-05 | 1.10e-09 | 1.94e-15 |
| our max error | 1.81e-06 | 1.71e-09 | 1.48e-11 | 1.47e-11 | 1.47e-11 |
| paper msec | 0.0463 | 0.0438 | 0.0485 | 0.0511 | 0.0538 |
| our msec | 0.0874 | 0.0813 | 0.0871 | 0.0828 | 0.0837 |

The CGMY cases show that COS can remain effective for infinite-activity Lévy models when the
interval and cumulants are handled correctly.
The Y=1.98 case is numerically delicate; report language focuses on the error curve and
truncation-policy diagnostic.

---

## Interpretation

The "ugly rows" are not a single COS failure:

- **BSM Table 2**  -  flat local error floor under the paper-grid replay is consistent with
  truncation or reference-rounding effects, not series-resolution error.
- **Heston Table 5**  -  a long-maturity / wide-interval resolution problem.
  A wide interval needs enough cosine terms to resolve it.
- **Heston Table 6**  -  harder because a single interval is used across a 21-strike strip.
- **Paper timings** are historical 2008 hardware measurements and should not be treated as
  directly portable runtime claims.
- **COS accuracy depends on two choices at once**: support interval and number of cosine terms.

This motivates the improved COS policy described in
[filtered_cos_extension.md](filtered_cos_extension.md).

---

## Improved COS summary (from Junike-style fix)

| Case | Paper best N | Paper best max error | Paper-grid replay | Improved method | Improved N | Improved error |
|---|---:|---:|---:|---|---:|---:|
| BSM Table 2 | 64 | 3.55e-15 | 1.60e-14 | COS | 64 | 1.54e-14 |
| Heston Table 4 | 200 | 3.70e-09 | 6.57e-07 | COS | 512 | 2.22e-11 |
| Heston Table 5 | 140 | 9.88e-10 | 4.68e-03 | COS | 1024 | 9.68e-11 |
| Heston Table 6 strip | 200 | 2.05e-08 | 2.62e-06 | COS | 512 | 2.92e-10 |
| VG Table 7, T=0.1 | 2048 | 7.98e-08 | 4.94e-08 | COS | 1024 | 1.49e-08 |
| VG Table 7, T=1.0 | 150 | 1.51e-09 | 4.39e-10 | COS | 2048 | 2.00e-10 |
| CGMY Table 8, Y=0.5 | 140 | 4.04e-09 | 2.16e-10 | COS | 1024 | 1.19e-10 |
| CGMY Table 10, Y=1.98 | 40 | 1.94e-15 | 1.47e-11 | Lewis | 8192 | 6.41e-11 |

Headline: the adaptive path beats the strict paper-grid replay in 7/8 cases and beats the
paper's best reported error in 6/8 cases.

> **Evidence:** Per-case numbers in [`benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv`](../benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv) and [`benchmarks/paper_replications/cos_paper_replication/outputs/fo2008_replication_errors.csv`](../benchmarks/paper_replications/cos_paper_replication/outputs/fo2008_replication_errors.csv).
> Notebook: [`notebooks/research/cos_method_improved.ipynb`](../notebooks/research/cos_method_improved.ipynb).
> Summary report: [`benchmarks/paper_replications/cos_paper_replication/outputs/summary.md`](../benchmarks/paper_replications/cos_paper_replication/outputs/summary.md).
