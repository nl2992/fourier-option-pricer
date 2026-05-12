# Adaptive Filtered-COS Extension

This document describes the adaptive filtered-COS layer  -  the primary **research extension**
in this project, going beyond the baseline Junike truncation policy.

**Important framing:** this is an **original project extension** inspired by prior spectral-filter
ideas. It is presented here as an in-house adaptive filtered-COS workflow motivated by the
literature rather than as a line-by-line replication of a single published paper.

**Demo notebook:** [`notebooks/research/adaptive_cos.ipynb`](../notebooks/research/adaptive_cos.ipynb)  
**Improved-truncation notebook:** [`notebooks/research/cos_method_improved.ipynb`](../notebooks/research/cos_method_improved.ipynb)

---

## Motivation

Junike-style truncation addresses the *interval-selection* component of COS pricing: given
the model's cumulants, it adaptively widens the integration domain `[a, b]` until the
tail-mass proxy falls below a user threshold `eps_trunc`. This is a necessary condition for
accuracy but not sufficient.

Two additional sources of error remain after interval selection is resolved:

1. **Finite-series truncation**  -  the COS expansion uses N terms.
   If the characteristic function decays slowly (heavy-tailed models, short maturities) or
   the payoff density has sharp features, the N-term series may carry visible Gibbs-like
   oscillations even when `[a, b]` is correctly chosen.

2. **Nonsmoothness at the truncation boundary**  -  the artificial periodisation introduced by
   COS creates a discontinuity at `a` and `b`. For models with non-Gaussian densities the
   boundary effect persists at moderate N.

Spectral filtering addresses both by damping the high-frequency COS coefficients before the
payoff dot product. The extension is *additive*: the filter is applied only when requested;
`filter_spec=None` reproduces the unfiltered output exactly.

**Inspiration:** Ruijter, Versteegh and Oosterlee (2015) applied spectral filters to a
Fourier option pricing technique and showed that exponential and raised-cosine filters
significantly reduce residual oscillation errors. Our extension follows their approach but
wraps it inside a deterministic policy-search selector rather than hard-coding a specific
filter.

---

## Spectral filter implementations

All filters are implemented in `foureng/utils/spectral_filters.py`.
Weight vector σ has length N; applied as `A[k] ← σ[k] · A[k]` before the payoff sum.
`σ[0] = 1` always (DC term is never modified).

| Filter | `σ_k` formula | Notes |
|--------|--------------|-------|
| `"none"` | 1 | Identity / no-op (default) |
| `"fejer"` | `1 − k/(N−1)` | First-order Cesàro summation |
| `"lanczos"` | `sinc(k/(N−1))` | `sinc` in the `np.sinc` sense |
| `"raised_cosine"` | `½(1 + cos(πk/(N−1)))` | Hann window |
| `"exponential"` | `exp(−α·(k/(N−1))^p)` | Order-p; default `α = −ln(ε_mach)` |

The exponential filter with p=8 is the default in `cos_filtered`. It keeps the
low-frequency terms effectively unchanged while sending the highest-frequency weight to
machine-ε.

---

## Adaptive policy grid-search selector

`foureng/experiments/cos_filter_grid_search.py` implements a deterministic selector that:

1. Builds a candidate set of `(COSGridPolicy, COSFilterSpec)` pairs.
2. Prices with each candidate and measures error against a high-resolution reference.
3. Returns the **fastest candidate satisfying the error tolerance** (or the lowest-error
   candidate if none satisfies it).

**Selection rule:**
```
if any candidate has max_abs_err ≤ tol and status == "ok":
    pick the one with lowest runtime_ms
else:
    pick the one with lowest max_abs_err (among status == "ok")
```

Because the no-filter Junike candidate is always included in the pool, the selector weakly
dominates fixed Junike-COS in the joint (error, runtime) metric under the tested grid.
It cannot pick a result worse than the best candidate in its set.

---

## FO2008 benchmark results

Results from [`notebooks/research/adaptive_cos.ipynb`](../notebooks/research/adaptive_cos.ipynb),
which benchmarks all three COS variants on five canonical FO2008 test cases at N=256.

| Case | Vanilla err | Junike err | Adaptive err | Filter selected | Interpretation |
|------|-------------|------------|--------------|-----------------|----------------|
| BSM T=1 | 1.78e-14 | 1.60e-14 | 1.60e-14 | none | All variants at machine precision |
| Heston T=1 | 8.42e-10 | 1.87e-11 | 4.82e-13 | none | Gain from adaptive N policy, not filter |
| Heston T=10 | 1.49e-10 | 2.51e-10 | 2.51e-10 | none | Wide interval needs more N; no filter help |
| VG T=0.1 | 3.34e-03 | 3.38e-03 | 3.20e-03 | raised_cosine | Filter gives a small real improvement |
| CGMY T=0.25 | 6.97e-07 | 5.85e-04 | 1.73e-08 | none | Junike widens interval; adaptive N recovers it |

**What these results show:**
- The spectral filter was actually selected (non-none) in only **1 of 5 cases** (VG T=0.1), and the improvement there is small (~5%).
- Large accuracy gains (Heston T=1, CGMY T=0.25) come from the **adaptive N selection policy**, not from spectral damping.
- For CGMY T=0.25, Junike truncation at N=256 is worse than vanilla because the wider interval requires more terms; the adaptive policy compensates by choosing a larger N.
- The filter's job is residual oscillation suppression once the interval and N are already well-chosen.

---

## Appropriate interpretation

> *"Junike helps truncation. Filtering helps residual finite-series / nonsmoothness cases.
> The adaptive selector chooses among vanilla COS, Junike-COS, and filtered Junike-COS."*

On the broader FO2008 test suite (8 cases, paper-grid N), the **Junike improved truncation** beats the naive paper-grid COS in **7/8 cases** and beats the paper's own best-N result in **6/8 cases**.
Full per-case data: [`benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv`](../benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv).
See also [fo2008_replication.md](fo2008_replication.md) for the full table.

Appropriate scope for the extension:
- The 7/8 and 6/8 results are attributable to the Junike truncation improvement, not to spectral filtering.
- The spectral filter provided a measurable (if small) gain in 1 of the 5 directly tested FO2008 cases.
- Filtered COS is a complementary stability control, most useful when the truncation window is already correct but finite-series oscillation remains.
- Junike interval selection and adaptive N policy are the primary accuracy drivers.

---

## Practical policy workflow

```python
import foureng as fe

# 1. Get cumulants
cums = fe.heston_cumulants(fwd, params)

# 2. Build improved grid
policy = fe.recommended_cos_policy("heston", params, mode="improved")
grid   = fe.cos_improved_grid(cums, model="heston", params=params)

# 3. Price with optional filter
filter_spec = fe.COSFilterSpec("exponential", order=8)
result = fe.filtered_cos_prices(phi, fwd, strikes, grid, filter_spec=filter_spec)

# 4. Or use the adaptive selector (picks filter automatically)
decision = fe.cos_adaptive_decision(fwd=fwd, params=params, model="heston",
                                     strikes=strikes, tol=1e-5)
```

---

## Output files

The demo notebook and `scripts/run_filtered_cos_extension.py` write:

| File | Description |
|------|-------------|
| `benchmarks/mc_vs_fourier_methods/outputs/cos_policy_search_showcase.csv` | Per-case grid-search results and adaptive result label |
| `benchmarks/mc_vs_fourier_methods/outputs/adaptive_filtered_cos_model_zoo.csv` | Model-zoo rerun summary |
| `benchmarks/mc_vs_fourier_methods/outputs/figures/cos_policy_search_showcase.png` | Showcase scatter (runtime vs error) |
| `benchmarks/mc_vs_fourier_methods/outputs/figures/adaptive_filtered_cos_model_zoo_errors.png` | Model-zoo error grouped bar chart |
| `benchmarks/mc_vs_fourier_methods/outputs/figures/adaptive_filtered_cos_model_zoo_runtime.png` | Model-zoo runtime grouped bar chart |

---

## Test coverage

| Test file | What it covers |
|-----------|---------------|
| `tests/methods/test_cos_spectral_filters.py` | Shape, finiteness, monotonicity, edge cases for all 5 filter types |
| `tests/methods/test_filtered_cos_pricing.py` | Backward compat (no-filter exact match), BSM accuracy, pipeline integration |
| `tests/methods/test_filtered_cos_outperforms_baselines.py` | Slow stress tests: VG T=0.1, CGMY T=0.25, BSM weak-dominance |

---

## Related documents

- [fo2008_replication.md](fo2008_replication.md)  -  where the improved COS beats the naive
  paper-grid replay
- [validation_hierarchy.md](validation_hierarchy.md)  -  how the filtered-COS tests are
  classified (numerical_stability, no exact numeric target)
- [appendix.md](../appendix.md) section 14  -  Junike/Junike-Pankrashkin theoretical background
