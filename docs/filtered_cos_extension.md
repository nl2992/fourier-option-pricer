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

## Appropriate interpretation

The key interpretation for this extension is:

> *"Junike helps truncation. Filtering helps residual finite-series / nonsmoothness cases.
> The adaptive selector chooses among vanilla COS, Junike-COS, and filtered Junike-COS."*

On the FO2008 test suite, the **Junike improved truncation** (which the selector always includes as a candidate) beats the naive paper-grid COS in **7/8 cases** and beats the paper's own best-N result in **6/8 cases**.
Full per-case data: [`benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv`](../benchmarks/cos_method_improved/outputs/cos_method_improved_paper_compare.csv).
The spectral filter adds a further stability layer in cases where the truncation window is already correct but residual oscillation from finite-series remains.
See also [fo2008_replication.md](fo2008_replication.md) for the table of paper-grid vs. improved-COS errors.
Appropriate scope for the extension:
- The 7/8 and 6/8 results are attributable to the Junike truncation improvement, not to spectral filtering.
- Filtered COS serves as a complementary stability control on top of Junike COS.
- The extension is a deterministic policy layer over explicit candidate methods and filters.
- Junike interval selection remains the primary driver of accuracy improvement.

The extension is best understood as a second control layer that can improve pricing speed or
accuracy in cases where the Junike truncation is adequate but the finite-series resolution
is still the bottleneck.

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
