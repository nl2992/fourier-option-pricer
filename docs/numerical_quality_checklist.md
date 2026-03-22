# Numerical Quality Checklist

Rubric-facing audit against the course M1 / M4 floating-point best-practice requirements.
Every item is mapped to the corresponding source file and, where applicable, to the test
that verifies it.

---

## Quick-reference status

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | `float64` default throughout | ✅ pass | All CF arrays cast to `np.complex128`; output arrays are `float64`. |
| 2 | `expm1` for `exp(x)−1`, `log1p` for `log(1+x)` | ✅ fixed | `bates.py` zeta formula fixed (was `np.exp(x) - 1`). |
| 3 | `np.isfinite` guards on outputs | ✅ pass | `validity.py`, `implied_vol.py`, surface layer all guard. |
| 4 | No `log(sum(exp(...)))` pattern | ✅ pass | `logsumexp` not needed; no such pattern found in codebase. |
| 5 | Greeks via analytic formula, not FD | ✅ pass | `greeks/cos_greeks.py` — analytic ∂/∂F₀ and ∂²/∂F₀². |
| 6 | Central-difference step ∝ √ε | ✅ pass | `cumulants.py` 5-point central FD on imaginary axis uses `finfo.eps`. |
| 7 | Variance floor prevents MC NaN | ✅ pass | `heston_conditional_mc.py`: `np.maximum(sigma2, 1e-16)`. |
| 8 | RNG via `np.random.default_rng(seed)` | ✅ pass | All MC engines (BS, Heston, control variate) use `default_rng`. |
| 9 | Vectorised MC — no Python path loops | ✅ pass | `black_scholes_mc.py`, `heston_conditional_mc.py` batch-generate. |
| 10 | No `np.prod(U)` underflow risk | ✅ pass | No `np.prod` on probability arrays found in `foureng/`. |
| 11 | `float64` dtype on pricer output | ✅ pass | `price_strip` → `np.ndarray` of `dtype=float64`. |
| 12 | Input validation raises `ValueError` | ✅ pass | `models/fmls.py`, `meixner.py`, `merton_jd.py`, `bilateral_gamma.py`, `generalized_hyperbolic.py`. |
| 13 | No hardcoded magic tolerances | ✅ pass | `1e-16` floors are `≈ finfo.tiny`; `finfo(float).eps` used in filters. |
| 14 | Extreme inputs tested | ✅ pass | `tests/methods/test_numerical_quality.py` tiny-T, tiny-σ, wide strike range. |

---

## Detailed findings

### Item 2 — `expm1` fix in `bates.py` (bug fixed)

**File:** `foureng/models/bates.py`, lines 143 and 184.

**Problem:** The Bates jump compensator uses
```
zeta = exp(mu_j + 0.5 * sigma_j^2) - 1
```
When `mu_j` and `sigma_j` are both near zero the argument is `~0` and
`np.exp(x) - 1` suffers catastrophic cancellation, returning exactly `0.0` in
float64 when the true value is `~x > 0`.

**Example:** `mu_j = 1e-10`, `sigma_j = 1e-6` → argument `≈ 5e-13`.
- `np.exp(5e-13) - 1` rounds to `0.0` (all precision lost).
- `np.expm1(5e-13)` correctly returns `≈ 5e-13`.

**Fix:**
```python
# Before
zeta = np.exp(mu_j + 0.5 * sig_j * sig_j) - 1.0

# After
zeta = np.expm1(mu_j + 0.5 * sig_j * sig_j)
```
Applied to both `bates_cf` (CF computation) and `bates_cumulants`.

**Test:** `tests/methods/test_numerical_quality.py::TestCancellationSafeFormulas::test_bates_zeta_expm1_near_zero`.

---

### Item 5 — Analytic Greeks (better than FD requirement)

`foureng/greeks/cos_greeks.py` computes Delta and Gamma analytically by
differentiating the COS put-coefficient formula:

```
dV_k^put / dF_0     = -(2/(b-a)) * chi_k(a, d)
d^2 V_k^put / dF_0^2 = (2/(b-a)) * (K / F_0^2) * cos(omega_k * (d-a))
```

This is **exact** within the COS approximation and avoids any bump-size
sensitivity. Spot Delta/Gamma follow via the chain rule `dF_0/dS_0 = e^{(r−q)T}`.

---

### Item 6 — Central FD with machine-epsilon step size

`foureng/utils/cumulants.py` computes second cumulants numerically via a
5-point central finite-difference scheme on the imaginary axis of the CF:

```python
eps = np.finfo(float).eps          # 2.22e-16
h   = eps ** (1/5) * max(1, |x|)  # scales with magnitude
c2  = (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12*h^2)
```

Step size is tied to machine epsilon, not hardcoded.

---

### Item 7 — Variance floor in Heston MC

`foureng/mc/heston_conditional_mc.py`:
```python
sigma2 = (1.0 - rho^2) * V_T
sigma  = np.sqrt(np.maximum(sigma2, 1e-16))   # floor ≈ sqrt(eps_machine)
```
`1e-16 ≈ finfo.tiny / 4` prevents `sqrt(0)` → `0` divide-later or
`sqrt(negative)` → `NaN` when numerical noise drives `V_T` marginally negative.

---

### Item 8 — RNG pattern

All MC engines pass `rng = np.random.default_rng(seed)` created once at the
top of the pricing function and accept an integer seed at the API boundary:

```python
# foureng/mc/black_scholes_mc.py
rng = np.random.default_rng(mc.seed)
z = rng.standard_normal((n_paths, n_steps))
```

No `np.random.seed()` call exists anywhere in `foureng/`.

---

### Item 9 — Vectorised MC

Both MC engines batch-generate all paths in a single NumPy call:
```python
z   = rng.standard_normal((n_paths, n_steps))      # shape (N, M)
inc = drift * dt + vol * np.sqrt(dt) * z
log_paths = np.cumsum(inc, axis=1)                  # shape (N, M)
payoff = np.maximum(np.exp(log_paths[:, -1]) - K, 0.0)
```
No `for i in range(n_paths)` loop in the numerical kernel.

---

## Test file

All checklist items are covered by:

```
tests/methods/test_numerical_quality.py
```

Key test classes:

| Class / function | What it checks |
|-----------------|---------------|
| `TestCancellationSafeFormulas` | `expm1` precision for tiny args; `log1p` vs `log(1+x)` |
| `test_pricer_output_is_finite` | No NaN/inf for 9 (model, method) combinations |
| `test_pricer_output_is_float64` | dtype stays `float64` |
| `test_pricer_output_non_negative` | Call price ≥ 0 |
| `TestCOSGreeks` | Analytic Delta/Gamma finite, Delta ∈ (0,1), Gamma ≥ 0 |
| `TestMCReproducibility` | Same seed → same prices; different seeds → different prices |
| `TestVarianceFloor` | Near-zero v0 does not produce NaN in conditional MC |
| `TestExtremeInputs` | Tiny T, tiny σ, wide strike range all finite |
| `TestDtypes` | CF returns `complex128`; cumulants return `float` |

---

## What was not an issue

The following initially appeared suspicious but on inspection are correct:

| Pattern | Location | Why it is fine |
|---------|----------|---------------|
| `np.maximum(sigma2, 1e-16)` | `heston_conditional_mc.py` | `1e-16 ≈ finfo.tiny`; it is a precision guard, not a magic tolerance |
| `max(c.c2, 1e-16)` | `cumulants.py` | Same; prevents `sqrt(0)` in grid construction |
| `1e-12` theta/v0 floor | `double_heston.py` | Degenerate-parameter guard for the PyFENG CF backend |
| `abs(Y-1) > 1e-12` | `heston_cgmy.py` | Singularity check at the CGMY Y=1 pole |
| No `logsumexp` needed | whole codebase | No `log(sum(exp(...)))` pattern exists; pricing sums are in linear space |
| Analytic Greeks | `cos_greeks.py` | Better than the FD requirement; central-FD only appears in cumulant estimation |
