# Numerical Limitations and Caveats

This document records the known numerical limitations of the pricing methods
in `foureng`. Understanding these boundaries helps you choose grid parameters,
select the right method for a given model, and interpret any unexpected results.

---

## 1. COS method: truncation interval selection

The COS method (Fang & Oosterlee 2008) approximates the risk-neutral density
as a cosine series on a finite interval `[a, b]`. Truncating outside this
interval introduces a systematic error that does **not** decrease with the
number of terms `N` — it can only be reduced by widening `[a, b]`.

**Standard rule** (`cos_auto_grid`):

```
a = c1 - L * sqrt(c2 + sqrt(|c4|))
b = c1 + L * sqrt(c2 + sqrt(|c4|))
```

with `L = 10` as the default. This works well for models with sub-exponential
tails (all diffusion and most Lévy models here). It can fail for:

- **Alpha-stable distributions with α < 2**: power-law tails mean the
  cumulant-based width rule will underestimate the required truncation. For
  FMLS with α < 1.9, use Carr-Madan or FRFT rather than COS — they work
  on the real frequency axis where the CF decays exponentially. See
  `tests/models/test_fmls_paper.py`.
- **Very long maturities or high vol-of-vol**: the distribution spreads out;
  check that `[a, b]` covers at least 5σ of the log-return distribution.

**Improved rule** (`cos_improved`, `cos_adaptive_decision`):

The Junike-Pankrashkin (2022) tolerance-based truncation selects `[a, b]` by
bounding the CF tail energy rather than using fixed cumulants. This is more
robust for heavy-tailed models but can trigger a fallback to Carr-Madan or
Lewis integration for very wide intervals (> 48 log-units). Watch the
`COSPolicyDecision.method` field to see which path was taken.

---

## 2. COS method: term count `N` and exponential convergence

For distributions with exponential tails, COS converges as `O(exp(-c·N))`.
In practice, `N = 64–256` gives errors < 1e-6 for vanilla European options
under Heston at typical maturities.

Exponential convergence degrades to algebraic if:

- The density has discontinuities in its derivative (e.g., compound-Poisson
  models with `lam` very small but non-zero — the density has a point mass
  mixing with a smooth component).
- The truncation interval is too narrow (Gibbs phenomenon).

The standard recommendation is `N ≥ 128` for production use. `N = 256` is
conservative and still fast.

---

## 3. Spectral filtering: when it helps and when it does not

Filters (Fejér, Lanczos, exponential, raised-cosine) reduce the Gibbs
oscillation at the edges of the COS interval by tapering the high-frequency
cosine coefficients. They are enabled via `method="cos_filtered"` and the
`COSFilterSpec` object.

**When filtering helps:**
- Models with moderate tails and a narrow-ish truncation interval.
- Reducing oscillation near deep-ITM or deep-OTM strikes.

**When filtering introduces bias:**
- The exponential filter with a high `order` parameter can damp low-frequency
  content and bias ATM prices. Keep `order ≤ 12` for vanilla options.
- Near discontinuities in the payoff (binary options, barrier payoffs):
  filtering smooths the discontinuity and introduces a systematic bias.
- For the Lanczos filter: known to over-smooth for small `N`; test with
  `N = 256` before comparing to `N = 64`.

Always compare filtered vs unfiltered prices on your model/strike range before
relying on filtering in production.

---

## 4. Carr-Madan FFT: damping parameter `alpha`

The Carr-Madan (1999) FFT requires a dampened integrand to make the call
payoff transform `L^2`-integrable. The damping parameter `alpha` must satisfy:

- `alpha > 0` (always required).
- `E[S_T^{alpha+1}] < ∞` (model-dependent moment condition).

For most models, `alpha = 1.5` is a safe default. It can fail if:

- The model has fat right tails (e.g., CGMY with `M` close to 1, or FMLS
  with `alpha` close to 1). Increase `M` or decrease `alpha`.
- `alpha` is set so large that the integrand oscillates before decaying —
  reduce `alpha`.

The `utils/validity.py` module provides `alpha_valid_upper_bound(model, params)`
to check the moment condition at construction time.

**Log-strike grid spacing `eta`:**

A smaller `eta` (finer frequency spacing) covers a wider strike range `[K_min, K_max]`
but reduces the number of distinct strikes on the uniform log-strike grid.
For typical equities: `eta = 0.25` with `N = 4096` gives a grid spanning
roughly exp(±π/eta) ≈ exp(±12.6) log-moneyness, more than enough.

---

## 5. FRFT: non-uniform log-strike grids

The FRFT (Chourdakis 2004) lifts the Nyquist constraint η·λ = 2π/N, allowing
independent control of frequency spacing `eta` and log-strike spacing `lam`.
This lets you target any strike range without wasting grid points on distant
strikes.

Accuracy risk: the FRFT is more sensitive to aliasing when `N` is small or
`lam` is large. Use `N ≥ 1024` and validate against Carr-Madan at a few
representative strikes.

---

## 6. PyFENG dependency caveats

Several models delegate to PyFENG for their characteristic function:
`bsm`, `heston`, `ousv`, `vg`, `cgmy`, `nig`, `sv32`, `rough_heston`.

Known compatibility issues:

1. **Rough Heston and `scipy.misc.derivative`**: `pyfeng.ex.RoughHestonFft`
   fails under SciPy ≥ 1.14 because `scipy.misc.derivative` was removed.
   `foureng` works around this by importing directly from `pyfeng.sv_fft`:
   ```python
   from pyfeng.sv_fft import RoughHestonFft
   ```
   If you upgrade PyFENG and see an `ImportError`, check whether PyFENG has
   moved `RoughHestonFft` to a different submodule.

2. **PyFENG `charfunc_logprice` convention**: PyFENG's name "logprice" is
   actually a log-forward CF (verified to ~1e-18 against the Lewis convention
   we use throughout). If a future PyFENG version changes this convention, a
   single correction factor `phi * exp(-1j*u*log(F0))` would be needed; this
   is documented with inline markers in `foureng/models/heston.py`.

3. **Minimum version**: `pyfeng >= 0.3.0` is required. The `constraints/minimum.txt`
   file pins this for CI. If you see unexpected `AttributeError` on a PyFENG
   model, check `pip show pyfeng` and upgrade if needed.

---

## 7. Parameter validation: known edge cases

All `Params` constructors now validate inputs at construction time and raise
`ValueError` with a diagnostic message. Some non-obvious edge cases:

- **CGMY with Y close to 1**: Y = 1 is excluded because the Lévy exponent
  has a logarithmic branch. The constructor rejects `|Y - 1| < 1e-12`.
- **NIG existence condition**: `1 - 2*theta*nu - sigma^2*nu > 0` must hold,
  otherwise the martingale correction `mu` is complex. The constructor checks
  this and raises before PyFENG can produce a NaN.
- **Kou/HestonKou `eta1 > 1`**: the Kou jump mean `E[J] = p/eta1 - (1-p)/eta2`
  requires `eta1 > 1` for the martingale correction to be finite.
- **Heston `nu = 0`**: allowed (degenerates to BSM), though the Feller
  condition `2*kappa*theta ≥ nu^2` becomes trivially satisfied.

---

## 8. Implied volatility inversion

Newton-safeguarded (`implied_vol_newton_safeguarded`) is faster but can fail
to converge near intrinsic value (deep ITM options) or near zero vega (deep
OTM, long maturity). Brent (`implied_vol_brent`) is slower but guaranteed to
converge within `[0.001, 5.0]` vol. Use Brent as a fallback when Newton
fails.

Both methods can return NaN if the input price is below intrinsic value or
above the no-arbitrage upper bound. Check `BSInputs.is_valid()` before calling
the inversion.

---

## References

See [papers.md](papers.md) for full citations of all methods.
