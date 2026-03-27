# Methodology and Results

## Bates and 3/2 replication layer

### MathWorks Reference Case (Bates)

The Bates (1996) validation layer uses a reference case derived from the
MathWorks Financial Toolbox functions `optByBatesNI` and `optByBatesFFT`.
The frozen reference prices are stored in `tests/refs/bates_mathworks_reference.json`
and cover five European call strikes on a 6-month contract:

| Strike | Reference call price |
|-------:|---------------------:|
|  76.00 |               7.5765 |
|  78.00 |               6.4020 |
|  80.00 |               5.3484 |
|  82.00 |               4.4173 |
|  84.00 |               3.6073 |

Market inputs: S0 = 80, r = 0.03, q = 0.02, T = 0.5.

Heston block: v0 = 0.04, theta = 0.05, kappa = 1.0, sigma_v = 0.2, rho = -0.7.

Jump block: lambda = 2.0 jumps/year, mean jump percentage = 2%, jump vol = 8%.

#### mu_j conversion formula

MathWorks specifies jumps in terms of `mean_jump_percentage` (the expected
proportional change in the price level per jump). The log-jump mean `mu_j`
used internally is:

    mu_j = log(1 + mean_jump_percentage) - 0.5 * sigma_j^2

For the reference case:

    mu_j = log(1.02) - 0.5 * 0.08^2 = 0.01660262729617973

The jump compensator ensures the forward price remains F0:

    zeta = exp(mu_j + 0.5 * sigma_j^2) - 1 = mean_jump_percentage = 0.02
    omega_j = -lambda * zeta

#### Tolerances

- COS, COS-improved, Lewis: atol = 1e-2 against MathWorks published values; 5e-4 for internal cross-engine tests.
- Carr-Madan, FRFT, filtered-COS: atol = 1e-3 against internal reference.

MathWorks specifies jumps through a proportional mean jump `MeanJ`. The repo converts it to the log-jump mean using `mu_j = log(1 + MeanJ) - 0.5 * JumpVol^2`, which matches the documented MathWorks convention. Any residual discrepancy should be treated as a date, basis, integration, grid, LittleTrap, or implementation-convention issue rather than automatically as a different jump-mean formula. The MathWorks reference prices are classified as `software_reference`; internally generated high-resolution prices are classified as `derived_reference`.

### 3/2 Stochastic Volatility

#### PyFENG regression target

The exact prices for the 3/2 model are difficult to obtain from a closed-form
formula. Instead, the primary validation target is the PyFENG `Sv32Fft` pricer
(backed by Lewis 2000), which provides a trusted numerical reference. The frozen
regression prices are stored in `tests/refs/sv32_pyfeng_reference.json`:

Parameters: v0 = 0.06, kappa = 20.48, theta = 0.218, nu = 3.20, rho = -0.99.
S0 = 100, r = 0, q = 0, T = 0.5. Strikes: [95, 100, 105].
Reference prices: [11.7235, 8.9978, 6.7091].

This is not a paper-replication benchmark. It is a backend regression target
for the current PyFENG-backed sv32 implementation. Exact reproducibility
relies on a fixed PyFENG version.

#### Baldeaux-Badran figure parameters

Baldeaux and Badran (2012) illustrate the 3/2 model's short-maturity smile
with a 9-calendar-day parameter set. These parameters are stored in
`tests/refs/sv32_baldeaux_badran_figure_params.json` for qualitative notebook
replication and smoke testing only. No hard exact prices are asserted against
this parameter set; only no-arbitrage shape constraints are checked
(monotonicity and convexity of calls in strike).

### Jump convention: log-forward, phi(-i) = 1

All characteristic functions in this project are defined under the log-forward
convention:

    X_T = log(S_T / F_0),   F_0 = S_0 * exp((r - q) * T)

The martingale condition requires:

    E[exp(X_T)] = E[S_T / F_0] = 1

which translates to the CF identity:

    phi(-i) = E[exp((-i)(-i) X_T)] = E[exp(X_T)] = 1

Both the Bates and 3/2 CF implementations satisfy this condition to numerical
precision (verified in the test suite as part of CF identity checks).
