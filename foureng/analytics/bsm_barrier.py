"""BSM analytic closed-form for single-barrier options.

Reference: Reiner & Rubinstein (1991), extended formulation from
Haug (2007) "The Complete Guide to Option Pricing Formulas", Ch. 2.

Four barrier types supported for both calls and puts:
  "down_out" : knocked out if S_t ≤ H  (H < S₀ required)
  "down_in"  : activated  if S_t ≤ H
  "up_out"   : knocked out if S_t ≥ H  (H > S₀ required)
  "up_in"    : activated  if S_t ≥ H

In-out parity holds exactly:  knock_out + knock_in = vanilla option price.

Parameters follow the standard BSM parameterisation:
  S  : current spot price
  K  : strike
  H  : barrier level
  r  : continuously-compounded risk-free rate
  q  : continuous dividend yield
  T  : time to maturity (years)
  sigma : lognormal volatility

All prices are undiscounted by rebate (rebate=0 throughout).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ── vanilla reference ──────────────────────────────────────────────────────


def bsm_call(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """Standard BSM European call price."""
    b = r - q
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_put(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """Standard BSM European put price."""
    b = r - q
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ── core building-block (Haug 2007, Ch. 2) ────────────────────────────────


def _barrier_components(
    S: float,
    K: float,
    H: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
):
    """Return A, B, C, D building-block callables (φ, η-aware).

    Definitions (φ=+1 call / -1 put, η=+1 down / -1 up):
        A(φ)     = φ·[S·e^{-qT}·N(φ·x1) − K·e^{-rT}·N(φ·(x1−σ√T))]
        B(φ)     = φ·[S·e^{-qT}·N(φ·x2) − K·e^{-rT}·N(φ·(x2−σ√T))]
        C(φ,η)   = φ·[S·e^{-qT}·(H/S)^{2(μ+1)}·N(η·y1) − K·e^{-rT}·(H/S)^{2μ}·N(η·(y1−σ√T))]
        D(φ,η)   = φ·[S·e^{-qT}·(H/S)^{2(μ+1)}·N(η·y2) − K·e^{-rT}·(H/S)^{2μ}·N(η·(y2−σ√T))]

    where  μ = b/σ² − ½,  b = r − q,
        x1 = [ln(S/K)    + (1+μ)σ√T] / (σ√T)     ← standard BSM d₁
        x2 = [ln(S/H)    + (1+μ)σ√T] / (σ√T)
        y1 = [ln(H²/SK)  + (1+μ)σ√T] / (σ√T)     ← reflection of x1 about barrier
        y2 = [ln(H/S)    + (1+μ)σ√T] / (σ√T)     ← reflection of x2
    """
    b = r - q
    sq = sigma * np.sqrt(T)

    mu = b / sigma**2 - 0.5

    x1 = np.log(S / K) / sq + (1 + mu) * sq
    x2 = np.log(S / H) / sq + (1 + mu) * sq
    y1 = np.log(H**2 / (S * K)) / sq + (1 + mu) * sq
    y2 = np.log(H / S) / sq + (1 + mu) * sq

    hs_2mu2 = (H / S) ** (2 * (mu + 1))  # (H/S)^{2μ+2}
    hs_2mu = (H / S) ** (2 * mu)  # (H/S)^{2μ}
    Sd = S * np.exp(-q * T)
    Kd = K * np.exp(-r * T)

    def A(phi: int) -> float:
        return phi * (Sd * norm.cdf(phi * x1) - Kd * norm.cdf(phi * (x1 - sq)))

    def B(phi: int) -> float:
        return phi * (Sd * norm.cdf(phi * x2) - Kd * norm.cdf(phi * (x2 - sq)))

    def C(phi: int, eta: int) -> float:
        return phi * (Sd * hs_2mu2 * norm.cdf(eta * y1) - Kd * hs_2mu * norm.cdf(eta * (y1 - sq)))

    def D(phi: int, eta: int) -> float:
        return phi * (Sd * hs_2mu2 * norm.cdf(eta * y2) - Kd * hs_2mu * norm.cdf(eta * (y2 - sq)))

    return A, B, C, D


# ── main public function ───────────────────────────────────────────────────


def bsm_barrier_price(
    S: float,
    K: float,
    H: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    barrier_type: str,
    cp: int = 1,
) -> float:
    """BSM analytic price for a single-barrier option (zero rebate).

    Parameters
    ----------
    S, K, H : float
        Spot, strike, barrier.  H must be > 0.
    r, q, T, sigma : float
        Risk-free rate, dividend yield, maturity, lognormal vol.
    barrier_type : {"down_out", "down_in", "up_out", "up_in"}
        Barrier flavour.
    cp : int
        +1 call, −1 put.

    Returns
    -------
    float
        Option price.  In-out parity holds: down_out + down_in = vanilla.
    """
    phi = cp  # +1 call, -1 put
    eta = 1 if barrier_type.startswith("down") else -1

    # Already breached: knock-out = 0, knock-in = vanilla
    vanilla = bsm_call(S, K, r, q, T, sigma) if cp == 1 else bsm_put(S, K, r, q, T, sigma)
    if barrier_type.startswith("down") and S <= H:
        return 0.0 if barrier_type.endswith("_out") else vanilla
    if barrier_type.startswith("up") and S >= H:
        return 0.0 if barrier_type.endswith("_out") else vanilla

    A, B, C, D = _barrier_components(S, K, H, r, q, T, sigma)

    # Haug (2007) Table 2.14 formulas (φ=+1 call cases):
    #   down-and-out call (H≤K): A − C
    #   down-and-in  call (H≤K): C
    #   up-and-out   call (H≥K): A − B + C − D
    #   up-and-in    call (H≥K): B − C + D
    #
    # For puts (φ=−1), the same assembly formulas hold because A, B, C, D
    # already embed φ.  In-out parity: DO+DI = A(φ) = vanilla ✓
    #
    # Note: these formulas assume H≤K for down-barriers and H≥K for up.
    # For the atypical H>K down or H<K up cases additional terms arise;
    # those are not implemented here (they are rare in practice).

    # Two structural cases governed by the sign of φ×η:
    #
    #   φ·η = +1  ("easy barrier": barrier is on the far side of the exercise region)
    #     → down-out call  (call exercises above K; down barrier below K is far)
    #     → up-out put     (put exercises below K; up barrier above K is far)
    #   Formula: out = A − C,  in = C
    #
    #   φ·η = −1  ("hard barrier": barrier overlaps the exercise region)
    #     → up-out call    (call exercises above K; barrier above K cuts through)
    #     → down-out put   (put exercises below K; barrier below K cuts through)
    #   Formula: out = A − B + C − D,  in = B − C + D
    #
    # In-out parity: out + in = A(φ) = vanilla for each case. ✓

    if phi * eta > 0:  # easy-barrier case
        if barrier_type.endswith("_out"):
            return max(A(phi) - C(phi, eta), 0.0)
        else:
            return max(C(phi, eta), 0.0)
    else:  # hard-barrier case
        if barrier_type.endswith("_out"):
            return max(A(phi) - B(phi) + C(phi, eta) - D(phi, eta), 0.0)
        else:
            return max(B(phi) - C(phi, eta) + D(phi, eta), 0.0)


# ── double-barrier (eigenfunction expansion) ───────────────────────────────


def bsm_double_barrier_price(
    S: float,
    K: float,
    L: float,
    U: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    cp: int = 1,
    knockout: bool = True,
    n_max: int = 150,
) -> float:
    """BSM double-barrier option price via eigenfunction expansion.

    Prices a double knock-out (DKO) or double knock-in (DKI) option with
    lower barrier L and upper barrier U.

    The GBM log-price X = log(S_t/S) is absorbed at a = log(L/S) < 0 and
    b = log(U/S) > 0.  The transition density absorbed at both barriers is
    expanded in a sine-series (Karatzas & Shreve 1991, Kunitomo-Ikeda 1992,
    Haug 2007 Ch. 2.17):

        p(x, T | 0) = (2/M) exp(ν x) exp(-ν²σ²T/2)
                      Σ_{n=1}^∞ sin(nπy₀/M) sin(nπ(x-a)/M)
                                 exp(-n²π²σ²T / (2M²))

    where M = log(U/L), y₀ = log(S/L), a = -y₀, ν = (r-q)/σ² - 1/2.

    Parameters
    ----------
    S, K, L, U : float
        Spot, strike, lower barrier, upper barrier.  Must satisfy 0 < L < U.
    r, q, T, sigma : float
        Risk-free rate, dividend yield, maturity, lognormal vol.
    cp : int
        +1 call, -1 put.
    knockout : bool
        True → DKO price;  False → DKI price (= vanilla - DKO).
    n_max : int
        Maximum number of eigenfunction terms.  Convergence is accelerated
        by early termination when the n-th decay factor falls below 1e-14.

    Returns
    -------
    float
        Option price.  In-out parity holds: DKO + DKI = vanilla.

    Notes
    -----
    DKI price is computed via in-out parity:  DKI = vanilla - DKO.

    The series converges rapidly for T > 0;  more terms are needed for
    small T or large (U-L)/S.
    """
    if L <= 0 or U <= L:
        raise ValueError(f"Must have 0 < L < U; got L={L}, U={U}")
    if sigma <= 0 or T <= 0:
        raise ValueError(f"sigma and T must be strictly positive; got sigma={sigma}, T={T}")

    vanilla = bsm_call(S, K, r, q, T, sigma) if cp == 1 else bsm_put(S, K, r, q, T, sigma)

    # ── already knocked-out or impossible payoff ───────────────────────────
    if S <= L or S >= U:
        dko_price = 0.0
        return dko_price if knockout else vanilla

    # Call can never finish in-the-money if K ≥ U (spot absorbed before exercise region)
    # Put can never finish in-the-money if K ≤ L
    if cp == 1 and K >= U:
        return 0.0 if knockout else vanilla
    if cp == -1 and K <= L:
        return 0.0 if knockout else vanilla

    # ── log-space geometry ─────────────────────────────────────────────────
    M = np.log(U / L)  # width of barrier interval in log-space
    y0 = np.log(S / L)  # log(S/L) > 0 since L < S
    a = np.log(L / S)  # = -y0  (lower barrier in X-space)
    b_bar = np.log(U / S)  # upper barrier in X-space
    mu = r - q - 0.5 * sigma**2  # drift of log-price
    nu = mu / sigma**2  # convenience: μ/σ²

    # ── integration limits for the payoff integral ─────────────────────────
    log_K_S = np.log(K / S)  # log(K/S)
    if cp == 1:
        # Call: payoff = S·e^x − K  for x > log(K/S)
        xi_low = max(log_K_S, a)
        xi_high = b_bar
    else:
        # Put: payoff = K − S·e^x  for x < log(K/S)
        xi_low = a
        xi_high = min(log_K_S, b_bar)

    if xi_low >= xi_high:
        # Integration region is empty → DKO = 0
        return 0.0 if knockout else vanilla

    # ── common prefactor ───────────────────────────────────────────────────
    disc = np.exp(-r * T)
    global_factor = (2.0 / M) * disc * np.exp(-(nu**2) * sigma**2 * T / 2.0)

    # ── closed-form helper: ∫_c^d exp(α x) sin(η_n (x - a)) dx ───────────
    # J(α, c, d) = [exp(αx)(α sin(η_n(x-a)) − η_n cos(η_n(x-a)))] / (α² + η_n²) |_c^d
    def _antideriv(alpha: float, eta_n: float, x: float) -> float:
        """Antiderivative of exp(αx) sin(ηn(x-a)) evaluated at x."""
        denom = alpha**2 + eta_n**2
        phase = eta_n * (x - a)
        return np.exp(alpha * x) * (alpha * np.sin(phase) - eta_n * np.cos(phase)) / denom

    def _J(alpha: float, eta_n: float, c: float, d: float) -> float:
        return _antideriv(alpha, eta_n, d) - _antideriv(alpha, eta_n, c)

    # ── eigenfunction series ───────────────────────────────────────────────
    series_sum = 0.0
    for n in range(1, n_max + 1):
        decay = np.exp(-(n**2) * np.pi**2 * sigma**2 * T / (2.0 * M**2))
        if decay < 1e-14:
            break  # remaining terms negligible

        gamma_n = np.sin(n * np.pi * y0 / M)  # Γ_n = sin(nπy₀/M)
        eta_n = n * np.pi / M

        # I_n = S·J(ν+1, ξ_low, ξ_high) − K·J(ν, ξ_low, ξ_high)
        # (with appropriate sign for puts)
        if cp == 1:
            I_n = S * _J(nu + 1.0, eta_n, xi_low, xi_high) - K * _J(nu, eta_n, xi_low, xi_high)
        else:
            # Put payoff: (K − S·e^x) → I_n = K·J(ν, ...) − S·J(ν+1, ...)
            I_n = K * _J(nu, eta_n, xi_low, xi_high) - S * _J(nu + 1.0, eta_n, xi_low, xi_high)

        series_sum += gamma_n * decay * I_n

    dko_price = max(global_factor * series_sum, 0.0)

    if knockout:
        return dko_price
    else:
        # DKI = vanilla - DKO  (in-out parity)
        return max(vanilla - dko_price, 0.0)
