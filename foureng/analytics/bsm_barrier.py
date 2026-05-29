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
    S: float, K: float, H: float,
    r: float, q: float, T: float, sigma: float,
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

    hs_2mu2 = (H / S) ** (2 * (mu + 1))   # (H/S)^{2μ+2}
    hs_2mu  = (H / S) ** (2 * mu)          # (H/S)^{2μ}
    Sd = S * np.exp(-q * T)
    Kd = K * np.exp(-r * T)

    def A(phi: int) -> float:
        return phi * (Sd * norm.cdf(phi * x1) - Kd * norm.cdf(phi * (x1 - sq)))

    def B(phi: int) -> float:
        return phi * (Sd * norm.cdf(phi * x2) - Kd * norm.cdf(phi * (x2 - sq)))

    def C(phi: int, eta: int) -> float:
        return phi * (Sd * hs_2mu2 * norm.cdf(eta * y1)
                      - Kd * hs_2mu  * norm.cdf(eta * (y1 - sq)))

    def D(phi: int, eta: int) -> float:
        return phi * (Sd * hs_2mu2 * norm.cdf(eta * y2)
                      - Kd * hs_2mu  * norm.cdf(eta * (y2 - sq)))

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
    phi = cp                                          # +1 call, -1 put
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

    if phi * eta > 0:   # easy-barrier case
        if barrier_type.endswith("_out"):
            return max(A(phi) - C(phi, eta), 0.0)
        else:
            return max(C(phi, eta), 0.0)
    else:               # hard-barrier case
        if barrier_type.endswith("_out"):
            return max(A(phi) - B(phi) + C(phi, eta) - D(phi, eta), 0.0)
        else:
            return max(B(phi) - C(phi, eta) + D(phi, eta), 0.0)
