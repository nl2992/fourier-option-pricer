"""Black-Scholes-Merton closed-form reference pricers.

Provides exact closed-form pricing for:
- European vanilla (BSM formula)
- Cash-or-nothing and asset-or-nothing digitals
- Discrete geometric Asian (Goldman-Sosin-Gatto style adjustment)
- Forward-starting options (by homogeneity / scale-invariance)
- Single-barrier options (Reiner-Rubinstein 1991)
- Floating- and fixed-strike lookback options (Goldman-Sosin-Gatto 1979)

All functions are purely analytic — no numerical integration required.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float):
    """Compute BSM d1 and d2."""
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    return d1, d2


# ---------------------------------------------------------------------------
# European BSM
# ---------------------------------------------------------------------------


def bsm_european(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    cp: int,
) -> float:
    """Standard Black-Scholes-Merton price.

    Parameters
    ----------
    S     : current spot price
    K     : strike price
    r     : continuously compounded risk-free rate
    q     : continuous dividend yield
    sigma : lognormal volatility
    T     : time to maturity (years)
    cp    : +1 call, -1 put

    Returns
    -------
    float
        BSM option price.
    """
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc = np.exp(-r * T)
    growth = np.exp(-q * T)
    if cp == 1:
        return S * growth * norm.cdf(d1) - K * disc * norm.cdf(d2)
    else:
        return K * disc * norm.cdf(-d2) - S * growth * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Digital BSM
# ---------------------------------------------------------------------------


def bsm_cash_or_nothing(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    cp: int,
    cash: float = 1.0,
) -> float:
    """Cash-or-nothing digital option price.

    Pays ``cash`` if S_T > K (call) or S_T < K (put).

    Call price  = cash * exp(-rT) * N( d2)
    Put  price  = cash * exp(-rT) * N(-d2)
    """
    _, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc = np.exp(-r * T)
    if cp == 1:
        return cash * disc * norm.cdf(d2)
    else:
        return cash * disc * norm.cdf(-d2)


def bsm_asset_or_nothing(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    cp: int,
) -> float:
    """Asset-or-nothing digital option price.

    Pays S_T if S_T > K (call) or S_T < K (put).

    Call price = S * exp(-qT) * N( d1)
    Put  price = S * exp(-qT) * N(-d1)
    """
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    growth = np.exp(-q * T)
    if cp == 1:
        return S * growth * norm.cdf(d1)
    else:
        return S * growth * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Geometric Asian BSM (discrete, equal spacing)
# ---------------------------------------------------------------------------


def bsm_geometric_asian(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    N: int,
    cp: int,
) -> float:
    """Closed-form price for a discrete geometric Asian option.

    N monitoring dates equally spaced at t_i = i*T/N, i = 1, ..., N.
    Uses the lognormality of the geometric average of a GBM path.

    Adjusted drift and variance:
        mu_G   = log(S) + (r - q - sigma^2/2) * T*(N+1)/(2*N)
        var_G  = sigma^2 * T * (N+1)*(2*N+1) / (6*N^2)
        F_G    = exp(mu_G + var_G/2)   (geometric-average "forward")
        sqrt_v = sqrt(var_G)
        d1_G   = (log(F_G/K) + var_G/2) / sqrt_v
        d2_G   = d1_G - sqrt_v
        call   = exp(-rT) * (F_G * N(d1_G) - K * N(d2_G))
        put via parity: put = call - exp(-rT)*(F_G - K)
    """
    mu_G = np.log(S) + (r - q - 0.5 * sigma**2) * T * (N + 1) / (2 * N)
    var_G = sigma**2 * T * (N + 1) * (2 * N + 1) / (6 * N**2)
    F_G = np.exp(mu_G + 0.5 * var_G)
    sqrt_v = np.sqrt(var_G)
    disc = np.exp(-r * T)

    if sqrt_v < 1e-14:
        # Zero volatility limit: deterministic payoff
        intrinsic = max(cp * (F_G - K), 0.0)
        return disc * intrinsic

    d1_G = (np.log(F_G / K) + 0.5 * var_G) / sqrt_v
    d2_G = d1_G - sqrt_v

    call = disc * (F_G * norm.cdf(d1_G) - K * norm.cdf(d2_G))
    if cp == 1:
        return call
    else:
        # Put-call parity for geometric Asian
        return call - disc * (F_G - K)


# ---------------------------------------------------------------------------
# Forward-starting BSM
# ---------------------------------------------------------------------------


def bsm_forward_start(
    S: float,
    r: float,
    q: float,
    sigma: float,
    start_time: float,
    maturity: float,
    alpha: float,
    cp: int,
) -> float:
    """BSM forward-starting option price.

    Strike is set at K = alpha * S_{start_time} at the start date.
    By scale-invariance (homogeneity of degree 1 in S):

        V(0) = exp(-q * start_time) * S
               * bsm_european(1, alpha, r, q, sigma, tenor, cp)

    where tenor = maturity - start_time.

    When start_time <= 0, the strike is already determined: K = alpha * S.
    """
    if start_time <= 0:
        # At or past the strike-setting date: vanilla with strike = alpha * S
        K = alpha * S
        return bsm_european(S, K, r, q, sigma, maturity, cp)

    tenor = maturity - start_time
    # Scale-invariance: V = e^{-q*t_s} * S * bsm_european(1, alpha, r, q, sigma, tenor, cp)
    unit_val = bsm_european(1.0, alpha, r, q, sigma, tenor, cp)
    return np.exp(-q * start_time) * S * unit_val


# ---------------------------------------------------------------------------
# Barrier BSM — Reiner-Rubinstein (1991)
# ---------------------------------------------------------------------------


def bsm_barrier(
    S: float,
    K: float,
    H: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    barrier_type: str,
    rebate: float = 0.0,
    cp: int = 1,
) -> float:
    """Single-barrier option price via Reiner-Rubinstein (1991).

    Parameters
    ----------
    S            : spot price
    K            : strike
    H            : barrier level
    r            : risk-free rate
    q            : dividend yield
    sigma        : volatility
    T            : time to maturity
    barrier_type : one of "down_out", "down_in", "up_out", "up_in"
    rebate       : cash paid at expiry if knock-out never fires (default 0).
                   For knock-in options the rebate parameter is ignored
                   (use parity: knock_in = vanilla - knock_out).
    cp           : +1 call, -1 put

    Notes
    -----
    For knock-in options, uses in-out parity:
        knock_in = vanilla - knock_out   (with rebate=0 for the KO)
    The rebate for knock-out is assumed paid at expiry if barrier is never hit.
    """
    if barrier_type not in {"down_out", "down_in", "up_out", "up_in"}:
        raise ValueError(
            f"bsm_barrier: barrier_type must be one of "
            "'down_out', 'down_in', 'up_out', 'up_in'; "
            f"got {barrier_type!r}"
        )

    # Handle already-breached cases
    if barrier_type == "down_out" and S <= H:
        return rebate * np.exp(-r * T)
    if barrier_type == "up_out" and S >= H:
        return rebate * np.exp(-r * T)
    if barrier_type == "down_in" and S <= H:
        return bsm_european(S, K, r, q, sigma, T, cp)
    if barrier_type == "up_in" and S >= H:
        return bsm_european(S, K, r, q, sigma, T, cp)

    # Knock-in via in-out parity
    if "in" in barrier_type:
        out_type = barrier_type.replace("_in", "_out")
        vanilla = bsm_european(S, K, r, q, sigma, T, cp)
        ko = bsm_barrier(S, K, H, r, q, sigma, T, out_type, rebate=0.0, cp=cp)
        return vanilla - ko

    # ---------------------------------------------------------------------------
    # Knock-out pricing via Reiner-Rubinstein
    # ---------------------------------------------------------------------------
    sqT = np.sqrt(T)
    mu = (r - q - 0.5 * sigma**2) / sigma**2
    lam = np.sqrt(mu**2 + 2.0 * r / sigma**2)

    x1 = np.log(S / K) / (sigma * sqT) + (1.0 + mu) * sigma * sqT
    x2 = np.log(S / H) / (sigma * sqT) + (1.0 + mu) * sigma * sqT
    y1 = np.log(H**2 / (S * K)) / (sigma * sqT) + (1.0 + mu) * sigma * sqT
    y2 = np.log(H / S) / (sigma * sqT) + (1.0 + mu) * sigma * sqT
    z = np.log(H / S) / (sigma * sqT) + lam * sigma * sqT

    phi = float(cp)

    def A(p):
        return p * S * np.exp(-q * T) * norm.cdf(p * x1) - p * K * np.exp(-r * T) * norm.cdf(
            p * (x1 - sigma * sqT)
        )

    def B(p):
        return p * S * np.exp(-q * T) * norm.cdf(p * x2) - p * K * np.exp(-r * T) * norm.cdf(
            p * (x2 - sigma * sqT)
        )

    def C(p, eta):
        return p * S * np.exp(-q * T) * (H / S) ** (2.0 * (mu + 1.0)) * norm.cdf(
            eta * y1
        ) - p * K * np.exp(-r * T) * (H / S) ** (2.0 * mu) * norm.cdf(eta * (y1 - sigma * sqT))

    def D(p, eta):
        return p * S * np.exp(-q * T) * (H / S) ** (2.0 * (mu + 1.0)) * norm.cdf(
            eta * y2
        ) - p * K * np.exp(-r * T) * (H / S) ** (2.0 * mu) * norm.cdf(eta * (y2 - sigma * sqT))

    def E(eta):
        return (
            rebate
            * np.exp(-r * T)
            * (
                norm.cdf(eta * (x2 - sigma * sqT))
                - (H / S) ** (2.0 * mu) * norm.cdf(eta * (y2 - sigma * sqT))
            )
        )

    def F_reb(eta):
        return rebate * (
            (H / S) ** (mu + lam) * norm.cdf(eta * z)
            + (H / S) ** (mu - lam) * norm.cdf(eta * (z - 2.0 * lam * sigma * sqT))
        )

    if barrier_type == "down_out":
        eta = 1.0
        if cp == 1:  # call
            if H <= K:
                return A(phi) - C(phi, eta) + F_reb(eta)
            else:
                return B(phi) - D(phi, eta) + E(eta) + F_reb(eta)
        else:  # put
            if H >= K:
                return A(phi) - B(phi) + D(phi, eta) - C(phi, eta) + F_reb(eta)
            else:
                return F_reb(eta)

    if barrier_type == "up_out":
        eta = -1.0
        if cp == 1:  # call
            if H >= K:
                return A(phi) - B(phi) + D(phi, eta) - C(phi, eta) + F_reb(eta)
            else:
                return F_reb(eta)
        else:  # put
            if H <= K:
                return A(phi) - C(phi, eta) + F_reb(eta)
            else:
                return B(phi) - D(phi, eta) + E(eta) + F_reb(eta)

    raise RuntimeError(f"unreachable: barrier_type={barrier_type!r}")


# ---------------------------------------------------------------------------
# Lookback BSM — Goldman-Sosin-Gatto (1979)
# ---------------------------------------------------------------------------


def _lookback_floating_call_formula(
    S: float, m: float, r: float, q: float, sigma: float, T: float
) -> float:
    """Floating-strike lookback call (Conze-Viswanathan 1991 / Haug 2007).

    Payoff S_T - min_{0..T} S_u. ``m`` is the running minimum, ``b = r - q``
    is the cost of carry (assumed non-zero here; the b->0 limit is handled by
    the caller).
    """
    sqT = np.sqrt(T)
    b = r - q
    growth = np.exp(-q * T)
    disc = np.exp(-r * T)
    xi = 2.0 * b / sigma**2
    a1 = (np.log(S / m) + (b + 0.5 * sigma**2) * T) / (sigma * sqT)
    a2 = a1 - sigma * sqT
    a1m = a1 - xi * sigma * sqT  # = a1 - (2b/sigma^2) * sigma*sqrt(T)
    return (
        S * growth * norm.cdf(a1)
        - m * disc * norm.cdf(a2)
        + S
        * disc
        * (sigma**2 / (2.0 * b))
        * ((S / m) ** (-xi) * norm.cdf(-a1m) - np.exp(b * T) * norm.cdf(-a1))
    )


def _lookback_floating_put_formula(
    S: float, M: float, r: float, q: float, sigma: float, T: float
) -> float:
    """Floating-strike lookback put (Conze-Viswanathan 1991 / Haug 2007).

    Payoff max_{0..T} S_u - S_T. ``M`` is the running maximum, ``b = r - q``
    is the cost of carry (assumed non-zero here; the b->0 limit is handled by
    the caller).
    """
    sqT = np.sqrt(T)
    b = r - q
    growth = np.exp(-q * T)
    disc = np.exp(-r * T)
    xi = 2.0 * b / sigma**2
    d1 = (np.log(S / M) + (b + 0.5 * sigma**2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    d1m = d1 - xi * sigma * sqT
    return (
        M * disc * norm.cdf(-d2)
        - S * growth * norm.cdf(-d1)
        + S
        * disc
        * (sigma**2 / (2.0 * b))
        * (-((S / M) ** (-xi)) * norm.cdf(d1m) + np.exp(b * T) * norm.cdf(d1))
    )


def bsm_lookback_floating(
    S: float,
    S_min_or_max: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    cp: int,
) -> float:
    """Floating-strike lookback option price (Goldman-Sosin-Gatto 1979).

    Payoff:
        call (cp=+1) : S_T - min_{0..T} S_u    (always non-negative)
        put  (cp=-1) : max_{0..T} S_u - S_T    (always non-negative)

    Parameters
    ----------
    S             : current spot price
    S_min_or_max  : running minimum (for call) or maximum (for put).
                    At inception equals S.
    r, q, sigma, T: standard BSM inputs
    cp            : +1 call, -1 put
    """
    sqT = np.sqrt(T)
    disc = np.exp(-r * T)

    if cp == 1:
        m = S_min_or_max  # running minimum

        if sigma * sqT < 1e-14:
            S_T_det = S * np.exp((r - q) * T)
            return disc * max(S_T_det - m, 0.0)

        rq = r - q
        if abs(rq) < 1e-10:
            # Use limiting formula at r=q (L'Hopital):
            # shift slightly to avoid division by zero, interpolate
            eps = 1e-6
            return 0.5 * (
                _lookback_floating_call_formula(S, m, r + eps, q, sigma, T)
                + _lookback_floating_call_formula(S, m, r - eps, q, sigma, T)
            )
        return _lookback_floating_call_formula(S, m, r, q, sigma, T)

    else:
        M = S_min_or_max  # running maximum

        if sigma * sqT < 1e-14:
            S_T_det = S * np.exp((r - q) * T)
            return disc * max(M - S_T_det, 0.0)

        rq = r - q
        if abs(rq) < 1e-10:
            eps = 1e-6
            return 0.5 * (
                _lookback_floating_put_formula(S, M, r + eps, q, sigma, T)
                + _lookback_floating_put_formula(S, M, r - eps, q, sigma, T)
            )
        return _lookback_floating_put_formula(S, M, r, q, sigma, T)


def bsm_lookback_fixed(
    S: float,
    K: float,
    S_min_or_max: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    cp: int,
) -> float:
    """Fixed-strike lookback option price (Goldman-Sosin-Gatto 1979).

    Payoff:
        call (cp=+1) : max(max_{0..T} S_u - K, 0)
        put  (cp=-1) : max(K - min_{0..T} S_u, 0)

    Parameters
    ----------
    S             : current spot price
    K             : fixed strike
    S_min_or_max  : running maximum (for call) or minimum (for put).
                    At inception equals S.
    r, q, sigma, T: standard BSM inputs
    cp            : +1 call, -1 put
    """
    sqT = np.sqrt(T)
    disc = np.exp(-r * T)
    growth = np.exp(-q * T)
    rq = r - q

    if abs(rq) < 1e-10:
        rq_safe = 1e-8  # avoid division by zero in formula; vanishing correction
    else:
        rq_safe = rq

    if cp == 1:
        M = S_min_or_max  # running maximum (at inception = S)

        if sigma * sqT < 1e-14:
            S_T_det = S * np.exp((r - q) * T)
            max_val = max(M, S_T_det)
            return disc * max(max_val - K, 0.0)

        # Reference level for the formula: eff_K = max(K, M) handles
        # the case where running max already exceeds the strike.
        # When M > K: payoff = max_T - K = (max_T - M) + (M - K) where max_T >= M.
        # The first term is a "fresh" lookback starting at M, the second is certain.
        eff_K = max(K, M)
        xi = 2.0 * rq_safe / sigma**2
        a1 = (np.log(S / eff_K) + (rq_safe + 0.5 * sigma**2) * T) / (sigma * sqT)
        a2 = a1 - sigma * sqT
        a1m = a1 - xi * sigma * sqT

        price = (
            S * growth * norm.cdf(a1)
            - eff_K * disc * norm.cdf(a2)
            + S
            * disc
            * (sigma**2 / (2.0 * rq_safe))
            * (-((S / eff_K) ** (-xi)) * norm.cdf(a1m) + np.exp(rq_safe * T) * norm.cdf(a1))
        )
        # Add certain component if M > K
        if M > K:
            price += (M - K) * disc
        return price

    else:
        m = S_min_or_max  # running minimum (at inception = S)

        if sigma * sqT < 1e-14:
            S_T_det = S * np.exp((r - q) * T)
            min_val = min(m, S_T_det)
            return disc * max(K - min_val, 0.0)

        # Reference level: eff_K = min(K, m)
        eff_K = min(K, m)
        xi = 2.0 * rq_safe / sigma**2
        a1 = (np.log(S / eff_K) + (rq_safe + 0.5 * sigma**2) * T) / (sigma * sqT)
        a2 = a1 - sigma * sqT
        a1m = a1 - xi * sigma * sqT

        price = (
            eff_K * disc * norm.cdf(-a2)
            - S * growth * norm.cdf(-a1)
            + S
            * disc
            * (sigma**2 / (2.0 * rq_safe))
            * ((S / eff_K) ** (-xi) * norm.cdf(-a1m) - np.exp(rq_safe * T) * norm.cdf(-a1))
        )
        # Add certain component if m < K
        if m < K:
            price += (K - m) * disc
        return price
