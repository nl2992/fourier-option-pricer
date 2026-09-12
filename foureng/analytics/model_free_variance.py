"""Model-free variance from a strip of European options (log-contract replication).

For ``X = log(S_T / F)`` the payoff ``-X`` is statically replicated by
out-of-the-money options (Carr & Madan 1998; Demeterfi, Derman, Kamal & Zou
1999; Neuberger 1994):

    -log(S_T / F) = -(S_T - F)/F + int_0^F (K - S_T)^+ / K^2 dK
                                 + int_F^inf (S_T - K)^+ / K^2 dK,

so with ``Q(K)`` the out-of-the-money price (put below ``F``, call above),

    sigma^2_LC = (2 / T) E[-X] = 2 / (D T) * int_0^inf Q(K) / K^2 dK.

For continuous price paths this is the fair strike of a continuously
monitored variance swap (``E[int sigma_t^2 dt] / T``). With jumps it is not:
it is the "VIX-squared" quantity, and the variance swap strike exceeds it by
the jump correction (Carr & Wu 2009); compare
:func:`~foureng.analytics.levy_variance.levy_variance_fair_strike`.

Functions
---------
``log_contract_variance_from_strip``
    Replication integral from quotes, by Simpson's rule on each side of the
    forward, with the forward inserted as a node so the kink of ``Q`` at
    ``F`` never falls inside an interval.
``vix_style_index``
    The CBOE VIX discretisation (white paper, 2019): ``Delta K / K^2``
    weights, ``K_0`` the first listed strike at or below the forward, and the
    ``(F/K_0 - 1)^2`` correction; returned as ``100 * sqrt(variance)``.
``log_contract_variance``
    The exact model counterpart ``-2 c_1 / T`` from the CF's first cumulant,
    for comparing replicated and model values (e.g. VIX calibration).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY

__all__ = ["log_contract_variance", "log_contract_variance_from_strip", "vix_style_index"]


def _otm_prices(
    strikes, fwd: ForwardSpec, calls, puts
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.asarray(strikes, dtype=float)
    if K.ndim != 1 or K.size < 3:
        raise ValueError("strikes must be a 1-D array with at least 3 entries")
    if np.any(K <= 0.0) or np.any(np.diff(K) <= 0.0):
        raise ValueError("strikes must be positive and strictly increasing")
    if calls is None and puts is None:
        raise ValueError("pass calls, puts, or both")
    given = [np.asarray(x, dtype=float) for x in (calls, puts) if x is not None]
    if any(x.shape != K.shape for x in given):
        raise ValueError("calls/puts must have the same shape as strikes")
    parity = fwd.disc * (fwd.F0 - K)  # C - P
    C = np.asarray(calls, dtype=float) if calls is not None else np.asarray(puts) + parity
    P = np.asarray(puts, dtype=float) if puts is not None else np.asarray(calls) - parity
    if not (K[0] < fwd.F0 < K[-1]):
        raise ValueError("the strike strip must bracket the forward")
    return K, C, P


def _simpson(y: np.ndarray, x: np.ndarray) -> float:
    """Composite Simpson rule on a non-uniform grid, independent of the SciPy version.

    Pairs of intervals use the three-point non-uniform Simpson weights; an odd
    number of intervals closes with Cartwright's (2017) last-interval
    correction (the rule SciPy >= 1.11 uses; SciPy 1.10 averaged two rules).
    """
    n = len(x) - 1
    if n < 2:  # a single interval: trapezoid
        return float(0.5 * (y[0] + y[-1]) * (x[-1] - x[0])) if n == 1 else 0.0
    m = n - (n % 2)  # intervals covered by whole pairs
    h = np.diff(x)
    h0, h1 = h[0:m:2], h[1:m:2]
    y0, y1, y2 = y[0:m:2], y[1 : m + 1 : 2], y[2 : m + 1 : 2]
    hs = h0 + h1
    total = np.sum(
        hs / 6.0 * ((2.0 - h1 / h0) * y0 + hs * hs / (h0 * h1) * y1 + (2.0 - h0 / h1) * y2)
    )
    if n % 2:
        a, b = h[-2], h[-1]
        total += (
            (2.0 * b * b + 3.0 * a * b) / (6.0 * (a + b)) * y[-1]
            + (b * b + 3.0 * a * b) / (6.0 * a) * y[-2]
            - b**3 / (6.0 * a * (a + b)) * y[-3]
        )
    return float(total)


def log_contract_variance_from_strip(
    strikes,
    fwd: ForwardSpec,
    *,
    calls=None,
    puts=None,
) -> float:
    """Annualised model-free variance ``2 E[-log(S_T/F)] / T`` from option quotes.

    Parameters
    ----------
    strikes :
        Strictly increasing strikes bracketing the forward ``fwd.F0``.
    fwd :
        Forward spec: ``F0``, ``disc`` and ``T`` of the quotes.
    calls, puts :
        Discounted call and/or put prices at ``strikes``. A missing side is
        filled in by put-call parity.

    Returns
    -------
    float
        Replicated variance. The strip is not extrapolated: mass beyond
        ``[strikes[0], strikes[-1]]`` is omitted, which biases the result
        down, so quote coverage should reach well into both wings.
    """
    K, C, P = _otm_prices(strikes, fwd, calls, puts)
    F = fwd.F0
    q_at_f = float(CubicSpline(K, C)(F))  # call = put at the forward
    lo, hi = K < F, K > F
    k_lo = np.append(K[lo], F)
    k_hi = np.insert(K[hi], 0, F)
    y_lo = np.append(P[lo], q_at_f) / k_lo**2
    y_hi = np.insert(C[hi], 0, q_at_f) / k_hi**2
    integral = _simpson(y_lo, k_lo) + _simpson(y_hi, k_hi)
    return float(2.0 * integral / (fwd.disc * fwd.T))


def vix_style_index(strikes, fwd: ForwardSpec, *, calls=None, puts=None) -> float:
    """CBOE VIX-methodology volatility index, ``100 * sqrt(sigma^2)``.

    ``sigma^2 = (2/T) sum_i Delta K_i / K_i^2 e^{rT} Q(K_i) - (1/T) (F/K_0 - 1)^2``
    with ``K_0`` the first strike at or below ``F``, ``Q(K_0)`` the average of
    the call and put there, and ``Delta K_i`` half the distance between the
    neighbouring strikes (one-sided at the ends). The CBOE's zero-bid
    truncation rule is a market-data filter and is left to the caller.
    """
    K, C, P = _otm_prices(strikes, fwd, calls, puts)
    F, T = fwd.F0, fwd.T
    i0 = int(np.searchsorted(K, F, side="right")) - 1
    k0 = K[i0]
    Q = np.where(K < k0, P, C)
    Q[i0] = 0.5 * (C[i0] + P[i0])
    dK = np.empty_like(K)
    dK[1:-1] = 0.5 * (K[2:] - K[:-2])
    dK[0], dK[-1] = K[1] - K[0], K[-1] - K[-2]
    var = (2.0 / T) * np.sum(dK / K**2 * Q) / fwd.disc - (F / k0 - 1.0) ** 2 / T
    return float(100.0 * np.sqrt(max(var, 0.0)))


def log_contract_variance(model: str, fwd: ForwardSpec, params) -> float:
    """Exact model value of the replicated quantity, ``-2 c_1 / T``.

    ``c_1 = E[log(S_T/F)]`` is the first cumulant of the registry CF. Equals the
    expected average instantaneous variance for continuous models (e.g. BSM
    ``sigma^2``; Heston ``theta + (v0 - theta)(1 - e^{-kappa T})/(kappa T)``).
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {model!r}")
    c1 = MODEL_REGISTRY[model].cumulants(fwd, params)[0]
    return float(-2.0 * c1 / fwd.T)
