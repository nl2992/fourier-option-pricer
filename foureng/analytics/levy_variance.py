"""Discrete variance-swap valuation under Levy models via CF cumulants.

The repo's realized-variance convention (shared with the Monte Carlo engine
and :func:`~foureng.analytics.bsm_variance.bsm_variance_swap`) is

    RV = (1/T) * sum_i ( log(S_{t_i} / S_{t_{i-1}}) )^2.

For a Levy model the log-return over ``dt_i`` is ``(r - q) dt_i + X_{dt_i}``
with independent increments, so each squared-return expectation is exact in
terms of the increment's first two cumulants ``c1_i = E[X_{dt_i}]`` and
``c2_i = Var[X_{dt_i}]``:

    E[R_i^2] = ( (r - q) dt_i + c1_i )^2 + c2_i,
    E[RV]    = (1/T) * sum_i E[R_i^2].

Jumps enter through ``c2`` (and the martingale compensator through ``c1``),
so the fair strike correctly exceeds the diffusion-only value -- the
discrete-monitoring analogue of the jump correction in Carr-Wu (2009).
With ``lam = 0`` every jump model collapses to the BSM closed form.

References
----------
Carr, P. & Wu, L. (2009). Variance risk premiums. *Review of Financial
Studies*, 22(3), 1311-1341.

Neuberger, A. (1994). The log contract. *Journal of Portfolio Management*,
20(2), 74-80. (Continuous-monitoring limit of the discrete fair strike.)
"""

from __future__ import annotations

import numpy as np

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY
from ..products.variance import VarianceSwap

#: Models whose CF exponent is linear in maturity, so per-increment cumulants
#: are the registry cumulants evaluated at T = dt.
LEVY_VARIANCE_MODELS = frozenset(
    {"bsm", "kou", "merton_jd", "vg", "nig", "cgmy", "meixner", "bilateral_gamma"}
)


def levy_variance_fair_strike(
    model: str,
    fwd: ForwardSpec,
    params,
    sampling_times,
    *,
    maturity: float | None = None,
) -> float:
    """Exact E[RV] (annualized fair variance strike) under a Levy model.

    Parameters
    ----------
    model :
        Registry key; must be in :data:`LEVY_VARIANCE_MODELS`.
    fwd :
        Market inputs; only ``r`` and ``q`` (and ``S0`` formally) are used.
    params :
        Model parameter dataclass.
    sampling_times :
        Strictly increasing positive observation dates.
    maturity :
        Annualization horizon ``T``; defaults to the last sampling date.
    """
    if model not in LEVY_VARIANCE_MODELS:
        raise ValueError(
            f"levy_variance_fair_strike: model {model!r} is not a supported Levy "
            f"model; choose from {sorted(LEVY_VARIANCE_MODELS)}"
        )
    t = np.asarray(sampling_times, dtype=np.float64)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("sampling_times must be a non-empty 1-D array")
    if np.any(t <= 0.0) or not np.all(np.diff(t) > 0.0):
        raise ValueError("sampling_times must be strictly increasing and positive")
    T = float(t[-1]) if maturity is None else float(maturity)
    if T <= 0.0:
        raise ValueError(f"maturity must be > 0; got {T}")

    entry = MODEL_REGISTRY[model]
    carry = fwd.r - fwd.q
    dt = np.diff(np.concatenate(([0.0], t)))

    expected_rv = 0.0
    for d in dt:
        c1, c2, _ = entry.cumulants(ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=float(d)), params)
        expected_rv += (carry * float(d) + c1) ** 2 + c2
    return expected_rv / T


def levy_variance_swap(
    model: str,
    fwd: ForwardSpec,
    params,
    product: VarianceSwap,
) -> float:
    """Discounted expectation of the variance-swap payoff under a Levy model.

    Mirrors :func:`~foureng.analytics.bsm_variance.bsm_variance_swap`:
    returns ``disc(T) * notional * E[RV]`` with ``T = product.maturity``.
    """
    expected_rv = levy_variance_fair_strike(
        model, fwd, params, product.sampling_times, maturity=product.maturity
    )
    disc = float(np.exp(-fwd.r * product.maturity))
    return float(disc * product.notional * expected_rv)


__all__ = ["LEVY_VARIANCE_MODELS", "levy_variance_fair_strike", "levy_variance_swap"]
