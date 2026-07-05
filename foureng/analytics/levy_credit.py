"""Structural CDS pricing under Levy models via PROJ first-passage survival.

Default is modeled structurally (Black & Cox 1976): the firm defaults the
first time its asset value, monitored on a discrete grid, is at or below a
barrier ``B``. The discretely monitored survival curve

    Q(t_k) = P( min_{j <= k} S_{t_j} > B )

is computed with the PROJ down-and-out unit-payoff recursion
(:func:`~foureng.pricers.proj.proj_survival_probability`), one run per
premium date, and the standard running-spread legs give the par spread:

    protection = (1 - R) * sum_i D(t_i) (Q(t_{i-1}) - Q(t_i))
    annuity    = sum_i D(t_i) alpha_i [ Q(t_i) + (Q(t_{i-1}) - Q(t_i)) / 2 ]
    par spread = protection / annuity,

with the half-period accrual-on-default convention in the annuity. With a
deterministic hazard (Q(t) = e^{-lambda t}) the formula collapses to the
credit-triangle ``spread ~ (1 - R) lambda``, which anchors the leg assembly
independently of the survival engine.

References
----------
Black, F. & Cox, J.C. (1976). Valuing corporate securities: some effects of
bond indenture provisions. *Journal of Finance*, 31(2), 351-367.

O'Kane, D. (2008). *Modelling Single-name and Multi-name Credit
Derivatives*. Wiley. (CDS leg conventions and the credit triangle.)
"""

from __future__ import annotations

import numpy as np

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY
from ..pricers.proj import proj_survival_probability

#: Models with independent stationary increments, for which the per-step CF
#: drives the PROJ first-passage recursion.
LEVY_CREDIT_MODELS = frozenset(
    {"bsm", "kou", "merton_jd", "vg", "nig", "cgmy", "meixner", "bilateral_gamma"}
)


def levy_survival_curve(
    model: str,
    fwd: ForwardSpec,
    params,
    *,
    default_barrier: float,
    horizons,
    monitoring_dt: float = 1.0 / 12.0,
    N: int = 1 << 13,
    alph: float = 7.0,
) -> np.ndarray:
    """Discretely monitored first-passage survival probabilities Q(t) per horizon.

    Each horizon uses ``round(t / monitoring_dt)`` equally spaced monitoring
    dates (at least one). Horizons must be positive and increasing.
    """
    if model not in LEVY_CREDIT_MODELS:
        raise ValueError(
            f"levy_survival_curve: model {model!r} is not a supported Levy model; "
            f"choose from {sorted(LEVY_CREDIT_MODELS)}"
        )
    t = np.asarray(horizons, dtype=np.float64)
    if t.ndim != 1 or t.size == 0 or np.any(t <= 0.0) or not np.all(np.diff(t) > 0.0):
        raise ValueError("horizons must be a non-empty, positive, increasing 1-D array")
    if monitoring_dt <= 0.0:
        raise ValueError(f"monitoring_dt must be > 0; got {monitoring_dt}")

    entry = MODEL_REGISTRY[model]
    out = np.empty(t.size)
    for i, horizon in enumerate(t):
        m = max(1, int(round(horizon / monitoring_dt)))
        dt = horizon / m
        fwd_dt = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=dt)
        drift = (fwd.r - fwd.q) * dt

        def step_cf(u, _fwd_dt=fwd_dt, _drift=drift):
            return np.exp(1j * u * _drift) * np.asarray(
                entry.cf(u, _fwd_dt, params), dtype=np.complex128
            )

        out[i] = proj_survival_probability(
            step_cf, S0=fwd.S0, B=float(default_barrier), M=m, N=N, alph=alph
        )
    return out


def cds_par_spread_from_survival(
    survival: np.ndarray,
    payment_times: np.ndarray,
    r: float,
    recovery: float,
) -> float:
    """Par spread from a survival curve on the premium dates (O'Kane legs)."""
    q = np.asarray(survival, dtype=np.float64)
    t = np.asarray(payment_times, dtype=np.float64)
    if q.shape != t.shape:
        raise ValueError("survival and payment_times must have the same shape")
    if not (0.0 <= recovery < 1.0):
        raise ValueError(f"recovery must be in [0, 1); got {recovery}")
    q_prev = np.concatenate(([1.0], q[:-1]))
    alpha = np.diff(np.concatenate(([0.0], t)))
    disc = np.exp(-r * t)
    defaults = np.maximum(q_prev - q, 0.0)
    protection = (1.0 - recovery) * float(np.sum(disc * defaults))
    annuity = float(np.sum(disc * alpha * (q + 0.5 * defaults)))
    if annuity <= 0.0:
        raise ValueError("cds_par_spread_from_survival: non-positive risky annuity")
    return protection / annuity


def levy_cds_spread(
    model: str,
    fwd: ForwardSpec,
    params,
    *,
    default_barrier: float,
    recovery: float,
    maturity: float,
    payments_per_year: int = 4,
    monitoring_dt: float = 1.0 / 12.0,
    N: int = 1 << 13,
    alph: float = 7.0,
) -> float:
    """Structural CDS par spread under a Levy model.

    Combines the PROJ first-passage survival curve on the premium grid with
    the standard running-spread legs (half-period accrual on default).
    """
    if maturity <= 0.0:
        raise ValueError(f"maturity must be > 0; got {maturity}")
    n_pay = max(1, int(round(payments_per_year * maturity)))
    pay_times = np.linspace(maturity / n_pay, maturity, n_pay)
    q = levy_survival_curve(
        model,
        fwd,
        params,
        default_barrier=default_barrier,
        horizons=pay_times,
        monitoring_dt=monitoring_dt,
        N=N,
        alph=alph,
    )
    # Enforce monotonicity against residual grid noise before leg assembly.
    q = np.minimum.accumulate(np.clip(q, 0.0, 1.0))
    return cds_par_spread_from_survival(q, pay_times, fwd.r, recovery)


__all__ = [
    "LEVY_CREDIT_MODELS",
    "cds_par_spread_from_survival",
    "levy_cds_spread",
    "levy_survival_curve",
]
