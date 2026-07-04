"""Exact forward-start option pricing for Levy models via the CF.

For a Levy log-return process (stationary independent increments), the
forward-start payoff with strike set at ``t1`` as ``K = alpha * S_{t1}``
factorizes exactly (Rubinstein 1990's homogeneity argument, model-free
within the Levy class):

    (S_T - alpha * S_{t1})^+  =  S_{t1} * (S_T / S_{t1} - alpha)^+,

and ``S_T / S_{t1}`` is independent of ``S_{t1}`` with the law of
``exp((r - q) tau + X_tau)``, ``tau = T - t1``. Hence

    V = S0 * e^{-q t1} * EuropeanPrice(S0=1, K=alpha, r, q, T=tau),

where the European leg is priced here with the COS engine on the model's
own CF -- exact for every Levy model in the registry, no vol-freezing or
moment matching. Under BSM this reproduces
:func:`~foureng.analytics.bsm_exotics.bsm_forward_start` to machine
precision; under jump models it prices the forward smile that the BSM
formula cannot see.

References
----------
Rubinstein, M. (1990). Pay now, choose later. *Risk*, 3(2), 13.

Musiela, M. & Rutkowski, M. (2005). *Martingale Methods in Financial
Modelling*, 2nd ed., Springer. (Forward-start factorization for exponential
Levy models.)
"""

from __future__ import annotations

import numpy as np

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY
from .cos import cos_auto_grid, cos_prices

#: Models with stationary independent increments, for which the
#: forward-start factorization is exact.
LEVY_FORWARD_START_MODELS = frozenset(
    {"bsm", "kou", "merton_jd", "vg", "nig", "cgmy", "meixner", "bilateral_gamma"}
)


def levy_forward_start_price(
    model: str,
    fwd: ForwardSpec,
    params,
    *,
    alpha: float,
    start_time: float,
    maturity: float,
    cp: int = 1,
    N: int = 1 << 11,
    L: float = 12.0,
) -> float:
    """Exact forward-start price under a Levy model.

    Parameters
    ----------
    model :
        Registry key; must be in :data:`LEVY_FORWARD_START_MODELS`.
    fwd :
        Market inputs; ``fwd.T`` is ignored in favor of ``maturity``.
    params :
        Model parameter dataclass.
    alpha :
        Strike ratio: the strike fixes at ``alpha * S_{t1}`` (> 0).
    start_time :
        Strike-setting date ``t1`` in years, ``0 <= t1 < maturity``.
        ``t1 = 0`` reduces to a vanilla with strike ``alpha * S0``.
    maturity :
        Final expiry ``T`` (> start_time).
    cp :
        ``+1`` call, ``-1`` put.
    N, L :
        COS grid size and truncation multiplier for the European leg.
    """
    if model not in LEVY_FORWARD_START_MODELS:
        raise ValueError(
            f"levy_forward_start_price: model {model!r} is not a supported Levy "
            f"model; choose from {sorted(LEVY_FORWARD_START_MODELS)}"
        )
    if cp not in (1, -1):
        raise ValueError(f"levy_forward_start_price: cp must be +1 or -1, got {cp}")
    if alpha <= 0.0:
        raise ValueError(f"levy_forward_start_price: alpha must be > 0, got {alpha}")
    if not (0.0 <= start_time < maturity):
        raise ValueError(
            f"levy_forward_start_price: need 0 <= start_time < maturity, got "
            f"start_time={start_time}, maturity={maturity}"
        )

    entry = MODEL_REGISTRY[model]
    tau = maturity - start_time

    # European leg on a unit-spot asset over the remaining tenor.
    fwd_tau = ForwardSpec(S0=1.0, r=fwd.r, q=fwd.q, T=tau)
    phi = lambda u: entry.cf(u, fwd_tau, params)  # noqa: E731
    grid = cos_auto_grid(entry.cumulants(fwd_tau, params), N=N, L=L)
    strikes = np.array([float(alpha)])
    call = float(cos_prices(phi, fwd_tau, strikes, grid).call_prices[0])
    if cp == -1:
        call = call - fwd_tau.disc * (fwd_tau.F0 - float(alpha))

    return float(fwd.S0 * np.exp(-fwd.q * start_time) * call)


__all__ = ["LEVY_FORWARD_START_MODELS", "levy_forward_start_price"]
