"""Exact cliquet pricing for Levy models via per-period CF collars.

For a Levy log-return process the period returns
``R_k = S_{t_k} / S_{t_{k-1}} - 1`` are independent, and the locally
collared return decomposes into a static call spread:

    clip(R, lf, lc) = lf + (R - lf)^+ - (R - lc)^+,

so each period expectation is two undiscounted European calls on a
unit-spot asset over that period. Without a *global* collar the payoff is
separable:

    additive:        V = D(T) * cp * sum_k  E[clip(R_k, lf, lc)]
    multiplicative:  V = D(T) * ( prod_k (1 + E[clip(R_k, lf, lc)]) - 1 )

(the product form uses independence of the periods). Both are exact for
every Levy model in the registry -- the same decomposition used by the
cliquet/EIA pricers in the PROJ literature (Kirkby & Deng 2016). A global
floor or cap couples the periods and needs a genuinely path-dependent
method; those cases are rejected here and belong to `cliquet_mc`.

Strikes ``1 + lf <= 0`` (e.g. no floor) are handled analytically:
``E[(R - a)^+] = E[R] - a`` because ``R > -1`` almost surely.

References
----------
Kirkby, J.L. & Deng, S. (2016). Static hedging and pricing of exotic
options with discrete averaging (cliquet/EIA decompositions in the PROJ
framework). *SSRN*.

Wilmott, P. (2002). Cliquet options and volatility models. *Wilmott
Magazine*, 6, 78-83.
"""

from __future__ import annotations

import numpy as np

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY
from ..products.cliquet import CliquetOption
from .cos import cos_auto_grid, cos_prices

#: Models with independent stationary increments, for which the per-period
#: collar decomposition is exact.
LEVY_CLIQUET_MODELS = frozenset(
    {"bsm", "kou", "merton_jd", "vg", "nig", "cgmy", "meixner", "bilateral_gamma"}
)


def _undiscounted_call(
    model: str, fwd: ForwardSpec, params, dt: float, K: float, N: int, L: float
) -> float:
    """E[(S_dt - K)^+] for a unit-spot asset over one period."""
    entry = MODEL_REGISTRY[model]
    fwd_dt = ForwardSpec(S0=1.0, r=fwd.r, q=fwd.q, T=dt)
    if K <= 0.0:
        return float(fwd_dt.F0 - K)
    phi = lambda u: entry.cf(u, fwd_dt, params)  # noqa: E731
    grid = cos_auto_grid(entry.cumulants(fwd_dt, params), N=N, L=L)
    call = float(cos_prices(phi, fwd_dt, np.array([K]), grid).call_prices[0])
    return call / fwd_dt.disc


def levy_cliquet_price(
    model: str,
    fwd: ForwardSpec,
    params,
    product: CliquetOption,
    *,
    N: int = 1 << 11,
    L: float = 12.0,
) -> float:
    """Exact locally collared cliquet price under a Levy model.

    Supports additive and multiplicative payoffs with per-period floor/cap.
    Global floors/caps couple the periods and are rejected -- use
    ``method='cliquet_mc'`` for those contracts.
    """
    if model not in LEVY_CLIQUET_MODELS:
        raise ValueError(
            f"levy_cliquet_price: model {model!r} is not a supported Levy model; "
            f"choose from {sorted(LEVY_CLIQUET_MODELS)}"
        )
    if np.isfinite(product.global_floor) or np.isfinite(product.global_cap):
        raise NotImplementedError(
            "levy_cliquet_price: global floors/caps couple the periods and are "
            "not separable; use method='cliquet_mc' for globally collared cliquets."
        )

    lf, lc = float(product.local_floor), float(product.local_cap)
    t = np.asarray(product.reset_times, dtype=np.float64)
    dts = np.diff(np.concatenate(([0.0], t)))

    # E[clip(R, lf, lc)] per period, caching by period length (reset grids
    # are usually equally spaced, so this prices one COS call spread total).
    cache: dict[float, float] = {}
    expectations = []
    for dt in dts:
        key = round(float(dt), 14)
        if key not in cache:
            fwd_dt = ForwardSpec(S0=1.0, r=fwd.r, q=fwd.q, T=float(dt))
            e_r = float(fwd_dt.F0 - 1.0)  # E[R] = e^{(r-q)dt} - 1
            if not np.isfinite(lf):
                # E[min(R, lc)] = E[R] - E[(R - lc)^+]
                val = e_r
                if np.isfinite(lc):
                    val -= _undiscounted_call(model, fwd, params, float(dt), 1.0 + lc, N, L)
            else:
                val = lf + _undiscounted_call(model, fwd, params, float(dt), 1.0 + lf, N, L)
                if np.isfinite(lc):
                    val -= _undiscounted_call(model, fwd, params, float(dt), 1.0 + lc, N, L)
            cache[key] = val
        expectations.append(cache[key])

    disc = float(np.exp(-fwd.r * product.maturity))
    if product.payoff_type == "additive":
        return disc * product.cp * float(np.sum(expectations))
    # multiplicative: independence factorizes the product of gross returns
    return disc * float(np.prod([1.0 + e for e in expectations]) - 1.0)


__all__ = ["LEVY_CLIQUET_MODELS", "levy_cliquet_price"]
