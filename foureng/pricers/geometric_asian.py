"""Exact discrete geometric-Asian pricing for Levy models via the CF product.

For a Levy log-return process ``X_t`` with independent increments, the
discretely monitored geometric average over dates ``t_1 < ... < t_M``

    G = ( prod_k S_{t_k} )^{1/M},
    log G = log S0 + (r - q) * mean(t_k) + A,
    A = (1/M) * sum_k X_{t_k} = sum_j w_j * DX_j,   w_j = (M - j + 1) / M,

is an explicit weighted sum of the independent increments
``DX_j = X_{t_j} - X_{t_{j-1}}``, so its characteristic function is the
finite product

    phi_A(u) = prod_j phi_{dt_j}(w_j * u)

of per-increment CFs -- *exactly*, with no lognormal approximation. Pricing
the geometric Asian therefore reduces to a European option on the synthetic
asset ``G`` with forward ``F_G = E[G]`` and normalized CF
``phi_Z(u) = phi_A(u) * exp(-i u log E[e^A])``, evaluated here with the COS
engine on a cumulant-based truncation interval. Under BSM this reproduces
the closed form in :func:`~foureng.analytics.bsm_asian.bsm_discrete_geometric_asian`
to near machine precision; under Kou/VG/NIG/CGMY it provides the exact
geometric price that the arithmetic-Asian control variate needs.

References
----------
Fusai, G. & Meucci, A. (2008). Pricing discretely monitored Asian options
under Levy processes. *Journal of Banking and Finance*, 32(10), 2076-2088.

Kemna, A.G.Z. & Vorst, A.C.F. (1990). A pricing method for options based on
average asset values. *Journal of Banking and Finance*, 14(1), 113-129.
"""

from __future__ import annotations

import numpy as np

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY
from .cos import cos_auto_grid, cos_prices

#: Models whose CF exponent is linear in maturity (true Levy processes), so
#: per-increment CFs are obtained by evaluating the registry CF at T = dt.
LEVY_GEOMETRIC_ASIAN_MODELS = frozenset(
    {"bsm", "kou", "merton_jd", "vg", "nig", "cgmy", "meixner", "bilateral_gamma"}
)


def levy_geometric_asian_price(
    model: str,
    fwd: ForwardSpec,
    params,
    *,
    strikes,
    monitoring_times,
    cp: int = 1,
    N: int = 1 << 11,
    L: float = 12.0,
) -> np.ndarray:
    """Price fixed-strike discrete geometric-Asian options exactly by CF.

    Parameters
    ----------
    model :
        Registry key; must be in :data:`LEVY_GEOMETRIC_ASIAN_MODELS`.
    fwd :
        Market inputs. ``fwd.T`` is ignored; discounting uses the final
        monitoring date (payment at the last fixing, matching
        ``bsm_discrete_geometric_asian``).
    params :
        Model parameter dataclass.
    strikes :
        Scalar or 1-D array of fixed strikes (> 0).
    monitoring_times :
        Strictly increasing positive fixing dates ``t_1 < ... < t_M``.
    cp :
        ``+1`` call, ``-1`` put.
    N, L :
        COS grid size and truncation multiplier for the final inversion.
    """
    if model not in LEVY_GEOMETRIC_ASIAN_MODELS:
        raise ValueError(
            f"levy_geometric_asian_price: model {model!r} is not a supported Levy "
            f"model; choose from {sorted(LEVY_GEOMETRIC_ASIAN_MODELS)}"
        )
    if cp not in (1, -1):
        raise ValueError(f"levy_geometric_asian_price: cp must be +1 or -1, got {cp}")

    t = np.asarray(monitoring_times, dtype=np.float64)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("monitoring_times must be a non-empty 1-D array")
    if np.any(t <= 0.0) or not np.all(np.diff(t) > 0.0):
        raise ValueError("monitoring_times must be strictly increasing and positive")

    K = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    if np.any(K <= 0.0):
        raise ValueError("all strikes must be > 0")

    entry = MODEL_REGISTRY[model]
    M = t.size
    dt = np.diff(np.concatenate(([0.0], t)))
    weights = (M - np.arange(M)) / M  # w_j = (M - j + 1)/M for j = 1..M
    t_bar = float(np.mean(t))
    maturity = float(t[-1])

    fwd_steps = [ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=float(d)) for d in dt]

    # log E[e^A] from the per-increment CFs at u = -i w_j (real by construction)
    log_mgf = 0.0
    for fw, w in zip(fwd_steps, weights):
        val = np.asarray(entry.cf(np.array([-1j * w]), fw, params)).ravel()[0]
        log_mgf += float(np.log(val).real)

    def phi_z(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        out = np.ones_like(u)
        for fw, w in zip(fwd_steps, weights):
            out = out * np.asarray(entry.cf(w * u, fw, params))
        return out * np.exp(-1j * u * log_mgf)

    # Cumulants of Z = A - log E[e^A]: weighted per-increment cumulants.
    c1 = c2 = c4 = 0.0
    for fw, w in zip(fwd_steps, weights):
        s1, s2, s4 = entry.cumulants(fw, params)
        c1 += w * s1
        c2 += w * w * s2
        c4 += w**4 * s4
    c1 -= log_mgf

    # Synthetic forward spec whose F0 equals E[G] and whose discount factor
    # matches payment at the final fixing date.
    forward_geo = fwd.S0 * float(np.exp((fwd.r - fwd.q) * t_bar + log_mgf))
    q_eff = fwd.r - float(np.log(forward_geo / fwd.S0)) / maturity
    fwd_geo = ForwardSpec(S0=fwd.S0, r=fwd.r, q=q_eff, T=maturity)

    grid = cos_auto_grid((c1, c2, c4), N=N, L=L)
    calls = np.asarray(cos_prices(phi_z, fwd_geo, K, grid).call_prices, dtype=np.float64)
    if cp == 1:
        return calls
    return calls - fwd_geo.disc * (forward_geo - K)


__all__ = ["LEVY_GEOMETRIC_ASIAN_MODELS", "levy_geometric_asian_price"]
