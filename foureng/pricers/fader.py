"""Fader option pricing for Levy models via COS density x remaining-life value.

Linearity splits the fade-in payoff date by date:

    V_in = D(T) * (1/M) * sum_k  E[ 1_{L < S_{t_k} < U} * vanilla(S_T, K) ].

For Levy log-returns, conditioning on ``X_{t_k} = x`` factorizes each term
(independent increments):

    E[1_A(X_{t_k}) * (S_T - K)^+]  =  int_A f_{t_k}(x) * C_k(x) dx,

where ``f_{t_k}`` is the COS density of ``X_{t_k}`` and ``C_k(x)`` is the
undiscounted remaining-life European value. The spot-shift homogeneity
``C_k(x) = e^x * g_k(K e^{-x})`` turns the family of conditional values into
a *single* COS strike-strip per monitoring date (strikes ``K e^{-x_i}`` at
the Gauss-Legendre nodes ``x_i``), so the whole fader costs ``2M`` COS runs.
Fade-out follows from fade-in/fade-out parity (their sum is the vanilla),
which holds exactly by construction of the notional split.

References
----------
Hakala, J. & Wystup, U. (2002). *Foreign Exchange Risk*. Risk Books.
(Fader/corridor structures.)

Fang, F. & Oosterlee, C.W. (2008). A novel pricing method for European
options based on Fourier-cosine series expansions. *SIAM J. Sci. Comput.*,
31(2), 826-848. (COS density recovery used for the date-k marginal.)
"""

from __future__ import annotations

import numpy as np

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY
from ..products.fader import FaderOption
from .cos import cos_auto_grid, cos_prices

#: Models with independent stationary increments, for which the date-by-date
#: factorization is exact.
LEVY_FADER_MODELS = frozenset(
    {"bsm", "kou", "merton_jd", "vg", "nig", "cgmy", "meixner", "bilateral_gamma"}
)


def _cos_density(phi_vals_grid, x, a, b):
    """COS density series at points ``x`` from CF values on the COS grid."""
    n = phi_vals_grid.shape[0]
    u = np.arange(n) * np.pi / (b - a)
    coeffs = np.real(phi_vals_grid * np.exp(-1j * u * a))
    coeffs[0] *= 0.5
    return (2.0 / (b - a)) * (coeffs[None, :] * np.cos(np.outer(x - a, u))).sum(axis=1)


def levy_fader_price(
    model: str,
    fwd: ForwardSpec,
    params,
    product: FaderOption,
    *,
    n_quad: int = 256,
    N: int = 1 << 11,
    L: float = 12.0,
) -> float:
    """Fade-in / fade-out option price under a Levy model.

    Parameters
    ----------
    model :
        Registry key; must be in :data:`LEVY_FADER_MODELS`.
    fwd :
        Market inputs; ``fwd.T`` is ignored in favor of ``product.maturity``.
    params :
        Model parameter dataclass.
    product :
        :class:`~foureng.products.fader.FaderOption`.
    n_quad :
        Gauss-Legendre nodes per monitoring date.
    N, L :
        COS terms and truncation multiplier for both the density and the
        remaining-life European strips.
    """
    if model not in LEVY_FADER_MODELS:
        raise ValueError(
            f"levy_fader_price: model {model!r} is not a supported Levy model; "
            f"choose from {sorted(LEVY_FADER_MODELS)}"
        )
    entry = MODEL_REGISTRY[model]
    T = float(product.maturity)
    K = float(product.strike)
    cp = product.cp
    t = np.asarray(product.monitoring_times, dtype=np.float64)
    M = t.size
    carry = fwd.r - fwd.q
    disc = float(np.exp(-fwd.r * T))
    fwd_T_log = float(np.log(fwd.S0) + carry * T)  # log forward at maturity

    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))

    total = 0.0
    for t_k in t:
        # Indicator in terms of x = X_{t_k}: L < S0 e^{carry t_k + x} < U
        lo = float(np.log(product.lower / fwd.S0) - carry * t_k)
        hi = float(np.log(product.upper / fwd.S0) - carry * t_k)

        fwd_tk = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=float(t_k))
        grid_k = cos_auto_grid(entry.cumulants(fwd_tk, params), N=N, L=L)
        # Density support: clip the range to the COS window.
        a_q, b_q = max(lo, grid_k.a), min(hi, grid_k.b)
        if b_q <= a_q:
            continue  # range carries no probability mass at this date
        x = 0.5 * (b_q - a_q) * nodes + 0.5 * (b_q + a_q)
        w = 0.5 * (b_q - a_q) * weights

        u_grid = np.arange(N) * np.pi / (grid_k.b - grid_k.a)
        phi_vals = np.asarray(entry.cf(u_grid, fwd_tk, params), dtype=np.complex128)
        dens = np.maximum(_cos_density(phi_vals, x, grid_k.a, grid_k.b), 0.0)

        tau = T - float(t_k)
        if tau > 1e-12:
            # Remaining-life undiscounted value via one COS strike strip:
            # C_k(x) = e^x * g(K e^{-x}), g(K') = E[(F_T e^{Y'} - K')^+].
            fwd_tau = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=tau)
            phi_tau = lambda u: entry.cf(u, fwd_tau, params)  # noqa: B023, E731
            grid_tau = cos_auto_grid(entry.cumulants(fwd_tau, params), N=N, L=L)
            strikes = K * np.exp(-x)
            # Price with a unit-forward spec scaled to F_T: use S0' so that
            # F0' = exp(fwd_T_log - x)... simpler: forward F_x = S0 e^{carry T + x}
            # per node; calls on fixed strike K with forward F_x equal
            # e^{x} * calls(forward F_T, strike K e^{-x}).
            fwd_price_spec = ForwardSpec(
                S0=float(np.exp(fwd_T_log - fwd.r * tau + fwd.q * tau)),
                r=fwd.r,
                q=fwd.q,
                T=tau,
            )  # F0 = exp(fwd_T_log)
            calls = (
                np.asarray(
                    cos_prices(phi_tau, fwd_price_spec, strikes, grid_tau).call_prices,
                    dtype=np.float64,
                )
                / fwd_price_spec.disc
            )  # undiscounted
            if cp == -1:
                calls = calls - (np.exp(fwd_T_log) - strikes)  # parity, undiscounted
            cond_val = np.exp(x) * calls
        else:
            # Monitoring at maturity: intrinsic value on the terminal forward.
            s_T = np.exp(fwd_T_log + x)
            cond_val = np.maximum(cp * (s_T - K), 0.0)

        total += float(np.sum(w * dens * cond_val))

    fade_in = disc * total / M

    if product.fade_type == "in":
        return fade_in
    # fade-out = vanilla - fade-in (exact notional split)
    fwd_T = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=T)
    phi_T = lambda u: entry.cf(u, fwd_T, params)  # noqa: E731
    grid_T = cos_auto_grid(entry.cumulants(fwd_T, params), N=max(N, 1 << 10), L=L)
    vanilla = float(cos_prices(phi_T, fwd_T, np.array([K]), grid_T).call_prices[0])
    if cp == -1:
        vanilla = vanilla - fwd_T.disc * (fwd_T.F0 - K)
    return float(max(vanilla - fade_in, 0.0))


__all__ = ["LEVY_FADER_MODELS", "levy_fader_price"]
