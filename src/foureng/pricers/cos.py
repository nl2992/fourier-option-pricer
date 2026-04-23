from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.base import CharFunc, ForwardSpec
from ..utils.cumulants import (
    Cumulants,
    cos_centered_half_width,
    cos_centered_interval,
    cos_resolution_terms,
    cos_tail_proxy,
    cos_truncation_interval,
)
from ..utils.grids import COSGrid, COSGridPolicy


@dataclass(frozen=True)
class COSResult:
    strikes: np.ndarray
    call_prices: np.ndarray


@dataclass(frozen=True)
class COSPolicyDecision:
    """Resolved adaptive COS decision.

    ``method`` is the engine the improved pipeline should use after inspecting
    the interval geometry. In favorable regimes that is still ``"cos"``; for
    very wide intervals the decision can switch to ``"lewis"`` or
    ``"carr_madan"`` instead of forcing COS into an unfavorable approximation
    domain.
    """

    grid: COSGrid
    method: str
    dx_target: float
    L_used: float
    tail_proxy: float
    tail_family: str
    reason: str = ""


def _canonical_cos_model_name(model: str | None) -> str | None:
    if model is None:
        return None
    aliases = {
        "variance_gamma": "vg",
    }
    return aliases.get(model, model)


def _cos_tail_family(model: str | None, params=None) -> str:
    model = _canonical_cos_model_name(model)
    if model in {None, "bsm", "heston", "ousv", "nig"}:
        return "gaussian_like"
    if model in {"kou", "bates", "heston_kou", "heston_cgmy"}:
        return "semi_heavy"
    if model == "vg":
        return "heavy"
    if model == "cgmy":
        Y = float(getattr(params, "Y", 0.0))
        return "semi_heavy" if Y < 1.0 else "heavy"
    return "semi_heavy"


def _default_dx_target(model: str | None, params=None, *, mode: str = "benchmark") -> float:
    """Model-dependent spatial resolution target for adaptive COS."""
    model = _canonical_cos_model_name(model)
    if model == "bsm":
        base = 0.020
    elif model == "heston":
        base = 0.020
    elif model in {"ousv", "nig"}:
        base = 0.020
    elif model == "vg":
        return 0.003 if mode == "benchmark" else 0.010
    elif model == "cgmy":
        Y = float(getattr(params, "Y", 0.0))
        base = 0.030 if Y < 1.0 else 0.055
    else:
        base = 0.035
    if mode == "surface":
        return 3.0 * base
    return base


def _default_L_seed(model: str | None, params=None, *, mode: str = "benchmark") -> float:
    """Initial L used by the adaptive interval policy."""
    model = _canonical_cos_model_name(model)
    if model == "vg":
        base = 10.0
    elif model == "cgmy":
        Y = float(getattr(params, "Y", 0.0))
        base = 10.0 if Y < 1.0 else 14.0
    elif model in {"kou", "bates", "heston_kou", "heston_cgmy"}:
        base = 10.0
    else:
        base = 8.0
    if mode == "surface":
        return max(6.0, 0.8 * base)
    return base


def _default_fallback_method(model: str | None) -> str | None:
    return None


def recommended_cos_policy(
    model: str | None = None,
    params=None,
    *,
    mode: str = "benchmark",
) -> COSGridPolicy:
    """Recommended adaptive COS policy for the given model/regime."""
    model = _canonical_cos_model_name(model or getattr(params, "name", None))
    tail_family = _cos_tail_family(model, params)
    truncation = "tolerance" if tail_family != "heavy" else "heuristic"
    return COSGridPolicy(
        mode=mode,
        truncation=truncation,
        centered=True,
        dx_target=_default_dx_target(model, params, mode=mode),
        L=_default_L_seed(model, params, mode=mode),
        eps_trunc=1e-10 if mode == "benchmark" else 1e-7,
        min_N=32,
        max_N=16384 if mode == "benchmark" else 4096,
        width_fallback=40.0 if mode == "benchmark" else 28.0,
        fallback_method=_default_fallback_method(model),
    )


def cos_adaptive_decision(
    cumulants: tuple[float, float, float],
    *,
    model: str | None = None,
    params=None,
    policy: COSGridPolicy | None = None,
    strike_count: int | None = None,
) -> COSPolicyDecision:
    """Resolve an adaptive COS interval/N choice from model cumulants."""
    model = _canonical_cos_model_name(model or getattr(params, "name", None))
    policy = policy or recommended_cos_policy(model, params, mode="benchmark")

    c1, c2, c4 = cumulants
    c = Cumulants(c1=float(c1), c2=float(c2), c4=float(c4))
    tail_family = _cos_tail_family(model, params)
    dx_target = (
        float(policy.dx_target)
        if policy.dx_target is not None
        else _default_dx_target(model, params, mode=policy.mode)
    )

    if policy.truncation == "paper":
        L_used = float(
            policy.paper_L
            if policy.paper_L is not None
            else (policy.L if policy.L is not None else _default_L_seed(model, params, mode=policy.mode))
        )
    else:
        L_used = float(
            policy.L
            if policy.L is not None
            else _default_L_seed(model, params, mode=policy.mode)
        )

    max_L = 96.0
    if policy.truncation == "tolerance":
        while True:
            if policy.centered:
                half_width = cos_centered_half_width(c, L=L_used)
            else:
                a_tmp, b_tmp = cos_truncation_interval(c, L=L_used)
                half_width = 0.5 * (b_tmp - a_tmp)
            tail = cos_tail_proxy(c, half_width, family=tail_family)
            if tail <= policy.eps_trunc or L_used >= max_L:
                tail_proxy = tail
                break
            L_used *= 1.25 if tail_family != "heavy" else 1.5
    else:
        if policy.centered:
            half_width = cos_centered_half_width(c, L=L_used)
        else:
            a_tmp, b_tmp = cos_truncation_interval(c, L=L_used)
            half_width = 0.5 * (b_tmp - a_tmp)
        tail_proxy = cos_tail_proxy(c, half_width, family=tail_family)

    if policy.centered:
        a, b = cos_centered_interval(c, L=L_used)
        center = c.c1
        label = f"{policy.truncation}_centered"
    else:
        a, b = cos_truncation_interval(c, L=L_used)
        center = 0.0
        label = policy.truncation

    width = b - a
    N = (
        int(policy.fixed_N)
        if policy.fixed_N is not None
        else cos_resolution_terms(
            width,
            dx_target,
            min_N=policy.min_N,
            max_N=policy.max_N,
        )
    )
    grid = COSGrid(N=N, a=float(a), b=float(b), center=float(center), label=label)

    method = "cos"
    reason = ""
    if policy.width_fallback > 0.0 and width > policy.width_fallback:
        fallback = policy.fallback_method or _default_fallback_method(model)
        if fallback is None:
            fallback = "lewis" if strike_count is None or strike_count <= 4 else "carr_madan"
        if fallback is not None:
            method = fallback
            reason = (
                f"interval width {width:.2f} exceeded "
                f"{policy.width_fallback:.2f}; switched to {fallback}"
            )

    return COSPolicyDecision(
        grid=grid,
        method=method,
        dx_target=dx_target,
        L_used=L_used,
        tail_proxy=tail_proxy,
        tail_family=tail_family,
        reason=reason,
    )


def cos_improved_grid(
    cumulants: tuple[float, float, float],
    *,
    model: str | None = None,
    params=None,
    policy: COSGridPolicy | None = None,
) -> COSGrid:
    """Convenience wrapper returning only the adaptive COS grid."""
    return cos_adaptive_decision(
        cumulants,
        model=model,
        params=params,
        policy=policy,
    ).grid


def _call_payoff_coeffs(a: float, b: float, N: int, K: np.ndarray, F0: float) -> np.ndarray:
    """Fourier-cosine coefficients V_k of the call payoff (S_T - K)^+.

    Retained for backward compatibility and for modules that explicitly want
    the call-payoff form (e.g. parameter-sensitivity Greeks in
    ``foureng.greeks.cos_greeks``). **Do not use for pricing at long
    maturities** — the ``e^b`` factor in the chi integral catastrophically
    loses precision for wide truncation intervals (``b > ~10``). Pricing
    goes through :func:`_put_payoff_coeffs` + put-call parity instead.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    y_star = np.log(K / F0)
    c = np.clip(y_star, a, b)
    d = b

    k = np.arange(N)
    omega = k * np.pi / (b - a)

    ca = c[None, :] - a
    da = d - a
    cos_cd = np.cos(omega[:, None] * da)
    sin_cd = np.sin(omega[:, None] * da)
    cos_cc = np.cos(omega[:, None] * ca)
    sin_cc = np.sin(omega[:, None] * ca)

    ed = np.exp(d)
    ec = np.exp(c)

    chi = (
        cos_cd * ed
        - cos_cc * ec
        + omega[:, None] * sin_cd * ed
        - omega[:, None] * sin_cc * ec
    ) / (1.0 + omega[:, None] ** 2)

    psi = np.empty_like(chi)
    psi[0, :] = d - c
    with np.errstate(divide="ignore", invalid="ignore"):
        psi[1:, :] = (sin_cd[1:, :] - sin_cc[1:, :]) / omega[1:, None]

    V = (2.0 / (b - a)) * (F0 * chi - K[None, :] * psi)

    mask = y_star >= b
    if np.any(mask):
        V[:, mask] = 0.0
    return V


def _put_payoff_coeffs(a: float, b: float, N: int, K: np.ndarray, F0: float) -> np.ndarray:
    """Fourier-cosine coefficients V_k of the put payoff (K - S_T)^+."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    y_star = np.log(K / F0)

    c = a
    d = np.minimum(y_star, b)

    k = np.arange(N)
    omega = k * np.pi / (b - a)

    da = d[None, :] - a
    ca = c - a

    cos_cd = np.cos(omega[:, None] * da)
    sin_cd = np.sin(omega[:, None] * da)
    cos_cc = np.cos(omega[:, None] * ca)
    sin_cc = np.sin(omega[:, None] * ca)

    ed = np.exp(d)
    ec = np.exp(c)

    chi = (
        cos_cd * ed[None, :]
        - cos_cc * ec
        + omega[:, None] * sin_cd * ed[None, :]
        - omega[:, None] * sin_cc * ec
    ) / (1.0 + omega[:, None] ** 2)

    psi = np.empty_like(chi)
    psi[0, :] = d - c
    with np.errstate(divide="ignore", invalid="ignore"):
        psi[1:, :] = (sin_cd[1:, :] - sin_cc[1:, :]) / omega[1:, None]

    V = (2.0 / (b - a)) * (K[None, :] * psi - F0 * chi)

    mask = y_star <= a
    if np.any(mask):
        V[:, mask] = 0.0
    return V


def cos_prices(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes: np.ndarray,
    grid: COSGrid,
    *,
    payoff_mode: str = "put_parity",
    call_direct_width_max: float = 20.0,
) -> COSResult:
    """Fang-Oosterlee (2008) COS method for European calls.

    Expansion variable: y = log(S_T/F0), with CF phi (our CharFunc protocol).
    Truncation interval [a,b] = grid.a, grid.b; number of cosine terms N = grid.N.

    ``grid.center`` optionally shifts the expansion variable to

        z = log(S_T / F_0) - center,

    which lets us use a symmetric interval around zero in the centered
    variable while still evaluating the true payoff. This is the path used by
    the improved adaptive COS policy.

    **Numerical implementation note**: by default we price the **put** via COS
    (payoff coefficients are O(K), bounded regardless of ``b``) and recover
    the call by put-call parity:

        C - P = D * (F0 - K),    i.e.    C = P + D * (F0 - K)

    where D = exp(-r*T). This avoids the catastrophic cancellation that
    plagues a direct COS-on-call for long maturities.

    ``payoff_mode`` controls the coefficient side:

    - ``"put_parity"`` : always use put coefficients + parity,
    - ``"call_direct"`` : use the direct call coefficients,
    - ``"auto"`` : use put+parity generally, but allow direct-call pricing
      for OTM calls when the interval is narrow enough that the ``e^b`` term
      is still numerically tame.
    """
    a, b, N = grid.a, grid.b, grid.N
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    center = float(getattr(grid, "center", 0.0))
    shifted_F0 = fwd.F0 * np.exp(center)

    k = np.arange(N)
    omega = k * np.pi / (b - a)

    phi_vals = phi(omega)
    if center != 0.0:
        phi_vals = phi_vals * np.exp(-1j * omega * center)
    A = np.real(phi_vals * np.exp(-1j * omega * a))
    A[0] *= 0.5

    if payoff_mode == "call_direct":
        V_call = _call_payoff_coeffs(a, b, N, strikes, shifted_F0)
        calls = fwd.disc * (A[:, None] * V_call).sum(axis=0)
    else:
        V_put = _put_payoff_coeffs(a, b, N, strikes, shifted_F0)
        puts = fwd.disc * (A[:, None] * V_put).sum(axis=0)
        calls = puts + fwd.disc * (fwd.F0 - strikes)

        if payoff_mode == "auto" and (b - a) <= call_direct_width_max:
            direct_mask = strikes >= fwd.F0
            if np.any(direct_mask):
                V_call = _call_payoff_coeffs(a, b, N, strikes, shifted_F0)
                direct_calls = fwd.disc * (A[:, None] * V_call).sum(axis=0)
                calls = np.where(direct_mask, direct_calls, calls)
        elif payoff_mode not in {"put_parity", "auto"}:
            raise ValueError(
                f"unknown payoff_mode {payoff_mode!r}; choose "
                "'put_parity' | 'call_direct' | 'auto'"
            )
    return COSResult(strikes=strikes, call_prices=calls)


def cos_auto_grid(cumulants: tuple[float, float, float], N: int, L: float = 10.0) -> COSGrid:
    """Build a COSGrid from model cumulants via FO 2008 truncation rule."""
    c1, c2, c4 = cumulants
    a, b = cos_truncation_interval(Cumulants(c1=c1, c2=c2, c4=c4), L=L)
    return COSGrid(N=N, a=a, b=b)
