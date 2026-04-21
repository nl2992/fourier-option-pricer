"""Closed-form Delta / Gamma / parameter-sensitivity under the COS expansion.

The European call price under Fang-Oosterlee (2008), with expansion variable
y = log(S_T / F_0), is

    C = D * sum_k' Re{ phi(omega_k) * exp(-i*omega_k*a) } * V_k

where D = exp(-r*T), omega_k = k*pi/(b-a), and the payoff coefficients

    V_k = (2/(b-a)) * [ F_0 * chi_k(c, b) - K * psi_k(c, b) ],   c = max(a, log(K/F_0))

with the FO2008 Appendix A formulas for chi_k, psi_k.

Delta (dC/dS_0) and Gamma (d^2C/dS_0^2) are obtained analytically from the
F_0-dependence of V_k (the CF phi(u) of log(S_T/F_0) does not depend on S_0
under our log-forward convention; all S_0 flow goes through F_0). After a
short calculation the k-th payoff derivative simplifies to

    dV_k / dF_0     = (2/(b-a)) * chi_k(c, b)
    d^2 V_k / dF_0^2 = (2/(b-a)) * (K / F_0^2) * cos(omega_k * (c - a))    if a < log(K/F_0) < b
                    = 0                                                    otherwise

The second-derivative indicator captures two degenerate cases: deep-ITM
(c pinned to a, so V_k is linear in F_0 -> Gamma = 0) and deep-OTM
(log(K/F_0) >= b, so V_k = 0 identically).

Because dF_0/dS_0 = exp((r-q)T), the spot Greeks are

    Delta = exp((r-q)T) * dC/dF_0
    Gamma = exp(2(r-q)T) * d^2C/dF_0^2

Parameter sensitivities (dC/dtheta for any model parameter) come for free:
since the grid and payoff coefficients do not depend on theta, we just
replace phi by dphi/dtheta in the same COS sum. Useful for Vega-style
Greeks when the caller can produce dphi/dtheta in closed form.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..char_func.base import CharFunc, ForwardSpec
from ..utils.grids import COSGrid


@dataclass(frozen=True)
class COSGreeks:
    strikes: np.ndarray
    call_prices: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray


def _chi_psi(a: float, b: float, N: int, K: np.ndarray, F0: float):
    """Return chi_k(c,b), psi_k(c,b), omega_k, c, and the in-interval mask.

    All arrays broadcast to shape (N, nK) except for omega (N,) and
    in_interval / c (nK,).
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    y_star = np.log(K / F0)                          # (nK,)
    in_interval = (y_star > a) & (y_star < b)
    c = np.clip(y_star, a, b)                         # (nK,)

    k = np.arange(N)
    omega = k * np.pi / (b - a)                       # (N,)

    ca = c[None, :] - a                               # (1, nK)
    da = b - a
    cos_cd = np.cos(omega[:, None] * da)              # (N, 1)
    sin_cd = np.sin(omega[:, None] * da)
    cos_cc = np.cos(omega[:, None] * ca)              # (N, nK)
    sin_cc = np.sin(omega[:, None] * ca)

    ed = np.exp(b)
    ec = np.exp(c)                                    # (nK,)

    chi = (cos_cd * ed - cos_cc * ec
           + omega[:, None] * sin_cd * ed
           - omega[:, None] * sin_cc * ec) / (1.0 + omega[:, None] ** 2)

    psi = np.empty_like(chi)
    psi[0, :] = b - c
    with np.errstate(divide="ignore", invalid="ignore"):
        psi[1:, :] = (sin_cd[1:, :] - sin_cc[1:, :]) / omega[1:, None]

    far_otm = y_star >= b
    if np.any(far_otm):
        chi[:, far_otm] = 0.0
        psi[:, far_otm] = 0.0

    return chi, psi, omega, c, cos_cc, in_interval, far_otm


def cos_price_and_greeks(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes: np.ndarray,
    grid: COSGrid,
) -> COSGreeks:
    """Price + (Delta, Gamma) for European calls under the COS expansion.

    Single COS sweep: evaluates phi on the same grid used for pricing and
    reuses the chi/psi payoff coefficients and their analytic F_0-derivatives.
    """
    a, b, N = grid.a, grid.b, grid.N
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    F0 = fwd.F0

    chi, psi, omega, c, cos_cc, in_interval, far_otm = _chi_psi(a, b, N, strikes, F0)

    phi_vals = phi(omega)
    A = np.real(phi_vals * np.exp(-1j * omega * a))
    A[0] *= 0.5                                       # first-term prime

    V = (2.0 / (b - a)) * (F0 * chi - strikes[None, :] * psi)
    dV_dF0 = (2.0 / (b - a)) * chi                    # (N, nK)

    d2V_dF02 = np.zeros_like(chi)
    if np.any(in_interval):
        coef = (2.0 / (b - a)) * (strikes[None, :] / (F0 * F0)) * cos_cc
        d2V_dF02[:, in_interval] = coef[:, in_interval]

    disc = fwd.disc
    price = disc * (A[:, None] * V).sum(axis=0)
    dC_dF0 = disc * (A[:, None] * dV_dF0).sum(axis=0)
    d2C_dF02 = disc * (A[:, None] * d2V_dF02).sum(axis=0)

    jac = np.exp((fwd.r - fwd.q) * fwd.T)
    delta = jac * dC_dF0
    gamma = (jac * jac) * d2C_dF02

    return COSGreeks(strikes=strikes, call_prices=price, delta=delta, gamma=gamma)


def cos_delta_gamma(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes: np.ndarray,
    grid: COSGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning just (Delta, Gamma)."""
    out = cos_price_and_greeks(phi, fwd, strikes, grid)
    return out.delta, out.gamma


def cos_parameter_sensitivity(
    dphi_dparam: CharFunc,
    fwd: ForwardSpec,
    strikes: np.ndarray,
    grid: COSGrid,
) -> np.ndarray:
    """Compute dC/dtheta for one scalar model parameter theta.

    Caller provides dphi/dtheta as a callable with the same signature as a
    CharFunc (returning complex array of shape (N,)). Because the COS grid
    (a, b, N) and the payoff coefficients V_k are theta-independent, the
    derivative price is just the COS sum with phi replaced by dphi/dtheta.

    Typical use: Vega-like sensitivities where dphi/dsigma is derivable in
    closed form from the model CF (e.g. Heston d/dv0, BS-equivalent d/dsigma,
    VG d/dsigma, Kou d/dsigma). This is what was referred to in earlier
    design notes as "Delta/Vega in one FFT call" -- with COS it is one COS
    call per Greek, sharing the same grid.
    """
    a, b, N = grid.a, grid.b, grid.N
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))

    k = np.arange(N)
    omega = k * np.pi / (b - a)

    dphi_vals = dphi_dparam(omega)
    A = np.real(dphi_vals * np.exp(-1j * omega * a))
    A[0] *= 0.5

    # Reuse the standard COS payoff coefficients V_k (strike-dependent).
    from ..pricers.cos import _call_payoff_coeffs
    V = _call_payoff_coeffs(a, b, N, strikes, fwd.F0)
    return fwd.disc * (A[:, None] * V).sum(axis=0)
