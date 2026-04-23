"""Closed-form Delta / Gamma / parameter-sensitivity under the COS expansion.

**Numerical convention**: the price is computed as a COS put plus put-call
parity (``C = P + D * (F_0 - K)``). This matches :func:`foureng.pricers.cos.cos_prices`
exactly and avoids the catastrophic cancellation that hits the direct
COS-on-call sum for long maturities (where the truncation bound ``b`` grows
large, making ``e^b`` O(1e15) and the ``chi_k`` payoff coefficients overflow
float64 precision). See ``pricers/cos.py`` for the detailed argument.

Expansion variable y = log(S_T / F_0):

    P = D * sum_k' Re{ phi(omega_k) * exp(-i*omega_k*a) } * V_k^put
    C = P + D * (F_0 - K)

where D = exp(-r*T), omega_k = k*pi/(b-a), and the put payoff coefficients

    V_k^put = (2/(b-a)) * [ K * psi_k(a, d) - F_0 * chi_k(a, d) ],   d = min(b, log(K/F_0))

with the FO2008 Appendix A formulas for chi_k, psi_k.

Delta (dC/dF_0) and Gamma (d^2C/dF_0^2) are obtained analytically from the
F_0-dependence of V_k^put and the linear parity term. The parity term
contributes D to dC/dF_0 and 0 to d^2C/dF_0^2. Applying Leibniz to the put
coefficient (with careful bookkeeping of the moving upper limit
d = log(K/F_0)) gives the clean result

    dV_k^put / dF_0     = -(2/(b-a)) * chi_k(a, d)
    d^2 V_k^put / dF_0^2 = (2/(b-a)) * (K / F_0^2) * cos(omega_k * (d - a))   if a < log(K/F_0) < b
                        = 0                                                   otherwise

so

    dC/dF_0     = D * sum_k' A_k * dV_k^put/dF_0    +    D
    d^2C/dF_0^2 = D * sum_k' A_k * d^2 V_k^put/dF_0^2

The Gamma indicator mask is identical to the direct-call formulation: deep-ITM
(y_star <= a => put worthless on [a,b] and parity gives C = D*(F_0-K), linear)
and deep-OTM (y_star >= b => d = b constant in F_0). Gamma = 0 in both cases.

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

from ..models.base import CharFunc, ForwardSpec
from ..utils.grids import COSGrid


@dataclass(frozen=True)
class COSGreeks:
    strikes: np.ndarray
    call_prices: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray


def _chi_psi_put(a: float, b: float, N: int, K: np.ndarray, F0: float):
    """Return put-integration chi_k(a,d), psi_k(a,d), V_k^put, omega, and masks.

    The put payoff is non-zero for y in [a, min(b, y*)] where y* = log(K/F0).
    All arrays broadcast to shape (N, nK); omega is (N,); d, in_interval,
    deep_itm, deep_otm are (nK,). ``cos_cd`` is cos(omega*(d-a)) — reused
    for the Gamma formula.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    y_star = np.log(K / F0)                          # (nK,)
    in_interval = (y_star > a) & (y_star < b)
    deep_itm = y_star <= a                           # put worthless on [a,b]
    deep_otm = y_star >= b                           # d clamped to b, F0-independent

    d = np.minimum(y_star, b)                        # (nK,) upper integration limit

    k = np.arange(N)
    omega = k * np.pi / (b - a)                       # (N,)

    da = d[None, :] - a                               # (1, nK)
    # ca = 0 since c = a; cos(0)=1, sin(0)=0.
    cos_cd = np.cos(omega[:, None] * da)              # (N, nK)
    sin_cd = np.sin(omega[:, None] * da)

    ed = np.exp(d)                                    # (nK,); bounded by K/F0
    ec = np.exp(a)                                    # scalar

    chi = (cos_cd * ed[None, :] - ec
           + omega[:, None] * sin_cd * ed[None, :]) / (1.0 + omega[:, None] ** 2)

    psi = np.empty_like(chi)
    psi[0, :] = d - a
    with np.errstate(divide="ignore", invalid="ignore"):
        psi[1:, :] = sin_cd[1:, :] / omega[1:, None]

    V_put = (2.0 / (b - a)) * (K[None, :] * psi - F0 * chi)

    # Deep-ITM put: integrand identically zero on [a,b].
    if np.any(deep_itm):
        chi[:, deep_itm] = 0.0
        psi[:, deep_itm] = 0.0
        V_put[:, deep_itm] = 0.0

    return chi, psi, V_put, omega, d, cos_cd, in_interval, deep_itm, deep_otm


def cos_price_and_greeks(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes: np.ndarray,
    grid: COSGrid,
) -> COSGreeks:
    """Price + (Delta, Gamma) for European calls under the COS expansion.

    Single COS sweep: evaluates phi on the same grid used for pricing and
    reuses the chi/psi put payoff coefficients and their analytic
    F_0-derivatives. The put + parity recovery avoids the ``exp(b)``
    cancellation that would plague a direct call-coefficient sum at long
    maturities; see this module's docstring for the full argument.
    """
    a, b, N = grid.a, grid.b, grid.N
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    F0 = fwd.F0

    chi, psi, V_put, omega, d, cos_cd, in_interval, deep_itm, deep_otm = _chi_psi_put(
        a, b, N, strikes, F0
    )

    phi_vals = phi(omega)
    A = np.real(phi_vals * np.exp(-1j * omega * a))
    A[0] *= 0.5                                       # first-term prime

    # dV_k^put / dF_0 = -(2/(b-a)) * chi_k(a, d); deep-ITM has chi=0 already.
    dV_put_dF0 = -(2.0 / (b - a)) * chi               # (N, nK)

    # d^2 V_k^put / dF_0^2: only in-interval contributes, same form as
    # the direct-call derivation because parity is linear in F_0.
    d2V_put_dF02 = np.zeros_like(chi)
    if np.any(in_interval):
        coef = (2.0 / (b - a)) * (strikes[None, :] / (F0 * F0)) * cos_cd
        d2V_put_dF02[:, in_interval] = coef[:, in_interval]

    disc = fwd.disc
    put_price = disc * (A[:, None] * V_put).sum(axis=0)
    dP_dF0 = disc * (A[:, None] * dV_put_dF0).sum(axis=0)
    d2P_dF02 = disc * (A[:, None] * d2V_put_dF02).sum(axis=0)

    # Put-call parity: C = P + D*(F0 - K); dC/dF0 = dP/dF0 + D; d2C = d2P.
    price = put_price + disc * (F0 - strikes)
    dC_dF0 = dP_dF0 + disc
    d2C_dF02 = d2P_dF02

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
