from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..models.base import CharFunc, ForwardSpec
from ..utils.grids import COSGrid
from ..utils.cumulants import Cumulants, cos_truncation_interval


@dataclass(frozen=True)
class COSResult:
    strikes: np.ndarray
    call_prices: np.ndarray


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
    y_star = np.log(K / F0)                       # (nK,)
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

    chi = (cos_cd * ed - cos_cc * ec
           + omega[:, None] * sin_cd * ed
           - omega[:, None] * sin_cc * ec) / (1.0 + omega[:, None] ** 2)

    psi = np.empty_like(chi)
    psi[0, :] = d - c
    with np.errstate(divide="ignore", invalid="ignore"):
        psi[1:, :] = (sin_cd[1:, :] - sin_cc[1:, :]) / omega[1:, None]

    V = (2.0 / (b - a)) * (F0 * chi - K[None, :] * psi)

    mask = (y_star >= b)
    if np.any(mask):
        V[:, mask] = 0.0
    return V


def _put_payoff_coeffs(a: float, b: float, N: int, K: np.ndarray, F0: float) -> np.ndarray:
    """Fourier-cosine coefficients V_k of the put payoff (K - S_T)^+ as a
    function of y = log(S_T/F0), on the interval [a,b], for each strike.

    FO 2008 Appendix A closed forms for chi_k, psi_k apply identically; for
    the put we integrate from c=a to d=min(b, log(K/F0)):

        V_k = (2/(b-a)) * ( K * psi_k(a, d) - F0 * chi_k(a, d) )

    **Why put and not call**: the call payoff integrates cos(...) * e^y on
    [log(K/F0), b]; if ``b`` is large (long maturity with the FO2008 L=32
    recipe gives b ≈ 35, so e^b ≈ 2e15), the chi_k magnitudes are
    astronomically large and must cancel against K*psi_k to produce a
    small price — float64 catastrophically loses precision. The put
    integrates on [a, log(K/F0)] where e^d ≤ K/F0 (strike-bounded), so
    the coefficients stay O(K). We then recover the call via put-call
    parity in ``cos_prices`` without any loss of precision.

    Vectorised over strikes -> returns shape (N, len(K)).
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    y_star = np.log(K / F0)                       # (nK,)

    # put payoff nonzero for y in [a, min(b, y*)]; zero elsewhere
    c = a                                         # scalar lower integration limit
    d = np.minimum(y_star, b)                     # (nK,) upper limit

    k = np.arange(N)
    omega = k * np.pi / (b - a)                   # (N,)

    da = d[None, :] - a                           # (1, nK)
    ca = c - a                                    # = 0

    cos_cd = np.cos(omega[:, None] * da)          # (N, nK)
    sin_cd = np.sin(omega[:, None] * da)
    cos_cc = np.cos(omega[:, None] * ca)          # (N, 1); ca=0 → cos=1
    sin_cc = np.sin(omega[:, None] * ca)          # ca=0 → sin=0

    ed = np.exp(d)                                # (nK,); bounded by K/F0
    ec = np.exp(c)                                # scalar; = e^a

    chi = (cos_cd * ed[None, :] - cos_cc * ec
           + omega[:, None] * sin_cd * ed[None, :]
           - omega[:, None] * sin_cc * ec) / (1.0 + omega[:, None] ** 2)

    psi = np.empty_like(chi)
    psi[0, :] = d - c                             # k=0 row
    with np.errstate(divide="ignore", invalid="ignore"):
        psi[1:, :] = (sin_cd[1:, :] - sin_cc[1:, :]) / omega[1:, None]

    V = (2.0 / (b - a)) * (K[None, :] * psi - F0 * chi)

    # If a strike is so deep OTM (put) that y* <= a, put is worthless on [a,b].
    mask = (y_star <= a)
    if np.any(mask):
        V[:, mask] = 0.0
    return V


def cos_prices(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes: np.ndarray,
    grid: COSGrid,
) -> COSResult:
    """Fang-Oosterlee (2008) COS method for European calls.

    Expansion variable: y = log(S_T/F0), with CF phi (our CharFunc protocol).
    Truncation interval [a,b] = grid.a, grid.b; number of cosine terms N = grid.N.

    **Numerical implementation note**: we price the **put** via COS
    (payoff coefficients are O(K), bounded regardless of ``b``) and
    recover the call by put-call parity:

        C - P = D * (F0 - K),    i.e.    C = P + D * (F0 - K)

    where D = exp(-r*T). This avoids the catastrophic cancellation that
    plagues a direct COS-on-call for long maturities (where the FO2008
    truncation b grows to tens and e^b overflows ~15 digits of
    precision). Verified: Heston T=10 with L=32 now converges to
    ~1e-3 at N=140 vs the previous blow-up to 1e+11.

    Price (per strike):
        P(K) = exp(-r*T) * sum_{k=0}^{N-1}' Re{ phi(k*pi/(b-a)) * exp(-i*k*pi*a/(b-a)) } * V_k^put
        C(K) = P(K) + exp(-r*T) * (F0 - K)
    """
    a, b, N = grid.a, grid.b, grid.N
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))

    k = np.arange(N)
    omega = k * np.pi / (b - a)                   # (N,)

    phi_vals = phi(omega)                          # (N,), complex
    A = np.real(phi_vals * np.exp(-1j * omega * a))  # (N,)
    A[0] *= 0.5                                    # first-term prime

    V_put = _put_payoff_coeffs(a, b, N, strikes, fwd.F0)  # (N, nK)
    puts = fwd.disc * (A[:, None] * V_put).sum(axis=0)    # (nK,)
    calls = puts + fwd.disc * (fwd.F0 - strikes)          # put-call parity
    return COSResult(strikes=strikes, call_prices=calls)


def cos_auto_grid(cumulants: tuple[float, float, float], N: int, L: float = 10.0) -> COSGrid:
    """Build a COSGrid from model cumulants via FO 2008 truncation rule."""
    c1, c2, c4 = cumulants
    a, b = cos_truncation_interval(Cumulants(c1=c1, c2=c2, c4=c4), L=L)
    return COSGrid(N=N, a=a, b=b)
