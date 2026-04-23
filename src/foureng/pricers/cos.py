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
    """Fourier-cosine coefficients V_k of the call payoff (S_T - K)^+ as a
    function of y = log(S_T/F0), on the interval [a,b], for each strike.

    FO 2008 Appendix A closed forms:
        omega_k = k*pi/(b-a)
        chi_k(c,d) = (1 + omega_k^2)^{-1} *
            [ cos(omega_k*(d-a))*e^d - cos(omega_k*(c-a))*e^c
             + omega_k*sin(omega_k*(d-a))*e^d - omega_k*sin(omega_k*(c-a))*e^c ]
        psi_0(c,d) = d - c
        psi_k(c,d) = (sin(omega_k*(d-a)) - sin(omega_k*(c-a))) / omega_k  (k>0)

    With c = max(a, log(K/F0)), d = b:
        V_k = (2/(b-a)) * ( F0 * chi_k(c,d) - K * psi_k(c,d) )

    Vectorised over strikes -> returns shape (N, len(K)).
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    y_star = np.log(K / F0)                       # (nK,)

    # c per strike; clip to [a, b] so that c >= b means OTM beyond support
    c = np.clip(y_star, a, b)                     # (nK,)
    d = b

    k = np.arange(N)
    omega = k * np.pi / (b - a)                   # (N,)

    # precompute terms
    ca = c[None, :] - a                           # (1,nK)
    da = d - a
    cos_cd = np.cos(omega[:, None] * da)          # (N,1)  broadcastable
    sin_cd = np.sin(omega[:, None] * da)
    cos_cc = np.cos(omega[:, None] * ca)          # (N,nK)
    sin_cc = np.sin(omega[:, None] * ca)

    ed = np.exp(d)
    ec = np.exp(c)                                # (nK,)

    # chi_k(c,d) : shape (N, nK)
    chi = (cos_cd * ed - cos_cc * ec
           + omega[:, None] * sin_cd * ed
           - omega[:, None] * sin_cc * ec) / (1.0 + omega[:, None] ** 2)

    # psi_k(c,d) : shape (N, nK);   k=0 row handled separately
    psi = np.empty_like(chi)
    psi[0, :] = d - c
    with np.errstate(divide="ignore", invalid="ignore"):
        psi[1:, :] = (sin_cd[1:, :] - sin_cc[1:, :]) / omega[1:, None]

    V = (2.0 / (b - a)) * (F0 * chi - K[None, :] * psi)

    # If a strike is so OTM that y* >= b, the call is (numerically) worthless;
    # we already clipped c to b so chi and psi are zero there. Sanity force-zero:
    mask = (y_star >= b)
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

    Price (per strike):
        C(K) = exp(-r*T) * sum_{k=0}^{N-1}' Re{ phi(k*pi/(b-a)) * exp(-i*k*pi*a/(b-a)) } * V_k
    where the prime means halve the k=0 term, and V_k are the payoff coefficients.
    """
    a, b, N = grid.a, grid.b, grid.N
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))

    k = np.arange(N)
    omega = k * np.pi / (b - a)                   # (N,)

    phi_vals = phi(omega)                          # (N,), complex
    A = np.real(phi_vals * np.exp(-1j * omega * a))  # (N,)
    A[0] *= 0.5                                    # first-term prime

    V = _call_payoff_coeffs(a, b, N, strikes, fwd.F0)     # (N, nK)
    prices = fwd.disc * (A[:, None] * V).sum(axis=0)      # (nK,)
    return COSResult(strikes=strikes, call_prices=prices)


def cos_auto_grid(cumulants: tuple[float, float, float], N: int, L: float = 10.0) -> COSGrid:
    """Build a COSGrid from model cumulants via FO 2008 truncation rule."""
    c1, c2, c4 = cumulants
    a, b = cos_truncation_interval(Cumulants(c1=c1, c2=c2, c4=c4), L=L)
    return COSGrid(N=N, a=a, b=b)
