from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..char_func.base import CharFunc, ForwardSpec
from ..utils.grids import FFTGrid
from ..utils.numerics import cm_simpson_weights
from ..utils.interp import interp_cubic


@dataclass(frozen=True)
class CarrMadanResult:
    k: np.ndarray            # log-strike grid (log K)
    call_prices: np.ndarray  # call prices on that grid
    K: np.ndarray            # strikes = exp(k)


def _phi_logprice(phi_logret: CharFunc, fwd: ForwardSpec, v: np.ndarray) -> np.ndarray:
    """Convert CF of log-return log(S_T/F0) to CF of log-price log(S_T)."""
    logF0 = np.log(fwd.F0)
    return np.exp(1j * v * logF0) * phi_logret(v)


def carr_madan_fft_prices(
    phi: CharFunc,
    fwd: ForwardSpec,
    grid: FFTGrid,
    k0: float = 0.0,
) -> CarrMadanResult:
    """Carr-Madan 1999 FFT: call prices on a log-strike grid.

    psi(v) = exp(-r*T) * phi_logS(v - (alpha+1)i)
             / (alpha^2 + alpha - v^2 + i*(2*alpha + 1)*v)
    C(k)   = exp(-alpha*k)/pi * Re{ FFT[ exp(i*v*(b - k0)) * psi(v) * w * eta ] }
    k_n    = (k0 - b) + n*lam,  with b = N*lam/2.

    Default k0=0 centres the log-strike grid at K=1; interpolation on log K
    will recover prices for any strike within [exp(-b), exp(+b)].
    """
    N, eta, alpha = grid.N, grid.eta, grid.alpha
    lam = grid.lam
    b = 0.5 * N * lam

    v = np.arange(N) * eta
    psi_logS = _phi_logprice(phi, fwd, v - 1j * (alpha + 1.0))
    denom = alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v
    psi = fwd.disc * psi_logS / denom

    w = cm_simpson_weights(N)
    x = np.exp(1j * (b - k0) * v) * psi * w * eta

    fft_vals = np.fft.fft(x)
    k = (k0 - b) + np.arange(N) * lam
    call = (np.exp(-alpha * k) / np.pi) * np.real(fft_vals)
    K = np.exp(k)
    return CarrMadanResult(k=k, call_prices=call, K=K)


def carr_madan_price_at_strikes(
    phi: CharFunc,
    fwd: ForwardSpec,
    grid: FFTGrid,
    strikes: np.ndarray,
    window_factor: float = 0.9,
) -> np.ndarray:
    """FFT prices on a dense log-strike grid, then cubic-spline interpolate.

    Grid is centred at k0 = log(F0) so the Fourier inversion sits symmetrically
    around the at-the-money strike. The returned grid covers log K in
    [k0 - b, k0 + b] with b = N*lam/2; edges suffer Gibbs / aliasing, so we
    restrict the interpolation support to the interior
        |log(K) - k0| < window_factor * b
    and REQUIRE queried strikes to lie inside that window (anything past it
    is a user error — make N larger or eta smaller to widen the grid).
    """
    k0 = float(np.log(fwd.F0))
    res = carr_madan_fft_prices(phi, fwd, grid, k0=k0)

    b = 0.5 * grid.N * grid.lam
    k_query = np.log(np.asarray(strikes, dtype=float))
    half = window_factor * b
    if np.any(np.abs(k_query - k0) >= half):
        raise ValueError(
            f"Strike(s) outside grid-consistent window "
            f"|log(K/F0)| < {window_factor}*b = {half:.4f}. "
            f"Widen the grid (larger N or smaller eta)."
        )

    mask = np.abs(res.k - k0) < half
    return interp_cubic(res.k[mask], res.call_prices[mask], k_query)
