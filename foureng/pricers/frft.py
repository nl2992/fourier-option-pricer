"""Chourdakis (2004) fractional FFT pricer for European calls.

The FRFT decouples the frequency step eta from the log-strike step lam by
introducing the fraction zeta = eta*lam / (2*pi). A standard FFT is the
special case zeta = 1/N; here both can be chosen freely to tune frequency
resolution and strike coverage independently.

The pricing formula mirrors Carr-Madan exactly; only the final convolution
changes from an FFT to an FRFT. Main interface functions:
    frft_prices            -- full log-strike grid as FRFTResult
    frft_price_at_strikes  -- ATM-centered grid + cubic-spline lookup at target strikes
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.base import CharFunc, ForwardSpec
from ..utils.frft import frft
from ..utils.grids import FRFTGrid
from ..utils.interp import interp_cubic
from ..utils.numerics import cm_simpson_weights, phi_logprice


@dataclass(frozen=True)
class FRFTResult:
    """Output of frft_prices.

    Attributes
    ----------
    k : np.ndarray
        Log-strike grid, uniform with spacing lam.
    call_prices : np.ndarray
        Damped call prices on the log-strike grid.
    K : np.ndarray
        Strikes, equal to exp(k).
    """

    k: np.ndarray
    call_prices: np.ndarray
    K: np.ndarray


def frft_prices(
    phi: CharFunc,
    fwd: ForwardSpec,
    grid: FRFTGrid,
    k0: float = 0.0,
) -> FRFTResult:
    """Chourdakis (2004) fractional FFT for Carr-Madan pricing.

    Decouples freq step eta from log-strike step lam via
        zeta = eta * lam / (2*pi)
    (Nyquist FFT is the zeta = 1/N special case; here zeta is free.)

    Pricing:
        psi(v) = exp(-rT) * phi_logS(v - (alpha+1)i)
                 / (alpha^2 + alpha - v^2 + i*(2*alpha+1)*v)
        C(k_n) = exp(-alpha*k_n)/pi * Re{ FRFT_zeta[ exp(i*v*(b-k0)) * psi(v) * w * eta ]_n }
    with v_j = j*eta, k_n = (k0 - b) + n*lam, b = N*lam/2.
    """
    N, eta, lam, alpha = grid.N, grid.eta, grid.lam, grid.alpha
    zeta = grid.zeta
    b = 0.5 * N * lam

    v = np.arange(N) * eta
    psi_logS = phi_logprice(phi, fwd, v - 1j * (alpha + 1.0))
    denom = alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v
    psi = fwd.disc * psi_logS / denom

    w = cm_simpson_weights(N)
    x = np.exp(1j * (b - k0) * v) * psi * w * eta

    G = frft(x, zeta)
    k = (k0 - b) + np.arange(N) * lam
    call = (np.exp(-alpha * k) / np.pi) * np.real(G)
    K = np.exp(k)
    return FRFTResult(k=k, call_prices=call, K=K)


def frft_price_at_strikes(
    phi: CharFunc,
    fwd: ForwardSpec,
    grid: FRFTGrid,
    strikes: np.ndarray,
    window_factor: float = 0.9,
) -> np.ndarray:
    """FRFT grid prices -> cubic-spline interpolate to target strikes.

    Same grid-consistent windowing as Carr-Madan: the grid is centred at
    k0 = log(F0), covers |log(K) - k0| <= b = N*lam/2, and the interpolation
    support is restricted to the interior |log(K) - k0| < window_factor*b to
    avoid Gibbs/aliasing at the edges. Queried strikes outside the window
    raise a ValueError — widen the grid (larger N or smaller eta/lam) instead
    of silently extrapolating.
    """
    k0 = float(np.log(fwd.F0))
    res = frft_prices(phi, fwd, grid, k0=k0)

    b = 0.5 * grid.N * grid.lam
    k_query = np.log(np.asarray(strikes, dtype=float))
    half = window_factor * b
    if np.any(np.abs(k_query - k0) >= half):
        raise ValueError(
            f"Strike(s) outside grid-consistent window "
            f"|log(K/F0)| < {window_factor}*b = {half:.4f}. "
            f"Widen the grid (larger N, smaller eta/lam)."
        )

    mask = np.abs(res.k - k0) < half
    return interp_cubic(res.k[mask], res.call_prices[mask], k_query)
