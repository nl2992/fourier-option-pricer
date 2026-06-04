"""CONV-style Fourier European option pricer.

The MATLAB reference project exposes a ``Fourier/CONV`` European routine.  This
module adds the corresponding public method surface for ``foureng`` using a
deterministic Fourier inversion of the risk-neutral share and cash
probabilities.  It is intentionally European-only; exotic CONV/PROJ recursions
belong in later product-specific engines.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from foureng.models.base import CharFunc, ForwardSpec
from foureng.utils.grids import CONVGrid


def conv_price_at_strikes(
    phi: CharFunc,
    fwd: ForwardSpec,
    grid: CONVGrid,
    strikes,
    *,
    cp: int = 1,
) -> np.ndarray:
    """Price European calls or puts by Fourier probability inversion.

    Parameters
    ----------
    phi
        Characteristic function of ``X_T = log(S_T / F_0)``.
    fwd
        Market inputs.
    grid
        Positive-frequency integration grid.
    strikes
        Strike array.
    cp
        ``+1`` for calls, ``-1`` for puts.  Puts are returned by parity after
        computing the call value.
    """
    if cp not in (1, -1):
        raise ValueError(f"conv_price_at_strikes: cp must be +1 or -1, got {cp}")

    K = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    if K.size == 0:
        raise ValueError("strikes must be non-empty")
    if np.any(K <= 0.0) or not np.all(np.isfinite(K)):
        raise ValueError("strikes must be finite and > 0")

    u = grid.u()
    iu = 1j * u
    log_moneyness = np.log(K / fwd.F0)

    phi_u = np.asarray(phi(u), dtype=np.complex128)
    phi_shift = np.asarray(phi(u - 1j), dtype=np.complex128)
    martingale = complex(np.asarray(phi(np.array([-1j])), dtype=np.complex128)[0])
    if not np.isfinite(martingale.real) or not np.isfinite(martingale.imag):
        raise FloatingPointError("phi(-i) is not finite; CONV share probability is undefined")
    if abs(martingale) < 1e-14:
        raise FloatingPointError("phi(-i) is numerically zero; CONV share probability is undefined")

    phase = np.exp(-1j * np.outer(log_moneyness, u))
    p2_integrand = np.real(phase * (phi_u / iu))
    p1_integrand = np.real(phase * (phi_shift / (iu * martingale)))

    p2 = 0.5 + trapezoid(p2_integrand, u, axis=1) / np.pi
    p1 = 0.5 + trapezoid(p1_integrand, u, axis=1) / np.pi

    call = fwd.disc * (fwd.F0 * p1 - K * p2)
    call = np.maximum(call, 0.0)
    if cp == 1:
        return np.asarray(call, dtype=np.float64)
    put = call - fwd.disc * (fwd.F0 - K)
    return np.asarray(np.maximum(put, 0.0), dtype=np.float64)


__all__ = ["conv_price_at_strikes"]
