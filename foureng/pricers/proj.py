"""First European PROJ-method façade.

This module intentionally implements the smallest stable public slice of the
MATLAB PROJ surface: European vanilla pricing under one-dimensional CF models.
The numerical core is currently COS-based, which gives a validated projection
baseline while reserving the ``proj`` method label for later B-spline/frame
projection recursions for barriers, Asians, and Bermudans.
"""

from __future__ import annotations

import numpy as np

from foureng.models.base import CharFunc, ForwardSpec
from foureng.pricers.cos import cos_auto_grid, cos_prices


def proj_european_price_at_strikes(
    phi: CharFunc,
    fwd: ForwardSpec,
    cumulants: tuple[float, float, float],
    strikes,
    *,
    cp: int = 1,
    N: int = 512,
    L: float = 10.0,
) -> np.ndarray:
    """European vanilla PROJ façade, currently using COS projection coefficients."""
    if cp not in (1, -1):
        raise ValueError(f"cp must be +1 or -1, got {cp}")
    K = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    grid = cos_auto_grid(cumulants, N=N, L=L)
    calls = cos_prices(phi, fwd, K, grid).call_prices
    if cp == 1:
        return np.asarray(calls, dtype=np.float64)
    puts = calls - fwd.disc * (fwd.F0 - K)
    return np.asarray(np.maximum(puts, 0.0), dtype=np.float64)


__all__ = ["proj_european_price_at_strikes"]
