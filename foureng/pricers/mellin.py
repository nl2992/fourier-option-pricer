"""Mellin-method European pricing façade.

The MATLAB reference project has model-specific Mellin transform files for VG,
NIG, FMLS, NTS, and bilateral gamma.  This first slice exposes a stable
European ``mellin`` method label for those model families and prices through
the same damped probability-inversion interface used for European transform
validation.  Model-specific Mellin contour formulae can replace this backend
incrementally without changing public dispatch.
"""

from __future__ import annotations

import numpy as np

from foureng.models.base import CharFunc, ForwardSpec
from foureng.pricers.conv import conv_price_at_strikes
from foureng.utils.grids import CONVGrid

MELLIN_SUPPORTED_MODELS = frozenset({"vg", "nig", "fmls", "bilateral_gamma"})


def mellin_price_at_strikes(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes,
    *,
    cp: int = 1,
    grid: CONVGrid | None = None,
) -> np.ndarray:
    """Price European options through the Mellin-method public façade."""
    conv_grid = grid or CONVGrid(N=4096, u_max=200.0)
    return np.asarray(conv_price_at_strikes(phi, fwd, conv_grid, strikes, cp=cp), dtype=np.float64)


__all__ = ["MELLIN_SUPPORTED_MODELS", "mellin_price_at_strikes"]
