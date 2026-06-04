"""SABR approximation pricers."""

from __future__ import annotations

import numpy as np

from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd
from foureng.models.base import ForwardSpec
from foureng.models.sabr import SabrParams, sabr_hagan_implied_vol


def sabr_hagan_price_at_strikes(
    fwd: ForwardSpec,
    params: SabrParams,
    strikes,
    *,
    cp: int = 1,
) -> np.ndarray:
    """Price European options by Hagan SABR implied vol plus Black formula."""
    if cp not in (1, -1):
        raise ValueError(f"cp must be +1 or -1, got {cp}")
    K = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    vols = sabr_hagan_implied_vol(fwd.F0, K, fwd.T, params)
    prices = [
        bs_price_from_fwd(
            float(vol),
            BSInputs(fwd.F0, float(k), fwd.T, fwd.r, fwd.q, is_call=(cp == 1)),
        )
        for vol, k in zip(vols, K)
    ]
    return np.asarray(prices, dtype=np.float64)


__all__ = ["sabr_hagan_price_at_strikes"]
