"""Model -> (T, K) price grid -> Black-76 IV grid.

Single-call convenience on top of ``cos_prices`` + ``implied_vol_lets_be_rational``.
Same-strike strip assumed across maturities; easy to generalize later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..iv.lets_be_rational import implied_vol_lets_be_rational
from ..models.base import ForwardSpec
from ..pricers.cos import cos_auto_grid, cos_prices


@dataclass(frozen=True)
class SurfaceSpec:
    """Grid of maturities and strikes to evaluate the model on.

    Same strike strip is used across every maturity (simplifies vectorisation;
    can be generalized by evaluating one T at a time with per-T strikes).
    """

    S0: float
    r: float
    q: float
    maturities: np.ndarray  # shape (nT,)
    strikes: np.ndarray  # shape (nK,)


def model_price_surface(
    spec: SurfaceSpec,
    cf_factory: Callable[[ForwardSpec], Callable],
    cumulant_factory: Callable[[ForwardSpec], tuple[float, float, float]],
    N: int = 256,
    L: float = 10.0,
) -> np.ndarray:
    """Compute a (nT, nK) grid of European call prices.

    cf_factory(fwd)       -> phi (a CharFunc) for that maturity
    cumulant_factory(fwd) -> (c1, c2, c4) for COS truncation
    """
    mats = np.asarray(spec.maturities, dtype=float)
    if np.any(mats <= 0.0):
        raise ValueError(f"All maturities must be > 0; got {mats[mats <= 0].tolist()}")
    nT = len(spec.maturities)
    nK = len(spec.strikes)
    out = np.empty((nT, nK), dtype=float)
    for i, T in enumerate(spec.maturities):
        fwd = ForwardSpec(S0=spec.S0, r=spec.r, q=spec.q, T=float(T))
        phi = cf_factory(fwd)
        cums = cumulant_factory(fwd)
        grid = cos_auto_grid(cums, N=N, L=L)
        out[i, :] = cos_prices(phi, fwd, spec.strikes, grid).call_prices
    return out


def model_iv_surface(
    spec: SurfaceSpec,
    cf_factory: Callable[[ForwardSpec], Callable],
    cumulant_factory: Callable[[ForwardSpec], tuple[float, float, float]],
    N: int = 256,
    L: float = 10.0,
) -> np.ndarray:
    """Compute a (nT, nK) Black-76 implied-vol grid.

    Prices come from ``model_price_surface``; IVs from the vectorised
    machine-precision inversion of Jäckel (2015) over the whole grid at once.
    Cells that fail to invert return NaN (a price outside the no-arbitrage
    bounds, typically numerical noise on a degenerate deep-ITM/OTM quote).
    """
    mats = np.asarray(spec.maturities, dtype=float)
    if np.any(mats <= 0.0):
        raise ValueError(f"All maturities must be > 0; got {mats[mats <= 0].tolist()}")
    prices = model_price_surface(spec, cf_factory, cumulant_factory, N=N, L=L)
    T = mats[:, None]
    K = np.asarray(spec.strikes, dtype=float)[None, :]
    F = spec.S0 * np.exp((spec.r - spec.q) * T)
    return implied_vol_lets_be_rational(prices, F, K, T, disc=np.exp(-spec.r * T), cp=1)
