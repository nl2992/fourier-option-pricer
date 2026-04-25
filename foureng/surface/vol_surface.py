"""Model -> (T, K) price grid -> Black-76 IV grid.

Single-call convenience on top of ``cos_prices`` + ``implied_vol_newton_safeguarded``.
Same-strike strip assumed across maturities; easy to generalize later.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable

from ..models.base import ForwardSpec
from ..iv.implied_vol import BSInputs, implied_vol_newton_safeguarded
from ..pricers.cos import cos_prices, cos_auto_grid


@dataclass(frozen=True)
class SurfaceSpec:
    """Grid of maturities and strikes to evaluate the model on.

    Same strike strip is used across every maturity (simplifies vectorisation;
    can be generalized by evaluating one T at a time with per-T strikes).
    """
    S0: float
    r: float
    q: float
    maturities: np.ndarray          # shape (nT,)
    strikes: np.ndarray             # shape (nK,)


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

    Prices come from ``model_price_surface``; IVs via safeguarded Newton.
    Cells that fail to invert return NaN (likely a degenerate / deep-OTM price).
    """
    prices = model_price_surface(spec, cf_factory, cumulant_factory, N=N, L=L)
    ivs = np.full_like(prices, np.nan)
    for i, T in enumerate(spec.maturities):
        for j, K in enumerate(spec.strikes):
            inp = BSInputs(F0=spec.S0 * np.exp((spec.r - spec.q) * float(T)),
                            K=float(K), T=float(T), r=spec.r, q=spec.q, is_call=True)
            ivs[i, j] = implied_vol_newton_safeguarded(float(prices[i, j]), inp)
    return ivs
