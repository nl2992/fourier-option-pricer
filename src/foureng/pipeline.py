from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .char_func.base import ForwardSpec, CharFunc
from .utils.grids import FFTGrid, FRFTGrid, COSGrid
from .pricers.carr_madan import carr_madan_price_at_strikes
from .pricers.frft import frft_price_at_strikes
from .pricers.cos import cos_prices


@dataclass(frozen=True)
class PhaseOutputs:
    strikes: np.ndarray
    prices: np.ndarray


def phase2_carr_madan(
    phi: CharFunc, fwd: ForwardSpec, strikes: np.ndarray, grid: FFTGrid
) -> PhaseOutputs:
    prices = carr_madan_price_at_strikes(phi, fwd, grid, strikes)
    return PhaseOutputs(strikes=np.asarray(strikes, float), prices=prices)


def phase3_frft(
    phi: CharFunc, fwd: ForwardSpec, strikes: np.ndarray, grid: FRFTGrid
) -> PhaseOutputs:
    prices = frft_price_at_strikes(phi, fwd, grid, strikes)
    return PhaseOutputs(strikes=np.asarray(strikes, float), prices=prices)


def phase4_cos(
    phi: CharFunc, fwd: ForwardSpec, strikes: np.ndarray, grid: COSGrid
) -> PhaseOutputs:
    res = cos_prices(phi, fwd, strikes, grid)
    return PhaseOutputs(strikes=res.strikes, prices=res.call_prices)
