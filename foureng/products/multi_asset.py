"""Multi-asset option products."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import ProductSpec


@dataclass(frozen=True)
class ExchangeOption(ProductSpec):
    """Margrabe exchange option: pays max(S1_T - S2_T, 0).

    Parameters
    ----------
    maturity : float
        Time to expiry in years. Must be > 0.
    spot2 : float
        Current spot of asset 2. Must be > 0.
    q2 : float
        Continuous dividend yield of asset 2.
    sigma2 : float
        Lognormal volatility of asset 2. Must be >= 0.
    rho : float
        Instantaneous correlation between the two asset log returns.
        Must lie in [-1, 1].

    Notes
    -----
    Margrabe (1978) gives a closed-form BSM formula.  Under general
    multi-asset models the price depends on the correlation between
    the two assets.
    """

    product_type: str = field(default="exchange", init=False, repr=False)
    maturity: float = 0.0
    spot2: float = 100.0
    q2: float = 0.0
    sigma2: float = 0.2
    rho: float = 0.0

    def __post_init__(self) -> None:
        if self.maturity <= 0:
            raise ValueError(f"ExchangeOption: maturity must be > 0, got {self.maturity}")
        if self.spot2 <= 0:
            raise ValueError(f"ExchangeOption: spot2 must be > 0, got {self.spot2}")
        if self.sigma2 < 0:
            raise ValueError(f"ExchangeOption: sigma2 must be >= 0, got {self.sigma2}")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"ExchangeOption: rho must lie in [-1, 1], got {self.rho}")


@dataclass(frozen=True)
class BasketOption(ProductSpec):
    """Basket option on a weighted sum of assets.

    Parameters
    ----------
    strike : float
        Strike. Must be > 0.
    maturity : float
        Time to expiry in years. Must be > 0.
    cp : int
        +1 call, -1 put.
    weights : np.ndarray
        Asset weights (must sum to a positive value). Shape (n_assets,).
    """

    product_type: str = field(default="basket", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    weights: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"BasketOption: strike must be > 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"BasketOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"BasketOption: cp must be +1 or -1, got {self.cp}")
        w = np.asarray(self.weights)
        if w.ndim != 1 or len(w) == 0:
            raise ValueError("BasketOption: weights must be a non-empty 1-D array")
        if np.sum(w) <= 0:
            raise ValueError("BasketOption: weights must sum to a positive value")


@dataclass(frozen=True)
class SpreadOption(ProductSpec):
    """Spread option: call on S1_T - S2_T with strike K.

    Parameters
    ----------
    strike : float
        Strike on the spread. May be zero (Kirk approximation / zero-strike case).
        Must be >= 0.
    maturity : float
        Time to expiry in years. Must be > 0.
    cp : int
        +1 call, -1 put.
    """

    product_type: str = field(default="spread", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1

    def __post_init__(self) -> None:
        if self.strike < 0:
            raise ValueError(f"SpreadOption: strike must be >= 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"SpreadOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"SpreadOption: cp must be +1 or -1, got {self.cp}")


@dataclass(frozen=True)
class BestOfOption(ProductSpec):
    """Best-of option: call on max(S1_T, S2_T, ...).

    Parameters
    ----------
    strike : float
        Strike. Must be > 0.
    maturity : float
        Time to expiry in years. Must be > 0.
    cp : int
        +1 call (pays max - K)+, -1 put (pays K - min)+.
    n_assets : int
        Number of underlying assets. Must be >= 2.
    """

    product_type: str = field(default="best_of", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    n_assets: int = 2

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"BestOfOption: strike must be > 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"BestOfOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"BestOfOption: cp must be +1 or -1, got {self.cp}")
        if self.n_assets < 2:
            raise ValueError(f"BestOfOption: n_assets must be >= 2, got {self.n_assets}")
