"""European vanilla option product."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ProductSpec


@dataclass(frozen=True)
class EuropeanOption(ProductSpec):
    """Plain-vanilla European call or put.

    Parameters
    ----------
    strike : float
        Strike price. Must be > 0.
    maturity : float
        Time to expiry in years. Must be > 0.
    cp : int
        +1 for call, -1 for put.
    """

    product_type: str = field(default="european", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"EuropeanOption: strike must be > 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"EuropeanOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"EuropeanOption: cp must be +1 or -1, got {self.cp}")
