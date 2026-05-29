"""American option product."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ProductSpec


@dataclass(frozen=True)
class AmericanOption(ProductSpec):
    """American-exercise put or call.

    Parameters
    ----------
    strike : float
        Strike price. Must be > 0.
    maturity : float
        Time to expiry in years. Must be > 0.
    cp : int
        +1 call, -1 put.

    Notes
    -----
    For a non-dividend-paying underlying, ``cp=+1`` (American call) has the
    same value as the corresponding European call.  This is a well-known
    structural property that should be verified by any implementing pricer.
    """

    product_type: str = field(default="american", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"AmericanOption: strike must be > 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"AmericanOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"AmericanOption: cp must be +1 or -1, got {self.cp}")
