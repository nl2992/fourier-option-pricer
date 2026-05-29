"""Lookback option products."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .base import ProductSpec


@dataclass(frozen=True)
class LookbackOption(ProductSpec):
    """Fixed- or floating-strike lookback option.

    Parameters
    ----------
    maturity : float
        Time to expiry in years. Must be > 0.
    cp : int
        +1 call, -1 put.
    strike_type : {"fixed", "floating"}
        * ``"fixed"``    — payoff = max(M - K, 0) for call, max(K - m, 0) for put
          where M = max S_t, m = min S_t.
        * ``"floating"`` — payoff = S_T - m for call, M - S_T for put.
    strike : float
        Strike for fixed-strike contracts. Ignored for floating-strike.
    monitoring : {"continuous", "discrete"}
        Whether the running extremum is monitored continuously or at set dates.
    """

    product_type: str = field(default="lookback", init=False, repr=False)
    maturity: float = 0.0
    cp: int = 1
    strike_type: Literal["fixed", "floating"] = "floating"
    strike: float = 0.0
    monitoring: Literal["continuous", "discrete"] = "continuous"

    def __post_init__(self) -> None:
        if self.maturity <= 0:
            raise ValueError(f"LookbackOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"LookbackOption: cp must be +1 or -1, got {self.cp}")
        if self.strike_type not in ("fixed", "floating"):
            raise ValueError(
                f"LookbackOption: strike_type must be 'fixed' or 'floating', "
                f"got {self.strike_type!r}"
            )
        if self.strike_type == "fixed" and self.strike <= 0:
            raise ValueError(
                f"LookbackOption: strike must be > 0 for fixed-strike, got {self.strike}"
            )
        if self.monitoring not in ("continuous", "discrete"):
            raise ValueError(
                f"LookbackOption: monitoring must be 'continuous' or 'discrete', "
                f"got {self.monitoring!r}"
            )
