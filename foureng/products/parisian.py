"""Parisian option product specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .base import ProductSpec


@dataclass(frozen=True)
class ParisianOption(ProductSpec):
    """Parisian option: barrier activated or cancelled by an excursion of length D.

    A Parisian barrier condition is triggered when the underlying spends a
    specified *consecutive* time D above (up-Parisian) or below (down-Parisian)
    a barrier level H.  This contrasts with a standard barrier option which is
    triggered by the first touch.

    Two sub-types are supported:

    * **standard** (resetting): the excursion clock resets each time the path
      re-crosses the barrier.  The condition fires when any single excursion
      exceeds D.
    * **cumulative**: the clock accumulates total time spent past the barrier
      (no resetting).  The condition fires when the total exceeds D.

    Parameters
    ----------
    strike : float
        Strike price. Must be > 0.
    barrier : float
        Barrier level H. Must be > 0.
    maturity : float
        Time to expiry in years. Must be > 0.
    excursion_window : float
        Parisian window D in years.  Must satisfy 0 < D < maturity.
    cp : int
        +1 call, -1 put.
    direction : {"up", "down"}
        "down" — excursion clock runs when S_t < H (barrier below spot).
        "up"   — excursion clock runs when S_t > H (barrier above spot).
    knockout : bool
        True  → option is cancelled when the Parisian condition fires.
        False → option is activated when the Parisian condition fires.
    parisian_type : {"standard", "cumulative"}
        Whether the excursion clock resets on each barrier re-crossing.
    rebate : float
        Cash paid immediately when a knock-out fires. Default 0.

    Notes
    -----
    In-out parity holds exactly:
        knockout_price + knockin_price = vanilla_price
    """

    product_type: str = field(default="parisian", init=False, repr=False)
    strike: float = 0.0
    barrier: float = 0.0
    maturity: float = 0.0
    excursion_window: float = 0.0
    cp: int = 1
    direction: Literal["up", "down"] = "down"
    knockout: bool = True
    parisian_type: Literal["standard", "cumulative"] = "standard"
    rebate: float = 0.0

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"ParisianOption: strike must be > 0, got {self.strike}")
        if self.barrier <= 0:
            raise ValueError(f"ParisianOption: barrier must be > 0, got {self.barrier}")
        if self.maturity <= 0:
            raise ValueError(f"ParisianOption: maturity must be > 0, got {self.maturity}")
        if not (0 < self.excursion_window < self.maturity):
            raise ValueError(
                f"ParisianOption: excursion_window must be in (0, maturity), "
                f"got {self.excursion_window} (maturity={self.maturity})"
            )
        if self.cp not in (1, -1):
            raise ValueError(f"ParisianOption: cp must be +1 or -1, got {self.cp}")
        if self.direction not in ("up", "down"):
            raise ValueError(
                f"ParisianOption: direction must be 'up' or 'down', got {self.direction!r}"
            )
        if self.parisian_type not in ("standard", "cumulative"):
            raise ValueError(
                f"ParisianOption: parisian_type must be 'standard' or 'cumulative', "
                f"got {self.parisian_type!r}"
            )
        if self.rebate < 0:
            raise ValueError(f"ParisianOption: rebate must be >= 0, got {self.rebate}")

    @property
    def is_knockout(self) -> bool:
        return self.knockout

    @property
    def is_knockin(self) -> bool:
        return not self.knockout
