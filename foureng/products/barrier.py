"""Barrier option products (single and double barrier)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .base import ProductSpec


@dataclass(frozen=True)
class BarrierOption(ProductSpec):
    """Single-barrier option (knock-in or knock-out).

    Parameters
    ----------
    strike : float
        Strike price. Must be > 0.
    barrier : float
        Barrier level. Must be > 0.
    maturity : float
        Time to expiry in years. Must be > 0.
    cp : int
        +1 call, -1 put.
    barrier_type : {"down_out", "up_out", "down_in", "up_in"}
        Direction and knockout/knockin designation.
    rebate : float
        Cash paid at the moment of barrier breach (for knock-outs) or at
        maturity if barrier is never hit (for knock-ins). Default 0.
    monitoring : {"continuous", "discrete"}
        Whether the barrier is checked continuously or at discrete dates.
    """

    product_type: str = field(default="barrier", init=False, repr=False)
    strike: float = 0.0
    barrier: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    barrier_type: Literal["down_out", "up_out", "down_in", "up_in"] = "down_out"
    rebate: float = 0.0
    monitoring: Literal["continuous", "discrete"] = "continuous"

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"BarrierOption: strike must be > 0, got {self.strike}")
        if self.barrier <= 0:
            raise ValueError(f"BarrierOption: barrier must be > 0, got {self.barrier}")
        if self.maturity <= 0:
            raise ValueError(f"BarrierOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"BarrierOption: cp must be +1 or -1, got {self.cp}")
        _valid_bt = {"down_out", "up_out", "down_in", "up_in"}
        if self.barrier_type not in _valid_bt:
            raise ValueError(
                f"BarrierOption: barrier_type must be one of {_valid_bt}, got {self.barrier_type!r}"
            )
        if self.monitoring not in ("continuous", "discrete"):
            raise ValueError(
                f"BarrierOption: monitoring must be 'continuous' or 'discrete', "
                f"got {self.monitoring!r}"
            )
        if self.rebate < 0:
            raise ValueError(f"BarrierOption: rebate must be >= 0, got {self.rebate}")
        # Semantic consistency: down-barriers must be below strike, up-barriers above.
        # (These are warnings rather than hard errors in many implementations;
        # we enforce them to catch accidental parameter swaps.)
        if self.barrier_type.startswith("down") and self.barrier >= self.strike:
            raise ValueError(
                f"BarrierOption: down-barrier ({self.barrier}) must be < strike "
                f"({self.strike}) for a standard down-barrier contract"
            )
        if self.barrier_type.startswith("up") and self.barrier <= self.strike:
            raise ValueError(
                f"BarrierOption: up-barrier ({self.barrier}) must be > strike "
                f"({self.strike}) for a standard up-barrier contract"
            )

    @property
    def is_knockout(self) -> bool:
        return self.barrier_type.endswith("_out")

    @property
    def is_knockin(self) -> bool:
        return self.barrier_type.endswith("_in")


@dataclass(frozen=True)
class DoubleBarrierOption(ProductSpec):
    """Double-barrier option with lower and upper barriers.

    Parameters
    ----------
    strike : float
        Strike price. Must be > 0.
    lower_barrier : float
        Lower barrier. Must be > 0 and < upper_barrier.
    upper_barrier : float
        Upper barrier. Must be > lower_barrier.
    maturity : float
        Time to expiry in years. Must be > 0.
    cp : int
        +1 call, -1 put.
    knockout : bool
        ``True`` for double-knock-out; ``False`` for double-knock-in.
    rebate : float
        Cash paid on breach. Default 0.
    monitoring : {"continuous", "discrete"}
        Monitoring convention.
    """

    product_type: str = field(default="double_barrier", init=False, repr=False)
    strike: float = 0.0
    lower_barrier: float = 0.0
    upper_barrier: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    knockout: bool = True
    rebate: float = 0.0
    monitoring: Literal["continuous", "discrete"] = "continuous"

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"DoubleBarrierOption: strike must be > 0, got {self.strike}")
        if self.lower_barrier <= 0:
            raise ValueError(
                f"DoubleBarrierOption: lower_barrier must be > 0, got {self.lower_barrier}"
            )
        if self.upper_barrier <= self.lower_barrier:
            raise ValueError(
                f"DoubleBarrierOption: upper_barrier ({self.upper_barrier}) must be > "
                f"lower_barrier ({self.lower_barrier})"
            )
        if self.maturity <= 0:
            raise ValueError(f"DoubleBarrierOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"DoubleBarrierOption: cp must be +1 or -1, got {self.cp}")
        if self.rebate < 0:
            raise ValueError(f"DoubleBarrierOption: rebate must be >= 0, got {self.rebate}")
