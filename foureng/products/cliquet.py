"""Cliquet (ratchet) and equity-indexed annuity product."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .base import ProductSpec


@dataclass(frozen=True)
class CliquetOption(ProductSpec):
    """Cliquet (ratchet) option: sum or product of capped/floored period returns.

    Parameters
    ----------
    maturity : float
        Total maturity in years. Must be > 0.
    reset_times : np.ndarray
        Dates at which each period return is measured.  The first period runs
        from 0 to reset_times[0]; the last period ends at maturity.
        All values must be in (0, maturity].
    local_floor : float
        Per-period minimum return. Default -inf (no floor).
    local_cap : float
        Per-period maximum return. Default +inf (no cap).
    global_floor : float
        Minimum total payoff. Default -inf.
    global_cap : float
        Maximum total payoff. Default +inf.
    payoff_type : {"additive", "multiplicative"}
        How period returns are combined.
    cp : int
        +1 for cliquet call (receives sum of positive returns), -1 for put.
        Affects sign convention only for ``payoff_type="additive"``.
    """

    product_type: str = field(default="cliquet", init=False, repr=False)
    maturity: float = 0.0
    reset_times: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    local_floor: float = float("-inf")
    local_cap: float = float("inf")
    global_floor: float = float("-inf")
    global_cap: float = float("inf")
    payoff_type: Literal["additive", "multiplicative"] = "additive"
    cp: int = 1

    def __post_init__(self) -> None:
        if self.maturity <= 0:
            raise ValueError(f"CliquetOption: maturity must be > 0, got {self.maturity}")
        t = np.asarray(self.reset_times)
        if t.ndim != 1 or len(t) == 0:
            raise ValueError("CliquetOption: reset_times must be a non-empty 1-D array")
        if np.any(t <= 0):
            raise ValueError("CliquetOption: all reset_times must be > 0")
        if np.any(t > self.maturity + 1e-12):
            raise ValueError(
                f"CliquetOption: all reset_times must be <= maturity ({self.maturity})"
            )
        if not np.all(np.diff(t) > 0):
            raise ValueError("CliquetOption: reset_times must be strictly increasing")
        if self.local_cap < self.local_floor:
            raise ValueError(
                f"CliquetOption: local_cap ({self.local_cap}) must be >= "
                f"local_floor ({self.local_floor})"
            )
        if self.global_cap < self.global_floor:
            raise ValueError(
                f"CliquetOption: global_cap ({self.global_cap}) must be >= "
                f"global_floor ({self.global_floor})"
            )
        if self.payoff_type not in ("additive", "multiplicative"):
            raise ValueError(
                f"CliquetOption: payoff_type must be 'additive' or 'multiplicative', "
                f"got {self.payoff_type!r}"
            )
        if self.cp not in (1, -1):
            raise ValueError(f"CliquetOption: cp must be +1 or -1, got {self.cp}")
