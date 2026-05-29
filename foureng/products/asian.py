"""Asian option products."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .base import ProductSpec


@dataclass(frozen=True)
class AsianOption(ProductSpec):
    """Discretely-monitored Asian (average-rate) option.

    Parameters
    ----------
    strike : float
        Strike (for fixed-strike Asian). Must be > 0.
    maturity : float
        Time to expiry in years. Must be > 0.
    cp : int
        +1 call, -1 put.
    average_type : {"arithmetic", "geometric"}
        Type of average.
    monitoring_times : np.ndarray
        Sorted monitoring dates in years. All must be in (0, maturity].
    strike_type : {"fixed", "floating"}
        Fixed-strike Asian uses the explicit ``strike``; floating-strike
        Asian uses the terminal spot as strike (alpha * average).
    alpha : float
        Moneyness factor for floating-strike contracts. Default 1.0.
    """

    product_type: str = field(default="asian", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    average_type: Literal["arithmetic", "geometric"] = "arithmetic"
    monitoring_times: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    strike_type: Literal["fixed", "floating"] = "fixed"
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.strike <= 0 and self.strike_type == "fixed":
            raise ValueError(f"AsianOption: strike must be > 0 for fixed-strike, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"AsianOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"AsianOption: cp must be +1 or -1, got {self.cp}")
        if self.average_type not in ("arithmetic", "geometric"):
            raise ValueError(
                f"AsianOption: average_type must be 'arithmetic' or 'geometric', "
                f"got {self.average_type!r}"
            )
        t = np.asarray(self.monitoring_times)
        if t.ndim != 1 or len(t) == 0:
            raise ValueError("AsianOption: monitoring_times must be a non-empty 1-D array")
        if np.any(t <= 0):
            raise ValueError("AsianOption: all monitoring_times must be > 0")
        if np.any(t > self.maturity + 1e-12):
            raise ValueError(
                f"AsianOption: all monitoring_times must be <= maturity ({self.maturity})"
            )
        if not np.all(np.diff(t) > 0):
            raise ValueError("AsianOption: monitoring_times must be strictly increasing")
        if self.alpha <= 0:
            raise ValueError(f"AsianOption: alpha must be > 0, got {self.alpha}")
