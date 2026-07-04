"""Fader (faded-notional) option product."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .base import ProductSpec


@dataclass(frozen=True)
class FaderOption(ProductSpec):
    """Fader option: vanilla payoff scaled by the range occupation fraction.

    The notional accrues over discrete monitoring dates: with ``n_in`` the
    number of dates on which ``lower < S_t < upper``,

        fade-in :  payoff = (n_in / M)       * vanilla(S_T, K)
        fade-out:  payoff = ((M - n_in) / M) * vanilla(S_T, K)

    so fade-in + fade-out = vanilla for the same range and dates.

    Parameters
    ----------
    strike : float
        Vanilla strike. Must be > 0.
    maturity : float
        Expiry in years. Must be > 0.
    cp : int
        +1 call, -1 put.
    lower, upper : float
        Fade range bounds, 0 < lower < upper.
    monitoring_times : np.ndarray
        Strictly increasing dates in (0, maturity].
    fade_type : {"in", "out"}
        Fade-in accrues notional inside the range; fade-out outside.
    """

    product_type: str = field(default="fader", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    lower: float = 0.0
    upper: float = 0.0
    monitoring_times: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    fade_type: Literal["in", "out"] = "in"

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"FaderOption: strike must be > 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"FaderOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"FaderOption: cp must be +1 or -1, got {self.cp}")
        if not (0.0 < self.lower < self.upper):
            raise ValueError(
                f"FaderOption: need 0 < lower < upper; got lower={self.lower}, upper={self.upper}"
            )
        t = np.asarray(self.monitoring_times)
        if t.ndim != 1 or len(t) == 0:
            raise ValueError("FaderOption: monitoring_times must be a non-empty 1-D array")
        if np.any(t <= 0):
            raise ValueError("FaderOption: all monitoring_times must be > 0")
        if np.any(t > self.maturity + 1e-12):
            raise ValueError(
                f"FaderOption: all monitoring_times must be <= maturity ({self.maturity})"
            )
        if not np.all(np.diff(t) > 0):
            raise ValueError("FaderOption: monitoring_times must be strictly increasing")
        if self.fade_type not in ("in", "out"):
            raise ValueError(
                f"FaderOption: fade_type must be 'in' or 'out', got {self.fade_type!r}"
            )
