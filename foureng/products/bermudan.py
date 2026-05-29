"""Bermudan option product."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import ProductSpec


@dataclass(frozen=True)
class BermudanOption(ProductSpec):
    """Bermudan put or call with a finite set of exercise dates.

    Parameters
    ----------
    strike : float
        Strike price. Must be > 0.
    maturity : float
        Final expiry in years. Must be > 0.
    cp : int
        +1 call, -1 put.
    exercise_times : np.ndarray
        Sorted early-exercise dates (years). All must be in (0, maturity].
        At least one exercise date is required.

    Notes
    -----
    * M = 1 (a single exercise date equal to maturity) reduces to a European.
    * For the COS Bermudan pricer (Fang & Oosterlee 2009) only models with a
      one-dimensional Markov state are supported without modification; Heston
      Bermudan COS requires the 2-D extension.
    """

    product_type: str = field(default="bermudan", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    exercise_times: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"BermudanOption: strike must be > 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"BermudanOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"BermudanOption: cp must be +1 or -1, got {self.cp}")
        t = np.asarray(self.exercise_times)
        if t.ndim != 1 or len(t) == 0:
            raise ValueError("BermudanOption: exercise_times must be a non-empty 1-D array")
        if np.any(t <= 0):
            raise ValueError("BermudanOption: all exercise_times must be > 0")
        if np.any(t > self.maturity + 1e-12):
            raise ValueError(
                f"BermudanOption: all exercise_times must be <= maturity ({self.maturity})"
            )
        if not np.all(np.diff(t) > 0):
            raise ValueError("BermudanOption: exercise_times must be strictly increasing")

    @property
    def n_exercise(self) -> int:
        return len(self.exercise_times)

    @property
    def is_european_equivalent(self) -> bool:
        """True when there is exactly one exercise date equal to maturity."""
        return (
            self.n_exercise == 1
            and abs(float(self.exercise_times[-1]) - self.maturity) < 1e-12
        )
