"""Variance swap and variance option products."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .base import ProductSpec


@dataclass(frozen=True)
class VarianceSwap(ProductSpec):
    """Variance swap: pays realised variance minus fair strike.

    Parameters
    ----------
    maturity : float
        Time to expiry in years. Must be > 0.
    sampling_times : np.ndarray
        Discrete sampling dates (years) for realised-variance computation.
        All must be in (0, maturity].
    notional : float
        Vega notional (variance notional = notional / (2 * sqrt(fair_vol))).
        Default 1.0.
    """

    product_type: str = field(default="variance_swap", init=False, repr=False)
    maturity: float = 0.0
    sampling_times: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    notional: float = 1.0

    def __post_init__(self) -> None:
        if self.maturity <= 0:
            raise ValueError(f"VarianceSwap: maturity must be > 0, got {self.maturity}")
        t = np.asarray(self.sampling_times)
        if t.ndim != 1 or len(t) == 0:
            raise ValueError("VarianceSwap: sampling_times must be a non-empty 1-D array")
        if np.any(t <= 0):
            raise ValueError("VarianceSwap: all sampling_times must be > 0")
        if np.any(t > self.maturity + 1e-12):
            raise ValueError(
                f"VarianceSwap: all sampling_times must be <= maturity ({self.maturity})"
            )


@dataclass(frozen=True)
class VarianceOption(ProductSpec):
    """Option on realised variance.

    Parameters
    ----------
    strike : float
        Strike on realised variance (annualised). Must be >= 0.
    maturity : float
        Time to expiry in years. Must be > 0.
    cp : int
        +1 variance call, -1 variance put.
    sampling_times : np.ndarray
        Discrete sampling dates for the realised-variance underlying.
    variance_type : {"realised", "integrated"}
        Whether the underlying is the discretely-sampled realised variance
        or the continuously-integrated variance.
    """

    product_type: str = field(default="variance_option", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    sampling_times: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    variance_type: Literal["realised", "integrated"] = "realised"

    def __post_init__(self) -> None:
        if self.strike < 0:
            raise ValueError(f"VarianceOption: strike must be >= 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"VarianceOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"VarianceOption: cp must be +1 or -1, got {self.cp}")
        if self.variance_type not in ("realised", "integrated"):
            raise ValueError(
                f"VarianceOption: variance_type must be 'realised' or 'integrated', "
                f"got {self.variance_type!r}"
            )
