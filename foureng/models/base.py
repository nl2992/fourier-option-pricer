"""Model-layer base types.

This module defines the three primitives shared by every model in the package:

* :class:`ForwardSpec` — market inputs (spot, rates, maturity).
* :class:`ModelSpec` — base dataclass for all model parameter classes.
* :class:`CharFunc` — callable protocol for characteristic functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ForwardSpec:
    """Market inputs for a European option in forward-measure form.

    Attributes
    ----------
    S0 : float
        Current spot price. Must be strictly positive.
    r : float
        Continuously compounded risk-free rate. Must be finite.
    q : float
        Continuous dividend yield (or foreign risk-free rate for FX). Must be finite.
    T : float
        Time to maturity in years. Must be non-negative.
    F0 : float
        Forward price S0 * exp((r - q) * T), computed automatically.
    disc : float
        Discount factor exp(-r * T), computed automatically.
    """

    S0: float
    r: float
    q: float
    T: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.S0) and self.S0 > 0):
            raise ValueError(f"ForwardSpec: S0 must be finite and > 0; got {self.S0}")
        if not np.isfinite(self.r):
            raise ValueError(f"ForwardSpec: r must be finite; got {self.r}")
        if not np.isfinite(self.q):
            raise ValueError(f"ForwardSpec: q must be finite; got {self.q}")
        if not (np.isfinite(self.T) and self.T >= 0):
            raise ValueError(f"ForwardSpec: T must be finite and >= 0; got {self.T}")

    @property
    def F0(self) -> float:
        return self.S0 * np.exp((self.r - self.q) * self.T)

    @property
    def disc(self) -> float:
        return float(np.exp(-self.r * self.T))


@dataclass(frozen=True)
class ModelSpec:
    name: str


class CharFunc(Protocol):
    """CF of the log-return X_T = log(S_T/F0) under Q.

    phi(u) := E^Q[ exp(i u X_T) ]
    """

    def __call__(self, u: np.ndarray) -> np.ndarray: ...
