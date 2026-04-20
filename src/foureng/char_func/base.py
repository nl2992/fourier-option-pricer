from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class ForwardSpec:
    """Deterministic forward and discount inputs."""
    S0: float
    r: float
    q: float
    T: float

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
