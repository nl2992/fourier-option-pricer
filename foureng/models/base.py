"""Model-layer base types.

This module defines the three primitives shared by every model in the package:

* :class:`ForwardSpec` — market inputs (spot, rates, maturity).
* :class:`ModelSpec` — base dataclass for all model parameter classes.
* :class:`CharFunc` — callable protocol for characteristic functions.

:class:`FourierModelBase` is a reserved base class for future class-based
model backends; the current API uses free functions (e.g. ``heston_cf``,
``vg_cf``) and is not affected by it.
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
        Current spot price.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuous dividend yield (or foreign risk-free rate for FX).
    T : float
        Time to maturity in years.
    F0 : float
        Forward price S0 * exp((r - q) * T), computed automatically.
    disc : float
        Discount factor exp(-r * T), computed automatically.
    """
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


class FourierModelBase:
    """Reserved base class for future class-based model backends.

    The current public API uses free functions (``bsm_cf``, ``heston_cf``,
    etc.) and does not require this class.  It is retained as an import
    anchor for ``foureng.models.registry`` and may be fleshed out in a
    future release.
    """

    model_name: str = ""
