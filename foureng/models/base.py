"""Model-layer base types.

Pass 1 scope — this module only holds ``ForwardSpec``, ``ModelSpec``, and
the ``CharFunc`` callable protocol. The ``FourierModelBase`` abstract
class is *declared here as a Pass-2 placeholder* so downstream modules
can begin importing the symbol, but it is deliberately unimplemented
until Pass 2 (backend normalization) when each model gains a concrete
class that inherits from it.

Keeping the placeholder here — instead of adding a second base module
later — means Pass 2 is a pure in-file fill-in, not another rename.
"""
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


class FourierModelBase:
    """Abstract model contract for the PyFENG-compatible façade layer.

    **Pass 1: placeholder.** This class is intentionally empty. It will
    be fleshed out in Pass 2 (backend normalization), when each
    ``models/<model>.py`` gains a concrete class that implements
    ``charfunc_logprice`` and ``cumulants``. Pass 3 (``sv_fft.py``,
    ``sv_cos.py``, ``sv_frft.py`` façades) then inherits from these
    classes.

    Intended Pass-2 surface (documented early so that reviews align):

        class FourierModelBase:
            model_name: str

            def charfunc_logprice(self, u, texp):
                raise NotImplementedError

            def cumulants(self, texp):
                raise NotImplementedError

    Do not use this class in production code yet — the current
    characteristic-function surface is still the free functions
    ``bsm_cf``, ``heston_cf``, etc.
    """

    model_name: str = ""
