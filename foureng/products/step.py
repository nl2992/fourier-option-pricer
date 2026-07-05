"""Step (occupation-time-discounted) option product."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .base import ProductSpec


@dataclass(frozen=True)
class StepOption(ProductSpec):
    """Proportional step option (Linetsky 1999), discretely monitored.

    The vanilla payoff is damped by the time the asset spends beyond the
    barrier: with ``n_beyond`` the number of monitoring dates on which the
    asset is beyond ``barrier`` (below it for ``step_type="down"``, above it
    for ``"up"``) and ``dt = maturity / n_monitoring``,

        payoff = exp(-rho * dt * n_beyond) * vanilla(S_T, K).

    ``rho = 0`` recovers the vanilla; ``rho -> infinity`` recovers the
    discretely monitored knock-out barrier -- step options interpolate
    between the two, avoiding the barrier's discontinuous delta.

    Parameters
    ----------
    strike : float
        Vanilla strike (> 0).
    maturity : float
        Expiry in years (> 0).
    cp : int
        +1 call, -1 put.
    barrier : float
        Step barrier level (> 0).
    rho : float
        Occupation-time damping rate (>= 0, per year beyond the barrier).
    step_type : {"down", "up"}
        Which side of the barrier accrues damping.
    n_monitoring : int
        Number of equally spaced monitoring dates (>= 1); 252 approximates
        continuous monitoring.
    """

    product_type: str = field(default="step", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    barrier: float = 0.0
    rho: float = 0.0
    step_type: Literal["down", "up"] = "down"
    n_monitoring: int = 252

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"StepOption: strike must be > 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"StepOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"StepOption: cp must be +1 or -1, got {self.cp}")
        if self.barrier <= 0:
            raise ValueError(f"StepOption: barrier must be > 0, got {self.barrier}")
        if not (np.isfinite(self.rho) and self.rho >= 0):
            raise ValueError(f"StepOption: rho must be finite and >= 0, got {self.rho}")
        if self.step_type not in ("down", "up"):
            raise ValueError(
                f"StepOption: step_type must be 'down' or 'up', got {self.step_type!r}"
            )
        if int(self.n_monitoring) < 1:
            raise ValueError(f"StepOption: n_monitoring must be >= 1, got {self.n_monitoring}")
