"""Forward-starting option product."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ProductSpec


@dataclass(frozen=True)
class ForwardStartOption(ProductSpec):
    """Forward-starting option: strike is set as ``alpha * S_{t_start}``.

    Parameters
    ----------
    start_time : float
        The date (years from now) at which the strike is determined.
        Must be in [0, maturity).  Zero reduces to a vanilla option with
        strike = alpha * S_0.
    maturity : float
        Final expiry in years. Must be > start_time.
    cp : int
        +1 call, -1 put.
    alpha : float
        Strike-setting ratio: K = alpha * S_{t_start}. Default 1.0 (ATM).
        Must be > 0.
    """

    product_type: str = field(default="forward_start", init=False, repr=False)
    start_time: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.start_time < 0:
            raise ValueError(f"ForwardStartOption: start_time must be >= 0, got {self.start_time}")
        if self.maturity <= 0:
            raise ValueError(f"ForwardStartOption: maturity must be > 0, got {self.maturity}")
        if self.maturity <= self.start_time:
            raise ValueError(
                f"ForwardStartOption: maturity ({self.maturity}) must be > "
                f"start_time ({self.start_time})"
            )
        if self.cp not in (1, -1):
            raise ValueError(f"ForwardStartOption: cp must be +1 or -1, got {self.cp}")
        if self.alpha <= 0:
            raise ValueError(f"ForwardStartOption: alpha must be > 0, got {self.alpha}")

    @property
    def tenor(self) -> float:
        return self.maturity - self.start_time
