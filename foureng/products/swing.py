"""Swing (multiple-exercise) option product."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ProductSpec


@dataclass(frozen=True)
class SwingOption(ProductSpec):
    """Swing option: ``n_rights`` vanilla exercises over ``n_exercise_dates``.

    The holder may exercise at most once per date, each exercise paying the
    vanilla intrinsic ``max(cp (S - K), 0)``. ``n_rights = 1`` is a Bermudan
    option; ``n_rights >= n_exercise_dates`` values as the sum of the
    per-date European options.

    Parameters
    ----------
    strike : float
        Exercise strike (> 0).
    maturity : float
        Final exercise date in years (> 0).
    cp : int
        +1 call rights, -1 put rights.
    n_rights : int
        Number of exercise rights (>= 1).
    n_exercise_dates : int
        Number of equally spaced exercise dates over (0, maturity] (>= 1).
    """

    product_type: str = field(default="swing", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    n_rights: int = 1
    n_exercise_dates: int = 12

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"SwingOption: strike must be > 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"SwingOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"SwingOption: cp must be +1 or -1, got {self.cp}")
        if int(self.n_rights) < 1:
            raise ValueError(f"SwingOption: n_rights must be >= 1, got {self.n_rights}")
        if int(self.n_exercise_dates) < 1:
            raise ValueError(
                f"SwingOption: n_exercise_dates must be >= 1, got {self.n_exercise_dates}"
            )
