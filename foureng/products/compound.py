"""Compound option product specification."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ProductSpec


@dataclass(frozen=True)
class CompoundOption(ProductSpec):
    """An option on an option (Geske 1979).

    Parameters
    ----------
    strike_outer : float
        Strike of the compound option itself (paid at T1).
    strike_inner : float
        Strike of the underlying option (paid at T2 if exercised).
    maturity_outer : float
        Expiry of the compound option (T1).  Must be < maturity_inner.
    maturity_inner : float
        Expiry of the underlying option (T2).
    cp_outer : int
        +1  = right to buy the inner option (call-on-call / call-on-put)
        -1  = right to sell the inner option (put-on-call / put-on-put)
    cp_inner : int
        +1 = underlying option is a call; -1 = underlying option is a put.
    """

    product_type: str = field(default="compound", init=False, repr=False)
    strike_outer: float = 0.0
    strike_inner: float = 0.0
    maturity_outer: float = 0.0
    maturity_inner: float = 0.0
    cp_outer: int = 1
    cp_inner: int = 1

    def __post_init__(self) -> None:
        if self.strike_outer < 0:
            raise ValueError(f"strike_outer must be ≥ 0, got {self.strike_outer}")
        if self.strike_inner <= 0:
            raise ValueError(f"strike_inner must be > 0, got {self.strike_inner}")
        if self.maturity_outer <= 0:
            raise ValueError(f"maturity_outer must be > 0, got {self.maturity_outer}")
        if self.maturity_inner <= 0:
            raise ValueError(f"maturity_inner must be > 0, got {self.maturity_inner}")
        if self.maturity_outer >= self.maturity_inner:
            raise ValueError(
                f"maturity_outer ({self.maturity_outer}) must be < maturity_inner ({self.maturity_inner})"
            )
        if self.cp_outer not in (1, -1):
            raise ValueError(f"cp_outer must be +1 or -1, got {self.cp_outer}")
        if self.cp_inner not in (1, -1):
            raise ValueError(f"cp_inner must be +1 or -1, got {self.cp_inner}")
