"""Chooser option product specification."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ProductSpec


@dataclass(frozen=True)
class ChooserOption(ProductSpec):
    """Simple chooser option (Rubinstein 1991).

    At time T_choice the holder decides whether the contract becomes a call
    or a put, both with the same strike K and expiry T_exp > T_choice.

    Parameters
    ----------
    strike : float
        Common strike for the call and put.
    maturity_choice : float
        Date at which the holder chooses (T_choice).  Must be < maturity_expiry.
    maturity_expiry : float
        Expiry of the resulting call or put (T_exp).
    """

    product_type: str = field(default="chooser", init=False, repr=False)
    strike: float = 0.0
    maturity_choice: float = 0.0
    maturity_expiry: float = 0.0

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"strike must be > 0, got {self.strike}")
        if self.maturity_choice <= 0:
            raise ValueError(f"maturity_choice must be > 0, got {self.maturity_choice}")
        if self.maturity_expiry <= 0:
            raise ValueError(f"maturity_expiry must be > 0, got {self.maturity_expiry}")
        if self.maturity_choice >= self.maturity_expiry:
            raise ValueError(
                f"maturity_choice ({self.maturity_choice}) must be < maturity_expiry ({self.maturity_expiry})"
            )
