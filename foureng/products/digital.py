"""Digital (binary) option products."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .base import ProductSpec


@dataclass(frozen=True)
class DigitalOption(ProductSpec):
    """Cash-or-nothing / asset-or-nothing digital option.

    Parameters
    ----------
    strike : float
        Strike price. Must be > 0.
    maturity : float
        Time to expiry in years. Must be > 0.
    cp : int
        +1 for call (pays if S_T > K), -1 for put (pays if S_T < K).
    payoff_type : {"cash_or_nothing", "asset_or_nothing"}
        ``"cash_or_nothing"`` pays 1 unit of cash; ``"asset_or_nothing"``
        pays 1 unit of the underlying.
    cash_amount : float
        Cash paid for ``"cash_or_nothing"`` contracts. Default 1.0.
    """

    product_type: str = field(default="digital", init=False, repr=False)
    strike: float = 0.0
    maturity: float = 0.0
    cp: int = 1
    payoff_type: Literal["cash_or_nothing", "asset_or_nothing"] = "cash_or_nothing"
    cash_amount: float = 1.0

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"DigitalOption: strike must be > 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"DigitalOption: maturity must be > 0, got {self.maturity}")
        if self.cp not in (1, -1):
            raise ValueError(f"DigitalOption: cp must be +1 or -1, got {self.cp}")
        if self.payoff_type not in ("cash_or_nothing", "asset_or_nothing"):
            raise ValueError(
                f"DigitalOption: payoff_type must be 'cash_or_nothing' or "
                f"'asset_or_nothing', got {self.payoff_type!r}"
            )
        if self.cash_amount <= 0:
            raise ValueError(
                f"DigitalOption: cash_amount must be > 0, got {self.cash_amount}"
            )
