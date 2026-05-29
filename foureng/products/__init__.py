"""Product abstraction layer.

All supported product dataclasses are importable from this package:

    from foureng.products import EuropeanOption, BarrierOption, AsianOption

Each product is a frozen dataclass that validates its own parameters on
construction and carries a ``product_type`` string used by the capability
registry.
"""

from .american import AmericanOption
from .asian import AsianOption
from .barrier import BarrierOption, DoubleBarrierOption
from .base import ProductSpec
from .bermudan import BermudanOption
from .cliquet import CliquetOption
from .digital import DigitalOption
from .european import EuropeanOption
from .forward_start import ForwardStartOption
from .lookback import LookbackOption
from .multi_asset import BasketOption, BestOfOption, ExchangeOption, SpreadOption
from .variance import VarianceOption, VarianceSwap

__all__ = [
    "ProductSpec",
    "EuropeanOption",
    "DigitalOption",
    "BarrierOption",
    "DoubleBarrierOption",
    "AsianOption",
    "BermudanOption",
    "AmericanOption",
    "LookbackOption",
    "ForwardStartOption",
    "VarianceSwap",
    "VarianceOption",
    "CliquetOption",
    "ExchangeOption",
    "BasketOption",
    "SpreadOption",
    "BestOfOption",
]
