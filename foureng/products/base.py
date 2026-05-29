"""Base types for the product abstraction layer.

Every concrete product class in ``foureng.products`` is a frozen dataclass
that inherits from :class:`ProductSpec`.  The ``product_type`` class variable
identifies the product family for the capability registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductSpec:
    """Root base class for all product specifications.

    Subclasses must set ``product_type`` to the canonical string used by the
    capability registry (e.g. ``"european"``, ``"barrier"``, ``"asian"``).
    """

    product_type: str = "base"
