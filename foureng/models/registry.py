"""Model backend registry.

The authoritative dispatch table lives in ``foureng.pipeline._MODELS`` as
a ``{name: (ParamsClass, cf_fn, cumulants_fn)}`` dict.  This module
exposes ``MODEL_BACKENDS`` as an extension point for future class-based
backends that inherit from :class:`~foureng.models.base.FourierModelBase`.
It is not consulted by the current free-function API and is safe to ignore.
"""
from __future__ import annotations
from typing import Type

from .base import FourierModelBase


MODEL_BACKENDS: dict[str, Type[FourierModelBase]] = {}
"""Name -> backend model class. Reserved for future class-based extensions."""


__all__ = ["MODEL_BACKENDS"]
