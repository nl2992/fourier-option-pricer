"""Model registry for the PyFENG-compatible façade layer.

**Pass 1: placeholder.** Today's dispatch lives in ``foureng.pipeline``
as a dict of ``(cf, cumulants, params_type)`` tuples keyed by model
name. In Pass 4 (pipeline migration) this module will own the
authoritative ``{name -> model class}`` registry so that
``foureng.sv_fft.HestonFft``, ``foureng.sv_cos.HestonCos``,
``foureng.sv_frft.HestonFrft`` and the ``price_strip`` entry point
share a single source of truth.

Populated in Pass 4; empty on purpose today.
"""
from __future__ import annotations
from typing import Dict, Type

from .base import FourierModelBase


MODEL_BACKENDS: Dict[str, Type[FourierModelBase]] = {}
"""Name -> backend model class. Populated by ``models/__init__.py`` in Pass 4."""


__all__ = ["MODEL_BACKENDS"]
