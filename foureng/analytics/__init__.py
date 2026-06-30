"""Analytic closed-form pricers for exotic options.

Single-barrier BSM (Reiner-Rubinstein / Haug 2007):
    bsm_barrier_price   -- down-and-out/in, up-and-out/in calls and puts

Double-barrier BSM (eigenfunction expansion, Kunitomo-Ikeda / Haug 2007):
    bsm_double_barrier_price  -- DKO/DKI calls and puts with two absorbing barriers

Asian options (BSM geometric/arithmetic approximations):
    See bsm_asian module.

Exotic BSM (binary, lookback, etc.):
    See bsm_exotics module.
"""

from .bsm_barrier import (
    bsm_barrier_price,
    bsm_call,
    bsm_double_barrier_price,
    bsm_put,
)

__all__ = [
    "bsm_call",
    "bsm_put",
    "bsm_barrier_price",
    "bsm_double_barrier_price",
]
