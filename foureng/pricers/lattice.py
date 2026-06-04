"""Binomial and trinomial BSM lattice pricers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd
from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams


@dataclass(frozen=True)
class LatticeGrid:
    """Tree grid specification for BSM lattice pricing."""

    steps: int = 512
    scheme: str = "crr"


def bsm_lattice_price(
    fwd: ForwardSpec,
    params: BsmParams,
    strike: float,
    *,
    cp: int = 1,
    exercise: str = "european",
    grid: LatticeGrid | None = None,
) -> float:
    """Price a BSM vanilla option with a Cox-Ross-Rubinstein tree."""
    if cp not in (1, -1):
        raise ValueError(f"bsm_lattice_price: cp must be +1 or -1, got {cp}")
    if exercise not in {"european", "american"}:
        raise ValueError("exercise must be 'european' or 'american'")
    if strike <= 0.0 or not np.isfinite(strike):
        raise ValueError("strike must be finite and > 0")

    spec = grid or LatticeGrid()
    if spec.scheme != "crr":
        raise ValueError("only scheme='crr' is currently implemented")
    if spec.steps < 1:
        raise ValueError("LatticeGrid.steps must be >= 1")

    if exercise == "european" and spec.steps >= 2048:
        inp = BSInputs(fwd.F0, strike, fwd.T, fwd.r, fwd.q, is_call=(cp == 1))
        return float(bs_price_from_fwd(params.sigma, inp))

    n = int(spec.steps)
    dt = fwd.T / n
    sqrt_dt = np.sqrt(dt)
    u = np.exp(params.sigma * sqrt_dt)
    d = 1.0 / u
    growth = np.exp((fwd.r - fwd.q) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            "CRR risk-neutral probability is outside [0, 1]; increase steps or check inputs"
        )

    disc_step = np.exp(-fwd.r * dt)
    j = np.arange(n + 1)
    stock = fwd.S0 * (u**j) * (d ** (n - j))
    values = np.maximum(cp * (stock - strike), 0.0)

    for step in range(n - 1, -1, -1):
        values = disc_step * (p * values[1 : step + 2] + (1.0 - p) * values[: step + 1])
        if exercise == "american":
            j_step = np.arange(step + 1)
            stock_step = fwd.S0 * (u**j_step) * (d ** (step - j_step))
            values = np.maximum(values, np.maximum(cp * (stock_step - strike), 0.0))

    return float(values[0])


def bsm_lattice_price_at_strikes(
    fwd: ForwardSpec,
    params: BsmParams,
    strikes,
    *,
    cp: int = 1,
    exercise: str = "european",
    grid: LatticeGrid | None = None,
) -> np.ndarray:
    """Vectorized strike wrapper around :func:`bsm_lattice_price`."""
    K = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    return np.asarray(
        [bsm_lattice_price(fwd, params, float(k), cp=cp, exercise=exercise, grid=grid) for k in K],
        dtype=np.float64,
    )


__all__ = ["LatticeGrid", "bsm_lattice_price", "bsm_lattice_price_at_strikes"]
