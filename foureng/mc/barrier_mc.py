"""Single-barrier MC pricer with Broadie-Glasserman-Kou (1999) continuity correction."""

from __future__ import annotations

import numpy as np

from ..pricers.analytic_bsm import bsm_european
from .path_engine import GBMPathSpec, simulate_gbm_paths

# BGK continuity-correction constant β ≈ -ζ(1/2) / sqrt(2π)
_BETA = 0.5826


def barrier_mc(
    S0: float,
    K: float,
    H: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_steps: int,
    barrier_type: str,
    rebate: float,
    cp: int,
    spec: GBMPathSpec,
) -> float:
    """Single-barrier option by MC with BGK continuity correction.

    Parameters
    ----------
    S0, K, H, r, q, sigma, T : standard barrier option inputs
    n_steps     : number of time steps
    barrier_type: one of "down_out", "down_in", "up_out", "up_in"
    rebate      : cash paid at expiry on knock-out paths
    cp          : +1 call, -1 put
    spec        : GBMPathSpec

    Returns
    -------
    float
        Discounted barrier option price.
    """
    _valid = {"down_out", "down_in", "up_out", "up_in"}
    if barrier_type not in _valid:
        raise ValueError(f"barrier_type must be one of {_valid}, got {barrier_type!r}")

    # Knock-in via in-out parity: price = vanilla - knock_out
    if barrier_type.endswith("_in"):
        out_type = barrier_type.replace("_in", "_out")
        vanilla = bsm_european(S0, K, r, q, sigma, T, cp)
        ko = barrier_mc(S0, K, H, r, q, sigma, T, n_steps, out_type, rebate=0.0, cp=cp, spec=spec)
        return vanilla - ko

    is_up = barrier_type.startswith("up")

    # BGK shift: discrete barrier → effective barrier for continuous-barrier equivalence
    sign = +1.0 if is_up else -1.0
    H_eff = H * np.exp(sign * _BETA * sigma * np.sqrt(T / n_steps))

    path_spec = GBMPathSpec(
        n_paths=spec.n_paths,
        n_steps=n_steps,
        seed=spec.seed,
        antithetic=spec.antithetic,
        batch_size=spec.batch_size,
    )
    paths = simulate_gbm_paths(S0, r, q, sigma, T, path_spec)

    if is_up:
        # Knocked out when path ever reaches or exceeds H_eff
        knocked = np.any(paths[:, 1:] >= H_eff, axis=1)
    else:
        # Knocked out when path ever drops to or below H_eff
        knocked = np.any(paths[:, 1:] <= H_eff, axis=1)

    S_T = paths[:, -1]
    disc = np.exp(-r * T)

    payoff = np.where(
        knocked,
        rebate,  # knocked out: receive rebate (paid at expiry)
        np.maximum(cp * (S_T - K), 0.0),
    )
    return float(disc * np.mean(payoff))
