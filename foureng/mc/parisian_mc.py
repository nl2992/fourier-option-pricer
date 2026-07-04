"""Monte Carlo pricer for Parisian options under GBM.

A Parisian barrier condition fires when the underlying spends a consecutive
time D (standard/resetting) or total cumulative time D (cumulative variant)
past a barrier H.

Reference: Chesney, Jeanblanc-Picqué & Yor (1997), "Brownian excursions and
Parisian barrier options."  Monte Carlo approach from Bernard, Le Courtois &
Quittard-Pinon (2005).

Implementation notes
--------------------
* Discrete monitoring: the excursion clock is tracked at each GBM time step.
  For accurate continuous-barrier emulation use n_steps ≥ 1000 (the
  discretisation error is O(1/sqrt(n_steps))).
* The Brownian-bridge correction for GBM is available via
  ``use_bb_correction=True`` and provides a first-order improvement for
  coarser grids.  It approximates the probability that the path crossed
  the barrier *between* two successive grid points during any sub-interval.
  BB correction applies only to the standard (resetting) type.
* For the **standard** Parisian, the excursion timer resets each time the
  path returns through H; the option is triggered by the first excursion
  that exceeds D.
* For the **cumulative** Parisian, the timer accumulates total time past H
  throughout [0, T].
"""

from __future__ import annotations

import numpy as np

from ..products.parisian import ParisianOption
from .paths import gbm_paths


def _excursion_triggered_standard(
    paths: np.ndarray,
    H: float,
    D: float,
    dt: float,
    direction: str,
    use_bb: bool,
    sigma: float,
) -> np.ndarray:
    """Return boolean array (n_paths,): True if standard Parisian condition fired."""
    n_paths, n_steps_p1 = paths.shape
    n_steps = n_steps_p1 - 1

    # Indicator: is path on the "excursion side" of H at each step?
    if direction == "down":
        on_side = paths < H  # (n_paths, n_steps+1)
    else:
        on_side = paths > H

    excursion_time = np.zeros(n_paths)
    triggered = np.zeros(n_paths, dtype=bool)

    for j in range(1, n_steps + 1):
        currently_on_side = on_side[:, j]

        if use_bb and sigma > 0:
            # Brownian-bridge correction: estimate probability that the path
            # crossed into the excursion side between steps j-1 and j.
            # For a down-Parisian: BB prob of touching H from above.
            S_prev = paths[:, j - 1]
            S_curr = paths[:, j]
            if direction == "down":
                log_above_prev = np.log(np.maximum(S_prev / H, 1e-300))
                log_above_curr = np.log(np.maximum(S_curr / H, 1e-300))
            else:
                log_above_prev = np.log(np.maximum(H / S_prev, 1e-300))
                log_above_curr = np.log(np.maximum(H / S_curr, 1e-300))

            cross_prob = np.exp(-2.0 * log_above_prev * log_above_curr / (sigma**2 * dt + 1e-300))
            cross_prob = np.clip(cross_prob, 0.0, 1.0)
            # If both endpoints are on the side, full dt counts; if neither, 0;
            # if crossing is probable, partial credit (expected crossing duration
            # approximated as dt * cross_prob * 0.5).
            both_on = on_side[:, j - 1] & on_side[:, j]
            entered = ~on_side[:, j - 1] & on_side[:, j]
            exited = on_side[:, j - 1] & ~on_side[:, j]
            neither_on = ~on_side[:, j - 1] & ~on_side[:, j]

            added_time = np.where(
                both_on,
                dt,
                np.where(
                    entered,
                    dt * 0.5,
                    np.where(exited, dt * 0.5, np.where(neither_on, dt * cross_prob * 0.5, 0.0)),
                ),
            )
            excursion_time = np.where(currently_on_side, excursion_time + added_time, 0.0)
        else:
            # Simple discrete monitoring: add dt when on the excursion side, reset otherwise.
            excursion_time = np.where(currently_on_side, excursion_time + dt, 0.0)

        triggered |= excursion_time >= D

    return triggered


def _excursion_triggered_cumulative(
    paths: np.ndarray,
    H: float,
    D: float,
    dt: float,
    direction: str,
) -> np.ndarray:
    """Return boolean array (n_paths,): True if cumulative Parisian condition fired."""
    if direction == "down":
        on_side = paths[:, 1:] < H  # exclude time-0 (usually S0 is not on the side)
    else:
        on_side = paths[:, 1:] > H

    total_time = on_side.sum(axis=1) * dt
    return total_time >= D


def parisian_mc_price(
    S0: float,
    K: float,
    H: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    D: float,
    *,
    cp: int = 1,
    direction: str = "down",
    knockout: bool = True,
    parisian_type: str = "standard",
    rebate: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 500,
    seed: int | None = None,
    antithetic: bool = True,
    use_bb_correction: bool = False,
) -> tuple[float, float]:
    """Price a Parisian option via Monte Carlo under GBM.

    Parameters
    ----------
    S0, K, H, r, q, T, sigma :
        Spot, strike, barrier, risk-free rate, dividend yield, maturity, vol.
    D : float
        Parisian excursion window in years.  Must satisfy 0 < D < T.
    cp : int
        +1 call, -1 put.
    direction : {"down", "up"}
        "down" (S < H triggers) or "up" (S > H triggers).
    knockout : bool
        True → option cancelled by Parisian event; False → option activated.
    parisian_type : {"standard", "cumulative"}
        Standard uses a resetting clock; cumulative accumulates total time.
    rebate : float
        Cash paid at maturity if knocked out. Default 0.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Time discretisation steps.  Higher → more accurate barrier tracking.
    seed : int or None
        RNG seed for reproducibility.
    antithetic : bool
        Antithetic variates variance reduction. n_paths must be even.
    use_bb_correction : bool
        Apply Brownian-bridge barrier correction (standard type only).

    Returns
    -------
    (price, std_error) : (float, float)
        MC estimate and 1-σ standard error.
    """
    if not (0 < D < T):
        raise ValueError(f"parisian_mc_price: D must be in (0, T); got D={D}, T={T}")
    if sigma <= 0:
        raise ValueError(f"parisian_mc_price: sigma must be > 0; got {sigma}")

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    paths = gbm_paths(S0, r, q, T, sigma, n_paths, n_steps, rng, antithetic=antithetic)

    if parisian_type == "standard":
        triggered = _excursion_triggered_standard(
            paths, H, D, dt, direction, use_bb=use_bb_correction, sigma=sigma
        )
    elif parisian_type == "cumulative":
        triggered = _excursion_triggered_cumulative(paths, H, D, dt, direction)
    else:
        raise ValueError(f"parisian_mc_price: unknown parisian_type={parisian_type!r}")

    S_T = paths[:, -1]
    vanilla = np.maximum(cp * (S_T - K), 0.0)

    if knockout:
        payoffs = np.where(triggered, rebate, vanilla)
    else:
        payoffs = np.where(triggered, vanilla, rebate)

    discounted = np.exp(-r * T) * payoffs
    price = float(discounted.mean())
    std_err = float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return price, std_err


def parisian_mc_price_from_product(
    product: ParisianOption,
    S0: float,
    r: float,
    q: float,
    sigma: float,
    *,
    n_paths: int = 50_000,
    n_steps: int = 500,
    seed: int | None = None,
    antithetic: bool = True,
    use_bb_correction: bool = False,
) -> tuple[float, float]:
    """Convenience wrapper accepting a :class:`ParisianOption` product spec.

    Returns
    -------
    (price, std_error) : (float, float)
    """
    return parisian_mc_price(
        S0=S0,
        K=product.strike,
        H=product.barrier,
        r=r,
        q=q,
        T=product.maturity,
        sigma=sigma,
        D=product.excursion_window,
        cp=product.cp,
        direction=product.direction,
        knockout=product.knockout,
        parisian_type=product.parisian_type,
        rebate=product.rebate,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
        use_bb_correction=use_bb_correction,
    )
