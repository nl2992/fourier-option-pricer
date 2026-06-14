"""Arithmetic Asian call/put MC pricer with geometric-average control variate."""

from __future__ import annotations

import numpy as np

from ..pricers.analytic_bsm import bsm_geometric_asian
from .path_engine import GBMPathSpec, simulate_gbm_paths


def asian_mc(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    N_mon: int,
    cp: int,
    spec: GBMPathSpec,
) -> float:
    """Arithmetic Asian option price by MC with geometric-average control variate.

    Parameters
    ----------
    S0, K, r, q, sigma, T : standard BSM inputs
    N_mon : number of equally-spaced monitoring dates
    cp    : +1 call, -1 put
    spec  : GBMPathSpec (n_paths, n_steps, seed, antithetic)

    Returns
    -------
    float
        Discounted arithmetic Asian price.
    """
    # Use N_mon steps so each column past col-0 is a monitoring date
    path_spec = GBMPathSpec(
        n_paths=spec.n_paths,
        n_steps=N_mon,
        seed=spec.seed,
        antithetic=spec.antithetic,
        batch_size=spec.batch_size,
    )
    paths = simulate_gbm_paths(S0, r, q, sigma, T, path_spec)

    # Monitoring columns: indices 1..N_mon (exclude S0 at col 0)
    mon = paths[:, 1:]

    arith_avg = mon.mean(axis=1)
    # Geometric average in log-space for numerical stability
    geo_avg = np.exp(np.mean(np.log(mon), axis=1))

    disc = np.exp(-r * T)

    arith_payoff = disc * np.maximum(cp * (arith_avg - K), 0.0)
    geo_payoff = disc * np.maximum(cp * (geo_avg - K), 0.0)

    # Analytic price of the geometric Asian (known closed form)
    geo_price_analytic = bsm_geometric_asian(S0, K, r, q, sigma, T, N_mon, cp)

    # Control variate: subtract (geo_mc - geo_analytic) * beta
    # Optimal beta estimated from sample covariance
    cov_matrix = np.cov(arith_payoff, geo_payoff, ddof=1)
    var_geo = cov_matrix[1, 1]
    if var_geo > 1e-30:
        beta = cov_matrix[0, 1] / var_geo
    else:
        beta = 0.0

    cv_payoff = arith_payoff - beta * (geo_payoff - geo_price_analytic)
    return float(np.mean(cv_payoff))
