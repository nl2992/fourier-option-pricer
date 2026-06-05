"""Unified Monte Carlo pricing engine for supported GBM products."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.base import ForwardSpec
from .paths import gbm_paths, gbm_paths_on_grid, gbm_terminal
from .payoffs import (
    asian_arithmetic_payoff,
    asian_geometric_payoff,
    barrier_payoff,
    cliquet_payoff,
    double_barrier_payoff,
    european_payoff,
    lookback_payoff,
    variance_option_payoff,
    variance_swap_payoff,
)


@dataclass(frozen=True)
class MCSpec:
    """Monte Carlo run configuration.

    Parameters
    ----------
    n_paths : int
        Number of sample paths.
    n_steps : int
        Discretisation steps per path.  ``1`` is exact for GBM Europeans.
    seed : int or None
        RNG seed for reproducibility.
    antithetic : bool
        Antithetic-variates variance reduction (``n_paths`` must be even).
    """

    n_paths: int = 100_000
    n_steps: int = 252
    seed: int | None = 42
    antithetic: bool = True


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo pricing result.

    Attributes
    ----------
    price : float
        Discounted expected payoff.
    stderr : float
        Standard error of the price estimate.
    ci_95 : tuple[float, float]
        Approximate 95% confidence interval (price ± 1.96*stderr).
    n_paths : int
    """

    price: float
    stderr: float
    ci_95: tuple[float, float]
    n_paths: int

    @property
    def relative_error(self) -> float:
        """Coefficient of variation: stderr / |price|."""
        return self.stderr / max(abs(self.price), 1e-12)


def mc_price(
    fwd: ForwardSpec,
    sigma: float,
    product,
    mc: MCSpec | None = None,
) -> MCResult:
    """Price a product via Monte Carlo using GBM paths.

    Parameters
    ----------
    fwd : ForwardSpec
        Market inputs.
    sigma : float
        GBM lognormal volatility.
    product :
        Any product with a ``product_type`` attribute.
    mc : MCSpec or None
        MC configuration; defaults to 100K paths, 252 steps, seed=42.

    Returns
    -------
    MCResult
    """
    if mc is None:
        mc = MCSpec()

    rng = np.random.default_rng(mc.seed)
    r, q, S0 = fwd.r, fwd.q, fwd.S0

    pt = getattr(product, "product_type", None)

    # ── European ───────────────────────────────────────────────────────────
    if pt == "european":
        T, K, cp = product.maturity, product.strike, product.cp
        n_paths = mc.n_paths if not mc.antithetic else mc.n_paths
        S_T = gbm_terminal(S0, r, q, T, sigma, n_paths, rng, antithetic=mc.antithetic)
        raw = european_payoff(S_T, K, cp)
        disc = np.exp(-r * T)
        return _result(raw * disc, n_paths)

    # ── Asian ──────────────────────────────────────────────────────────────
    if pt == "asian":
        T, K, cp = product.maturity, product.strike, product.cp
        if hasattr(product, "monitoring_times"):
            paths = gbm_paths_on_grid(
                S0,
                r,
                q,
                np.asarray(product.monitoring_times, dtype=np.float64),
                sigma,
                mc.n_paths,
                rng,
                antithetic=mc.antithetic,
            )
        else:
            paths = gbm_paths(
                S0, r, q, T, sigma, mc.n_paths, mc.n_steps, rng, antithetic=mc.antithetic
            )
        avg_type = getattr(product, "average_type", "arithmetic")
        if avg_type == "geometric":
            raw = asian_geometric_payoff(paths, K, cp)
        else:
            raw = asian_arithmetic_payoff(paths, K, cp)
        disc = np.exp(-r * T)
        return _result(raw * disc, mc.n_paths)

    # ── Single Barrier ─────────────────────────────────────────────────────
    if pt == "barrier":
        T, K, H, cp = product.maturity, product.strike, product.barrier, product.cp
        bt = product.barrier_type
        if product.monitoring == "discrete" and hasattr(product, "monitoring_times"):
            n_steps = len(product.monitoring_times)
        else:
            n_steps = mc.n_steps
        paths = gbm_paths(S0, r, q, T, sigma, mc.n_paths, n_steps, rng, antithetic=mc.antithetic)
        dt = T / n_steps
        raw = barrier_payoff(paths, K, H, bt, cp, use_bb_correction=True, sigma=sigma, dt=dt)
        disc = np.exp(-r * T)
        return _result(raw * disc, mc.n_paths)

    # ── Double Barrier ────────────────────────────────────────────────────
    if pt == "double_barrier":
        T, K, cp = product.maturity, product.strike, product.cp
        paths = gbm_paths(S0, r, q, T, sigma, mc.n_paths, mc.n_steps, rng, antithetic=mc.antithetic)
        raw = double_barrier_payoff(
            paths,
            K,
            product.lower_barrier,
            product.upper_barrier,
            cp,
            knockout=product.knockout,
        )
        disc = np.exp(-r * T)
        return _result(raw * disc, mc.n_paths)

    # ── Lookback ───────────────────────────────────────────────────────────
    if pt == "lookback":
        T = product.maturity
        paths = gbm_paths(S0, r, q, T, sigma, mc.n_paths, mc.n_steps, rng, antithetic=mc.antithetic)
        raw = lookback_payoff(
            paths,
            product.cp,
            strike_type=product.strike_type,
            strike=product.strike,
        )
        disc = np.exp(-r * T)
        return _result(raw * disc, mc.n_paths)

    # ── Variance swap / option ─────────────────────────────────────────────
    if pt == "variance_swap":
        paths = gbm_paths_on_grid(
            S0,
            r,
            q,
            np.asarray(product.sampling_times, dtype=np.float64),
            sigma,
            mc.n_paths,
            rng,
            antithetic=mc.antithetic,
        )
        raw = variance_swap_payoff(paths, product.sampling_times, notional=product.notional)
        disc = np.exp(-r * product.maturity)
        return _result(raw * disc, mc.n_paths)

    if pt == "variance_option":
        paths = gbm_paths_on_grid(
            S0,
            r,
            q,
            np.asarray(product.sampling_times, dtype=np.float64),
            sigma,
            mc.n_paths,
            rng,
            antithetic=mc.antithetic,
        )
        raw = variance_option_payoff(
            paths,
            product.sampling_times,
            product.strike,
            product.cp,
            variance_type=product.variance_type,
            sigma=sigma,
        )
        disc = np.exp(-r * product.maturity)
        return _result(raw * disc, mc.n_paths)

    # ── Cliquet ────────────────────────────────────────────────────────────
    if pt == "cliquet":
        paths = gbm_paths_on_grid(
            S0,
            r,
            q,
            np.asarray(product.reset_times, dtype=np.float64),
            sigma,
            mc.n_paths,
            rng,
            antithetic=mc.antithetic,
        )
        raw = cliquet_payoff(
            paths,
            product.cp,
            local_floor=product.local_floor,
            local_cap=product.local_cap,
            global_floor=product.global_floor,
            global_cap=product.global_cap,
            payoff_type=product.payoff_type,
        )
        disc = np.exp(-r * product.maturity)
        return _result(raw * disc, mc.n_paths)

    # ── Bermudan ───────────────────────────────────────────────────────────
    if pt == "bermudan":
        # Exercise boundary requires backward induction (LSMC); not implemented here.
        raise NotImplementedError(
            "mc_price: Bermudan options require LSMC (Longstaff-Schwartz). "
            "Use cos_bermudan_price instead for the COS backward-induction method."
        )

    raise NotImplementedError(
        f"mc_price: product_type={pt!r} is not yet supported. "
        "Supported: 'european', 'asian', 'barrier', 'double_barrier', "
        "'lookback', 'variance_swap', 'variance_option', 'cliquet'."
    )


# ── helper ─────────────────────────────────────────────────────────────────


def _result(discounted_payoffs: np.ndarray, n_paths: int) -> MCResult:
    price = float(discounted_payoffs.mean())
    se = float(discounted_payoffs.std() / np.sqrt(n_paths))
    return MCResult(
        price=price,
        stderr=se,
        ci_95=(price - 1.96 * se, price + 1.96 * se),
        n_paths=n_paths,
    )
