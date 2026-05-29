"""Unified Monte Carlo pricing engine for path-dependent products.

``mc_price`` dispatches on ``product.product_type`` and calls the
appropriate path generator + payoff function.  Variance reduction
(antithetic variates) is available via the ``MCSpec`` flag.

Supported products:
    european      -- one-step exact GBM
    asian         -- arithmetic or geometric average
    barrier       -- single-barrier knock-out / knock-in

Not yet supported (raise NotImplementedError):
    bermudan, variance_swap, lookback, cliquet, …
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.base import ForwardSpec
from ..products.asian import AsianOption
from ..products.barrier import BarrierOption
from ..products.bermudan import BermudanOption
from ..products.european import EuropeanOption
from .paths import gbm_paths, gbm_terminal
from .payoffs import (
    asian_arithmetic_payoff,
    asian_geometric_payoff,
    barrier_payoff,
    european_payoff,
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
        Any product with a ``product_type`` attribute.  Supported:
        ``"european"``, ``"asian"``, ``"barrier"``, ``"bermudan"``.
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
        paths = gbm_paths(S0, r, q, T, sigma, mc.n_paths, mc.n_steps, rng,
                          antithetic=mc.antithetic)
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
        paths = gbm_paths(S0, r, q, T, sigma, mc.n_paths, n_steps, rng,
                          antithetic=mc.antithetic)
        dt = T / n_steps
        raw = barrier_payoff(paths, K, H, bt, cp,
                             use_bb_correction=True, sigma=sigma, dt=dt)
        disc = np.exp(-r * T)
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
        "Supported: 'european', 'asian', 'barrier'."
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
