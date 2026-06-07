"""Unified Monte Carlo pricing engine for supported GBM products."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.base import ForwardSpec
from .paths import correlated_gbm_terminal, gbm_paths, gbm_paths_on_grid, gbm_terminal
from .payoffs import (
    asian_arithmetic_payoff,
    asian_geometric_payoff,
    barrier_payoff,
    basket_payoff,
    best_of_payoff,
    cliquet_payoff,
    double_barrier_payoff,
    european_payoff,
    lookback_payoff,
    spread_payoff,
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
    basis_degree : int
        Polynomial degree for Longstaff-Schwartz continuation regression.
    """

    n_paths: int = 100_000
    n_steps: int = 252
    seed: int | None = 42
    antithetic: bool = True
    basis_degree: int = 2


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

    # ── American / Bermudan via Longstaff-Schwartz ───────────────────────
    if pt == "american":
        exercise_times = np.linspace(
            product.maturity / mc.n_steps,
            product.maturity,
            mc.n_steps,
            dtype=np.float64,
        )
        paths = gbm_paths_on_grid(
            S0,
            r,
            q,
            exercise_times,
            sigma,
            mc.n_paths,
            rng,
            antithetic=mc.antithetic,
        )
        return _lsmc_result(
            paths,
            exercise_times,
            strike=product.strike,
            cp=product.cp,
            r=r,
            basis_degree=mc.basis_degree,
        )

    if pt == "bermudan":
        exercise_times = np.asarray(product.exercise_times, dtype=np.float64)
        paths = gbm_paths_on_grid(
            S0,
            r,
            q,
            exercise_times,
            sigma,
            mc.n_paths,
            rng,
            antithetic=mc.antithetic,
        )
        return _lsmc_result(
            paths,
            exercise_times,
            strike=product.strike,
            cp=product.cp,
            r=r,
            basis_degree=mc.basis_degree,
        )

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

    # ── Multi-asset terminal payoffs ──────────────────────────────────────
    if pt in {"exchange", "basket", "spread", "best_of"}:
        if pt == "exchange":
            terminal = correlated_gbm_terminal(
                np.array([S0, product.spot2], dtype=np.float64),
                r,
                np.array([q, product.q2], dtype=np.float64),
                product.maturity,
                np.array([sigma, product.sigma2], dtype=np.float64),
                np.array([[1.0, product.rho], [product.rho, 1.0]], dtype=np.float64),
                mc.n_paths,
                rng,
                antithetic=mc.antithetic,
            )
            raw = spread_payoff(terminal, 0.0, 1)
            disc = np.exp(-r * product.maturity)
            return _result(raw * disc, mc.n_paths)

        if pt == "basket":
            spots = np.concatenate(([S0], np.asarray(product.other_spots, dtype=np.float64)))
            dividend_yields = np.concatenate(
                ([q], np.asarray(product.other_dividend_yields, dtype=np.float64))
            )
            volatilities = np.concatenate(
                ([sigma], np.asarray(product.other_volatilities, dtype=np.float64))
            )
            corr_matrix = np.asarray(product.corr_matrix, dtype=np.float64)
            terminal = correlated_gbm_terminal(
                spots,
                r,
                dividend_yields,
                product.maturity,
                volatilities,
                corr_matrix,
                mc.n_paths,
                rng,
                antithetic=mc.antithetic,
            )
            raw = basket_payoff(terminal, product.strike, product.weights, product.cp)
            disc = np.exp(-r * product.maturity)
            return _result(raw * disc, mc.n_paths)

        if pt == "spread":
            terminal = correlated_gbm_terminal(
                np.array([S0, product.spot2], dtype=np.float64),
                r,
                np.array([q, product.q2], dtype=np.float64),
                product.maturity,
                np.array([sigma, product.sigma2], dtype=np.float64),
                np.array([[1.0, product.rho], [product.rho, 1.0]], dtype=np.float64),
                mc.n_paths,
                rng,
                antithetic=mc.antithetic,
            )
            raw = spread_payoff(terminal, product.strike, product.cp)
            disc = np.exp(-r * product.maturity)
            return _result(raw * disc, mc.n_paths)

        spots = np.concatenate(([S0], np.asarray(product.other_spots, dtype=np.float64)))
        dividend_yields = np.concatenate(
            ([q], np.asarray(product.other_dividend_yields, dtype=np.float64))
        )
        volatilities = np.concatenate(
            ([sigma], np.asarray(product.other_volatilities, dtype=np.float64))
        )
        corr_matrix = np.asarray(product.corr_matrix, dtype=np.float64)
        terminal = correlated_gbm_terminal(
            spots,
            r,
            dividend_yields,
            product.maturity,
            volatilities,
            corr_matrix,
            mc.n_paths,
            rng,
            antithetic=mc.antithetic,
        )
        raw = best_of_payoff(terminal, product.strike, product.cp)
        disc = np.exp(-r * product.maturity)
        return _result(raw * disc, mc.n_paths)

    raise NotImplementedError(
        f"mc_price: product_type={pt!r} is not yet supported. "
        "Supported: 'european', 'american', 'bermudan', 'asian', 'barrier', 'double_barrier', "
        "'lookback', 'variance_swap', 'variance_option', 'cliquet', "
        "'exchange', 'basket', 'spread', 'best_of'."
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


def _lsmc_result(
    paths: np.ndarray,
    exercise_times: np.ndarray,
    *,
    strike: float,
    cp: int,
    r: float,
    basis_degree: int,
) -> MCResult:
    """Longstaff-Schwartz price for vanilla American/Bermudan options."""
    if cp not in (1, -1):
        raise ValueError(f"cp must be +1 or -1, got {cp}")
    if basis_degree < 0:
        raise ValueError(f"basis_degree must be >= 0, got {basis_degree}")

    intrinsic = np.maximum(cp * (paths[:, 1:] - strike), 0.0)
    cashflow = intrinsic[:, -1].copy()
    exercise_time = np.full(paths.shape[0], exercise_times[-1], dtype=np.float64)

    for j in range(len(exercise_times) - 2, -1, -1):
        t_j = exercise_times[j]
        s_j = paths[:, j + 1]
        immediate = intrinsic[:, j]
        alive = exercise_time > t_j + 1e-12
        itm = alive & (immediate > 0.0)
        if not np.any(itm):
            continue

        discounted_cont = cashflow[alive] * np.exp(-r * (exercise_time[alive] - t_j))
        s_alive = s_j[alive]
        continuation_hat = np.zeros(np.count_nonzero(alive), dtype=np.float64)

        fit_mask = immediate[alive] > 0.0
        if np.count_nonzero(fit_mask) > basis_degree:
            x_fit = s_alive[fit_mask]
            y_fit = discounted_cont[fit_mask]
            design_fit = np.polynomial.polynomial.polyvander(x_fit, basis_degree)
            coeffs, *_ = np.linalg.lstsq(design_fit, y_fit, rcond=None)
            continuation_hat = np.polynomial.polynomial.polyvander(s_alive, basis_degree) @ coeffs
        else:
            continuation_hat.fill(float(discounted_cont.mean()))

        exercise_now_alive = fit_mask & (immediate[alive] >= continuation_hat)
        if not np.any(exercise_now_alive):
            continue

        alive_idx = np.flatnonzero(alive)
        exercise_idx = alive_idx[exercise_now_alive]
        cashflow[exercise_idx] = immediate[exercise_idx]
        exercise_time[exercise_idx] = t_j

    discounted = cashflow * np.exp(-r * exercise_time)
    return _result(discounted, paths.shape[0])
