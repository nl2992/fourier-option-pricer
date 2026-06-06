"""BSM analytic formulas for variance-linked products under repo conventions."""

from __future__ import annotations

import numpy as np

from ..models.base import ForwardSpec
from ..models.bsm import BsmParams
from ..products.variance import VarianceOption, VarianceSwap


def bsm_variance_swap(
    fwd: ForwardSpec,
    params: BsmParams,
    product: VarianceSwap,
) -> float:
    """Exact discounted expectation of the repo's realised-variance swap payoff.

    The Monte Carlo engine defines realised variance as:

        RV = sum_i (log(S_ti / S_t{i-1}))^2 / T

    Under GBM each log increment has mean
    ``(r - q - 0.5 * sigma^2) * dt_i`` and variance ``sigma^2 * dt_i``,
    so the exact expectation is:

        E[RV] = sigma^2 + ((r - q - 0.5 * sigma^2)^2 / T) * sum_i dt_i^2

    Parameters
    ----------
    fwd, params :
        Market inputs and BSM volatility.
    product :
        Variance swap product with sampling times and notional.
    """
    times = np.asarray(product.sampling_times, dtype=np.float64)
    dt = np.diff(np.concatenate(([0.0], times)))
    drift = fwd.r - fwd.q - 0.5 * params.sigma**2
    expected_rv = params.sigma**2 + (drift**2 / product.maturity) * float(np.sum(dt**2))
    return float(fwd.disc * product.notional * expected_rv)


def bsm_variance_option_integrated(
    fwd: ForwardSpec,
    params: BsmParams,
    product: VarianceOption,
) -> float:
    """Discounted intrinsic value of an integrated-variance option under BSM.

    With constant volatility under BSM, the annualised integrated variance is
    deterministic and equal to ``sigma^2`` under the repo's convention.
    """
    intrinsic = max(product.cp * (params.sigma**2 - product.strike), 0.0)
    return float(fwd.disc * intrinsic)


__all__ = ["bsm_variance_swap", "bsm_variance_option_integrated"]
