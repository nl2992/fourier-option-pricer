"""Additional small edge-case tests for project robustness."""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams
from foureng.mc.black_scholes_mc import MCSpec, european_call_mc
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid


def test_carr_madan_rejects_nonpositive_strikes():
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    params = BsmParams(sigma=0.2)

    with pytest.raises(ValueError, match="strikes must be > 0"):
        price_strip(
            "bsm",
            "carr_madan",
            np.array([90.0, 0.0, 110.0]),
            fwd,
            params,
            grid=FFTGrid(N=4096, eta=0.25, alpha=1.5),
        )


def test_bsm_cos_prices_are_monotone_and_above_intrinsic_value():
    fwd = ForwardSpec(S0=100.0, r=0.02, q=0.01, T=1.0)
    params = BsmParams(sigma=0.25)
    strikes = np.linspace(70.0, 130.0, 13)

    prices = price_strip("bsm", "cos", strikes, fwd, params)
    lower_bound = np.maximum(fwd.disc * (fwd.F0 - strikes), 0.0)

    assert np.all(np.isfinite(prices))
    assert np.all(prices >= lower_bound - 1e-9)
    assert np.all(np.diff(prices) <= 1e-9)


def test_bsm_monte_carlo_is_reproducible_with_fixed_seed():
    strikes = np.array([90.0, 100.0, 110.0])
    spec = MCSpec(n_paths=25_000, seed=1234)

    first = european_call_mc(100.0, strikes, 1.0, 0.03, 0.01, 0.2, spec)
    second = european_call_mc(100.0, strikes, 1.0, 0.03, 0.01, 0.2, spec)

    assert np.array_equal(first, second)
