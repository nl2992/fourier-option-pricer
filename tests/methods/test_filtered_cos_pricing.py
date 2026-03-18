"""Tests for the filtered-COS pricing extension.

Covers:
- filter_spec=None exactly matches old cos_prices output (backward compat).
- COSFilterSpec("none") exactly matches old cos_prices output.
- Filtered COS does not materially damage BSM accuracy (< 1e-6).
- price_strip("vg", "cos_filtered", ...) returns finite, non-negative prices.
- price_strip accepts grid=(COSGridPolicy, COSFilterSpec) tuple.
- Existing "cos" and "cos_improved" outputs are unchanged.
- All four filter types produce finite prices on a BSM example.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams, bsm_cf, bsm_cumulants
from foureng.models.variance_gamma import VGParams
from foureng.pipeline import price_strip
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.utils.grids import COSGridPolicy
from foureng.utils.spectral_filters import COSFilterSpec

pytestmark = [pytest.mark.numerical_stability]


# ---------------------------------------------------------------------------
# Backward-compatibility: filter_spec=None / "none" → exact old output
# ---------------------------------------------------------------------------


def test_no_filter_matches_existing_cos_exactly():
    """cos_prices(..., filter_spec=None) must match filter-less output exactly."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.25)
    strikes = np.linspace(70.0, 130.0, 13)

    phi = lambda u: bsm_cf(u, fwd, p)
    grid = cos_auto_grid(bsm_cumulants(fwd, p), N=512, L=10.0)

    old = cos_prices(phi, fwd, strikes, grid).call_prices
    new = cos_prices(phi, fwd, strikes, grid, filter_spec=None).call_prices

    assert np.allclose(new, old, atol=0.0, rtol=0.0), (
        f"filter_spec=None changed output: max diff = {np.max(np.abs(new - old)):.3e}"
    )


def test_none_filter_spec_matches_existing_cos_exactly():
    """cos_prices(..., filter_spec=COSFilterSpec('none')) must match old output."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.25)
    strikes = np.linspace(70.0, 130.0, 13)

    phi = lambda u: bsm_cf(u, fwd, p)
    grid = cos_auto_grid(bsm_cumulants(fwd, p), N=512, L=10.0)

    old = cos_prices(phi, fwd, strikes, grid).call_prices
    new = cos_prices(phi, fwd, strikes, grid, filter_spec=COSFilterSpec("none")).call_prices

    assert np.allclose(new, old, atol=0.0, rtol=0.0), (
        f"COSFilterSpec('none') changed output: max diff = {np.max(np.abs(new - old)):.3e}"
    )


# ---------------------------------------------------------------------------
# Accuracy: filtered COS should not damage BSM to more than 1e-6
# ---------------------------------------------------------------------------


def test_filtered_cos_does_not_damage_bsm_materially():
    """Exponential-filtered COS on BSM should match Black-Scholes to 1e-6."""
    from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd

    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.25)
    strikes = np.linspace(80.0, 120.0, 9)

    phi = lambda u: bsm_cf(u, fwd, p)
    grid = cos_auto_grid(bsm_cumulants(fwd, p), N=1024, L=10.0)

    got = cos_prices(
        phi,
        fwd,
        strikes,
        grid,
        filter_spec=COSFilterSpec("exponential", order=12),
    ).call_prices

    ref = np.array(
        [
            bs_price_from_fwd(
                0.25,
                BSInputs(F0=fwd.F0, K=float(K), T=fwd.T, r=fwd.r, q=fwd.q, is_call=True),
            )
            for K in strikes
        ]
    )

    max_err = float(np.max(np.abs(got - ref)))
    assert max_err < 1e-6, f"Filtered COS damaged BSM accuracy: max|err| = {max_err:.3e}"


# ---------------------------------------------------------------------------
# All four non-trivial filter types: prices finite and non-negative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["fejer", "lanczos", "raised_cosine", "exponential"])
def test_all_filter_types_give_finite_nonneg_bsm_prices(name):
    """Every filter type must produce finite, non-negative call prices on BSM."""
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.01, T=0.5)
    p = BsmParams(sigma=0.20)
    strikes = np.linspace(80.0, 120.0, 9)

    phi = lambda u: bsm_cf(u, fwd, p)
    grid = cos_auto_grid(bsm_cumulants(fwd, p), N=512, L=10.0)

    prices = cos_prices(
        phi,
        fwd,
        strikes,
        grid,
        filter_spec=COSFilterSpec(name=name),
    ).call_prices

    assert np.all(np.isfinite(prices)), f"Non-finite prices with filter={name}: {prices}"
    assert np.all(prices >= -1e-8), f"Negative prices with filter={name}: {prices.min():.4f}"


# ---------------------------------------------------------------------------
# Pipeline integration: price_strip("vg", "cos_filtered", ...)
# ---------------------------------------------------------------------------


def test_pipeline_cos_filtered_runs_on_vg():
    """price_strip("vg", "cos_filtered", ...) must return finite, non-negative prices."""
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.1)
    p = VGParams(sigma=0.12, nu=0.2, theta=-0.14)
    strikes = np.array([90.0, 100.0, 110.0])

    prices = price_strip("vg", "cos_filtered", strikes, fwd, p)

    assert prices.shape == strikes.shape
    assert np.all(np.isfinite(prices)), f"Non-finite prices: {prices}"
    assert np.all(prices >= -1e-10), f"Negative prices: {prices.min():.6f}"


def test_pipeline_cos_filtered_accepts_policy_and_filter_tuple():
    """price_strip accepts grid=(COSGridPolicy, COSFilterSpec)."""
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.1)
    p = VGParams(sigma=0.12, nu=0.2, theta=-0.14)
    strikes = np.array([90.0, 100.0, 110.0])

    policy = COSGridPolicy(
        mode="benchmark",
        truncation="tolerance",
        centered=True,
        dx_target=0.01,
        L=10.0,
        eps_trunc=1e-10,
        max_N=4096,
        width_fallback=0.0,
    )

    prices = price_strip(
        "vg",
        "cos_filtered",
        strikes,
        fwd,
        p,
        grid=(policy, COSFilterSpec("exponential", order=8)),
    )

    assert prices.shape == strikes.shape
    assert np.all(np.isfinite(prices)), f"Non-finite prices: {prices}"


def test_pipeline_cos_filtered_accepts_explicit_grid_and_filter_tuple():
    """price_strip accepts grid=(COSGrid, COSFilterSpec)."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.25)
    strikes = np.linspace(80.0, 120.0, 5)

    from foureng.models.bsm import bsm_cumulants

    cos_grid = cos_auto_grid(bsm_cumulants(fwd, p), N=512, L=10.0)

    prices = price_strip(
        "bsm",
        "cos_filtered",
        strikes,
        fwd,
        p,
        grid=(cos_grid, COSFilterSpec("lanczos")),
    )

    assert prices.shape == strikes.shape
    assert np.all(np.isfinite(prices))


def test_pipeline_cos_filtered_grid_none_uses_default_policy():
    """cos_filtered with grid=None should use the recommended policy."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.25)
    strikes = np.array([90.0, 100.0, 110.0])

    prices = price_strip("bsm", "cos_filtered", strikes, fwd, p, grid=None)

    assert prices.shape == strikes.shape
    assert np.all(np.isfinite(prices))


# ---------------------------------------------------------------------------
# Existing cos and cos_improved must be unchanged
# ---------------------------------------------------------------------------


def test_existing_cos_output_unchanged():
    """Adding cos_filtered must not change cos output on a simple BSM case."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.25)
    strikes = np.linspace(80.0, 120.0, 9)

    phi = lambda u: bsm_cf(u, fwd, p)
    grid = cos_auto_grid(bsm_cumulants(fwd, p), N=512, L=10.0)

    baseline = cos_prices(phi, fwd, strikes, grid).call_prices
    after = cos_prices(phi, fwd, strikes, grid, filter_spec=None).call_prices

    assert np.allclose(baseline, after, atol=0.0, rtol=0.0)


def test_existing_cos_improved_output_unchanged():
    """cos_improved output must be identical before and after the extension."""
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.25)
    strikes = np.linspace(80.0, 120.0, 9)

    prices_before = price_strip("bsm", "cos_improved", strikes, fwd, p)

    # Run cos_filtered and then cos_improved again to confirm no side effects
    _ = price_strip("bsm", "cos_filtered", strikes, fwd, p)
    prices_after = price_strip("bsm", "cos_improved", strikes, fwd, p)

    assert np.allclose(prices_before, prices_after, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Filtered COS error should not be worse than unfiltered on smooth BSM
# (sanity check: smooth model + adequate N → filter does not hurt much)
# ---------------------------------------------------------------------------


def test_filtered_cos_no_worse_than_unfiltered_bsm_atm():
    """For smooth BSM at ATM, filtered COS error should be ≤ 1e-5."""
    from foureng.iv.implied_vol import BSInputs, bs_price_from_fwd

    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    p = BsmParams(sigma=0.20)
    K = np.array([100.0])

    phi = lambda u: bsm_cf(u, fwd, p)
    grid = cos_auto_grid(bsm_cumulants(fwd, p), N=512, L=10.0)

    ref = bs_price_from_fwd(
        0.20,
        BSInputs(F0=fwd.F0, K=100.0, T=fwd.T, r=fwd.r, q=fwd.q, is_call=True),
    )

    filt = cos_prices(
        phi,
        fwd,
        K,
        grid,
        filter_spec=COSFilterSpec("exponential", order=8),
    ).call_prices[0]

    assert abs(filt - ref) < 1e-5, f"Filtered ATM error = {abs(filt - ref):.3e}"
