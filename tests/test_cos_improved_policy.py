from __future__ import annotations

import numpy as np

from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams, bsm_cf, bsm_cumulants
from foureng.models.heston import HestonParams, heston_cf, heston_cumulants
from foureng.models.variance_gamma import VGParams
from foureng.pipeline import price_strip
from foureng.pricers.cos import cos_adaptive_decision, cos_auto_grid, cos_prices
from foureng.pricers.lewis import lewis_call_prices
from foureng.utils.cumulants import Cumulants, cos_centered_half_width
from foureng.utils.grids import COSGrid, COSGridPolicy


def test_centered_cos_grid_matches_uncentered_bsm():
    """Centered COS should be numerically equivalent to the old uncentered grid.

    This pins the variable-shift implementation in ``cos_prices``:
    shifting the state variable by its mean must not change the option value.
    """
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.5)
    p = BsmParams(sigma=0.25)
    strikes = np.array([80.0, 100.0, 120.0])
    cums = bsm_cumulants(fwd, p)
    phi = lambda u: bsm_cf(u, fwd, p)

    grid_old = cos_auto_grid(cums, N=512, L=10.0)
    calls_old = cos_prices(phi, fwd, strikes, grid_old).call_prices

    c1, c2, c4 = cums
    half_width = cos_centered_half_width(Cumulants(c1=c1, c2=c2, c4=c4), L=10.0)
    grid_centered = COSGrid(N=512, a=-half_width, b=half_width, center=c1, label="centered")
    calls_centered = cos_prices(phi, fwd, strikes, grid_centered).call_prices

    assert np.allclose(calls_centered, calls_old, atol=1e-12, rtol=0.0)


def test_cos_adaptive_decision_scales_n_with_interval_width():
    """Adaptive COS should increase N when the chosen interval gets wider."""
    policy = COSGridPolicy(
        mode="benchmark",
        truncation="heuristic",
        centered=True,
        dx_target=0.05,
        L=10.0,
        width_fallback=0.0,
    )
    d_small = cos_adaptive_decision((0.0, 0.04, 0.0), model="bsm", policy=policy)
    d_large = cos_adaptive_decision((0.0, 1.00, 0.0), model="bsm", policy=policy)

    assert d_large.grid.N > d_small.grid.N
    assert d_small.grid.dx <= 0.05
    assert d_large.grid.dx <= 0.05


def test_price_strip_cos_improved_can_fall_back_to_lewis():
    """Wide Heston intervals should be able to route through Lewis cleanly."""
    fwd = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=10.0)
    p = HestonParams(kappa=1.5768, theta=0.0398, nu=0.5751, rho=-0.5711, v0=0.0175)
    strikes = np.array([100.0])

    policy = COSGridPolicy(
        mode="benchmark",
        truncation="paper",
        centered=True,
        paper_L=32.0,
        dx_target=0.02,
        width_fallback=10.0,
        fallback_method="lewis",
    )
    decision = cos_adaptive_decision(
        heston_cumulants(fwd, p),
        model="heston",
        params=p,
        policy=policy,
    )
    assert decision.method == "lewis"

    got = price_strip("heston", "cos_improved", strikes, fwd, p, grid=policy)
    ref = lewis_call_prices(
        lambda u: heston_cf(u, fwd, p),
        strikes,
        spot=fwd.S0,
        texp=fwd.T,
        intr=fwd.r,
        divr=fwd.q,
        method="trapz",
        u_max=200.0,
        n_u=max(4096, decision.grid.N),
    )
    assert np.allclose(got, ref, atol=1e-11, rtol=0.0)


def test_price_strip_cos_improved_uses_stable_direct_call_regime_for_heston_strip():
    """Narrow centered Heston intervals should avoid parity cancellation."""
    fwd = ForwardSpec(S0=100.0, r=0.0, q=0.0, T=1.0)
    p = HestonParams(kappa=1.5768, theta=0.0398, nu=0.5751, rho=-0.5711, v0=0.0175)
    strikes = np.arange(50.0, 151.0, 5.0)

    got = price_strip("heston", "cos_improved", strikes, fwd, p)
    ref = lewis_call_prices(
        lambda u: heston_cf(u, fwd, p),
        strikes,
        spot=fwd.S0,
        texp=fwd.T,
        intr=fwd.r,
        divr=fwd.q,
        method="trapz",
        u_max=250.0,
        n_u=8192,
    )
    assert np.allclose(got, ref, atol=1e-8, rtol=0.0)


def test_price_strip_cos_improved_short_maturity_vg_is_tightly_resolved():
    """Benchmark-mode VG should not under-resolve the short-maturity FO case."""
    fwd = ForwardSpec(S0=100.0, r=0.1, q=0.0, T=0.1)
    p = VGParams(sigma=0.12, nu=0.2, theta=-0.14)
    strikes = np.array([90.0])

    got = price_strip("vg", "cos_improved", strikes, fwd, p)
    ref = np.array([10.993703187])
    assert np.allclose(got, ref, atol=1e-6, rtol=0.0)
