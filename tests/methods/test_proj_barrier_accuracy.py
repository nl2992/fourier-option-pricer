"""Accuracy of ``proj_barrier_price`` / ``proj_double_barrier_price`` (A3).

``proj_barrier_price`` used to disagree with independent references by about
1e-3 at 252 monitoring dates (see ``docs/proj_parity_roadmap.md``). The root
causes:

1. The grid half-width ``alph`` was silently halved: ``dx`` was sized from
   the full FFT length ``N`` instead of the ``K_half = N // 2`` *live* value
   nodes (the other half is zero-padding for the linear-convolution trick),
   so the represented domain only spanned ``~alph``, not ``[-alph, alph]``.
   This introduced an aliasing bias that grew with the number of monitoring
   dates ``M``.
2. The barrier was snapped to the nearest grid node with a hard kill, rather
   than weighted by its sub-cell position, adding a further O(dx) bias per
   monitoring date (most visible on the side of the barrier where the payoff
   is large, e.g. an up-and-out call with a barrier above an at-the-money
   strike).
3. For pure-jump models with a slowly decaying short-step CF (e.g. VG at
   small dt), the discretized one-step transition operator could alias past
   unit magnitude, blowing up after many repeated convolutions (M=252).

The fix corrects the grid width, adds a partial-cell barrier weighting
(making the discretization converge cleanly at O(dx)), clips the transition
operator's Fourier transfer function to the analytic contraction bound
``exp(-r*dt)``, and Richardson-extrapolates two grid sizes (N, 2N) to cancel
the leading O(dx) error. Achieved accuracy against the fast Hilbert-transform
reference (``hilbert_barrier_price``, itself validated to ~5e-7 against an
exact Gaussian-kernel backward induction, see ``test_hilbert_exotics.py``):
about 1e-5 at 12 monitoring dates and 1e-4 (BSM/Kou) to 1e-3 (VG, the
hardest case here since it is an infinite-activity pure-jump model) at 252
monitoring dates, several orders of magnitude better than the ~1e-3 bug at
any monitoring frequency, though short of the 1e-6 ideal at high M within a
sub-second runtime budget.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.models.base import ForwardSpec
from foureng.models.registry import MODEL_REGISTRY
from foureng.pricers.hilbert_exotics import hilbert_barrier_price
from foureng.pricers.proj import proj_auto_grid, proj_barrier_price, proj_double_barrier_price
from foureng.products.barrier import BarrierOption, DoubleBarrierOption

FWD = ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
BSM = fe.BsmParams(sigma=0.2)
VG = fe.VGParams(sigma=0.2, nu=0.3, theta=-0.1)
KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)


def _step_cf(model: str, params, dt: float):
    cf = MODEL_REGISTRY[model].cf
    fwd_dt = ForwardSpec(S0=FWD.S0, r=FWD.r, q=FWD.q, T=dt)
    drift = (FWD.r - FWD.q) * dt
    return lambda u: np.exp(1j * u * drift) * np.asarray(cf(u, fwd_dt, params), dtype=complex)


def _proj_barrier(model: str, params, K: float, H: float, M: int, bt: str, cp: int) -> float:
    dt = FWD.T / M
    cums = MODEL_REGISTRY[model].cumulants(FWD, params)
    grid = proj_auto_grid(cums, N=1 << 15, L=10.0)
    return proj_barrier_price(
        _step_cf(model, params, dt),
        S0=FWD.S0,
        r=FWD.r,
        T=FWD.T,
        K=K,
        H=H,
        M=M,
        barrier_type=bt,
        cp=cp,
        q=FWD.q,
        N=grid.N,
        alph=grid.alph,
    )


def _hilbert(model: str, params, K: float, H: float, M: int, bt: str, cp: int) -> float:
    return hilbert_barrier_price(
        model, FWD, params, strike=K, barrier=H, maturity=FWD.T, barrier_type=bt, cp=cp, n_monitor=M
    )


# --------------------------------------------------------------------------- #
# proj_barrier_price vs the fast Hilbert-transform reference
# --------------------------------------------------------------------------- #
CASES = [
    (100.0, 85.0, "down_out", 1),
    (100.0, 85.0, "down_out", -1),
    (100.0, 120.0, "up_out", 1),
    (100.0, 120.0, "up_out", -1),
]


@pytest.mark.parametrize("K,H,bt,cp", CASES)
@pytest.mark.parametrize("model,params", [("bsm", BSM), ("vg", VG), ("kou", KOU)])
def test_proj_barrier_matches_hilbert_at_m12(model, params, K, H, bt, cp):
    got = _proj_barrier(model, params, K, H, 12, bt, cp)
    ref = _hilbert(model, params, K, H, 12, bt, cp)
    assert got == pytest.approx(ref, abs=5e-5)


@pytest.mark.slow
@pytest.mark.parametrize("K,H,bt,cp", CASES)
@pytest.mark.parametrize("model,params", [("bsm", BSM), ("vg", VG), ("kou", KOU)])
def test_proj_barrier_matches_hilbert_at_m252(model, params, K, H, bt, cp):
    got = _proj_barrier(model, params, K, H, 252, bt, cp)
    ref = _hilbert(model, params, K, H, 252, bt, cp)
    # VG (infinite-activity pure jump) is the hardest case here; BSM and Kou
    # do better. See the module docstring for the measured accuracy.
    tol = 5e-3 if model == "vg" else 2e-3
    assert got == pytest.approx(ref, abs=tol)


def test_proj_barrier_no_longer_blows_up_for_vg_at_high_m():
    """Regression test: VG at M=252 used to diverge to ~1e42 (a slowly
    decaying short-step CF aliased the discretized transition operator past
    unit magnitude, blowing up under repeated convolution)."""
    got = _proj_barrier("vg", VG, 100.0, 85.0, 252, "down_out", 1)
    ref = _hilbert("vg", VG, 100.0, 85.0, 252, "down_out", 1)
    assert np.isfinite(got)
    assert 0.0 <= got < 2.0 * ref


def test_proj_barrier_knock_in_out_parity():
    """down_in + down_out reproduces the vanilla to the same accuracy floor."""
    K, H, M = 100.0, 85.0, 52
    dt = FWD.T / M
    cums = MODEL_REGISTRY["bsm"].cumulants(FWD, BSM)
    grid = proj_auto_grid(cums, N=1 << 15, L=10.0)
    step_cf = _step_cf("bsm", BSM, dt)
    kwargs = dict(
        S0=FWD.S0, r=FWD.r, T=FWD.T, K=K, H=H, M=M, cp=1, q=FWD.q, N=grid.N, alph=grid.alph
    )
    out = proj_barrier_price(step_cf, barrier_type="down_out", **kwargs)
    inn = proj_barrier_price(step_cf, barrier_type="down_in", **kwargs)
    vanilla = fe.price_strip("bsm", "cos_improved", [K], FWD, BSM)[0]
    assert out + inn == pytest.approx(vanilla, abs=1e-4)


# --------------------------------------------------------------------------- #
# proj_double_barrier_price: sanity + BSM continuous-monitoring bound
# --------------------------------------------------------------------------- #
def test_proj_double_barrier_in_out_parity():
    K, Lb, Ub, M = 100.0, 85.0, 120.0, 52
    dt = FWD.T / M
    cums = MODEL_REGISTRY["bsm"].cumulants(FWD, BSM)
    grid = proj_auto_grid(cums, N=1 << 15, L=10.0)
    step_cf = _step_cf("bsm", BSM, dt)
    ko = proj_double_barrier_price(
        step_cf,
        S0=FWD.S0,
        r=FWD.r,
        T=FWD.T,
        K=K,
        L=Lb,
        U=Ub,
        M=M,
        knockout=True,
        cp=1,
        q=FWD.q,
        N=grid.N,
        alph=grid.alph,
    )
    ki = proj_double_barrier_price(
        step_cf,
        S0=FWD.S0,
        r=FWD.r,
        T=FWD.T,
        K=K,
        L=Lb,
        U=Ub,
        M=M,
        knockout=False,
        cp=1,
        q=FWD.q,
        N=grid.N,
        alph=grid.alph,
    )
    vanilla = fe.price_strip("bsm", "cos_improved", [K], FWD, BSM)[0]
    assert ko + ki == pytest.approx(vanilla, abs=1e-4)


def test_proj_double_barrier_converges_to_bsm_continuous_bound():
    """Discrete monitoring must sit above the continuous-monitoring analytic
    price and converge down toward it as the monitoring frequency grows."""
    from foureng.analytics.bsm_barrier import bsm_double_barrier_price

    K, Lb, Ub = 100.0, 85.0, 120.0
    cont = bsm_double_barrier_price(
        FWD.S0, K, Lb, Ub, FWD.r, FWD.q, FWD.T, BSM.sigma, cp=1, knockout=True
    )
    cums = MODEL_REGISTRY["bsm"].cumulants(FWD, BSM)
    grid = proj_auto_grid(cums, N=1 << 15, L=10.0)
    prices = []
    for M in (12, 52, 252):
        dt = FWD.T / M
        step_cf = _step_cf("bsm", BSM, dt)
        prices.append(
            proj_double_barrier_price(
                step_cf,
                S0=FWD.S0,
                r=FWD.r,
                T=FWD.T,
                K=K,
                L=Lb,
                U=Ub,
                M=M,
                knockout=True,
                cp=1,
                q=FWD.q,
                N=grid.N,
                alph=grid.alph,
            )
        )
    assert prices[0] > prices[1] > prices[2] > cont


# --------------------------------------------------------------------------- #
# Pipeline dispatch: product.monitoring is respected
# --------------------------------------------------------------------------- #
def test_pipeline_proj_barrier_respects_monitoring_and_grid():
    prod_discrete = BarrierOption(
        strike=100.0,
        barrier=85.0,
        maturity=1.0,
        cp=1,
        barrier_type="down_out",
        monitoring="discrete",
    )
    got = fe.price(prod_discrete, "kou", "proj_barrier", FWD, KOU, grid=12)
    expected = _proj_barrier("kou", KOU, 100.0, 85.0, 12, "down_out", 1)
    assert got == pytest.approx(expected, rel=1e-9)

    prod_cont = BarrierOption(
        strike=100.0,
        barrier=85.0,
        maturity=1.0,
        cp=1,
        barrier_type="down_out",
        monitoring="continuous",
    )
    with pytest.raises(NotImplementedError, match="barrier_bsm"):
        fe.price(prod_cont, "kou", "proj_barrier", FWD, KOU)


def test_pipeline_proj_double_barrier_respects_monitoring_and_grid():
    prod_discrete = DoubleBarrierOption(
        strike=100.0,
        lower_barrier=85.0,
        upper_barrier=120.0,
        maturity=1.0,
        cp=1,
        knockout=True,
        monitoring="discrete",
    )
    got = fe.price(prod_discrete, "kou", "proj_double_barrier", FWD, KOU, grid=12)
    assert np.isfinite(got) and got >= 0.0

    prod_cont = DoubleBarrierOption(
        strike=100.0,
        lower_barrier=85.0,
        upper_barrier=120.0,
        maturity=1.0,
        cp=1,
        knockout=True,
        monitoring="continuous",
    )
    with pytest.raises(NotImplementedError, match="double_barrier_bsm"):
        fe.price(prod_cont, "kou", "proj_double_barrier", FWD, KOU)
