"""Accuracy of the PROJ Bermudan, step, swing and survival recursions (A5).

These four recursions shared the grid-domain bug fixed for ``proj_barrier``
under A3 (see ``tests/methods/test_proj_barrier_accuracy.py``): ``dx`` was
sized from the full FFT length ``N`` instead of the ``K_half = N // 2`` live
value nodes, silently halving the represented domain ``[-alph, alph]``. Where
a recursion also snaps a boundary (barrier or damping level) to the nearest
grid node, that snap picks up an additional O(dx) bias per monitoring date,
just like the barrier pricer's hard knock-out did.

The fix, applied identically to each recursion:

- size ``dx`` from ``K_half`` so the live nodes span the full domain;
- where there is a boundary to snap (``proj_step_price``,
  ``proj_survival_probability``), weight the straddling node by its sub-cell
  position instead of a hard kill (``_apply_barrier_kill``, generalized with
  a ``dead_mult`` parameter so the same weighting covers both a hard
  knock-out and step's soft occupation-time damping);
- clip the one-step transfer function to the analytic contraction bound
  (``exp(-r*dt)`` for discounted recursions, ``1.0`` for the undiscounted
  survival probability) so a slowly decaying short-step CF cannot alias past
  it under repeated convolution;
- Richardson-extrapolate two grid sizes (``N``, ``2*N``) to cancel the
  leading O(dx) error, for every recursion except the swing option's plain
  node-wise exercise max, which already leaves only an O(dx^2) kink error
  (Richardson is still applied there too, since correcting the domain halves
  the node density for a given ``N`` and extrapolation recovers it).

``proj_bermudan_put``'s exercise boundary is already handled by a continuous
quadratic interpolation (not a node snap), so only the domain fix, transfer
clipping and Richardson apply there.

Measured accuracy after the fix (see the module-level assertions below for
the exact tolerances used):

- Bermudan put vs. the exact FO2009 COS-Bermudan engine (``cos_bermudan_price``):
  ~1e-12 (bsm), ~1e-6 (vg), ~1e-11 (kou) at 12 and 52 monitoring dates.
- Step option: the rho=0 limit matches the European vanilla
  (``cos_improved``) to ~1e-4, and the rho->infinity limit matches the fixed
  ``proj_barrier_price`` knock-out to ~1e-3 at 52 and 252 monitoring dates.
- Swing option: one right matches ``cos_bermudan_price`` to ~1e-4, and
  ``n_rights >= n_dates`` matches the sum of the per-date European options to
  ~1e-3.
- Survival probability: converges toward the Broadie-Glasserman-Kou (1997)
  continuity-corrected BSM first-passage probability (the standard analytic
  approximation for discretely monitored barriers) to ~1e-3 - 4e-3.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

import foureng as fe
from foureng.models.base import ForwardSpec
from foureng.models.registry import MODEL_REGISTRY
from foureng.pipeline import price_strip
from foureng.pricers.cos_bermudan import cos_bermudan_price_strip
from foureng.pricers.proj import (
    proj_auto_grid,
    proj_barrier_price,
    proj_bermudan_put,
    proj_step_price,
    proj_survival_probability,
    proj_swing_price,
)

pytestmark = [pytest.mark.derived_reference]

FWD = ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
BSM = fe.BsmParams(sigma=0.2)
VG = fe.VGParams(sigma=0.2, nu=0.3, theta=-0.1)
KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)


def _step_cf(model: str, params, S0: float, r: float, q: float, dt: float):
    cf = MODEL_REGISTRY[model].cf
    fwd_dt = ForwardSpec(S0=S0, r=r, q=q, T=dt)
    drift = (r - q) * dt
    return lambda u: np.exp(1j * u * drift) * np.asarray(cf(u, fwd_dt, params), dtype=complex)


# --------------------------------------------------------------------------- #
# proj_bermudan_put vs the exact FO2009 COS-Bermudan engine
# --------------------------------------------------------------------------- #
BERMUDAN_CASES = [
    ("bsm", BSM, 1e-9),
    ("vg", VG, 1e-5),
    ("kou", KOU, 1e-7),
]


@pytest.mark.parametrize("model,params,tol", BERMUDAN_CASES)
@pytest.mark.parametrize("M", [12, 52])
def test_proj_bermudan_matches_cos_bermudan(model, params, tol, M):
    S0, r, q, T, W = FWD.S0, FWD.r, FWD.q, FWD.T, 100.0
    fwd = ForwardSpec(S0=S0, r=r, q=q, T=T)
    dt = T / M
    ex_times = np.arange(1, M + 1) * dt
    cums = MODEL_REGISTRY[model].cumulants(fwd, params)
    alph = 12.0 * np.sqrt(abs(cums[1]) + np.sqrt(abs(cums[2])))

    proj = proj_bermudan_put(
        _step_cf(model, params, S0, r, q, dt), S0=S0, r=r, T=T, W=W, M=M, N=2**15, alph=alph
    )
    cos = cos_bermudan_price_strip(model, fwd, params, np.array([W]), T, ex_times, cp=-1, N=2048)[0]
    assert proj == pytest.approx(cos, abs=tol)


# --------------------------------------------------------------------------- #
# proj_step_price: rho=0 -> vanilla, rho->infinity -> proj_barrier_price
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model,params", [("bsm", BSM), ("kou", KOU)])
@pytest.mark.parametrize("M", [52, pytest.param(252, marks=pytest.mark.slow)])
def test_proj_step_zero_rho_matches_vanilla(model, params, M):
    K, B = 100.0, 90.0
    cums = MODEL_REGISTRY[model].cumulants(FWD, params)
    grid = proj_auto_grid(cums, N=1 << 15, L=10.0)
    dt = FWD.T / M
    vanilla = float(price_strip(model, "cos_improved", np.array([K]), FWD, params)[0])
    step0 = proj_step_price(
        _step_cf(model, params, FWD.S0, FWD.r, FWD.q, dt),
        S0=FWD.S0,
        r=FWD.r,
        T=FWD.T,
        K=K,
        B=B,
        rho=0.0,
        M=M,
        q=FWD.q,
        N=grid.N,
        alph=grid.alph,
    )
    assert step0 == pytest.approx(vanilla, abs=1e-3)


@pytest.mark.parametrize("model,params", [("bsm", BSM), ("kou", KOU)])
@pytest.mark.parametrize("M", [52, pytest.param(252, marks=pytest.mark.slow)])
def test_proj_step_infinite_rho_matches_barrier(model, params, M):
    K, B = 100.0, 90.0
    cums = MODEL_REGISTRY[model].cumulants(FWD, params)
    grid = proj_auto_grid(cums, N=1 << 15, L=10.0)
    dt = FWD.T / M
    sc = _step_cf(model, params, FWD.S0, FWD.r, FWD.q, dt)
    ko_ref = proj_barrier_price(
        sc, S0=FWD.S0, r=FWD.r, T=FWD.T, K=K, H=B, M=M, barrier_type="down_out", cp=1, q=FWD.q
    )
    stepinf = proj_step_price(
        sc,
        S0=FWD.S0,
        r=FWD.r,
        T=FWD.T,
        K=K,
        B=B,
        rho=1e9,
        M=M,
        q=FWD.q,
        N=grid.N,
        alph=grid.alph,
    )
    assert stepinf == pytest.approx(ko_ref, abs=1.5e-3)


# --------------------------------------------------------------------------- #
# proj_swing_price: n_rights=1 -> Bermudan, n_rights>=n_dates -> sum of
# per-date Europeans
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model,params", [("bsm", BSM), ("kou", KOU)])
@pytest.mark.parametrize("cp", [1, -1])
def test_proj_swing_single_right_matches_cos_bermudan(model, params, cp):
    M = 12
    fwd = FWD
    dt = fwd.T / M
    cums = MODEL_REGISTRY[model].cumulants(fwd, params)
    grid = proj_auto_grid(cums, N=1 << 15, L=10.0)
    swing1 = proj_swing_price(
        _step_cf(model, params, fwd.S0, fwd.r, fwd.q, dt),
        S0=fwd.S0,
        r=fwd.r,
        T=fwd.T,
        K=100.0,
        M=M,
        n_rights=1,
        cp=cp,
        q=fwd.q,
        N=grid.N,
        alph=grid.alph,
    )
    ex_times = np.arange(1, M + 1) * dt
    berm = cos_bermudan_price_strip(
        model, fwd, params, np.array([100.0]), fwd.T, ex_times, cp=cp, N=2048
    )[0]
    assert swing1 == pytest.approx(berm, abs=2e-4)


@pytest.mark.parametrize("model,params", [("bsm", BSM), ("kou", KOU)])
@pytest.mark.parametrize("cp", [1, -1])
def test_proj_swing_full_rights_is_sum_of_europeans(model, params, cp):
    M = 12
    fwd = FWD
    dt = fwd.T / M
    cums = MODEL_REGISTRY[model].cumulants(fwd, params)
    grid = proj_auto_grid(cums, N=1 << 15, L=10.0)
    swingM = proj_swing_price(
        _step_cf(model, params, fwd.S0, fwd.r, fwd.q, dt),
        S0=fwd.S0,
        r=fwd.r,
        T=fwd.T,
        K=100.0,
        M=M,
        n_rights=M,
        cp=cp,
        q=fwd.q,
        N=grid.N,
        alph=grid.alph,
    )
    total = 0.0
    for m in range(1, M + 1):
        t_m = fwd.T * m / M
        fwd_m = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=t_m)
        total += float(price_strip(model, "cos", np.array([100.0]), fwd_m, params, cp=cp)[0])
    assert swingM == pytest.approx(total, abs=1e-3)


# --------------------------------------------------------------------------- #
# proj_survival_probability vs the Broadie-Glasserman-Kou (1997)
# continuity-corrected BSM first-passage probability
# --------------------------------------------------------------------------- #
def _bsm_continuous_survival(
    S0: float, B: float, r: float, q: float, sigma: float, T: float
) -> float:
    """Closed-form P(min_{[0,T]} S_t > B) for continuously monitored GBM
    (reflection principle for a drifted Brownian motion)."""
    mu = r - q - 0.5 * sigma * sigma
    h = np.log(S0 / B)
    d1 = (h + mu * T) / (sigma * np.sqrt(T))
    d2 = (-h + mu * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) - np.exp(-2.0 * mu * h / sigma**2) * norm.cdf(d2))


# Broadie, Glasserman & Kou (1997), "A continuity correction for discrete
# barrier options", Mathematical Finance 7(4): shifting the barrier by
# exp(-beta * sigma * sqrt(dt)) (away from spot, for a lower barrier) turns
# the continuous-monitoring formula into an O(dt) accurate approximation of
# the discretely monitored probability, isolating the residual discrete
# monitoring effect (O(1/sqrt(M)) if left uncorrected) from genuine PROJ grid
# error.
_BETA_BGK = 0.5826


@pytest.mark.parametrize("M", [12, 52, pytest.param(252, marks=pytest.mark.slow)])
def test_proj_survival_probability_matches_bgk_continuity_correction(M):
    S0, B, sigma, r, q, T = 100.0, 70.0, 0.25, 0.05, 0.0, 1.0
    params = fe.BsmParams(sigma=sigma)
    dt = T / M
    surv = proj_survival_probability(_step_cf("bsm", params, S0, r, q, dt), S0=S0, B=B, M=M)
    B_adj = B * np.exp(-_BETA_BGK * sigma * np.sqrt(dt))
    bgk = _bsm_continuous_survival(S0, B_adj, r, q, sigma, T)
    assert surv == pytest.approx(bgk, abs=5e-3)


def test_proj_survival_probability_in_valid_range():
    """Regression guard: the recursion must stay a probability regardless of
    grid size, unlike the pre-fix domain bug which could compound bias with M."""
    params = fe.BsmParams(sigma=0.3)
    for M in (12, 52, 252):
        dt = 1.0 / M
        surv = proj_survival_probability(
            _step_cf("bsm", params, 100.0, 0.05, 0.0, dt), S0=100.0, B=60.0, M=M
        )
        assert 0.0 <= surv <= 1.0
