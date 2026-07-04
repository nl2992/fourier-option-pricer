"""PROJ double-barrier pricer validation.

Independent references:
- pushing one barrier far away must reproduce the single-barrier PROJ price;
- pushing both far away must reproduce the vanilla COS European price;
- under BSM with fine monitoring the price must approach the continuously
  monitored eigenfunction-expansion closed form from above (discrete KO bias);
- under Kou it must match a discretely monitored Monte Carlo with the same
  monitoring grid inside the confidence band.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.analytics.bsm_barrier import bsm_double_barrier_price
from foureng.models.registry import MODEL_REGISTRY
from foureng.pricers.proj import proj_double_barrier_price
from foureng.products.barrier import DoubleBarrierOption

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
_KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.6, eta1=25.0, eta2=15.0)


def _step_cf(model, params, dt):
    cf = MODEL_REGISTRY[model].cf
    fwd_dt = fe.ForwardSpec(S0=_FWD.S0, r=_FWD.r, q=_FWD.q, T=dt)
    drift = (_FWD.r - _FWD.q) * dt

    def inner(u):
        return np.exp(1j * u * drift) * np.asarray(cf(u, fwd_dt, params), dtype=np.complex128)

    return inner


def _price(model, params, *, K=100.0, L=80.0, U=125.0, M=252, cp=1, knockout=True, alph=7.0):
    return proj_double_barrier_price(
        _step_cf(model, params, _FWD.T / M),
        S0=_FWD.S0,
        r=_FWD.r,
        T=_FWD.T,
        K=K,
        L=L,
        U=U,
        M=M,
        knockout=knockout,
        cp=cp,
        q=_FWD.q,
        alph=alph,
    )


def test_far_barriers_reduce_to_vanilla():
    params = fe.BsmParams(sigma=0.2)
    v = _price("bsm", params, L=1e-7, U=1e9)
    european = float(fe.price_strip("bsm", "cos", np.array([100.0]), _FWD, params)[0])
    # M-step convolution accumulates O(M * dx^4) projection bias on a fixed
    # grid, so the far-barrier vanilla is close to, not identical to, COS.
    assert v == pytest.approx(european, abs=5e-3)


@pytest.mark.parametrize("cp", [1, -1])
def test_far_upper_barrier_matches_single_down_out(cp):
    """U -> infinity: double KO == single down-and-out at the same L."""
    from foureng.pricers.proj import proj_barrier_price

    step = _step_cf("kou", _KOU, _FWD.T / 252)
    double = _price("kou", _KOU, L=85.0, U=1e9, cp=cp)
    single = proj_barrier_price(
        step,
        S0=_FWD.S0,
        r=_FWD.r,
        T=_FWD.T,
        K=100.0,
        H=85.0,
        M=252,
        barrier_type="down_out",
        cp=cp,
        q=_FWD.q,
    )
    assert double == pytest.approx(single, abs=2e-3)


def test_bsm_matches_bgk_corrected_closed_form():
    """The discretely monitored PROJ price must match the eigenfunction
    closed form evaluated at Broadie-Glasserman-Kou continuity-corrected
    barriers L*exp(-beta*sigma*sqrt(dt)), U*exp(+beta*sigma*sqrt(dt)),
    and decrease monotonically toward the continuous limit."""
    params = fe.BsmParams(sigma=0.2)
    beta = 0.5826
    cont = bsm_double_barrier_price(
        _FWD.S0, 100.0, 80.0, 125.0, _FWD.r, _FWD.q, _FWD.T, params.sigma, cp=1, knockout=True
    )
    prev = np.inf
    for M in (126, 252):
        dt = _FWD.T / M
        proj = _price("bsm", params, M=M)
        shift = beta * params.sigma * np.sqrt(dt)
        bgk = bsm_double_barrier_price(
            _FWD.S0,
            100.0,
            80.0 * np.exp(-shift),
            125.0 * np.exp(shift),
            _FWD.r,
            _FWD.q,
            _FWD.T,
            params.sigma,
            cp=1,
            knockout=True,
        )
        assert proj == pytest.approx(bgk, rel=1e-2), f"M={M}: proj {proj} vs BGK {bgk}"
        assert cont - 1e-4 <= proj <= prev + 1e-6
        prev = proj


def test_kou_matches_discretely_monitored_monte_carlo():
    rng = np.random.default_rng(17)
    n_paths, M = 200_000, 26
    L, U, K = 80.0, 125.0, 100.0
    dt = _FWD.T / M
    sig, lam, p_up, e1, e2 = _KOU.sigma, _KOU.lam, _KOU.p, _KOU.eta1, _KOU.eta2
    zeta = p_up * e1 / (e1 - 1.0) + (1.0 - p_up) * e2 / (e2 + 1.0) - 1.0
    drift = (_FWD.r - _FWD.q - 0.5 * sig * sig - lam * zeta) * dt

    log_s = np.full(n_paths, np.log(_FWD.S0))
    alive = np.ones(n_paths, dtype=bool)
    for _ in range(M):
        x = drift + sig * np.sqrt(dt) * rng.standard_normal(n_paths)
        n_jumps = rng.poisson(lam * dt, size=n_paths)
        for j in range(1, int(n_jumps.max()) + 1):
            mask = n_jumps >= j
            n_active = int(mask.sum())
            up = rng.random(n_active) < p_up
            jump = np.where(
                up,
                rng.exponential(1.0 / e1, size=n_active),
                -rng.exponential(1.0 / e2, size=n_active),
            )
            x[mask] += jump
        log_s += x
        s = np.exp(log_s)
        alive &= (s > L) & (s < U)

    payoff = np.exp(-_FWD.r * _FWD.T) * np.where(alive, np.maximum(np.exp(log_s) - K, 0.0), 0.0)
    mc, se = float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_paths))
    proj = _price("kou", _KOU, K=K, L=L, U=U, M=M)
    assert abs(proj - mc) < 4.0 * se, f"PROJ {proj:.4f} vs MC {mc:.4f} +/- {se:.4f}"


def test_knock_out_bounds_and_corridor_monotonicity():
    vanilla = float(fe.price_strip("kou", "cos", np.array([100.0]), _FWD, _KOU)[0])
    wide = _price("kou", _KOU, L=70.0, U=140.0)
    narrow = _price("kou", _KOU, L=90.0, U=112.0)
    assert 0.0 <= narrow <= wide <= vanilla + 1e-3


def test_in_out_parity():
    ko = _price("kou", _KOU, knockout=True)
    ki = _price("kou", _KOU, knockout=False)
    vanilla_same_engine = _price("kou", _KOU, L=1e-7, U=1e9)
    assert ko + ki == pytest.approx(vanilla_same_engine, abs=1e-8)


def test_pipeline_dispatch():
    product = DoubleBarrierOption(
        strike=100.0,
        lower_barrier=80.0,
        upper_barrier=125.0,
        maturity=_FWD.T,
        cp=1,
        knockout=True,
    )
    via_price = fe.price(product, "kou", "proj_double_barrier", _FWD, _KOU)
    assert via_price > 0.0
    vanilla = float(fe.price_strip("kou", "cos", np.array([100.0]), _FWD, _KOU)[0])
    assert via_price < vanilla


def test_validation_errors():
    step = _step_cf("kou", _KOU, _FWD.T / 12)
    with pytest.raises(ValueError):
        proj_double_barrier_price(step, S0=100.0, r=0.03, T=1.0, K=100.0, L=110.0, U=125.0, M=12)
    with pytest.raises(ValueError):
        proj_double_barrier_price(
            step, S0=100.0, r=0.03, T=1.0, K=100.0, L=80.0, U=125.0, M=12, cp=0
        )
    with pytest.raises(ValueError):
        proj_double_barrier_price(
            step, S0=100.0, r=0.03, T=1.0, K=100.0, L=80.0, U=125.0, M=12, N=1000
        )
    product = DoubleBarrierOption(
        strike=100.0,
        lower_barrier=80.0,
        upper_barrier=125.0,
        maturity=1.0,
        rebate=1.0,
    )
    with pytest.raises(NotImplementedError):
        fe.price(product, "kou", "proj_double_barrier", _FWD, _KOU)
