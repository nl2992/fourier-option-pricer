"""Structural CDS / first-passage survival validation.

Independent references:
- with one monitoring date the survival probability is an exact normal CDF
  under BSM;
- the full survival curve must match a first-passage Monte Carlo on the
  same monthly monitoring grid (BSM and Kou);
- the leg assembly reproduces the credit triangle spread = (1-R)*lambda for
  a synthetic exponential survival curve, independent of the PROJ engine;
- structural monotonicities: barrier up => spread up, recovery up => spread
  down, far barrier => spread ~ 0.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

import foureng as fe
from foureng.analytics.levy_credit import (
    cds_par_spread_from_survival,
    levy_cds_spread,
    levy_survival_curve,
)

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.0, T=2.0)
_KOU = fe.KouParams(sigma=0.2, lam=1.0, p=0.4, eta1=20.0, eta2=8.0)
_BSM = fe.BsmParams(sigma=0.25)
_B = 70.0
_QUARTERS = np.linspace(0.25, 2.0, 8)


def test_single_date_survival_is_normal_cdf():
    """M=1: survival = P(S_dt > B), a plain lognormal tail probability."""
    dt = 0.25
    q = levy_survival_curve(
        "bsm", _FWD, _BSM, default_barrier=_B, horizons=np.array([dt]), monitoring_dt=dt
    )
    sig = _BSM.sigma
    d = (np.log(_FWD.S0 / _B) + (_FWD.r - _FWD.q - 0.5 * sig * sig) * dt) / (sig * np.sqrt(dt))
    assert q[0] == pytest.approx(float(norm.cdf(d)), abs=5e-5)


@pytest.mark.parametrize("model, params", [("bsm", _BSM), ("kou", _KOU)])
def test_survival_curve_matches_first_passage_monte_carlo(model, params):
    rng = np.random.default_rng(41)
    n_paths, n_steps = 200_000, 24  # monthly over 2y
    dt = _FWD.T / n_steps

    if model == "bsm":
        sig = params.sigma

        def increment(n):
            return (_FWD.r - _FWD.q - 0.5 * sig * sig) * dt + sig * np.sqrt(
                dt
            ) * rng.standard_normal(n)
    else:
        sig, lam, p_up, e1, e2 = params.sigma, params.lam, params.p, params.eta1, params.eta2
        zeta = p_up * e1 / (e1 - 1.0) + (1.0 - p_up) * e2 / (e2 + 1.0) - 1.0

        def increment(n):
            x = (_FWD.r - _FWD.q - 0.5 * sig * sig - lam * zeta) * dt + sig * np.sqrt(
                dt
            ) * rng.standard_normal(n)
            n_jumps = rng.poisson(lam * dt, size=n)
            for j in range(1, int(n_jumps.max()) + 1):
                mask = n_jumps >= j
                n_active = int(mask.sum())
                up = rng.random(n_active) < p_up
                x[mask] += np.where(
                    up,
                    rng.exponential(1.0 / e1, size=n_active),
                    -rng.exponential(1.0 / e2, size=n_active),
                )
            return x

    log_b = np.log(_B / _FWD.S0)
    log_s = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    mc_survival = []
    for k in range(1, n_steps + 1):
        log_s += increment(n_paths)
        alive &= log_s > log_b
        if k % 3 == 0:  # quarter ends
            mc_survival.append(alive.mean())
    mc_survival = np.asarray(mc_survival)

    q = levy_survival_curve(
        model, _FWD, params, default_barrier=_B, horizons=_QUARTERS, monitoring_dt=dt
    )
    se = np.sqrt(mc_survival * (1.0 - mc_survival) / n_paths)
    assert np.all(np.abs(q - mc_survival) < 4.0 * se + 2e-3), (
        f"max diff {np.max(np.abs(q - mc_survival)):.5f}"
    )


def test_credit_triangle_on_synthetic_survival():
    """Exponential survival must reproduce spread = (1-R) lambda up to the
    discrete-accrual convention."""
    lam, recovery, r = 0.02, 0.4, 0.03
    times = np.linspace(0.25, 5.0, 20)
    q = np.exp(-lam * times)
    spread = cds_par_spread_from_survival(q, times, r, recovery)
    assert spread == pytest.approx((1.0 - recovery) * lam, rel=5e-3)


def test_survival_monotone_in_horizon_and_barrier():
    q = levy_survival_curve(
        "kou", _FWD, _KOU, default_barrier=_B, horizons=_QUARTERS, monitoring_dt=1.0 / 12.0
    )
    assert np.all(np.diff(q) <= 1e-10)
    assert np.all((q >= 0.0) & (q <= 1.0))
    q_low_b = levy_survival_curve(
        "kou", _FWD, _KOU, default_barrier=50.0, horizons=_QUARTERS, monitoring_dt=1.0 / 12.0
    )
    assert np.all(q_low_b >= q - 1e-10)


def test_spread_monotonicities():
    kw = dict(recovery=0.4, maturity=2.0, payments_per_year=4, monitoring_dt=1.0 / 12.0)
    s_low = levy_cds_spread("kou", _FWD, _KOU, default_barrier=55.0, **kw)
    s_high = levy_cds_spread("kou", _FWD, _KOU, default_barrier=75.0, **kw)
    assert 0.0 < s_low < s_high
    s_r0 = levy_cds_spread(
        "kou",
        _FWD,
        _KOU,
        default_barrier=70.0,
        recovery=0.0,
        maturity=2.0,
        payments_per_year=4,
        monitoring_dt=1.0 / 12.0,
    )
    s_r6 = levy_cds_spread(
        "kou",
        _FWD,
        _KOU,
        default_barrier=70.0,
        recovery=0.6,
        maturity=2.0,
        payments_per_year=4,
        monitoring_dt=1.0 / 12.0,
    )
    assert s_r6 < s_r0
    assert s_r6 == pytest.approx(0.4 * s_r0, rel=1e-10)  # spread linear in (1-R)


def test_far_barrier_gives_negligible_spread():
    s = levy_cds_spread(
        "bsm", _FWD, fe.BsmParams(sigma=0.15), default_barrier=10.0, recovery=0.4, maturity=1.0
    )
    assert 0.0 <= s < 1e-6


def test_jumps_widen_spreads():
    """Adding downward jumps at equal diffusion vol must raise the spread."""
    kw = dict(default_barrier=70.0, recovery=0.4, maturity=2.0)
    s_diff = levy_cds_spread("bsm", _FWD, fe.BsmParams(sigma=0.2), **kw)
    s_jump = levy_cds_spread("kou", _FWD, _KOU, **kw)
    assert s_jump > s_diff


def test_validation_errors():
    heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.35, rho=-0.6, v0=0.04)
    with pytest.raises(ValueError):
        levy_survival_curve("heston", _FWD, heston, default_barrier=_B, horizons=np.array([1.0]))
    with pytest.raises(ValueError):
        levy_survival_curve("kou", _FWD, _KOU, default_barrier=120.0, horizons=np.array([1.0]))
    with pytest.raises(ValueError):
        cds_par_spread_from_survival(np.array([0.9]), np.array([1.0]), 0.03, 1.5)
    with pytest.raises(ValueError):
        levy_cds_spread("kou", _FWD, _KOU, default_barrier=_B, recovery=0.4, maturity=-1.0)
