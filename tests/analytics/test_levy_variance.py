"""Levy discrete variance-swap fair strike validation.

References that pin the formula down independently:
- under BSM it must reproduce bsm_variance_swap exactly;
- a jump model with intensity zero must collapse to BSM;
- under Kou it must sit inside the Monte Carlo confidence band;
- Merton has hand-computable squared-return moments.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.analytics.bsm_variance import bsm_variance_swap
from foureng.analytics.levy_variance import levy_variance_fair_strike, levy_variance_swap
from foureng.products.variance import VarianceSwap

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
_TIMES = np.linspace(1.0 / 12.0, 1.0, 12)
_KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.6, eta1=25.0, eta2=15.0)


def _swap(notional: float = 1.0) -> VarianceSwap:
    return VarianceSwap(maturity=1.0, sampling_times=_TIMES, notional=notional)


def test_bsm_matches_closed_form_exactly():
    params = fe.BsmParams(sigma=0.22)
    levy = levy_variance_swap("bsm", _FWD, params, _swap(notional=2.5))
    ref = bsm_variance_swap(_FWD, params, _swap(notional=2.5))
    assert levy == pytest.approx(ref, abs=1e-12)


def test_zero_intensity_jump_models_collapse_to_bsm():
    sigma = 0.2
    bsm_val = levy_variance_fair_strike("bsm", _FWD, fe.BsmParams(sigma=sigma), _TIMES)
    kou_val = levy_variance_fair_strike(
        "kou", _FWD, fe.KouParams(sigma=sigma, lam=0.0, p=0.5, eta1=20.0, eta2=20.0), _TIMES
    )
    merton_val = levy_variance_fair_strike(
        "merton_jd", _FWD, fe.MertonJDParams(sigma=sigma, lam=0.0, muj=-0.1, sigj=0.2), _TIMES
    )
    assert kou_val == pytest.approx(bsm_val, rel=1e-12)
    assert merton_val == pytest.approx(bsm_val, rel=1e-12)


def test_merton_matches_hand_computed_moments():
    """Merton squared-return moments are elementary:
    per period, mean = (r - q + omega + lam*muj)*dt and
    variance = (sigma^2 + lam*(muj^2 + sigj^2))*dt."""
    p = fe.MertonJDParams(sigma=0.15, lam=0.8, muj=-0.05, sigj=0.1)
    dt = float(_TIMES[0])
    omega = -0.5 * p.sigma**2 - p.lam * (np.exp(p.muj + 0.5 * p.sigj**2) - 1.0)
    mean = (_FWD.r - _FWD.q + omega + p.lam * p.muj) * dt
    var = (p.sigma**2 + p.lam * (p.muj**2 + p.sigj**2)) * dt
    expected = 12.0 * (mean**2 + var) / 1.0  # 12 equal periods, T = 1
    got = levy_variance_fair_strike("merton_jd", _FWD, p, _TIMES)
    assert got == pytest.approx(expected, rel=1e-10)


def test_jumps_increase_fair_strike():
    sigma = 0.2
    no_jumps = levy_variance_fair_strike("bsm", _FWD, fe.BsmParams(sigma=sigma), _TIMES)
    with_jumps = levy_variance_fair_strike(
        "kou", _FWD, fe.KouParams(sigma=sigma, lam=2.0, p=0.5, eta1=20.0, eta2=20.0), _TIMES
    )
    assert with_jumps > no_jumps


def test_kou_matches_monte_carlo():
    rng = np.random.default_rng(11)
    n_paths, M = 150_000, 12
    dt = 1.0 / M
    sig, lam, p_up, e1, e2 = _KOU.sigma, _KOU.lam, _KOU.p, _KOU.eta1, _KOU.eta2
    zeta = p_up * e1 / (e1 - 1.0) + (1.0 - p_up) * e2 / (e2 + 1.0) - 1.0
    drift = (_FWD.r - _FWD.q - 0.5 * sig * sig - lam * zeta) * dt

    returns = drift + sig * np.sqrt(dt) * rng.standard_normal((n_paths, M))
    n_jumps = rng.poisson(lam * dt, size=(n_paths, M))
    for j in range(1, int(n_jumps.max()) + 1):
        mask = n_jumps >= j
        n_active = int(mask.sum())
        up = rng.random(n_active) < p_up
        jump = np.where(
            up,
            rng.exponential(1.0 / e1, size=n_active),
            -rng.exponential(1.0 / e2, size=n_active),
        )
        add = np.zeros((n_paths, M))
        add[mask] = jump
        returns += add

    rv = (returns**2).sum(axis=1) / 1.0
    mc, se = float(rv.mean()), float(rv.std(ddof=1) / np.sqrt(n_paths))
    analytic = levy_variance_fair_strike("kou", _FWD, _KOU, _TIMES)
    assert abs(analytic - mc) < 4.0 * se, f"analytic {analytic:.6f} vs MC {mc:.6f} +/- {se:.6f}"


def test_pipeline_dispatch_and_discounting():
    direct = levy_variance_swap("kou", _FWD, _KOU, _swap(notional=3.0))
    via_price = fe.price(_swap(notional=3.0), "kou", "variance_levy_analytic", _FWD, _KOU)
    assert via_price == pytest.approx(direct, abs=0.0)
    # discounted at product maturity
    strike = levy_variance_fair_strike("kou", _FWD, _KOU, _TIMES, maturity=1.0)
    assert direct == pytest.approx(np.exp(-_FWD.r * 1.0) * 3.0 * strike, rel=1e-14)


def test_rejects_unsupported_model_and_bad_times():
    heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.35, rho=-0.6, v0=0.04)
    with pytest.raises(ValueError):
        levy_variance_fair_strike("heston", _FWD, heston, _TIMES)
    with pytest.raises(ValueError):
        levy_variance_fair_strike("kou", _FWD, _KOU, np.array([0.5, 0.2]))
    with pytest.raises(NotImplementedError):
        fe.price(_swap(), "heston", "variance_mc", _FWD, heston)
