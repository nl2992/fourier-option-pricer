"""Markov regime-switching BSM model validation.

Degeneracy structure gives exact references:
- one regime  -> the CF and prices must reduce to plain BSM;
- zero generator -> the CF is the initial-probability mixture of BSM CFs;
- conditional on occupation times the price is BSM at the occupation-weighted
  variance, so every price lies between the all-low-vol and all-high-vol BSM
  prices.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.models.regime_switching import (
    RegimeSwitchingBsmParams,
    regime_switching_cf,
    regime_switching_cumulants,
)

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.01, T=1.0)
_STRIKES = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
_U = np.linspace(-40.0, 40.0, 161)

_TWO_REGIME = RegimeSwitchingBsmParams(
    sigmas=(0.15, 0.35),
    generator=((-1.0, 1.0), (2.0, -2.0)),
    initial_probs=(0.7, 0.3),
)


def test_single_regime_reduces_to_bsm_cf():
    rs = RegimeSwitchingBsmParams(sigmas=(0.2,), generator=((0.0,),), initial_probs=(1.0,))
    phi_rs = regime_switching_cf(_U, _FWD, rs)
    phi_bsm = fe.bsm_cf(_U, _FWD, fe.BsmParams(sigma=0.2))
    np.testing.assert_allclose(phi_rs, phi_bsm, atol=1e-12, rtol=0.0)


def test_zero_generator_is_bsm_mixture():
    rs = RegimeSwitchingBsmParams(
        sigmas=(0.1, 0.4),
        generator=((0.0, 0.0), (0.0, 0.0)),
        initial_probs=(0.6, 0.4),
    )
    phi_rs = regime_switching_cf(_U, _FWD, rs)
    mixture = 0.6 * fe.bsm_cf(_U, _FWD, fe.BsmParams(sigma=0.1)) + 0.4 * fe.bsm_cf(
        _U, _FWD, fe.BsmParams(sigma=0.4)
    )
    np.testing.assert_allclose(phi_rs, mixture, atol=1e-12, rtol=0.0)


def test_martingale_normalization():
    phi_minus_i = regime_switching_cf(np.array([-1j]), _FWD, _TWO_REGIME)
    np.testing.assert_allclose(phi_minus_i, [1.0 + 0.0j], atol=1e-12)


def test_cumulants_single_regime_match_bsm():
    rs = RegimeSwitchingBsmParams(sigmas=(0.25,), generator=((0.0,),), initial_probs=(1.0,))
    c1, c2, c4 = regime_switching_cumulants(_FWD, rs)
    b1, b2, b4 = fe.bsm_cumulants(_FWD, fe.BsmParams(sigma=0.25))
    assert abs(c1 - b1) < 1e-8
    assert abs(c2 - b2) < 1e-6
    assert abs(c4 - b4) < 1e-4


def test_two_regime_price_between_single_regime_bsm_prices():
    rs_prices = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, _TWO_REGIME)
    lo = fe.price_strip("bsm", "cos", _STRIKES, _FWD, fe.BsmParams(sigma=0.15))
    hi = fe.price_strip("bsm", "cos", _STRIKES, _FWD, fe.BsmParams(sigma=0.35))
    assert np.all(rs_prices >= np.minimum(lo, hi) - 1e-10)
    assert np.all(rs_prices <= np.maximum(lo, hi) + 1e-10)


def test_cos_agrees_with_carr_madan_and_hilbert():
    cos = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, _TWO_REGIME)
    cm = fe.price_strip(
        "regime_switching",
        "carr_madan",
        _STRIKES,
        _FWD,
        _TWO_REGIME,
        grid=fe.FFTGrid(N=16384, eta=0.05, alpha=1.5),
    )
    hil = fe.price_strip("regime_switching", "hilbert", _STRIKES, _FWD, _TWO_REGIME)
    np.testing.assert_allclose(cos, cm, atol=2e-4, rtol=0.0)
    np.testing.assert_allclose(cos, hil, atol=1e-6, rtol=0.0)


def test_put_call_parity():
    calls = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, _TWO_REGIME, cp=1)
    puts = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, _TWO_REGIME, cp=-1)
    np.testing.assert_allclose(calls - puts, _FWD.disc * (_FWD.F0 - _STRIKES), atol=1e-8)


def test_fast_switching_approaches_stationary_average_variance():
    """With very fast switching, integrated variance concentrates at the
    stationary mean, so the price approaches BSM at that effective vol."""
    speed = 400.0
    rs = RegimeSwitchingBsmParams(
        sigmas=(0.15, 0.35),
        generator=((-speed, speed), (speed, -speed)),
        initial_probs=(0.5, 0.5),
    )
    # symmetric chain -> stationary distribution (1/2, 1/2)
    var_eff = 0.5 * 0.15**2 + 0.5 * 0.35**2
    bsm_eff = fe.price_strip("bsm", "cos", _STRIKES, _FWD, fe.BsmParams(sigma=np.sqrt(var_eff)))
    rs_prices = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, rs)
    np.testing.assert_allclose(rs_prices, bsm_eff, atol=5e-2, rtol=0.0)


def test_param_validation():
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(sigmas=(), generator=(), initial_probs=())
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(
            sigmas=(0.2, -0.1), generator=((0, 0), (0, 0)), initial_probs=(0.5, 0.5)
        )
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(
            sigmas=(0.2, 0.3), generator=((-1.0, 0.5), (1.0, -1.0)), initial_probs=(0.5, 0.5)
        )
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(
            sigmas=(0.2, 0.3), generator=((-1.0, 1.0), (-2.0, 2.0)), initial_probs=(0.5, 0.5)
        )
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(
            sigmas=(0.2, 0.3), generator=((-1.0, 1.0), (1.0, -1.0)), initial_probs=(0.9, 0.3)
        )


# ── per-regime Merton jumps (regime-switching jump-diffusion) ──────────────

_TWO_REGIME_JD = RegimeSwitchingBsmParams(
    sigmas=(0.15, 0.30),
    generator=((-1.0, 1.0), (2.0, -2.0)),
    initial_probs=(0.7, 0.3),
    jump_intensities=(0.5, 2.0),
    jump_means=(-0.05, -0.10),
    jump_stds=(0.10, 0.15),
)


def test_zero_intensity_jumps_equal_no_jump_cf():
    with_zero = RegimeSwitchingBsmParams(
        sigmas=(0.15, 0.35),
        generator=((-1.0, 1.0), (2.0, -2.0)),
        initial_probs=(0.7, 0.3),
        jump_intensities=(0.0, 0.0),
        jump_means=(-0.1, 0.1),
        jump_stds=(0.2, 0.2),
    )
    phi_jd = regime_switching_cf(_U, _FWD, with_zero)
    phi_plain = regime_switching_cf(_U, _FWD, _TWO_REGIME)
    np.testing.assert_allclose(phi_jd, phi_plain, atol=1e-14, rtol=0.0)


def test_single_regime_with_jumps_reduces_to_merton():
    rs = RegimeSwitchingBsmParams(
        sigmas=(0.2,),
        generator=((0.0,),),
        initial_probs=(1.0,),
        jump_intensities=(0.8,),
        jump_means=(-0.05,),
        jump_stds=(0.1,),
    )
    merton = fe.MertonJDParams(sigma=0.2, lam=0.8, muj=-0.05, sigj=0.1)
    phi_rs = regime_switching_cf(_U, _FWD, rs)
    phi_mjd = fe.merton_jd_cf(_U, _FWD, merton)
    np.testing.assert_allclose(phi_rs, phi_mjd, atol=1e-12, rtol=0.0)

    c_rs = regime_switching_cumulants(_FWD, rs)
    c_mjd = fe.merton_jd_cumulants(_FWD, merton)
    assert abs(c_rs[0] - c_mjd[0]) < 1e-8
    assert abs(c_rs[1] - c_mjd[1]) < 1e-6
    assert abs(c_rs[2] - c_mjd[2]) < 1e-4


def test_martingale_normalization_with_jumps():
    phi_minus_i = regime_switching_cf(np.array([-1j]), _FWD, _TWO_REGIME_JD)
    np.testing.assert_allclose(phi_minus_i, [1.0 + 0.0j], atol=1e-12)


def test_zero_generator_is_merton_mixture():
    rs = RegimeSwitchingBsmParams(
        sigmas=(0.1, 0.3),
        generator=((0.0, 0.0), (0.0, 0.0)),
        initial_probs=(0.6, 0.4),
        jump_intensities=(0.5, 1.5),
        jump_means=(-0.05, -0.08),
        jump_stds=(0.1, 0.12),
    )
    phi_rs = regime_switching_cf(_U, _FWD, rs)
    m1 = fe.MertonJDParams(sigma=0.1, lam=0.5, muj=-0.05, sigj=0.1)
    m2 = fe.MertonJDParams(sigma=0.3, lam=1.5, muj=-0.08, sigj=0.12)
    mixture = 0.6 * fe.merton_jd_cf(_U, _FWD, m1) + 0.4 * fe.merton_jd_cf(_U, _FWD, m2)
    np.testing.assert_allclose(phi_rs, mixture, atol=1e-12, rtol=0.0)


def test_jump_prices_cross_engine_and_parity():
    cos = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, _TWO_REGIME_JD)
    hil = fe.price_strip("regime_switching", "hilbert", _STRIKES, _FWD, _TWO_REGIME_JD)
    np.testing.assert_allclose(cos, hil, atol=2e-6, rtol=0.0)
    puts = fe.price_strip("regime_switching", "cos", _STRIKES, _FWD, _TWO_REGIME_JD, cp=-1)
    np.testing.assert_allclose(cos - puts, _FWD.disc * (_FWD.F0 - _STRIKES), atol=1e-8)


@pytest.mark.slow
def test_two_regime_jd_matches_monte_carlo():
    """Exact simulation: draw the chain path via exponential holding times,
    then condition on occupation times -- diffusion variance, compensators,
    and per-regime Poisson jump counts are all exact, so there is no
    time-discretization bias at all."""
    rng = np.random.default_rng(23)
    n = 200_000
    p = _TWO_REGIME_JD
    sig = np.asarray(p.sigmas)
    lam = np.asarray(p.jump_intensities)
    muj = np.asarray(p.jump_means)
    sj = np.asarray(p.jump_stds)
    zeta = np.expm1(muj + 0.5 * sj * sj)
    rates = -np.diag(np.asarray(p.generator))

    tau = np.zeros((n, 2))
    state = (rng.random(n) < p.initial_probs[1]).astype(int)
    t = np.zeros(n)
    active = np.ones(n, dtype=bool)
    while active.any():
        idx = np.where(active)[0]
        hold = rng.exponential(1.0 / rates[state[idx]])
        end = np.minimum(t[idx] + hold, _FWD.T)
        tau[idx, state[idx]] += end - t[idx]
        t[idx] = end
        done = end >= _FWD.T - 1e-15
        state[idx[~done]] = 1 - state[idx[~done]]
        active[idx[done]] = False

    var = (sig**2 * tau).sum(axis=1)
    comp = ((-0.5 * sig**2 - lam * zeta) * tau).sum(axis=1)
    n_jumps = rng.poisson(lam * tau)
    jump_sum = np.zeros(n)
    for j in (0, 1):
        jump_sum += (
            rng.normal(0.0, 1.0, n) * sj[j] * np.sqrt(n_jumps[:, j]) + n_jumps[:, j] * muj[j]
        )
    x = comp + np.sqrt(var) * rng.standard_normal(n) + jump_sum

    s_t = _FWD.S0 * np.exp((_FWD.r - _FWD.q) * _FWD.T + x)
    K = 100.0
    payoff = np.exp(-_FWD.r * _FWD.T) * np.maximum(s_t - K, 0.0)
    mc, se = float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n))
    cf_price = float(
        fe.price_strip("regime_switching", "cos", np.array([K]), _FWD, _TWO_REGIME_JD)[0]
    )
    assert abs(cf_price - mc) < 4.0 * se, f"CF {cf_price:.4f} vs MC {mc:.4f} +/- {se:.4f}"


def test_jump_param_validation():
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(
            sigmas=(0.2, 0.3),
            generator=((-1.0, 1.0), (1.0, -1.0)),
            initial_probs=(0.5, 0.5),
            jump_intensities=(0.5,),
        )
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(
            sigmas=(0.2, 0.3),
            generator=((-1.0, 1.0), (1.0, -1.0)),
            initial_probs=(0.5, 0.5),
            jump_intensities=(-0.5, 0.5),
            jump_means=(0.0, 0.0),
            jump_stds=(0.1, 0.1),
        )
    with pytest.raises(ValueError):
        RegimeSwitchingBsmParams(
            sigmas=(0.2, 0.3),
            generator=((-1.0, 1.0), (1.0, -1.0)),
            initial_probs=(0.5, 0.5),
            jump_intensities=(0.5, 0.5),
            jump_means=(0.0, 0.0),
            jump_stds=(0.1, -0.1),
        )
