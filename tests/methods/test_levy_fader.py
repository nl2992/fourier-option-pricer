"""Levy fader option pricer validation.

Independent references:
- an all-encompassing range makes fade-in == vanilla for any model;
- a single monitoring date at maturity has an exact BSM decomposition into
  undiscounted calls and N(d2) digitals;
- BSM and Kou full-path Monte Carlo on the same monitoring grid;
- fade-in + fade-out == vanilla, and fade-in grows with the range.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

import foureng as fe
from foureng.pricers.fader import levy_fader_price
from foureng.products.fader import FaderOption

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
_TIMES = np.linspace(1.0 / 12.0, 1.0, 12)
_KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.6, eta1=25.0, eta2=15.0)
_BSM = fe.BsmParams(sigma=0.2)


def _product(**kw) -> FaderOption:
    base = dict(
        strike=100.0,
        maturity=1.0,
        cp=1,
        lower=85.0,
        upper=120.0,
        monitoring_times=_TIMES,
        fade_type="in",
    )
    base.update(kw)
    return FaderOption(**base)


@pytest.mark.parametrize("model, params", [("bsm", _BSM), ("kou", _KOU)])
def test_all_encompassing_range_equals_vanilla(model, params):
    prod = _product(lower=1e-6, upper=1e6)
    fade_in = levy_fader_price(model, _FWD, params, prod)
    vanilla = float(fe.price_strip(model, "cos", np.array([100.0]), _FWD, params)[0])
    assert fade_in == pytest.approx(vanilla, abs=1e-3)


def test_single_date_at_maturity_matches_bsm_decomposition():
    """1_{L<S_T<U} (S_T-K)^+ with K < L decomposes into undiscounted calls
    plus digitals: E = [C_u(L) + (L-K) P(S_T>L)] - [C_u(U) + (U-K) P(S_T>U)]."""
    K, Lb, Ub = 90.0, 95.0, 115.0
    prod = _product(strike=K, lower=Lb, upper=Ub, monitoring_times=np.array([1.0]))
    got = levy_fader_price("bsm", _FWD, _BSM, prod)

    F, T, sig = _FWD.F0, _FWD.T, _BSM.sigma

    def _call_u(k):
        d1 = (np.log(F / k) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        return F * norm.cdf(d1) - k * norm.cdf(d2)

    def _digital(k):
        d2 = (np.log(F / k) - 0.5 * sig * sig * T) / (sig * np.sqrt(T))
        return norm.cdf(d2)

    expected = _FWD.disc * (
        (_call_u(Lb) + (Lb - K) * _digital(Lb)) - (_call_u(Ub) + (Ub - K) * _digital(Ub))
    )
    assert got == pytest.approx(expected, abs=5e-6)


@pytest.mark.parametrize("model, params", [("bsm", _BSM), ("kou", _KOU)])
def test_matches_full_path_monte_carlo(model, params):
    rng = np.random.default_rng(31)
    n_paths = 200_000
    prod = _product()
    all_times = np.concatenate((_TIMES, [] if _TIMES[-1] == 1.0 else [1.0]))
    dts = np.diff(np.concatenate(([0.0], all_times)))

    if model == "bsm":
        sig = params.sigma

        def increment(dt, n):
            return (_FWD.r - _FWD.q - 0.5 * sig * sig) * dt + sig * np.sqrt(
                dt
            ) * rng.standard_normal(n)
    else:
        sig, lam, p_up, e1, e2 = params.sigma, params.lam, params.p, params.eta1, params.eta2
        zeta = p_up * e1 / (e1 - 1.0) + (1.0 - p_up) * e2 / (e2 + 1.0) - 1.0

        def increment(dt, n):
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

    log_s = np.full(n_paths, np.log(_FWD.S0))
    n_in = np.zeros(n_paths)
    for t_k, dt in zip(all_times, dts):
        log_s += increment(dt, n_paths)
        if t_k in _TIMES:
            s = np.exp(log_s)
            n_in += ((s > prod.lower) & (s < prod.upper)).astype(float)

    s_T = np.exp(log_s)
    payoff = np.exp(-_FWD.r * 1.0) * (n_in / len(_TIMES)) * np.maximum(s_T - prod.strike, 0.0)
    mc, se = float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_paths))
    cf = levy_fader_price(model, _FWD, params, prod)
    assert abs(cf - mc) < 4.0 * se, f"CF {cf:.4f} vs MC {mc:.4f} +/- {se:.4f}"


def test_fade_in_plus_fade_out_is_vanilla():
    fade_in = levy_fader_price("kou", _FWD, _KOU, _product(fade_type="in"))
    fade_out = levy_fader_price("kou", _FWD, _KOU, _product(fade_type="out"))
    vanilla = float(fe.price_strip("kou", "cos", np.array([100.0]), _FWD, _KOU)[0])
    assert fade_in + fade_out == pytest.approx(vanilla, abs=1e-3)


def test_fade_in_monotone_in_range():
    narrow = levy_fader_price("kou", _FWD, _KOU, _product(lower=95.0, upper=108.0))
    wide = levy_fader_price("kou", _FWD, _KOU, _product(lower=80.0, upper=130.0))
    vanilla = float(fe.price_strip("kou", "cos", np.array([100.0]), _FWD, _KOU)[0])
    assert 0.0 < narrow < wide < vanilla


def test_put_fader_via_pipeline():
    prod = _product(cp=-1)
    via_price = fe.price(prod, "kou", "fader_cf", _FWD, _KOU)
    direct = levy_fader_price("kou", _FWD, _KOU, prod)
    assert via_price == pytest.approx(direct, abs=0.0)
    vanilla_put = float(fe.price_strip("kou", "cos", np.array([100.0]), _FWD, _KOU, cp=-1)[0])
    assert 0.0 < via_price < vanilla_put + 1e-10


def test_validation_and_errors():
    heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.35, rho=-0.6, v0=0.04)
    with pytest.raises(ValueError):
        levy_fader_price("heston", _FWD, heston, _product())
    with pytest.raises(ValueError):
        FaderOption(strike=100.0, maturity=1.0, lower=120.0, upper=85.0, monitoring_times=_TIMES)
    with pytest.raises(ValueError):
        FaderOption(
            strike=100.0,
            maturity=1.0,
            lower=85.0,
            upper=120.0,
            monitoring_times=_TIMES,
            fade_type="sideways",
        )
    with pytest.raises(NotImplementedError):
        fe.price(_product(), "kou", "monte_carlo", _FWD, _KOU)
