"""PROJ swing-option validation.

Two exact degeneracies anchor the dynamic program:
- n_rights = 1 is the Bermudan option (compared against the independent
  COS-Bermudan engine);
- n_rights = n_dates makes every ITM date exercisable, so the value is the
  sum of the per-date European options (compared against COS Europeans).
In between, the value must be increasing and concave in the number of
rights and subadditive against the Bermudan.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.models.registry import MODEL_REGISTRY
from foureng.products.swing import SwingOption

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
_KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.6, eta1=25.0, eta2=15.0)
_BSM = fe.BsmParams(sigma=0.2)
_M = 12


def _swing(model, params, *, n_rights, cp=1, K=100.0, M=_M):
    prod = SwingOption(strike=K, maturity=_FWD.T, cp=cp, n_rights=n_rights, n_exercise_dates=M)
    return fe.price(prod, model, "proj_swing", _FWD, params)


@pytest.mark.parametrize("model, params", [("bsm", _BSM), ("kou", _KOU)])
@pytest.mark.parametrize("cp", [1, -1])
def test_full_rights_is_sum_of_europeans(model, params, cp):
    """n_rights = M: exercise whenever ITM at every date, so the value is
    the sum of the M European options with maturities t_1..t_M."""
    swing = _swing(model, params, n_rights=_M, cp=cp)
    total = 0.0
    for m in range(1, _M + 1):
        t_m = _FWD.T * m / _M
        fwd_m = fe.ForwardSpec(S0=_FWD.S0, r=_FWD.r, q=_FWD.q, T=t_m)
        total += float(fe.price_strip(model, "cos", np.array([100.0]), fwd_m, params, cp=cp)[0])
    assert swing == pytest.approx(total, rel=2e-3)


def test_single_right_put_matches_cos_bermudan():
    """n_rights = 1 must reproduce the Bermudan put from the independent
    COS-Bermudan engine."""
    from foureng.products.bermudan import BermudanOption

    swing = _swing("kou", _KOU, n_rights=1, cp=-1)
    product = BermudanOption(
        strike=100.0,
        maturity=_FWD.T,
        cp=-1,
        exercise_times=np.linspace(_FWD.T / _M, _FWD.T, _M),
    )
    bermudan = fe.price(product, "kou", "cos_bermudan", _FWD, _KOU)
    assert swing == pytest.approx(bermudan, rel=5e-3)


def test_single_right_put_matches_proj_bermudan():
    M = 12
    dt = _FWD.T / M
    cf = MODEL_REGISTRY["kou"].cf
    fwd_dt = fe.ForwardSpec(S0=_FWD.S0, r=_FWD.r, q=_FWD.q, T=dt)
    drift = (_FWD.r - _FWD.q) * dt

    def step_cf(u):
        return np.exp(1j * u * drift) * np.asarray(cf(u, fwd_dt, _KOU), dtype=np.complex128)

    swing = _swing("kou", _KOU, n_rights=1, cp=-1)
    # alph=2.0: the 0.5 default truncates Kou's jump tails at this horizon
    bermudan = fe.proj_bermudan_put(step_cf, S0=_FWD.S0, r=_FWD.r, T=_FWD.T, W=100.0, M=M, alph=2.0)
    assert swing == pytest.approx(bermudan, rel=1e-3)


def test_increasing_and_concave_in_rights():
    values = [_swing("kou", _KOU, n_rights=n) for n in range(1, 7)]
    diffs = np.diff(values)
    assert np.all(diffs > 0.0)  # each extra right adds value
    assert np.all(np.diff(diffs) <= 1e-8)  # with diminishing marginal value
    assert values[1] <= 2.0 * values[0] + 1e-10  # subadditive vs Bermudan


def test_excess_rights_are_worthless():
    """More rights than dates cannot add value."""
    exact = _swing("kou", _KOU, n_rights=_M)
    excess = _swing("kou", _KOU, n_rights=_M + 5)
    assert excess == pytest.approx(exact, abs=1e-12)


def test_swing_bounded_below_by_bermudan_and_european():
    european = float(fe.price_strip("kou", "cos", np.array([100.0]), _FWD, _KOU)[0])
    bermudan_like = _swing("kou", _KOU, n_rights=1)
    two_rights = _swing("kou", _KOU, n_rights=2)
    assert two_rights > bermudan_like >= european - 5e-3


def test_validation_errors():
    with pytest.raises(ValueError):
        SwingOption(strike=-1.0, maturity=1.0)
    with pytest.raises(ValueError):
        SwingOption(strike=100.0, maturity=1.0, n_rights=0)
    heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.35, rho=-0.6, v0=0.04)
    prod = SwingOption(strike=100.0, maturity=1.0, n_rights=2, n_exercise_dates=12)
    with pytest.raises(NotImplementedError):
        fe.price(prod, "heston", "proj_swing", _FWD, heston)
    with pytest.raises(NotImplementedError):
        fe.price(prod, "kou", "monte_carlo", _FWD, _KOU)
