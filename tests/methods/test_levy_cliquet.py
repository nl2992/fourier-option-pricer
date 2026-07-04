"""Levy cliquet (per-period CF collar) pricer validation.

Independent references:
- no local collar has a closed form: additive value = D(T) * sum(F_dt - 1),
  multiplicative value = D(T) * (e^{(r-q)T} - 1);
- under BSM the CF price must sit inside the existing cliquet_mc engine's
  confidence band;
- under Kou it must sit inside an in-test jump Monte Carlo band;
- a one-period cliquet floored at 0 is an ATM-forward call.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.pricers.cliquet import levy_cliquet_price
from foureng.products.cliquet import CliquetOption

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
_TIMES = np.linspace(0.25, 1.0, 4)
_KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.6, eta1=25.0, eta2=15.0)


def _product(**kw) -> CliquetOption:
    base = dict(maturity=1.0, reset_times=_TIMES, payoff_type="additive", cp=1)
    base.update(kw)
    return CliquetOption(**base)


def test_no_collar_additive_closed_form():
    product = _product()
    got = levy_cliquet_price("kou", _FWD, _KOU, product)
    dts = np.diff(np.concatenate(([0.0], _TIMES)))
    expected = np.exp(-_FWD.r * 1.0) * float(np.sum(np.exp((_FWD.r - _FWD.q) * dts) - 1.0))
    assert got == pytest.approx(expected, rel=1e-12)


def test_no_collar_multiplicative_closed_form():
    product = _product(payoff_type="multiplicative")
    got = levy_cliquet_price("vg", _FWD, fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14), product)
    expected = np.exp(-_FWD.r * 1.0) * (np.exp((_FWD.r - _FWD.q) * 1.0) - 1.0)
    assert got == pytest.approx(expected, rel=1e-10)


def test_single_period_floor_zero_is_forward_atm_call():
    """One period, lf=0, lc=inf: payoff = (R)^+ = (S_dt - 1)^+ on unit spot,
    paid at maturity == period end, i.e. exactly the COS European call at K=1."""
    product = CliquetOption(
        maturity=1.0, reset_times=np.array([1.0]), local_floor=0.0, payoff_type="additive"
    )
    got = levy_cliquet_price("kou", _FWD, _KOU, product)
    fwd_unit = fe.ForwardSpec(S0=1.0, r=_FWD.r, q=_FWD.q, T=1.0)
    call = float(fe.price_strip("kou", "cos", np.array([1.0]), fwd_unit, _KOU)[0])
    assert got == pytest.approx(call, abs=1e-10)


def test_bsm_matches_cliquet_mc():
    product = _product(local_floor=0.0, local_cap=0.06)
    params = fe.BsmParams(sigma=0.2)
    cf = levy_cliquet_price("bsm", _FWD, params, product)
    mc = fe.mc_price(
        fe.ForwardSpec(S0=_FWD.S0, r=_FWD.r, q=_FWD.q, T=1.0),
        params.sigma,
        product,
        fe.MCSpec(n_paths=400_000, seed=5, antithetic=True),
    )
    assert abs(cf - mc.price) < 4.0 * max(mc.stderr, 1e-6), (
        f"CF {cf:.6f} vs MC {mc.price:.6f} +/- {mc.stderr:.6f}"
    )


def test_kou_matches_monte_carlo():
    rng = np.random.default_rng(3)
    n_paths = 200_000
    product = _product(local_floor=-0.02, local_cap=0.05)
    sig, lam, p_up, e1, e2 = _KOU.sigma, _KOU.lam, _KOU.p, _KOU.eta1, _KOU.eta2
    zeta = p_up * e1 / (e1 - 1.0) + (1.0 - p_up) * e2 / (e2 + 1.0) - 1.0

    dts = np.diff(np.concatenate(([0.0], _TIMES)))
    total = np.zeros(n_paths)
    for dt in dts:
        drift = (_FWD.r - _FWD.q - 0.5 * sig * sig - lam * zeta) * dt
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
        total += np.clip(np.exp(x) - 1.0, product.local_floor, product.local_cap)

    payoff = np.exp(-_FWD.r * 1.0) * total
    mc, se = float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_paths))
    cf = levy_cliquet_price("kou", _FWD, _KOU, product)
    assert abs(cf - mc) < 4.0 * se, f"CF {cf:.6f} vs MC {mc:.6f} +/- {se:.6f}"


def test_cp_sign_flip_additive():
    product_call = _product(local_floor=0.0, local_cap=0.08, cp=1)
    product_put = _product(local_floor=0.0, local_cap=0.08, cp=-1)
    v1 = levy_cliquet_price("kou", _FWD, _KOU, product_call)
    v2 = levy_cliquet_price("kou", _FWD, _KOU, product_put)
    assert v2 == pytest.approx(-v1, abs=1e-14)


def test_collar_bounds():
    """Collared value is bounded by the discounted collar levels."""
    product = _product(local_floor=-0.03, local_cap=0.05)
    v = levy_cliquet_price("kou", _FWD, _KOU, product)
    n = len(_TIMES)
    disc = np.exp(-_FWD.r * 1.0)
    assert disc * n * -0.03 <= v <= disc * n * 0.05


def test_pipeline_dispatch():
    product = _product(local_floor=0.0, local_cap=0.06)
    via_price = fe.price(product, "kou", "cliquet_cf", _FWD, _KOU)
    direct = levy_cliquet_price("kou", _FWD, _KOU, product)
    assert via_price == pytest.approx(direct, abs=0.0)


def test_rejects_global_collar_and_unsupported_model():
    with pytest.raises(NotImplementedError):
        levy_cliquet_price("kou", _FWD, _KOU, _product(global_floor=0.0))
    with pytest.raises(NotImplementedError):
        levy_cliquet_price("kou", _FWD, _KOU, _product(global_cap=0.5))
    heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.35, rho=-0.6, v0=0.04)
    with pytest.raises(ValueError):
        levy_cliquet_price("heston", _FWD, heston, _product())
