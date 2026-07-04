"""Levy forward-start pricer validation.

The homogeneity factorization is exact for stationary independent
increments, so:
- under BSM it must match the Rubinstein (1990) closed form;
- with start_time = 0 it must equal the vanilla European price at strike
  alpha * S0 for any Levy model;
- under Kou it must sit inside the Monte Carlo confidence band.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.analytics.bsm_exotics import bsm_forward_start
from foureng.pricers.forward_start import levy_forward_start_price
from foureng.products.forward_start import ForwardStartOption

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.04, q=0.015, T=1.0)
_KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.6, eta1=25.0, eta2=15.0)
_T1, _T2 = 0.4, 1.0


@pytest.mark.parametrize("cp", [1, -1])
@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.2])
def test_bsm_matches_rubinstein_closed_form(cp, alpha):
    params = fe.BsmParams(sigma=0.25)
    cf_price = levy_forward_start_price(
        "bsm", _FWD, params, alpha=alpha, start_time=_T1, maturity=_T2, cp=cp
    )
    ref = bsm_forward_start(_FWD.S0, alpha, _T1, _T2, _FWD.r, _FWD.q, params.sigma, cp=cp)
    assert cf_price == pytest.approx(ref, abs=1e-8)


@pytest.mark.parametrize(
    "model, params",
    [
        ("kou", _KOU),
        ("vg", fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14)),
        ("merton_jd", fe.MertonJDParams(sigma=0.15, lam=0.5, muj=-0.1, sigj=0.2)),
    ],
)
def test_zero_start_time_reduces_to_european(model, params):
    alpha = 1.1
    fs = levy_forward_start_price(
        model, _FWD, params, alpha=alpha, start_time=0.0, maturity=_T2, cp=1
    )
    european = float(fe.price_strip(model, "cos", np.array([alpha * _FWD.S0]), _FWD, params)[0])
    assert fs == pytest.approx(european, abs=5e-7)


def test_kou_matches_monte_carlo():
    rng = np.random.default_rng(13)
    n_paths = 200_000
    sig, lam, p_up, e1, e2 = _KOU.sigma, _KOU.lam, _KOU.p, _KOU.eta1, _KOU.eta2
    zeta = p_up * e1 / (e1 - 1.0) + (1.0 - p_up) * e2 / (e2 + 1.0) - 1.0

    def kou_increment(dt: float, n: int) -> np.ndarray:
        drift = (_FWD.r - _FWD.q - 0.5 * sig * sig - lam * zeta) * dt
        x = drift + sig * np.sqrt(dt) * rng.standard_normal(n)
        n_jumps = rng.poisson(lam * dt, size=n)
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
        return x

    alpha = 1.0
    s_t1 = _FWD.S0 * np.exp(kou_increment(_T1, n_paths))
    s_t2 = s_t1 * np.exp(kou_increment(_T2 - _T1, n_paths))
    payoff = np.exp(-_FWD.r * _T2) * np.maximum(s_t2 - alpha * s_t1, 0.0)
    mc, se = float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_paths))

    cf = levy_forward_start_price(
        "kou", _FWD, _KOU, alpha=alpha, start_time=_T1, maturity=_T2, cp=1
    )
    assert abs(cf - mc) < 4.0 * se, f"CF {cf:.4f} vs MC {mc:.4f} +/- {se:.4f}"


def test_parity_in_alpha():
    """call - put = S0 e^{-q t1} disc_tau (F_tau - alpha) for the unit leg."""
    tau = _T2 - _T1
    f_tau = float(np.exp((_FWD.r - _FWD.q) * tau))
    disc_tau = float(np.exp(-_FWD.r * tau))
    for alpha in (0.9, 1.0, 1.1):
        call = levy_forward_start_price(
            "kou", _FWD, _KOU, alpha=alpha, start_time=_T1, maturity=_T2, cp=1
        )
        put = levy_forward_start_price(
            "kou", _FWD, _KOU, alpha=alpha, start_time=_T1, maturity=_T2, cp=-1
        )
        expected = _FWD.S0 * np.exp(-_FWD.q * _T1) * disc_tau * (f_tau - alpha)
        assert call - put == pytest.approx(expected, abs=1e-10)


def test_pipeline_dispatch():
    product = ForwardStartOption(start_time=_T1, maturity=_T2, cp=1, alpha=1.05)
    via_price = fe.price(product, "kou", "forward_start_cf", _FWD, _KOU)
    direct = levy_forward_start_price(
        "kou", _FWD, _KOU, alpha=1.05, start_time=_T1, maturity=_T2, cp=1
    )
    assert via_price == pytest.approx(direct, abs=0.0)


def test_rejects_unsupported_cases():
    heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.35, rho=-0.6, v0=0.04)
    with pytest.raises(ValueError):
        levy_forward_start_price("heston", _FWD, heston, alpha=1.0, start_time=_T1, maturity=_T2)
    with pytest.raises(ValueError):
        levy_forward_start_price("kou", _FWD, _KOU, alpha=-1.0, start_time=_T1, maturity=_T2)
    with pytest.raises(ValueError):
        levy_forward_start_price("kou", _FWD, _KOU, alpha=1.0, start_time=1.5, maturity=_T2)
    with pytest.raises(ValueError):
        levy_forward_start_price("kou", _FWD, _KOU, alpha=1.0, start_time=_T1, maturity=_T2, cp=0)
