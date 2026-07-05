"""PROJ step-option (occupation-time damping) validation.

The step option interpolates between two exactly known limits:
- rho = 0 is the vanilla European;
- rho -> infinity is the discretely monitored knock-out barrier (same PROJ
  engine, so the limit is exact node for node);
and in between it must match a full-path Monte Carlo with the discrete
occupation-time payoff exp(-rho dt n_beyond) * vanilla.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.models.registry import MODEL_REGISTRY
from foureng.pricers.proj import proj_barrier_price, proj_step_price
from foureng.products.step import StepOption

pytestmark = [pytest.mark.derived_reference]

_FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
_KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.6, eta1=25.0, eta2=15.0)
_BSM = fe.BsmParams(sigma=0.2)


def _step_cf(model, params, dt):
    cf = MODEL_REGISTRY[model].cf
    fwd_dt = fe.ForwardSpec(S0=_FWD.S0, r=_FWD.r, q=_FWD.q, T=dt)
    drift = (_FWD.r - _FWD.q) * dt

    def inner(u):
        return np.exp(1j * u * drift) * np.asarray(cf(u, fwd_dt, params), dtype=np.complex128)

    return inner


def _price(model, params, *, K=100.0, B=90.0, rho=2.0, M=52, step_type="down", cp=1):
    return proj_step_price(
        _step_cf(model, params, _FWD.T / M),
        S0=_FWD.S0,
        r=_FWD.r,
        T=_FWD.T,
        K=K,
        B=B,
        rho=rho,
        M=M,
        step_type=step_type,
        cp=cp,
        q=_FWD.q,
    )


@pytest.mark.parametrize("model, params", [("bsm", _BSM), ("kou", _KOU)])
def test_zero_rho_is_vanilla(model, params):
    v = _price(model, params, rho=0.0)
    vanilla = float(fe.price_strip(model, "cos", np.array([100.0]), _FWD, params)[0])
    assert v == pytest.approx(vanilla, abs=5e-3)


@pytest.mark.parametrize("cp", [1, -1])
def test_infinite_rho_is_knock_out_barrier(cp):
    """Soft killing with exp(-rho dt) ~ 0 must reproduce the hard knock-out
    from the same engine on the same grid."""
    M = 52
    step = _price("kou", _KOU, rho=1e9, M=M, cp=cp)
    ko = proj_barrier_price(
        _step_cf("kou", _KOU, _FWD.T / M),
        S0=_FWD.S0,
        r=_FWD.r,
        T=_FWD.T,
        K=100.0,
        H=90.0,
        M=M,
        barrier_type="down_out",
        cp=cp,
        q=_FWD.q,
    )
    assert step == pytest.approx(ko, abs=2e-3)


def test_monotone_decreasing_in_rho():
    prices = [_price("kou", _KOU, rho=rho) for rho in (0.0, 0.5, 2.0, 10.0, 100.0)]
    assert all(a >= b - 1e-10 for a, b in zip(prices, prices[1:]))
    vanilla = float(fe.price_strip("kou", "cos", np.array([100.0]), _FWD, _KOU)[0])
    assert prices[0] <= vanilla + 5e-3
    assert prices[-1] >= 0.0


@pytest.mark.parametrize("model, params", [("bsm", _BSM), ("kou", _KOU)])
def test_matches_full_path_monte_carlo(model, params):
    rng = np.random.default_rng(37)
    n_paths, M = 200_000, 26
    rho, B, K = 3.0, 90.0, 100.0
    dt = _FWD.T / M

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

    log_s = np.full(n_paths, np.log(_FWD.S0))
    n_below = np.zeros(n_paths)
    for _ in range(M):
        log_s += increment(n_paths)
        n_below += (np.exp(log_s) < B).astype(float)

    s_T = np.exp(log_s)
    payoff = np.exp(-_FWD.r * _FWD.T) * np.exp(-rho * dt * n_below) * np.maximum(s_T - K, 0.0)
    mc, se = float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_paths))
    proj = _price(model, params, K=K, B=B, rho=rho, M=M)
    assert abs(proj - mc) < 4.0 * se + 2e-3, f"PROJ {proj:.4f} vs MC {mc:.4f} +/- {se:.4f}"


def test_up_step_damps_above_barrier():
    """An up-step with a barrier far above spot is barely damped; with a
    barrier at spot it is damped hard."""
    far = _price("kou", _KOU, B=200.0, step_type="up", rho=5.0)
    near = _price("kou", _KOU, B=100.0, step_type="up", rho=5.0)
    vanilla = float(fe.price_strip("kou", "cos", np.array([100.0]), _FWD, _KOU)[0])
    assert near < far <= vanilla + 5e-3


def test_pipeline_dispatch_and_validation():
    prod = StepOption(
        strike=100.0,
        maturity=1.0,
        cp=1,
        barrier=90.0,
        rho=2.0,
        step_type="down",
        n_monitoring=52,
    )
    via_price = fe.price(prod, "kou", "proj_step", _FWD, _KOU)
    # the pipeline builds its own cumulant-based grid, so agreement is to
    # grid accuracy rather than bitwise
    direct = _price("kou", _KOU, rho=2.0, M=52)
    assert via_price == pytest.approx(direct, abs=5e-3)

    with pytest.raises(ValueError):
        StepOption(strike=100.0, maturity=1.0, barrier=90.0, rho=-1.0)
    with pytest.raises(ValueError):
        StepOption(strike=100.0, maturity=1.0, barrier=90.0, step_type="sideways")
    with pytest.raises(ValueError):
        proj_step_price(
            _step_cf("kou", _KOU, 0.1),
            S0=100.0,
            r=0.03,
            T=1.0,
            K=100.0,
            B=90.0,
            rho=1.0,
            M=10,
            N=1000,
        )
    with pytest.raises(NotImplementedError):
        fe.price(prod, "kou", "monte_carlo", _FWD, _KOU)
