"""Lifted Heston (Abi Jaber 2019): Heston reduction, kernel, convergence, MC."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gamma

import foureng as fe

FWD = fe.ForwardSpec(S0=100.0, r=0.0, q=0.0, T=1.0)
ROUGH = fe.LiftedHestonParams(v0=0.02, kappa=0.3, theta=0.02, nu=0.3, rho=-0.7, H=0.1)
K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])


def test_one_factor_without_decay_is_heston():
    heston = fe.HestonParams(kappa=1.5, theta=0.04, nu=0.3, rho=-0.7, v0=0.05)
    lifted = fe.LiftedHestonParams(
        v0=0.05, kappa=1.5, theta=0.04, nu=0.3, rho=-0.7, weights=(1.0,), speeds=(0.0,)
    )
    u = np.linspace(0.0, 60.0, 31)
    np.testing.assert_allclose(
        fe.lifted_heston_cf(u, FWD, lifted), fe.heston_cf_form2(u, FWD, heston), atol=1e-10
    )
    np.testing.assert_allclose(
        fe.price_strip("lifted_heston", "cos_improved", K, FWD, lifted),
        fe.price_strip("heston", "cos_improved", K, FWD, heston),
        atol=1e-8,
    )


def test_kernel_approximates_the_fractional_kernel():
    """Abi Jaber's n = 20, r_20 = 2.5 geometric kernel is within a few percent of
    t^{alpha-1} / Gamma(alpha) over three decades of t."""
    c, x = fe.lifted_kernel(0.1, 20, 2.5)
    t = np.logspace(-3, 0, 13)
    ratio = (np.exp(-np.outer(t, x)) @ c) / (t ** (0.6 - 1.0) / gamma(0.6))
    assert np.all(np.abs(ratio - 1.0) < 0.06)


def test_rough_prices_converge_in_time_steps_and_are_martingale():
    u = np.linspace(0.0, 40.0, 21)
    a = fe.lifted_heston_cf(u, FWD, ROUGH, n_steps=512)
    b = fe.lifted_heston_cf(u, FWD, ROUGH, n_steps=2048)
    assert np.max(np.abs(a - b)) < 5e-7
    assert fe.lifted_heston_cf(np.array([-1j]), FWD, ROUGH)[0] == pytest.approx(1.0, abs=1e-12)
    cums = fe.lifted_heston_cumulants(FWD, ROUGH)
    fine = fe.price_strip(
        "lifted_heston", "cos", K, FWD, ROUGH, grid=fe.cos_auto_grid(cums, N=1024, L=14.0)
    )
    cos = fe.price_strip("lifted_heston", "cos_improved", K, FWD, ROUGH)
    np.testing.assert_allclose(cos, fine, atol=2e-6)


def test_high_frequencies_are_resolved_or_negligible():
    phi = fe.lifted_heston_cf(np.array([100.0, 400.0, 800.0, 6400.0]), FWD, ROUGH)
    assert np.all(np.isfinite(phi))
    assert abs(phi[1]) < 1e-11 and abs(phi[2]) < 1e-20 and phi[3] == 0.0


@pytest.mark.slow
def test_two_factor_kernel_matches_euler_monte_carlo():
    fwd = fe.ForwardSpec(S0=100.0, r=0.02, q=0.0, T=1.0)
    w, x = np.array([0.6, 0.9]), np.array([0.5, 8.0])
    p = fe.LiftedHestonParams(
        v0=0.04, kappa=1.2, theta=0.05, nu=0.5, rho=-0.6, weights=tuple(w), speeds=tuple(x)
    )
    strikes = np.array([85.0, 100.0, 115.0])
    rng = np.random.default_rng(4)
    n, steps = 100_000, 1000
    dt = fwd.T / steps
    U, X = np.zeros((n, 2)), np.zeros(n)
    for k in range(steps):
        g0 = p.v0 + p.kappa * p.theta * np.sum(w * (1 - np.exp(-x * k * dt)) / x)
        V = np.maximum(g0 + U @ w, 0.0)
        z1 = rng.standard_normal(n)
        z2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n)
        X += -0.5 * V * dt + np.sqrt(V * dt) * z2
        U = (U + (-p.kappa * V * dt + p.nu * np.sqrt(V * dt) * z1)[:, None]) / (1 + x * dt)
    S = fwd.F0 * np.exp(X)
    pay = fwd.disc * np.maximum(S[None, :] - strikes[:, None], 0.0)
    cv = S - fwd.F0
    est = pay - np.array([np.cov(r, cv)[0, 1] / cv.var() for r in pay])[:, None] * cv
    se = est.std(axis=1) / np.sqrt(n)
    got = fe.price_strip("lifted_heston", "cos_improved", strikes, fwd, p)
    assert np.all(np.abs(got - est.mean(axis=1)) < 4.0 * se)


def test_validation():
    with pytest.raises(ValueError, match="H must be"):
        fe.LiftedHestonParams(v0=0.02, kappa=0.3, theta=0.02, nu=0.3, rho=-0.7, H=0.6)
    with pytest.raises(ValueError, match="both weights and speeds"):
        fe.LiftedHestonParams(v0=0.02, kappa=0.3, theta=0.02, nu=0.3, rho=-0.7, weights=(1.0,))
    with pytest.raises(ValueError, match="rho"):
        fe.LiftedHestonParams(v0=0.02, kappa=0.3, theta=0.02, nu=0.3, rho=1.2)
