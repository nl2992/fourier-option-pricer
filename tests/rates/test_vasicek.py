"""Vasicek short-rate: closed-form bonds, CFs, and cumulants.

Coverage
--------
* Parameter validation.
* ``P(0, T)`` matches the closed-form Vasicek bond price on the Brigo-Mercurio
  reference parameter set.
* ``phi_{I_T}(i) == P(0, T)`` (Laplace-transform self-consistency).
* ``phi_{I_T}(0) == 1``.
* Cumulants (mean and variance of ``I_T``) agree with the analytic formula and
  with a Monte Carlo simulation of ``I_T`` at moderate T.
* Numerical stability at ``kappa*T -> 0`` via the small-argument Taylor branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.rates import (
    VasicekParams,
    vasicek_discount_bond,
    vasicek_integrated_rate_cf,
    vasicek_integrated_rate_cumulants,
)


def _bm_params() -> VasicekParams:
    """A middle-of-the-road parameter set used across tests."""
    return VasicekParams(kappa=0.5, theta=0.04, sigma=0.01, r0=0.03)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_kappa_positive_required():
    with pytest.raises(ValueError):
        VasicekParams(kappa=0.0, theta=0.04, sigma=0.01, r0=0.03)


def test_sigma_positive_required():
    with pytest.raises(ValueError):
        VasicekParams(kappa=0.5, theta=0.04, sigma=0.0, r0=0.03)


def test_theta_can_be_negative_or_zero():
    # theta is unrestricted (rate can mean-revert to a negative level in
    # low-rate regimes -- exactly the situation Vasicek was criticised for
    # and exactly the reason it stays useful for illustrative work).
    p = VasicekParams(kappa=0.5, theta=-0.005, sigma=0.01, r0=0.0)
    assert p.theta == -0.005


# ---------------------------------------------------------------------------
# Bond price
# ---------------------------------------------------------------------------


def test_bond_at_T_zero_is_one():
    p = _bm_params()
    assert vasicek_discount_bond(p, 0.0) == 1.0


def test_bond_positive_and_less_than_one_for_positive_rate():
    p = _bm_params()
    for T in [0.1, 1.0, 5.0, 10.0]:
        P = vasicek_discount_bond(p, T)
        assert 0.0 < P < 1.0


def test_bond_matches_analytic_formula_direct():
    """Recompute P(0, T) in-line to guard against typos in the module."""
    p = _bm_params()
    T = 5.0
    B = (1.0 - np.exp(-p.kappa * T)) / p.kappa
    long_run = p.theta - 0.5 * p.sigma * p.sigma / (p.kappa * p.kappa)
    log_A = long_run * (B - T) - 0.25 * p.sigma * p.sigma * B * B / p.kappa
    P_expected = float(np.exp(log_A - B * p.r0))
    P_actual = vasicek_discount_bond(p, T)
    assert np.isclose(P_actual, P_expected, atol=1e-14, rtol=0.0)


# ---------------------------------------------------------------------------
# CF self-consistency
# ---------------------------------------------------------------------------


def test_cf_at_zero_is_one():
    p = _bm_params()
    for T in [0.1, 1.0, 5.0]:
        v = vasicek_integrated_rate_cf(0.0, p, T)
        assert np.allclose(v, 1.0 + 0j, atol=1e-14)


def test_cf_at_u_i_equals_bond_price():
    """P(0, T) = E[exp(-I_T)] = phi_{I_T}(i) (Laplace transform view)."""
    p = _bm_params()
    for T in [0.1, 1.0, 5.0, 10.0]:
        cf_at_i = vasicek_integrated_rate_cf(1j, p, T)
        P = vasicek_discount_bond(p, T)
        assert np.isclose(np.real(cf_at_i), P, atol=1e-12, rtol=0.0), T
        assert np.isclose(np.imag(cf_at_i), 0.0, atol=1e-12), T


def test_cf_conjugate_symmetry_on_real_axis():
    """phi(-u) = conj(phi(u)) for real u (real-valued I_T)."""
    p = _bm_params()
    T = 2.0
    us = np.array([0.5, 1.0, 5.0, 20.0])
    phi_p = vasicek_integrated_rate_cf(us, p, T)
    phi_m = vasicek_integrated_rate_cf(-us, p, T)
    assert np.allclose(phi_m, np.conj(phi_p), atol=1e-14)


# ---------------------------------------------------------------------------
# Cumulants
# ---------------------------------------------------------------------------


def test_cumulants_positive_variance():
    p = _bm_params()
    _, var = vasicek_integrated_rate_cumulants(p, 5.0)
    assert var > 0.0


def test_cumulants_match_monte_carlo():
    """Simulate I_T under Vasicek and compare mean/variance to closed form."""
    rng = np.random.default_rng(42)
    p = VasicekParams(kappa=0.6, theta=0.05, sigma=0.02, r0=0.04)
    T = 4.0
    n_steps = 2000
    n_paths = 40_000
    dt = T / n_steps
    r = np.full(n_paths, p.r0)
    integral = np.zeros(n_paths)
    for _ in range(n_steps):
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        integral += r * dt
        r = r + p.kappa * (p.theta - r) * dt + p.sigma * dW
    mc_mean = float(integral.mean())
    mc_var = float(integral.var(ddof=1))

    mean, var = vasicek_integrated_rate_cumulants(p, T)
    # 40k paths at dt=T/2000 gives Monte Carlo standard error on the mean
    # around 5e-5 for these parameters; keep tolerances comfortable.
    assert abs(mc_mean - mean) < 5e-3
    assert abs(mc_var - var) / var < 0.05


# ---------------------------------------------------------------------------
# Numerical edge cases
# ---------------------------------------------------------------------------


def test_small_kappa_T_stable():
    """B(T) uses a Taylor fallback for kappa*T -> 0."""
    p = VasicekParams(kappa=1e-10, theta=0.03, sigma=0.001, r0=0.02)
    P = vasicek_discount_bond(p, 0.1)
    assert 0.0 < P < 1.0
    _, var = vasicek_integrated_rate_cumulants(p, 0.1)
    assert var >= 0.0
