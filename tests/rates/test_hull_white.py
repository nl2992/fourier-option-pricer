"""Hull-White one-factor: curve-fitted bonds, Gaussian CF, LevFin bridge.

Coverage
--------
* Parameter validation and default flat-forward curve.
* Bond price equals the caller-supplied ``initial_discount`` exactly.
* ``phi_{I_T}(i) == P(0, T)`` (self-consistency).
* Variance of ``I_T`` matches Vasicek in the flat-forward limit
  (theta held constant so both models generate the same Gaussian integral).
* Custom callable initial curves are respected (LevFin bridge use case).
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.rates import (
    HullWhiteParams,
    VasicekParams,
    hull_white_discount_bond,
    hull_white_integrated_rate_cf,
    hull_white_integrated_rate_cumulants,
    vasicek_integrated_rate_cumulants,
)


def _flat_params() -> HullWhiteParams:
    return HullWhiteParams(a=0.5, sigma=0.01, r0=0.03)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_a_positive_required():
    with pytest.raises(ValueError):
        HullWhiteParams(a=0.0, sigma=0.01, r0=0.03)


def test_sigma_positive_required():
    with pytest.raises(ValueError):
        HullWhiteParams(a=0.5, sigma=0.0, r0=0.03)


# ---------------------------------------------------------------------------
# Bond price
# ---------------------------------------------------------------------------


def test_bond_at_T_zero_is_one():
    assert hull_white_discount_bond(_flat_params(), 0.0) == 1.0


def test_bond_equals_flat_forward_default():
    p = _flat_params()
    for T in [0.1, 1.0, 5.0, 10.0]:
        assert np.isclose(
            hull_white_discount_bond(p, T),
            np.exp(-p.r0 * T),
            atol=1e-14,
        )


def test_bond_equals_custom_initial_curve():
    """P(0, T) is exactly the caller's curve  -  the point of Hull-White."""

    # A downward-sloping "yield curve" (higher discount factors at long tenors
    # via lower long-end forwards)  -  contrived just to be non-flat.
    def curve(T: float) -> float:
        f0 = 0.03
        slope = -0.001  # long-forward drift.
        y = f0 + slope * T  # instantaneous forward.
        return float(np.exp(-y * T))

    p = HullWhiteParams(a=0.4, sigma=0.008, r0=0.03, initial_discount=curve)
    for T in [0.5, 2.0, 7.0]:
        assert np.isclose(
            hull_white_discount_bond(p, T),
            curve(T),
            atol=1e-14,
        )


# ---------------------------------------------------------------------------
# CF self-consistency
# ---------------------------------------------------------------------------


def test_cf_at_zero_is_one():
    p = _flat_params()
    for T in [0.1, 1.0, 5.0]:
        v = hull_white_integrated_rate_cf(0.0, p, T)
        assert np.allclose(v, 1.0 + 0j, atol=1e-14)


def test_cf_at_u_i_equals_bond_price():
    p = _flat_params()
    for T in [0.1, 1.0, 5.0, 10.0]:
        cf_at_i = hull_white_integrated_rate_cf(1j, p, T)
        P = hull_white_discount_bond(p, T)
        # phi(i) = exp(-mean + var/2); the exp amplifies the ~1e-16 mean
        # rounding to ~mean * 1e-16 = ~1e-17 on P, but the intermediate
        # complex sqrt/log adds a couple more digits of noise; 1e-10 is
        # safely below any real bug and well above float rounding.
        assert np.isclose(np.real(cf_at_i), P, atol=1e-10, rtol=0.0), T
        assert np.isclose(np.imag(cf_at_i), 0.0, atol=1e-12), T


# ---------------------------------------------------------------------------
# Vasicek limit
# ---------------------------------------------------------------------------


def test_flat_forward_variance_matches_vasicek():
    """With a flat forward at r0, HW's variance of I_T matches Vasicek's."""
    a = 0.4
    sigma = 0.015
    r0 = 0.03
    hw = HullWhiteParams(a=a, sigma=sigma, r0=r0)
    # Vasicek with theta = r0 gives the same Gaussian process (mean-reverting
    # to the flat level), so Var[I_T] must coincide.
    va = VasicekParams(kappa=a, theta=r0, sigma=sigma, r0=r0)
    for T in [0.25, 1.0, 5.0]:
        _, var_hw = hull_white_integrated_rate_cumulants(hw, T)
        _, var_va = vasicek_integrated_rate_cumulants(va, T)
        assert np.isclose(var_hw, var_va, rtol=1e-12, atol=1e-16)


def test_flat_forward_hw_mean_equals_vasicek_mean_plus_convexity():
    """HW fitted to a flat forward has E[I_T] = Vasicek-mean + Var/2.

    Vasicek with theta = r0 has E[I_T] = r0 * T *exactly*, which under the
    Gaussian bond formula ``exp(-mean + var/2)`` implies a bond price of
    ``exp(-r0*T + var/2)``  -  not the flat market curve ``exp(-r0*T)``.
    Hull-White fixes this by shifting the mean by ``+ Var/2`` so that
    ``phi_{I_T}(i) = exp(-r0*T)`` exactly.  The two means differ by
    exactly the convexity correction.
    """
    a = 0.4
    sigma = 0.015
    r0 = 0.03
    hw = HullWhiteParams(a=a, sigma=sigma, r0=r0)
    va = VasicekParams(kappa=a, theta=r0, sigma=sigma, r0=r0)
    for T in [0.25, 1.0, 5.0]:
        mean_hw, var_hw = hull_white_integrated_rate_cumulants(hw, T)
        mean_va, var_va = vasicek_integrated_rate_cumulants(va, T)
        # Variance identical.
        assert np.isclose(var_hw, var_va, rtol=1e-12, atol=0.0)
        # Mean differs by the convexity correction Var/2.
        assert np.isclose(mean_hw - mean_va, 0.5 * var_va, atol=1e-14, rtol=0.0)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


def test_small_aT_variance_finite_and_non_negative():
    p = HullWhiteParams(a=1e-8, sigma=0.01, r0=0.03)
    _, var = hull_white_integrated_rate_cumulants(p, 0.5)
    assert np.isfinite(var) and var >= 0.0
