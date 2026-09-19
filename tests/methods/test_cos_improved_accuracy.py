"""Accuracy regression for the cos_improved default grid (A4).

Before this fix, ``price_strip(model, "cos_improved", ...)`` measured about
1e-9 absolute error against the high-precision contour engine on Heston and
4.6e-6 on a jump-heavy affine spec (see ``docs/fourier_coverage_plan.md``,
item A4). Three defaults in ``foureng.pricers.cos`` were responsible:

- ``cos_tail_proxy``'s gaussian-like tail estimate is many orders of
  magnitude smaller than the real put+parity COS error at the old L=8 seed,
  so the tolerance loop in ``cos_adaptive_decision`` exited on its very
  first check without ever growing the truncation interval.
- The old ``eps_trunc=1e-10`` let the loop stop early for semi-heavy tail
  families (kou, bates, jump-heavy affine specs) too, at an L that still
  left ~1e-7-1e-9 of real price error.
- ``cos_prices``' ``payoff_mode="auto"`` gated its per-strike direct-call
  optimization on the *total* interval width (``b - a <=
  call_direct_width_max``, default 20.0) instead of the upper bound ``b``
  alone; a centered grid can have ``b - a`` comfortably below that cap while
  ``b`` itself already exceeds the Le Floc'h threshold (8.0) at which the
  ``e^b`` term in the direct-call payoff coefficients loses precision. That
  let direct-call fire on wide jump-model grids and lose several digits.
- The old ``width_fallback=40.0`` routed those same wide grids to the
  Lewis/Carr-Madan fallback engines, which measured *less* accurate than
  COS itself once the payoff-mode bug above is fixed.

This module checks that representative models now reach about 1e-10
absolute against ``method="contour"`` (accurate to about 1e-13 relative),
while staying fast enough to run outside the ``slow`` marker.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe

pytestmark = [pytest.mark.derived_reference, pytest.mark.numerical_stability]

FWD = fe.ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
STRIKES = np.linspace(FWD.F0 * 0.5, FWD.F0 * 1.8, 15)

_TARGET_ATOL = 1e-10

_CASES = {
    "heston": fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04),
    "bates": fe.BatesParams(
        kappa=2.0, theta=0.04, nu=0.5, rho=-0.7, v0=0.04, lam_j=0.5, mu_j=-0.1, sigma_j=0.15
    ),
    "vg": fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14),
    "kou": fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0),
    "cgmy": fe.CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.8),
}


@pytest.mark.parametrize("model", list(_CASES))
def test_cos_improved_reaches_contour_accuracy(model):
    """cos_improved should match the contour reference to about 1e-10 at T=1."""
    params = _CASES[model]
    got = fe.price_strip(model, "cos_improved", STRIKES, FWD, params)
    ref = fe.price_strip(model, "contour", STRIKES, FWD, params)
    np.testing.assert_allclose(got, ref, atol=_TARGET_ATOL, rtol=0.0)


def test_cos_improved_reaches_contour_accuracy_on_original_heston_case():
    """The exact Heston case from the A4 problem statement (was ~1e-9)."""
    fwd = fe.ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
    params = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)
    strikes = np.arange(80.0, 120.001, 5.0)

    got = fe.price_strip("heston", "cos_improved", strikes, fwd, params)
    ref = fe.price_strip("heston", "contour", strikes, fwd, params)
    np.testing.assert_allclose(got, ref, atol=_TARGET_ATOL, rtol=0.0)


def test_cos_improved_reaches_sinc_accuracy_on_jump_heavy_affine_case():
    """The jump-heavy affine spec from the A4 problem statement (was ~4.6e-6).

    This spec (variance-proportional jump intensity) has no closed-form CF
    registered under a named model, so it is priced through the generic
    ``"affine"`` model and checked against ``"sinc"`` rather than
    ``"contour"``, matching ``tests/models/test_affine_model.py``.
    """
    kappa, theta, nu, rho, v0 = 1.5, 0.04, 0.5, -0.7, 0.05

    def _heston(drift=0.02, v_drift=-0.5, **jumps):
        H1 = np.zeros((2, 2, 2))
        H1[1] = [[1.0, rho * nu], [rho * nu, nu * nu]]
        return fe.AffineParams(
            x0=[np.log(100.0), v0],
            K0=[drift, kappa * theta],
            K1=[[0.0, v_drift], [0.0, -kappa]],
            H0=np.zeros((2, 2)),
            H1=H1,
            **jumps,
        )

    def _lognormal_jumps(mu, sig):
        return lambda c: np.exp(c[..., 0] * mu + 0.5 * sig * sig * c[..., 0] ** 2)

    mu_j, sig_j, lam1 = -0.08, 0.1, 20.0
    zeta = np.exp(mu_j + 0.5 * sig_j**2) - 1.0
    spec = _heston(
        v_drift=-0.5 - lam1 * zeta, l1=[0.0, lam1], jump_transform=_lognormal_jumps(mu_j, sig_j)
    )
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.9)
    strikes = np.array([85.0, 100.0, 115.0])

    got = fe.price_strip("affine", "cos_improved", strikes, fwd, spec)
    ref = fe.price_strip("affine", "sinc", strikes, fwd, spec)
    np.testing.assert_allclose(got, ref, atol=_TARGET_ATOL, rtol=0.0)


def test_vg_short_maturity_below_nu_is_documented_and_not_required_to_hit_target():
    """VG with T < nu has a singular density; COS converges only algebraically.

    This is the exception ``cos_improved``'s docstring calls out: it is not
    held to the 1e-10 bar, only to a much looser sanity bound, and it must
    not regress far below its current best-effort floor (~1e-7).
    """
    fwd = fe.ForwardSpec(S0=100.0, r=0.1, q=0.0, T=0.1)
    params = fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14)
    assert fwd.T < params.nu

    strikes = np.array([90.0])
    got = fe.price_strip("vg", "cos_improved", strikes, fwd, params)
    ref = fe.price_strip("vg", "contour", strikes, fwd, params)
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0.0)
