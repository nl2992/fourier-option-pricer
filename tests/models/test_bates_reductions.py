"""Bates model reduction tests — BSM and MertonJD limits.

NOTE: The Bates → Heston reduction (lam_j=0) is already covered in
``test_bates_reduces_to_heston.py``. This file covers the two additional
structural reductions:

5.2 Bates → BSM  (near-constant variance, zero jumps)
5.3 Bates → Merton JD  (near-constant variance, nonzero jumps)

PyFENG's Heston backend does not support nu=0 exactly (division by zero).
We use nu=1e-4 (effectively zero vol-of-vol: Feller ratio >> 1) so the
variance process is virtually deterministic. The resulting prices agree
with BSM / Merton JD to better than 1e-4.
"""
from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.pricers.cos import cos_auto_grid, cos_prices
from foureng.models.bates import bates_cumulants

pytestmark = pytest.mark.paper

# ---------------------------------------------------------------------------
# Shared forward spec and strikes
# ---------------------------------------------------------------------------

_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
_STRIKES = np.array([90.0, 95.0, 100.0, 105.0, 110.0])


# ---------------------------------------------------------------------------
# 5.2  Bates → BSM  (nu ≈ 0, lam_j=0)
# ---------------------------------------------------------------------------


def test_bates_reduces_to_bsm():
    """Bates with nu→0, lam_j=0 must give the same prices as BSM (atol=1e-4).

    With nu=1e-4 the variance process is effectively deterministic at v0=theta=0.04,
    so sigma_eff = sqrt(v0) = 0.2. With lam_j=0 the jump component vanishes.
    The CF therefore reduces to the BSM log-normal CF to machine precision.

    Note: nu=0 exactly is not supported by the PyFENG Heston backend; nu=1e-4
    causes a ~1e-8 Feller-condition induced correction that is negligible.
    """
    bates_params = fe.BatesParams(
        kappa=2.0, theta=0.04, nu=1e-4, rho=0.0, v0=0.04,
        lam_j=0.0, mu_j=0.0, sigma_j=0.0,
    )
    bsm_params = fe.BsmParams(sigma=0.2)  # sqrt(0.04)

    phi_b = lambda u: fe.bates_cf(u, _FWD, bates_params)
    cums_b = bates_cumulants(_FWD, bates_params)
    grid = cos_auto_grid(cums_b, N=256, L=10.0)
    prices_bates = cos_prices(phi_b, _FWD, _STRIKES, grid).call_prices

    prices_bsm = fe.price_strip("bsm", "cos", _STRIKES, _FWD, bsm_params)

    np.testing.assert_allclose(
        prices_bates, prices_bsm, atol=1e-4,
        err_msg="Bates(nu~0, lam_j=0) should reduce to BSM",
    )


# ---------------------------------------------------------------------------
# 5.3  Bates → Merton JD  (nu ≈ 0, lam_j > 0)
# ---------------------------------------------------------------------------


def test_bates_reduces_to_merton_jd():
    """Bates with nu→0, nonzero jumps must agree with Merton JD (atol=1e-4).

    Setting nu=1e-4 freezes variance near v0=theta=0.04, so the diffusion is
    near-plain GBM with sigma=0.2. The jump block is the same compound-Poisson
    construction used in both models, so prices agree to better than 1e-4.
    """
    bates_params = fe.BatesParams(
        kappa=2.0, theta=0.04, nu=1e-4, rho=0.0, v0=0.04,
        lam_j=2.0, mu_j=0.01660262729617973, sigma_j=0.08,
    )
    merton_params = fe.MertonJDParams(
        sigma=0.2, lam=2.0,
        muj=0.01660262729617973, sigj=0.08,
    )

    phi_b = lambda u: fe.bates_cf(u, _FWD, bates_params)
    cums_b = bates_cumulants(_FWD, bates_params)
    grid = cos_auto_grid(cums_b, N=256, L=10.0)
    prices_bates = cos_prices(phi_b, _FWD, _STRIKES, grid).call_prices

    prices_merton = fe.price_strip("merton_jd", "cos", _STRIKES, _FWD, merton_params)

    np.testing.assert_allclose(
        prices_bates, prices_merton, atol=1e-4,
        err_msg="Bates(nu~0, lam_j=2) should reduce to Merton JD",
    )
