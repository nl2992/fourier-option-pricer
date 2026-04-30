"""Phase 6 tests: MC with control variates.

Demonstrate:
  - BS call with S_T control: variance reduction ratio >= 2 at moderate moneyness.
  - BS call CV price unbiased (agrees with analytic BS to MC tolerance).
  - Heston with BS-control: variance reduction ratio >= 2 in a low-vol-of-vol
    regime where the BS-control correlation is high.
  - Heston CV price agrees with the COS reference (the truth for Heston under
    our validation stack) to MC tolerance at 100k paths.
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cf_form2, heston_cumulants
from foureng.pricers.cos import cos_prices, cos_auto_grid
from foureng.iv.implied_vol import bs_price_from_fwd, BSInputs
from foureng.mc.control_variate import bs_call_cv, heston_call_bs_control


def test_bs_cv_reduces_variance():
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.03, 0.01, 0.25
    n = 100_000
    res = bs_call_cv(S0, K, T, r, q, sigma, n, seed=42)
    assert res.var_reduction >= 2.0, (
        f"Expected >=2x variance reduction, got {res.var_reduction:.2f}x"
    )

    # Unbiasedness: CV mean within 4 standard errors of the analytic BS price.
    F = S0 * np.exp((r - q) * T)
    ref = bs_price_from_fwd(sigma, BSInputs(F0=F, K=K, T=T, r=r, q=q, is_call=True))
    assert abs(res.price_cv - ref) < 4.0 * res.se_cv, (
        f"CV price {res.price_cv:.5f} vs BS {ref:.5f}; se_cv={res.se_cv:.2e}"
    )
    # And tighter than the plain MC: plain|price_plain - ref| / se_plain typically ~1.
    assert res.se_cv < res.se_plain, "CV stderr should be strictly smaller"


def test_heston_bs_control_variance_reduction():
    S0, K, T, r, q = 100.0, 100.0, 0.5, 0.02, 0.0
    # Low vol-of-vol: Heston ~ BS, control correlation should be ~1.
    p = HestonParams(kappa=2.0, theta=0.04, nu=0.10, rho=-0.3, v0=0.04)
    res = heston_call_bs_control(S0, K, T, r, q, p, n_paths=50_000,
                                  n_steps=50, seed=7)
    assert res.var_reduction >= 2.0, (
        f"Expected >=2x on low-vov Heston, got {res.var_reduction:.2f}x"
    )

    # Cross-check vs COS (the validated truth under our stack).
    fwd = ForwardSpec(S0=S0, r=r, q=q, T=T)
    cums = heston_cumulants(fwd, p)
    grid = cos_auto_grid(cums, N=256, L=10.0)
    phi = lambda u: heston_cf_form2(u, fwd, p)
    ref = float(cos_prices(phi, fwd, np.array([K]), grid).call_prices[0])

    # 4-sigma band around the CV estimate.
    assert abs(res.price_cv - ref) < 4.0 * res.se_cv, (
        f"Heston CV price {res.price_cv:.5f} vs COS truth {ref:.5f}; "
        f"se_cv={res.se_cv:.2e} (se_plain={res.se_plain:.2e})"
    )


def test_bs_cv_reproducible_with_seed():
    out_a = bs_call_cv(100.0, 95.0, 0.5, 0.02, 0.0, 0.2, 20_000, seed=13)
    out_b = bs_call_cv(100.0, 95.0, 0.5, 0.02, 0.0, 0.2, 20_000, seed=13)
    assert out_a.price_cv == out_b.price_cv
    assert out_a.var_reduction == out_b.var_reduction
