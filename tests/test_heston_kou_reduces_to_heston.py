"""Model-reduction gate: Heston-Kou with ``lam_j = 0`` reduces to Heston.

When the Poisson intensity is zero the Kou jump factor collapses to 1
for every ``u``, so ``phi_HKou(u) == phi_Heston(u)`` analytically and
the price strip must agree to machine precision.

Parallel to :mod:`test_bates_reduces_to_heston`.
"""
from __future__ import annotations
import numpy as np

from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cf
from foureng.models.heston_kou import HestonKouParams, heston_kou_cf
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid


def _hkou_lam0(d) -> HestonKouParams:
    # Kou jump params must satisfy eta1 > 1, eta2 > 0 to pass validation,
    # even when lam_j = 0 and the jump block is irrelevant numerically.
    return HestonKouParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"],
        lam_j=0.0, p_j=0.4, eta1=10.0, eta2=5.0,
    )


def test_hkou_cf_equals_heston_cf_when_lamj_zero(lewis_heston):
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    hk = _hkou_lam0(d)

    u = np.array([-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 5.0])
    phi_H  = heston_cf(u, fwd, hp)
    phi_HK = heston_kou_cf(u, fwd, hk)
    assert np.max(np.abs(phi_HK - phi_H)) == 0.0


def test_hkou_price_equals_heston_price_when_lamj_zero_cos(lewis_heston):
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    hk = _hkou_lam0(d)

    K = d["strikes"]
    C_h  = price_strip("heston",     "cos", K, fwd, hp)
    C_hk = price_strip("heston_kou", "cos", K, fwd, hk)
    err = float(np.max(np.abs(C_h - C_hk)))
    assert err < 1e-12, f"COS Heston-Kou(lam=0) vs Heston max|err| = {err:.3e}"


def test_hkou_price_equals_heston_price_when_lamj_zero_frft(lewis_heston):
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    hk = _hkou_lam0(d)

    K = d["strikes"]
    grid = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)
    C_h  = price_strip("heston",     "frft", K, fwd, hp, grid=grid)
    C_hk = price_strip("heston_kou", "frft", K, fwd, hk, grid=grid)
    err = float(np.max(np.abs(C_h - C_hk)))
    assert err < 1e-12, f"FRFT Heston-Kou(lam=0) vs Heston max|err| = {err:.3e}"


def test_hkou_price_equals_heston_price_when_lamj_zero_cm(lewis_heston):
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    hk = _hkou_lam0(d)

    K = d["strikes"]
    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    C_h  = price_strip("heston",     "carr_madan", K, fwd, hp, grid=grid)
    C_hk = price_strip("heston_kou", "carr_madan", K, fwd, hk, grid=grid)
    err = float(np.max(np.abs(C_h - C_hk)))
    assert err < 1e-12, f"CM Heston-Kou(lam=0) vs Heston max|err| = {err:.3e}"
