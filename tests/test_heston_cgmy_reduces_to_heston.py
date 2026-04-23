"""Model-reduction gate: Heston-CGMY with ``C = 0`` reduces to Heston.

Setting the CGMY activity to zero kills the Lévy exponent:

    psi(u) = 0 * Gamma(-Y) * [ ... ] = 0   for all u,

so ``phi_jump(u) = exp(T * (psi(u) - i u psi(-i))) = 1`` identically and
``phi_HCGMY(u) == phi_Heston(u)``. The price strip must be bit-identical.

Parallels :mod:`test_bates_reduces_to_heston` and
:mod:`test_heston_kou_reduces_to_heston`. (The CGMY validator was
relaxed to permit ``C = 0`` specifically to enable this gate.)
"""
from __future__ import annotations
import numpy as np

from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cf
from foureng.models.heston_cgmy import HestonCGMYParams, heston_cgmy_cf
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid


def _hcgmy_nojump(d) -> HestonCGMYParams:
    """Heston-CGMY with C=0 (no jumps). G/M/Y still have to pass validation."""
    return HestonCGMYParams(
        kappa=d["kappa"], theta=d["theta"], nu=d["nu"], rho=d["rho"], v0=d["v0"],
        C=0.0, G=5.0, M=5.0, Y=0.7,
    )


def test_hcgmy_cf_equals_heston_cf_when_C_zero(lewis_heston):
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    hc = _hcgmy_nojump(d)

    u = np.array([-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 5.0])
    phi_H  = heston_cf(u, fwd, hp)
    phi_HC = heston_cgmy_cf(u, fwd, hc)
    assert np.max(np.abs(phi_HC - phi_H)) == 0.0, \
        "CGMY(C=0) CF not bit-identical to Heston"


def test_hcgmy_price_equals_heston_price_when_C_zero_cos(lewis_heston):
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    hc = _hcgmy_nojump(d)

    K = d["strikes"]
    C_h = price_strip("heston",      "cos", K, fwd, hp)
    C_c = price_strip("heston_cgmy", "cos", K, fwd, hc)
    err = float(np.max(np.abs(C_h - C_c)))
    assert err < 1e-12, f"COS Heston-CGMY(C=0) vs Heston max|err| = {err:.3e}"


def test_hcgmy_price_equals_heston_price_when_C_zero_frft(lewis_heston):
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    hc = _hcgmy_nojump(d)

    K = d["strikes"]
    grid = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)
    C_h = price_strip("heston",      "frft", K, fwd, hp, grid=grid)
    C_c = price_strip("heston_cgmy", "frft", K, fwd, hc, grid=grid)
    err = float(np.max(np.abs(C_h - C_c)))
    assert err < 1e-12, f"FRFT Heston-CGMY(C=0) vs Heston max|err| = {err:.3e}"


def test_hcgmy_price_equals_heston_price_when_C_zero_cm(lewis_heston):
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    hc = _hcgmy_nojump(d)

    K = d["strikes"]
    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    C_h = price_strip("heston",      "carr_madan", K, fwd, hp, grid=grid)
    C_c = price_strip("heston_cgmy", "carr_madan", K, fwd, hc, grid=grid)
    err = float(np.max(np.abs(C_h - C_c)))
    assert err < 1e-12, f"CM Heston-CGMY(C=0) vs Heston max|err| = {err:.3e}"
