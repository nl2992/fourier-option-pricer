"""Model-reduction gate: Bates with ``lam_j = 0`` must reduce to Heston.

When the Poisson intensity is zero the jump factor collapses to 1 for
*every* frequency, so ``phi_Bates(u) == phi_Heston(u)`` analytically.
The corresponding price strip must agree to machine precision. This is
the single most informative sanity test for Bates: it pins the SVJ
extension down to a deterministic identity rather than a numerical
oracle.

Uses the Lewis Heston parameter set so the continuous block is shared
with :func:`test_phase2_carr_madan_vg.test_heston_carr_madan_lewis`.
"""
from __future__ import annotations
import numpy as np

from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cf
from foureng.models.bates import BatesParams, bates_cf
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid


def test_bates_cf_equals_heston_cf_when_lamj_zero(lewis_heston):
    """CF identity — checked pointwise on a handful of frequencies."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                      rho=d["rho"], v0=d["v0"])
    bp = BatesParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"],
                     lam_j=0.0, mu_j=-0.1, sigma_j=0.15)  # jump params irrelevant

    u = np.array([-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 5.0])
    phi_H = heston_cf(u, fwd, hp)
    phi_B = bates_cf(u, fwd, bp)
    assert np.max(np.abs(phi_B - phi_H)) == 0.0, "Bates(lam=0) CF not bit-identical to Heston"


def test_bates_price_equals_heston_price_when_lamj_zero_cos(lewis_heston):
    """COS strip reduction — deterministic engine, should be bit-identical."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                      rho=d["rho"], v0=d["v0"])
    bp = BatesParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"],
                     lam_j=0.0, mu_j=-0.05, sigma_j=0.2)

    K = d["strikes"]
    C_h = price_strip("heston", "cos", K, fwd, hp)
    C_b = price_strip("bates",  "cos", K, fwd, bp)
    err = float(np.max(np.abs(C_h - C_b)))
    assert err < 1e-12, f"COS Bates(lam=0) vs Heston max|err| = {err:.3e}"


def test_bates_price_equals_heston_price_when_lamj_zero_frft(lewis_heston):
    """FRFT strip reduction — second Fourier engine, same identity."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                      rho=d["rho"], v0=d["v0"])
    bp = BatesParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"],
                     lam_j=0.0, mu_j=0.0, sigma_j=1e-8)

    K = d["strikes"]
    grid = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)
    C_h = price_strip("heston", "frft", K, fwd, hp, grid=grid)
    C_b = price_strip("bates",  "frft", K, fwd, bp, grid=grid)
    err = float(np.max(np.abs(C_h - C_b)))
    assert err < 1e-12, f"FRFT Bates(lam=0) vs Heston max|err| = {err:.3e}"


def test_bates_price_equals_heston_price_when_lamj_zero_cm(lewis_heston):
    """Carr-Madan FFT strip reduction — third engine."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    hp = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                      rho=d["rho"], v0=d["v0"])
    bp = BatesParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"],
                     lam_j=0.0, mu_j=0.3, sigma_j=0.5)

    K = d["strikes"]
    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    C_h = price_strip("heston", "carr_madan", K, fwd, hp, grid=grid)
    C_b = price_strip("bates",  "carr_madan", K, fwd, bp, grid=grid)
    err = float(np.max(np.abs(C_h - C_b)))
    assert err < 1e-12, f"CM Bates(lam=0) vs Heston max|err| = {err:.3e}"
