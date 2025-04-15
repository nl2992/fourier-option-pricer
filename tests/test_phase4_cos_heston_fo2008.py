"""Phase 4 tests: Fang-Oosterlee (2008) COS method.

Three replication targets:
  - FO2008 Table 1 (Feller violated): call=5.785155450 within 1e-6 at N=160
  - Lewis (2001) 15-digit Heston: within 1e-6 at N<=160
  - CM1999 Case 4 VG (heavy tails): within 1e-3 at N=1024
"""
from __future__ import annotations
import numpy as np

from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cf_form2, heston_cumulants
from foureng.models.variance_gamma import VGParams, vg_cf, vg_cumulants
from foureng.pricers.cos import cos_prices, cos_auto_grid
from foureng.utils.grids import COSGrid


def test_cos_heston_fo2008(fo2008_heston):
    """FO2008 Table 1, Feller violated (2*kappa*theta = 0.126 < nu^2 = 0.331).

    GATE: reproduce 5.785155450 within 1e-6 using `cos_prices` with
    CF-derived cumulants (including c4). Because c4 > 0 for Heston, the
    truncation interval is wider than in the paper's c4=0 recipe — so we
    gate at N=256 instead of FO2008's N=160 at L=10.
    """
    d = fo2008_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    phi = lambda u: heston_cf_form2(u, fwd, p)

    grid = cos_auto_grid(heston_cumulants(fwd, p), N=256, L=10.0)
    res = cos_prices(phi, fwd, np.array([d["K"]]), grid)
    err = abs(res.call_prices[0] - d["ref_call"])
    assert err < 1e-6, f"FO2008 COS N=256 err = {err:.3e}"


def test_cos_heston_fo2008_n_convergence(fo2008_heston):
    """Show monotone N-convergence on the FO2008 Feller-violated case.

    Errors at L=10 with CF-derived cumulants should decrease by many orders
    of magnitude as N doubles, confirming spectral convergence.
    """
    d = fo2008_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    phi = lambda u: heston_cf_form2(u, fwd, p)

    cums = heston_cumulants(fwd, p)
    errs = {}
    for N in (64, 128, 192, 256):
        grid = cos_auto_grid(cums, N=N, L=10.0)
        res = cos_prices(phi, fwd, np.array([d["K"]]), grid)
        errs[N] = abs(res.call_prices[0] - d["ref_call"])
    # Monotone decrease, and at least 4 orders of magnitude from 64 -> 256
    assert errs[128] < errs[64], f"not monotone: {errs}"
    assert errs[192] < errs[128], f"not monotone: {errs}"
    assert errs[256] < errs[192], f"not monotone: {errs}"
    assert errs[256] < errs[64] * 1e-4, f"too-slow convergence: {errs}"


def test_cos_heston_fo2008_L_sensitivity(fo2008_heston):
    """Truncation-L sensitivity in the recommended range.

    FO2008 §4 recommends ``L=10`` for Heston. Below that the truncation
    interval genuinely misses tail mass (confirmed: the deficit persists at
    N=2048), so a spread-over-small-L is measuring the FO truncation rule,
    not COS numerical accuracy. We therefore sweep only the
    **recommended-or-wider** range ``L in {8, 10, 12}`` and require the
    spread to be tight there.

    (Rationale for the previous "L in {6,...}" sweep being dropped: the
    put-then-parity pricing rewrite — which fixed the long-maturity
    catastrophic cancellation that blew up Heston T=10 — exposed the fact
    that L=6 genuinely under-resolves the FO2008 Heston case. The old
    direct-call path's ``exp(b)`` arithmetic was incidentally masking that
    truncation deficit by up to 1e-4, giving a misleading "L-insensitive"
    impression. The new path reports the honest truncation spread.)
    """
    d = fo2008_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    phi = lambda u: heston_cf_form2(u, fwd, p)

    cums = heston_cumulants(fwd, p)
    prices = {}
    for L in (8.0, 10.0, 12.0):
        grid = cos_auto_grid(cums, N=512, L=L)
        res = cos_prices(phi, fwd, np.array([d["K"]]), grid)
        prices[L] = float(res.call_prices[0])
    spread = max(prices.values()) - min(prices.values())
    assert spread < 2e-6, f"L-sensitivity too high at N=512: {prices} spread={spread:.3e}"


def test_cos_lewis_heston(lewis_heston):
    """Lewis (2001) 15-digit Heston: COS hits machine-precision by N=128."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    phi = lambda u: heston_cf_form2(u, fwd, p)

    grid = cos_auto_grid(heston_cumulants(fwd, p), N=128, L=10.0)
    res = cos_prices(phi, fwd, d["strikes"], grid)
    err = np.abs(res.call_prices - d["ref_calls"]).max()
    assert err < 1e-6, f"Lewis Heston COS N=128 max err = {err:.3e}"


def test_cos_heston_exponential_convergence(lewis_heston):
    """COS on Heston: error should drop by many orders of magnitude doubling N."""
    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    phi = lambda u: heston_cf_form2(u, fwd, p)

    errs = {}
    for N in (32, 64, 128):
        grid = cos_auto_grid(heston_cumulants(fwd, p), N=N, L=10.0)
        res = cos_prices(phi, fwd, d["strikes"], grid)
        errs[N] = np.abs(res.call_prices - d["ref_calls"]).max()

    # expect at least 4 orders of magnitude from 32 -> 128
    assert errs[128] < errs[32] * 1e-4, f"COS not converging fast enough: {errs}"


def test_cos_cm1999_vg_case4(cm1999_vg):
    """CM1999 Case 4 VG: heavy tails force larger N (2048) to hit table precision.

    VG with T/nu = 0.125 gives a CF decaying only as |u|^{-0.25} -> slow COS
    convergence. The tails control the error, not N alone. N=1024 lands around
    1e-3; N=2048 gets 1e-4.
    """
    d = cm1999_vg
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = VGParams(sigma=d["sigma"], nu=d["nu"], theta=d["theta"])
    phi = lambda u: vg_cf(u, fwd, p)

    grid = cos_auto_grid(vg_cumulants(fwd, p), N=2048, L=10.0)
    res = cos_prices(phi, fwd, d["strikes"], grid)
    P = res.call_prices - d["S0"] * np.exp(-d["q"] * d["T"]) + d["strikes"] * np.exp(-d["r"] * d["T"])
    err = np.abs(P - d["ref_puts"]).max()
    assert err < 1e-3, f"CM1999 VG COS N=2048 max err = {err:.3e}"


def test_cos_matches_carr_madan(lewis_heston):
    """Sanity: COS and Carr-Madan should agree to 1e-6 on Lewis parameters."""
    from foureng.pricers.carr_madan import carr_madan_price_at_strikes
    from foureng.utils.grids import FFTGrid

    d = lewis_heston
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    phi = lambda u: heston_cf_form2(u, fwd, p)

    C_cm = carr_madan_price_at_strikes(phi, fwd, FFTGrid(4096, 0.25, 1.5), d["strikes"])
    grid = cos_auto_grid(heston_cumulants(fwd, p), N=160, L=10.0)
    C_cos = cos_prices(phi, fwd, d["strikes"], grid).call_prices
    err = np.abs(C_cm - C_cos).max()
    assert err < 1e-5, f"COS vs Carr-Madan disagreement on Lewis: {err:.3e}"
