"""COS(Kou) validation by double cross-check against FRFT and Carr-Madan FFT.

Strategy: run COS on a sweep of Kou regimes (low/moderate/high jump intensity,
crash-heavy) at two maturities. For each case, price the same strike strip
with two independent Fourier inversions already in the repo (FRFT and FFT)
and verify:

  A) FRFT agrees with FFT at high resolution (they should — both are damped-
     integrand Carr-Madan, just different quadratures),
  B) COS at moderate N agrees with the FRFT reference,
  C) COS error decreases as N grows and is stable across the truncation-L
     parameter.

This is additive to ``test_phase4_cos_kou.py`` (which uses a single Kou case
and only Carr-Madan as reference): here we stress multiple regimes and bring
FRFT in as a second witness. Useful because Kou has a rational-CF pole
structure that can bite damping-based methods but not COS, so disagreement
patterns localise whether a bug is in the CF, the damping, or the truncation.
"""
from __future__ import annotations
import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.kou import KouParams, kou_cf, kou_cumulants
from foureng.pricers.cos import cos_prices, cos_auto_grid
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.frft import frft_price_at_strikes
from foureng.utils.grids import FFTGrid, FRFTGrid


def _strikes_around_forward(fwd: ForwardSpec, kmin: float, kmax: float, n: int) -> np.ndarray:
    """Log-moneyness grid centred on the forward, in a safe interpolation window."""
    return fwd.F0 * np.exp(np.linspace(kmin, kmax, n))


# Eight regimes: low/moderate/high jump intensity plus a crash-heavy case,
# each at two maturities. All satisfy eta1 > 1, eta2 > 0, so kou_cf accepts.
KOU_CASES = [
    dict(name="lowjump_T0p5",  T=0.5, sigma=0.20, lam=0.05, p=0.40, eta1=10.0, eta2=6.0),
    dict(name="lowjump_T1",    T=1.0, sigma=0.20, lam=0.05, p=0.40, eta1=10.0, eta2=6.0),
    dict(name="modjump_T0p5",  T=0.5, sigma=0.18, lam=0.30, p=0.40, eta1=10.0, eta2=6.0),
    dict(name="modjump_T1",    T=1.0, sigma=0.18, lam=0.30, p=0.40, eta1=10.0, eta2=6.0),
    dict(name="highjump_T0p5", T=0.5, sigma=0.12, lam=0.80, p=0.40, eta1=12.0, eta2=7.0),
    dict(name="highjump_T1",   T=1.0, sigma=0.12, lam=0.80, p=0.40, eta1=12.0, eta2=7.0),
    dict(name="crash_T0p5",    T=0.5, sigma=0.15, lam=0.40, p=0.25, eta1=14.0, eta2=5.0),
    dict(name="crash_T1",      T=1.0, sigma=0.15, lam=0.40, p=0.25, eta1=14.0, eta2=5.0),
]


def _setup(case: dict):
    fwd = ForwardSpec(S0=100.0, r=0.02, q=0.0, T=case["T"])
    p = KouParams(sigma=case["sigma"], lam=case["lam"], p=case["p"],
                  eta1=case["eta1"], eta2=case["eta2"])
    phi = lambda u: kou_cf(u, fwd, p)
    return fwd, p, phi


@pytest.mark.parametrize("case", KOU_CASES, ids=[c["name"] for c in KOU_CASES])
def test_kou_frft_and_fft_agree(case):
    """FRFT and FFT should agree tightly — they're both damped-integrand CM."""
    fwd, p, phi = _setup(case)
    strikes = _strikes_around_forward(fwd, -0.20, 0.20, 9)

    # alpha=1.5 is safe: Kou pole requires alpha < eta1 - 1 (min eta1 here = 10)
    grid_frft = FRFTGrid(N=4096, eta=0.02, lam=0.002, alpha=1.5)
    grid_fft = FFTGrid(N=2**14, eta=0.02, alpha=1.5)

    C_frft = frft_price_at_strikes(phi, fwd, grid_frft, strikes)
    C_fft = carr_madan_price_at_strikes(phi, fwd, grid_fft, strikes)

    assert np.all(np.isfinite(C_frft))
    assert np.all(np.isfinite(C_fft))
    err = float(np.max(np.abs(C_frft - C_fft)))
    assert err <= 2e-4, f"[{case['name']}] FRFT vs FFT disagree: {err:.3e}"


@pytest.mark.parametrize("case", KOU_CASES, ids=[c["name"] for c in KOU_CASES])
def test_kou_cos_matches_frft_reference(case):
    """COS at N=512, L=10 should match FRFT at 5e-4 across all regimes."""
    fwd, p, phi = _setup(case)
    strikes = _strikes_around_forward(fwd, -0.20, 0.20, 9)

    grid_frft = FRFTGrid(N=4096, eta=0.02, lam=0.002, alpha=1.5)
    C_frft = frft_price_at_strikes(phi, fwd, grid_frft, strikes)

    cums = kou_cumulants(fwd, p)
    grid_cos = cos_auto_grid(cums, N=512, L=10.0)
    C_cos = cos_prices(phi, fwd, strikes, grid_cos).call_prices

    assert np.all(np.isfinite(C_cos))
    err = float(np.max(np.abs(C_cos - C_frft)))
    assert err <= 5e-4, f"[{case['name']}] COS vs FRFT: {err:.3e}"


@pytest.mark.parametrize("case", KOU_CASES[:4], ids=[c["name"] for c in KOU_CASES[:4]])
def test_kou_cos_converges_in_N(case):
    """Using N=1024 COS as self-reference, require monotone-ish decrease and
    N=512 within 1e-3 of the proxy reference."""
    fwd, p, phi = _setup(case)
    strikes = _strikes_around_forward(fwd, -0.15, 0.15, 7)
    cums = kou_cumulants(fwd, p)
    L = 10.0

    C_ref = cos_prices(phi, fwd, strikes, cos_auto_grid(cums, N=1024, L=L)).call_prices

    errs: list[float] = []
    for N in (64, 128, 256, 512):
        C = cos_prices(phi, fwd, strikes, cos_auto_grid(cums, N=N, L=L)).call_prices
        errs.append(float(np.max(np.abs(C - C_ref))))

    assert errs[-1] < errs[0], f"[{case['name']}] no improvement with N: {errs}"
    assert errs[-1] <= 1e-3, f"[{case['name']}] N=512 still far from proxy: {errs[-1]:.3e}"


@pytest.mark.parametrize("case", KOU_CASES[:4], ids=[c["name"] for c in KOU_CASES[:4]])
def test_kou_cos_stable_in_L(case):
    """At N=1024 the prices should not depend on L within a narrow range."""
    fwd, p, phi = _setup(case)
    strikes = _strikes_around_forward(fwd, -0.10, 0.10, 5)
    cums = kou_cumulants(fwd, p)

    prices: list[np.ndarray] = []
    for L in (6.0, 8.0, 10.0, 12.0):
        C = cos_prices(phi, fwd, strikes, cos_auto_grid(cums, N=1024, L=L)).call_prices
        prices.append(C)
    stack = np.vstack(prices)
    spread = float((stack.max(axis=0) - stack.min(axis=0)).max())
    assert spread <= 2e-4, f"[{case['name']}] L-sensitivity too high: {spread:.3e}"
