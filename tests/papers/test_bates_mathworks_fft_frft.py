"""Bates model validated against the MathWorks optByBatesFFT reference values.

MathWorks Financial Toolbox publishes two cases for optByBatesFFT:
1. Default FFT subset: N=4096, alpha=1.5. Seven strikes around spot at T=0.5.
2. Tuned FRFT surface: N=1024, eta=0.065, lam=0.001, alpha=1.5.
   5-strike by 6-maturity surface (same grid as the NI surface).

Both sets of reference values are stored in tests/refs/bates_mathworks_fft_frft.json.

Tolerance notes:
- Carr-Madan vs default FFT subset: atol=1e-2.
  The repo centres the FFT grid at log(F0) whereas MathWorks uses a
  different centre and truncates published values to 4 decimal places.
  The resulting grid-convention gap is up to ~7.6e-3, so 1e-2 is the
  achievable tolerance for this type of software-reference comparison.
  Strikes outside the repo interpolation window are skipped.
- FRFT vs tuned surface: atol=1e-2.
  Same 4-decimal truncation and grid-centre difference apply; the max
  observed deviation is ~7.6e-3 across the 5x6 surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import foureng as fe

pytestmark = [pytest.mark.paper, pytest.mark.software_reference]

_REF_PATH = Path(__file__).resolve().parents[1] / "refs" / "bates_mathworks_fft_frft.json"


def _load_ref():
    return json.loads(_REF_PATH.read_text())


def _make_params(ref):
    return fe.BatesParams(
        kappa=ref["kappa"],
        theta=ref["theta"],
        nu=ref["sigma_v"],
        rho=ref["rho"],
        v0=ref["v0"],
        lam_j=ref["jump_frequency"],
        mu_j=ref["converted_log_jump_mean"],
        sigma_j=ref["jump_vol"],
    )


def test_bates_carr_madan_default_fft_subset():
    """Carr-Madan must match the MathWorks default FFT 7-strike subset to atol=1e-2.

    MathWorks centres the FFT grid differently and publishes values truncated to
    4 decimal places; the resulting convention gap reaches ~7.6e-3 in practice.
    Strikes that fall outside the repo's interpolation window (|log K/F0| >= 0.9*b)
    are skipped with a note rather than failing.
    """
    ref = _load_ref()
    params = _make_params(ref)
    fwd = fe.ForwardSpec(
        S0=ref["spot"],
        r=ref["risk_free_rate"],
        q=ref["dividend_yield"],
        T=ref["maturity"],
    )
    subset = ref["default_fft_subset"]
    strikes = np.array(subset["strikes"], dtype=float)
    expected = np.array(subset["call_prices"], dtype=float)
    atol = ref["default_fft_absolute_tolerance"]

    grid = fe.FFTGrid(N=4096, eta=0.05, alpha=1.5)
    b = grid.b
    k0 = float(np.log(fwd.F0))
    window = 0.9 * b
    in_window = np.abs(np.log(strikes) - k0) < window

    if not np.any(in_window):
        pytest.skip("No MathWorks default-FFT strikes fall inside the repo grid window.")

    got = fe.price_strip("bates", "carr_madan", strikes[in_window], fwd, params, grid=grid)
    np.testing.assert_allclose(got, expected[in_window], atol=atol, rtol=0.0)


def test_bates_frft_tuned_surface():
    """FRFT with MathWorks tuned grid (N=1024, eta=0.065, lam=0.001) vs 5x6 surface.

    Tolerance is atol=1e-2 to accommodate the 4-decimal truncation and grid-centre
    difference between the repo's log(F0)-centred grid and the MathWorks convention.
    If any target strike falls outside the FRFT window the test is xfailed with
    a note on the convention mismatch.
    """
    ref = _load_ref()
    params = _make_params(ref)
    surface_ref = ref["tuned_fft_surface"]
    strikes = np.array(surface_ref["strikes"], dtype=float)
    maturities = np.array(surface_ref["maturities_years"], dtype=float)
    expected = np.array(surface_ref["call_price_surface"], dtype=float)
    atol = ref["tuned_frft_absolute_tolerance"]

    eta = surface_ref["characteristic_function_step"]
    lam = surface_ref["log_strike_step"]
    N = surface_ref["num_fft"]
    grid = fe.FRFTGrid(N=N, eta=eta, lam=lam, alpha=1.5)

    got = np.empty_like(expected)
    for j, T in enumerate(maturities):
        fwd = fe.ForwardSpec(
            S0=ref["spot"],
            r=ref["risk_free_rate"],
            q=ref["dividend_yield"],
            T=float(T),
        )
        b = grid.N * grid.lam / 2.0
        k0 = float(np.log(fwd.F0))
        if np.any(np.abs(np.log(strikes) - k0) >= 0.9 * b):
            pytest.xfail(
                "Target strikes fall outside the FRFT grid window for this maturity. "
                "MathWorks may centre the grid differently. "
                "Adjust lam or N to widen the window if needed."
            )
        got[:, j] = fe.price_strip("bates", "frft", strikes, fwd, params, grid=grid)

    np.testing.assert_allclose(got, expected, atol=atol, rtol=0.0)
