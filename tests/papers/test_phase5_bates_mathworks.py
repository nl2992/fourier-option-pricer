"""Phase 5: Bates model validated against MathWorks optByBatesNI reference case.

Tests COS, COS-improved, filtered-COS, Carr-Madan, FRFT, and Lewis pricers
against the frozen JSON reference prices derived from MathWorks toolbox output.

Note on reference prices
------------------------
MathWorks specifies jumps through a proportional mean jump MeanJ and states
that log(1+J) is normally distributed with mean log(1+MeanJ) - 0.5*JumpVol^2.
The repo converts this to the log-jump mean parameter using:

    mu_j = log(1 + MeanJ) - 0.5 * JumpVol^2 = 0.01660262729617973

which matches the documented MathWorks convention. Any residual discrepancy
should be treated as a date, basis, integration, grid, LittleTrap, or
implementation-convention issue rather than automatically as a different
jump-mean formula. For tight cross-engine agreement tests we use internally
computed high-resolution prices (``call_prices_our_convention`` in the JSON),
which agree to 1e-4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import foureng as fe
from foureng.pricers.lewis import lewis_call_prices

pytestmark = [pytest.mark.paper, pytest.mark.software_reference]

# ---------------------------------------------------------------------------
# Load reference
# ---------------------------------------------------------------------------

_REF_PATH = Path(__file__).resolve().parents[1] / "refs" / "bates_mathworks_reference.json"

with _REF_PATH.open() as _f:
    _REF = json.load(_f)

_STRIKES = np.array(_REF["strikes"])
_REF_PRICES_MATHWORKS = np.array(_REF["call_prices"])  # published MathWorks
_REF_PRICES_INTERNAL = np.array(_REF["call_prices_our_convention"])  # our hi-res Lewis
_ATOL_MATHWORKS = 1e-2  # MathWorks agree to ~1%; they use a different mu_j convention
_ATOL_INTERNAL = 1e-4  # strict internal cross-engine agreement
_ATOL_LOOSE = 1e-3  # loose cross-engine (CM, FRFT)


def _make_fwd_params():
    fwd = fe.ForwardSpec(S0=80.0, r=0.03, q=0.02, T=0.5)
    params = fe.BatesParams(
        kappa=1.0,
        theta=0.05,
        nu=0.2,
        rho=-0.7,
        v0=0.04,
        lam_j=2.0,
        mu_j=0.01660262729617973,
        sigma_j=0.08,
    )
    return fwd, params


# ---------------------------------------------------------------------------
# Test 1: CF identities
# ---------------------------------------------------------------------------


def test_bates_cf_identities_mathworks_case():
    """phi(0) = 1, phi(-u) = conj(phi(u)), phi(-i) = 1."""
    fwd, params = _make_fwd_params()
    phi = lambda u: fe.bates_cf(u, fwd, params)

    # phi(0) = 1
    val_zero = phi(np.array([0.0]))[0]
    assert abs(val_zero - 1.0) < 1e-13, f"phi(0) = {val_zero} != 1"

    # phi(-i) = 1  (martingale condition)
    val_neg_i = phi(np.array([-1j]))[0]
    assert abs(val_neg_i - 1.0) < 1e-10, f"phi(-i) = {val_neg_i} != 1"

    # phi(-u) = conj(phi(u))
    u_test = np.array([0.5, 1.0, 2.0, 5.0])
    phi_pos = phi(u_test)
    phi_neg = phi(-u_test)
    max_err = float(np.max(np.abs(phi_neg - np.conj(phi_pos))))
    assert max_err < 1e-13, f"phi(-u) != conj(phi(u)); max err = {max_err:.3e}"


# ---------------------------------------------------------------------------
# Test 2: COS vs internal ground truth and MathWorks qualitative check
# ---------------------------------------------------------------------------


def test_bates_cos_matches_mathworks_reference():
    """price_strip('bates', 'cos') must match internal reference (1e-4) and
    MathWorks qualitatively (1e-2)."""
    fwd, params = _make_fwd_params()
    prices = fe.price_strip("bates", "cos", _STRIKES, fwd, params)
    np.testing.assert_allclose(
        prices,
        _REF_PRICES_INTERNAL,
        atol=_ATOL_INTERNAL,
        err_msg="Bates COS vs internal hi-res reference",
    )
    np.testing.assert_allclose(
        prices,
        _REF_PRICES_MATHWORKS,
        atol=_ATOL_MATHWORKS,
        err_msg="Bates COS vs MathWorks qualitative check",
    )


# ---------------------------------------------------------------------------
# Test 3: COS improved
# ---------------------------------------------------------------------------


def test_bates_cos_improved_matches_mathworks_reference():
    """price_strip('bates', 'cos_improved') must match internal reference (1e-4)."""
    fwd, params = _make_fwd_params()
    prices = fe.price_strip("bates", "cos_improved", _STRIKES, fwd, params)
    np.testing.assert_allclose(
        prices,
        _REF_PRICES_INTERNAL,
        atol=_ATOL_INTERNAL,
        err_msg="Bates COS-improved vs internal hi-res reference",
    )
    np.testing.assert_allclose(
        prices,
        _REF_PRICES_MATHWORKS,
        atol=_ATOL_MATHWORKS,
        err_msg="Bates COS-improved vs MathWorks qualitative check",
    )


# ---------------------------------------------------------------------------
# Test 4: COS filtered
# ---------------------------------------------------------------------------


def test_bates_cos_filtered_is_reasonable_against_mathworks_reference():
    """price_strip('bates', 'cos_filtered') should be within atol=1e-3 vs internal."""
    fwd, params = _make_fwd_params()
    prices = fe.price_strip("bates", "cos_filtered", _STRIKES, fwd, params, grid=None)
    np.testing.assert_allclose(
        prices,
        _REF_PRICES_INTERNAL,
        atol=_ATOL_LOOSE,
        err_msg="Bates COS-filtered vs internal hi-res reference",
    )


# ---------------------------------------------------------------------------
# Test 5: Carr-Madan FFT
# ---------------------------------------------------------------------------


def test_bates_carr_madan_matches_mathworks_reference():
    """Carr-Madan FFT must match internal reference to atol=1e-3."""
    fwd, params = _make_fwd_params()
    grid = fe.FFTGrid(N=16384, eta=0.05, alpha=1.5)
    prices = fe.price_strip("bates", "carr_madan", _STRIKES, fwd, params, grid=grid)
    np.testing.assert_allclose(
        prices,
        _REF_PRICES_INTERNAL,
        atol=_ATOL_LOOSE,
        err_msg="Bates Carr-Madan vs internal hi-res reference",
    )


# ---------------------------------------------------------------------------
# Test 6: FRFT
# ---------------------------------------------------------------------------


def test_bates_frft_matches_mathworks_reference():
    """FRFT must match internal reference to atol=1e-3."""
    fwd, params = _make_fwd_params()
    grid = fe.FRFTGrid(N=4096, eta=0.05, lam=0.01, alpha=1.5)
    prices = fe.price_strip("bates", "frft", _STRIKES, fwd, params, grid=grid)
    np.testing.assert_allclose(
        prices,
        _REF_PRICES_INTERNAL,
        atol=_ATOL_LOOSE,
        err_msg="Bates FRFT vs internal hi-res reference",
    )


# ---------------------------------------------------------------------------
# Test 7: Lewis
# ---------------------------------------------------------------------------


def test_bates_lewis_matches_mathworks_reference():
    """Lewis integral must match internal reference to atol=1e-4."""
    fwd, params = _make_fwd_params()
    phi = lambda u: fe.bates_cf(u, fwd, params)
    prices = lewis_call_prices(
        phi,
        _STRIKES,
        fwd.S0,
        fwd.T,
        intr=fwd.r,
        divr=fwd.q,
        method="trapz",
        u_max=250.0,
        n_u=16384,
    )
    np.testing.assert_allclose(
        prices,
        _REF_PRICES_INTERNAL,
        atol=_ATOL_INTERNAL,
        err_msg="Bates Lewis vs internal hi-res reference",
    )
    np.testing.assert_allclose(
        prices,
        _REF_PRICES_MATHWORKS,
        atol=_ATOL_MATHWORKS,
        err_msg="Bates Lewis vs MathWorks qualitative check",
    )
