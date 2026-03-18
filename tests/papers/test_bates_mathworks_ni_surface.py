"""Bates model validated against the MathWorks optByBatesNI 5x6 surface.

The MathWorks Financial Toolbox publishes a 5-strike by 6-maturity call price
surface for the Bates model using optByBatesNI. Reference values are stored in
tests/refs/bates_mathworks_ni_surface.json.

Jump convention: MathWorks specifies MeanJ = 0.02 (proportional mean jump).
The repo converts to log-jump mean via:
    mu_j = log(1 + MeanJ) - 0.5 * JumpVol^2 = 0.01660262729617973

Tolerance notes:
- Initial target is atol=1e-2. Once both engines pass comfortably the target
  can be tightened to 5e-4.
- If the short-maturity row matches but longer maturities drift, investigate
  COS truncation width and year-fraction convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import foureng as fe
from foureng.pricers.lewis import lewis_call_prices

pytestmark = [pytest.mark.paper, pytest.mark.software_reference]

_REF_PATH = Path(__file__).resolve().parents[1] / "refs" / "bates_mathworks_ni_surface.json"


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


@pytest.mark.parametrize("method,atol", [("cos_improved", 1e-2), ("lewis", 1e-2)])
def test_bates_mathworks_ni_surface(method, atol):
    """COS-improved and Lewis must match the MathWorks 5x6 NI surface to atol=1e-2.

    Surface shape: rows = strikes [76, 78, 80, 82, 84],
                   cols = maturities [0.5, 1.0, 1.5, 2.0, 2.5, 3.0].
    """
    ref = _load_ref()
    params = _make_params(ref)

    strikes = np.array(ref["strikes"], dtype=float)
    maturities = np.array(ref["maturities_years"], dtype=float)
    expected = np.array(ref["call_price_surface"], dtype=float)  # shape (nK, nT)

    got = np.empty_like(expected)

    for j, T in enumerate(maturities):
        fwd = fe.ForwardSpec(
            S0=ref["spot"],
            r=ref["risk_free_rate"],
            q=ref["dividend_yield"],
            T=float(T),
        )

        if method == "lewis":
            phi = lambda u, _fwd=fwd, _p=params: fe.bates_cf(u, _fwd, _p)
            got[:, j] = lewis_call_prices(
                phi,
                strikes,
                spot=fwd.S0,
                texp=fwd.T,
                intr=fwd.r,
                divr=fwd.q,
                method="trapz",
                u_max=250.0,
                n_u=16384,
            )
        else:
            got[:, j] = fe.price_strip("bates", method, strikes, fwd, params)

    np.testing.assert_allclose(got, expected, atol=atol, rtol=0.0)
