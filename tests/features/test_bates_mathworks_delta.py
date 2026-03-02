"""Bates delta validated against MathWorks optSensByBatesNI.

MathWorks publishes Delta = [0.6807, 0.6234, 0.5630, 0.5011, 0.4392] for
the five-strike Bates reference case (S0=80, T=0.5). Reference values are
stored in tests/refs/bates_mathworks_delta.json.

Delta is computed by central finite difference:
    dS = 1e-3 * S0
    Delta = (C(S0 + dS) - C(S0 - dS)) / (2 * dS)

using COS-improved prices. The tolerance atol=5e-3 matches the MathWorks
published 4-decimal precision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import foureng as fe

pytestmark = [pytest.mark.software_reference, pytest.mark.numerical_stability]

_REF_PATH = Path(__file__).resolve().parents[1] / "refs" / "bates_mathworks_delta.json"


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


def test_bates_mathworks_delta_central_fd():
    """Central finite-difference delta must match MathWorks optSensByBatesNI to atol=5e-3."""
    ref = _load_ref()
    params = _make_params(ref)
    S0 = ref["spot"]
    r = ref["risk_free_rate"]
    q = ref["dividend_yield"]
    T = ref["maturity"]
    strikes = np.array(ref["strikes"], dtype=float)
    expected_delta = np.array(ref["delta"], dtype=float)
    atol = ref["absolute_tolerance"]

    dS = 1e-3 * S0

    fwd_up = fe.ForwardSpec(S0=S0 + dS, r=r, q=q, T=T)
    fwd_dn = fe.ForwardSpec(S0=S0 - dS, r=r, q=q, T=T)

    prices_up = fe.price_strip("bates", "cos_improved", strikes, fwd_up, params)
    prices_dn = fe.price_strip("bates", "cos_improved", strikes, fwd_dn, params)

    delta = (prices_up - prices_dn) / (2.0 * dS)

    np.testing.assert_allclose(delta, expected_delta, atol=atol, rtol=0.0)
