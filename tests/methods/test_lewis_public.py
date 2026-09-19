"""Public method="lewis" dispatch (Lewis 2001) in price_strip.

Lewis was previously reachable only as an internal fallback inside the COS
adaptive policy. These tests cover the new public route: agreement with the
contour reference engine at T=1 across models, put-call parity, dispatch
correctness, and the capability registry entry.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.core.capabilities import METHOD_REGISTRY
from foureng.models.registry import MODEL_REGISTRY
from foureng.pricers.lewis import lewis_call_prices

FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
STRIKES = np.linspace(60.0, 160.0, 21)

MODELS = {
    "bsm": fe.BsmParams(sigma=0.2),
    "heston": fe.HestonParams(kappa=2.0, theta=0.04, nu=0.6, rho=-0.7, v0=0.04),
    "bates": fe.BatesParams(
        kappa=2.0, theta=0.04, nu=0.5, rho=-0.7, v0=0.04, lam_j=0.5, mu_j=-0.1, sigma_j=0.15
    ),
    "vg": fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14),
    "nig": fe.NigParams(sigma=0.2, nu=0.3, theta=-0.1),
    "kou": fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0),
    "cgmy": fe.CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.8),
    "merton_jd": fe.MertonJDParams(sigma=0.2, lam=1.0, muj=-0.05, sigj=0.1),
}


@pytest.mark.parametrize("model", list(MODELS))
def test_agrees_with_contour_at_t1(model):
    params = MODELS[model]
    got = fe.price_strip(model, "lewis", STRIKES, FWD, params, cp=1)
    ref = fe.price_strip(model, "contour", STRIKES, FWD, params, cp=1)
    np.testing.assert_allclose(got, ref, atol=1e-8, rtol=0)


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("texp", [0.1, 1.0, 5.0])
def test_agrees_with_contour_across_maturities(model, texp):
    params = MODELS[model]
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=texp)
    got = fe.price_strip(model, "lewis", STRIKES, fwd, params, cp=1)
    ref = fe.price_strip(model, "contour", STRIKES, fwd, params, cp=1)
    np.testing.assert_allclose(got, ref, atol=1e-7, rtol=0)


@pytest.mark.parametrize("model", ["heston", "vg", "kou"])
def test_puts_match_contour_and_parity(model):
    params = MODELS[model]
    calls = fe.price_strip(model, "lewis", STRIKES, FWD, params, cp=1)
    puts = fe.price_strip(model, "lewis", STRIKES, FWD, params, cp=-1)
    ref_puts = fe.price_strip(model, "contour", STRIKES, FWD, params, cp=-1)
    np.testing.assert_allclose(puts, ref_puts, atol=1e-8, rtol=0)
    np.testing.assert_allclose(calls - puts, FWD.disc * (FWD.F0 - STRIKES), atol=1e-12, rtol=0)


def test_dispatch_matches_direct_function_call():
    """price_strip(model, "lewis", ...) must be a thin wrapper over lewis_call_prices."""
    params = MODELS["kou"]
    phi = lambda u: MODEL_REGISTRY["kou"].cf(u, FWD, params)
    via_pipeline = fe.price_strip("kou", "lewis", STRIKES, FWD, params, cp=1)
    direct = lewis_call_prices(phi, STRIKES, spot=FWD.S0, texp=FWD.T, intr=FWD.r, divr=FWD.q)
    np.testing.assert_array_equal(via_pipeline, direct)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        fe.price_strip("not_a_model", "lewis", STRIKES, FWD, fe.BsmParams(0.2))


def test_capability_registry_entry():
    assert "lewis" in METHOD_REGISTRY
    spec = METHOD_REGISTRY["lewis"]
    assert spec.requires_cf is True
    assert spec.supports_products == frozenset({"european"})
    assert spec.supports_exercise == frozenset({"european"})
    assert spec.supports_path_dependent is False
    assert "Lewis" in spec.notes
