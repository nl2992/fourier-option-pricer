"""SWIFT Shannon-wavelet pricer (Ortiz-Gracia & Oosterlee 2016)."""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.models.registry import MODEL_REGISTRY
from foureng.pricers.swift import _scale, swift_price_at_strikes

FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.5)
MODELS = {
    "bsm": (fe.BsmParams(0.2), 1e-12),
    "heston": (fe.HestonParams(kappa=2.0, theta=0.04, nu=0.6, rho=-0.7, v0=0.04), 1e-11),
    "vg": (fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14), 1e-11),
    "kou": (fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0), 1e-9),
    "nig": (fe.NigParams(sigma=0.2, nu=0.3, theta=-0.1), 1e-10),
}
K = np.linspace(70.0, 140.0, 15)


@pytest.mark.parametrize("model", list(MODELS))
def test_matches_the_contour_reference(model):
    params, tol = MODELS[model]
    for cp in (1, -1):
        got = fe.price_strip(model, "swift", K, FWD, params, cp=cp)
        ref = fe.price_strip(model, "contour", K, FWD, params, cp=cp)
        np.testing.assert_allclose(got, ref, atol=tol)


def test_exponential_convergence_in_the_scale():
    params = MODELS["bsm"][0]
    ref = fe.price_strip("bsm", "contour", K, FWD, params)
    errs = [
        np.max(np.abs(fe.price_strip("bsm", "swift", K, FWD, params, grid=m) - ref))
        for m in (3, 4, 5)
    ]
    assert errs[0] > 1e3 * errs[1] and errs[2] < 1e-12


def test_scale_is_capped_for_slowly_decaying_cfs():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.02)
    params = fe.VGParams(sigma=0.12, nu=0.5, theta=-0.14)  # dt/nu tiny: algebraic CF decay
    phi = lambda u: MODEL_REGISTRY["vg"].cf(u, fwd, params)
    c1, c2, c4 = MODEL_REGISTRY["vg"].cumulants(fwd, params)
    width = 2 * 12 * np.sqrt(c2 + np.sqrt(abs(c4)))
    m = _scale(phi, None, width)
    assert 2.0**m * width <= 4096
    assert np.all(np.isfinite(swift_price_at_strikes(phi, fwd, K, (c1, c2, c4))))


def test_invalid_inputs():
    phi = lambda u: fe.bsm_cf(u, FWD, fe.BsmParams(0.2))
    cums = fe.bsm_cumulants(FWD, fe.BsmParams(0.2))
    with pytest.raises(ValueError, match="cp"):
        swift_price_at_strikes(phi, FWD, K, cums, cp=2)
    with pytest.raises(ValueError, match="strikes"):
        swift_price_at_strikes(phi, FWD, [0.0], cums)
