"""method='proj_asian' routes to the exact ASCOS engine (item A2).

``proj_asian`` used to run a Monte Carlo estimator with a geometric control
variate. It now delegates to the same exact engine as ``method='asian_cos'``
(:func:`foureng.pricers.arithmetic_asian.levy_arithmetic_asian_price`), so the
two methods must agree to machine precision, warn on use, and propagate
errors instead of swallowing them.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.products.asian import AsianOption

S0, R, Q = 100.0, 0.05, 0.02
FWD = fe.ForwardSpec(S0=S0, r=R, q=Q, T=1.0)
STRIKE = 100.0
MONITORING_TIMES = np.arange(1, 13) / 12.0  # 12 monthly dates

MODEL_PARAMS = {
    "bsm": fe.BsmParams(sigma=0.2),
    "vg": fe.VGParams(sigma=0.12, nu=0.3, theta=-0.14),
    "kou": fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0),
}


def _asian(cp: int = 1) -> AsianOption:
    return AsianOption(strike=STRIKE, maturity=1.0, cp=cp, monitoring_times=MONITORING_TIMES)


@pytest.mark.parametrize("model", ["bsm", "vg", "kou"])
@pytest.mark.parametrize("cp", [1, -1])
def test_proj_asian_matches_asian_cos_exactly(model, cp):
    params = MODEL_PARAMS[model]
    product = _asian(cp)
    with pytest.warns(DeprecationWarning, match="asian_cos"):
        proj_asian_price = fe.price(product, model, "proj_asian", FWD, params)
    asian_cos_price = fe.price(product, model, "asian_cos", FWD, params)
    assert proj_asian_price == pytest.approx(asian_cos_price, abs=1e-12)


def test_proj_asian_emits_deprecation_warning():
    product = _asian()
    with pytest.warns(
        DeprecationWarning, match="method='proj_asian' now uses the exact ASCOS engine"
    ):
        fe.price(product, "bsm", "proj_asian", FWD, MODEL_PARAMS["bsm"])


def test_proj_asian_rejects_geometric_average_without_swallowing():
    """Errors must propagate; proj_asian must not silently return 0.0."""
    geo_product = AsianOption(
        strike=STRIKE,
        maturity=1.0,
        cp=1,
        monitoring_times=MONITORING_TIMES,
        average_type="geometric",
    )
    with pytest.raises(NotImplementedError, match="arithmetic average Asians only"):
        fe.price(geo_product, "bsm", "proj_asian", FWD, MODEL_PARAMS["bsm"])


def test_proj_asian_rejects_unsupported_model_without_swallowing():
    heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.4, rho=-0.6, v0=0.04)
    with pytest.raises(NotImplementedError, match="1-D Lévy models"):
        fe.price(_asian(), "heston", "proj_asian", FWD, heston)


def test_proj_asian_price_cv_propagates_errors_instead_of_returning_zero():
    """The old bare `except Exception: return 0.0` in the geometric-CV helper is gone."""
    from foureng.pricers.proj import proj_asian_price_cv

    def broken_phi(u):
        raise ValueError("boom: deliberately broken characteristic function")

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="boom"):
            proj_asian_price_cv(
                broken_phi,
                FWD,
                MODEL_PARAMS["vg"],
                "vg",
                K=STRIKE,
                T=1.0,
                M=12,
                cp=1,
                n_paths=1_000,
                seed=0,
            )


def test_proj_asian_price_cv_still_importable_and_warns():
    """proj_asian_price_cv stays importable but is deprecated and no longer swallows errors."""
    from foureng.models.registry import MODEL_REGISTRY
    from foureng.pricers.proj import proj_asian_price_cv

    entry = MODEL_REGISTRY["bsm"]
    params = MODEL_PARAMS["bsm"]

    def phi(u):
        return entry.cf(u, FWD, params)

    with pytest.warns(DeprecationWarning, match="Monte Carlo estimator"):
        price = proj_asian_price_cv(
            phi, FWD, params, "bsm", K=STRIKE, T=1.0, M=12, cp=1, n_paths=2_000, seed=0
        )
    assert price > 0.0
