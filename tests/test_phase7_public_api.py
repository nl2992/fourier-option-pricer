"""Phase 7 smoke tests: public API is importable and end-to-end callable.

Purpose:
  - Every symbol advertised in ``foureng.__all__`` resolves to a real object.
  - A minimal "end user" workflow (spec a Heston, price with COS, compute
    Delta/Gamma, invert to IV) works through the top-level imports alone.
  - ``foureng.__version__`` is a PEP-440-ish string.
"""
from __future__ import annotations
import re
import numpy as np


def test_public_symbols_importable():
    import foureng
    for name in foureng.__all__:
        assert hasattr(foureng, name), f"public API missing: {name}"


def test_version_string():
    import foureng
    assert isinstance(foureng.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", foureng.__version__), foureng.__version__


def test_end_to_end_via_public_api():
    """Workflow: Heston params -> COS price + Greeks -> BS IV inversion."""
    import foureng as fe

    fwd = fe.ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
    p = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)
    phi = lambda u: fe.heston_cf_form2(u, fwd, p)

    cums = fe.heston_cumulants(fwd, p)
    grid = fe.cos_auto_grid(cums, N=128, L=10.0)

    K = np.array([100.0])
    price = fe.cos_prices(phi, fwd, K, grid).call_prices[0]
    greeks = fe.cos_price_and_greeks(phi, fwd, K, grid)

    # Prices must agree across entry points
    assert abs(price - greeks.call_prices[0]) < 1e-12

    # Sanity: Delta in (0,1) for a vanilla call, Gamma positive
    assert 0.0 < greeks.delta[0] < 1.0
    assert greeks.gamma[0] > 0.0

    # IV round-trip
    inp = fe.BSInputs(F0=fwd.F0, K=100.0, T=fwd.T, r=fwd.r, q=fwd.q, is_call=True)
    iv = fe.implied_vol_newton_safeguarded(float(price), inp)
    assert np.isfinite(iv) and 0.0 < iv < 1.0

    # Round-trip back to price
    price_back = fe.bs_price_from_fwd(iv, inp)
    assert abs(price_back - price) < 1e-8


def test_control_variate_api():
    """Smoke: the Phase 6 CV routines work via the top-level import."""
    import foureng as fe
    r = fe.bs_call_cv(100.0, 100.0, 1.0, 0.03, 0.0, 0.2, 20_000, seed=1)
    assert r.se_cv < r.se_plain
    assert r.var_reduction > 1.0
