"""Frozen regression test: Bates prices vs the seeded oracle prices.

Compares every supported pricing method against
:data:`foureng.refs.paper_refs.BATES_REGRESSION_STRIP_V1`. If a future
refactor silently shifts prices — a new compensator convention, a
grid-boundary off-by-one, a PyFENG upgrade with a different CF
convention — this test is what catches it.

Two tiers of tolerance:
  * ``carr_madan`` at the oracle's own grid : expected bit-identical
    (that exact CM call is how the frozen array was seeded), so the
    tolerance is numerical round-off only.
  * ``cos`` / ``frft`` : independent engines agreeing with the oracle
    to ~1e-9 on this strip; gate at 1e-7 to catch anything bigger.
"""
from __future__ import annotations
import numpy as np

from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid


# The exact grid used when the reference array was generated (see
# RegressionStrip.ref_method). Matching it makes the CM test a
# numerical identity check.
_ORACLE_CM_GRID = FFTGrid(N=32768, eta=0.10, alpha=1.5)


def test_bates_regression_cm_oracle_grid(bates_regression_v1):
    """CM at the oracle grid — numerical identity, tolerance = round-off."""
    ref = bates_regression_v1
    C = price_strip(ref.model, "carr_madan",
                    ref.strikes, ref.fwd, ref.params, grid=_ORACLE_CM_GRID)
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-12, (
        f"{ref.name}: CM@oracle max|err| = {err:.3e} (expected < 1e-12)"
    )


def test_bates_regression_cos(bates_regression_v1):
    ref = bates_regression_v1
    C = price_strip(ref.model, "cos", ref.strikes, ref.fwd, ref.params)
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-7, f"{ref.name}: COS max|err| = {err:.3e}"


def test_bates_regression_frft(bates_regression_v1):
    ref = bates_regression_v1
    grid = FRFTGrid(N=16384, eta=0.10, lam=0.0025, alpha=1.5)
    C = price_strip(ref.model, "frft", ref.strikes, ref.fwd, ref.params, grid=grid)
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-7, f"{ref.name}: FRFT max|err| = {err:.3e}"
