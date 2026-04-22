"""Frozen regression test: Heston-CGMY prices vs seeded oracle.

Parallels :mod:`test_bates_regression_strip`. CGMY has the roughest CF
of the three SVJ extensions, but the frozen-array settings agree with
every method to < 1e-7 on this parameter regime; keep the tolerance
consistent with the Bates / Heston-Kou variants so a real drift stands
out cleanly.
"""
from __future__ import annotations
import numpy as np

from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid


_ORACLE_CM_GRID = FFTGrid(N=32768, eta=0.10, alpha=1.5)


def test_heston_cgmy_regression_cm_oracle_grid(heston_cgmy_regression_v1):
    ref = heston_cgmy_regression_v1
    C = price_strip(ref.model, "carr_madan",
                    ref.strikes, ref.fwd, ref.params, grid=_ORACLE_CM_GRID)
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-12, f"{ref.name}: CM@oracle max|err| = {err:.3e}"


def test_heston_cgmy_regression_cos(heston_cgmy_regression_v1):
    ref = heston_cgmy_regression_v1
    C = price_strip(ref.model, "cos", ref.strikes, ref.fwd, ref.params)
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-7, f"{ref.name}: COS max|err| = {err:.3e}"


def test_heston_cgmy_regression_frft(heston_cgmy_regression_v1):
    ref = heston_cgmy_regression_v1
    grid = FRFTGrid(N=16384, eta=0.10, lam=0.0025, alpha=1.5)
    C = price_strip(ref.model, "frft", ref.strikes, ref.fwd, ref.params, grid=grid)
    err = float(np.max(np.abs(C - ref.prices)))
    assert err < 1e-7, f"{ref.name}: FRFT max|err| = {err:.3e}"
