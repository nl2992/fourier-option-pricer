"""Carr & Madan (1999) paper-facing tests."""
from __future__ import annotations

import numpy as np

from foureng.models.base import ForwardSpec
from foureng.models.variance_gamma import VGParams, vg_cf
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.utils.grids import FFTGrid


def test_vg_carr_madan_cm1999_case4(cm1999_vg):
    d = cm1999_vg
    fwd = ForwardSpec(S0=d["S0"], r=d["r"], q=d["q"], T=d["T"])
    params = VGParams(sigma=d["sigma"], nu=d["nu"], theta=d["theta"])
    phi = lambda u: vg_cf(u, fwd, params)

    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    calls = carr_madan_price_at_strikes(phi, fwd, grid, d["strikes"])
    puts = calls - d["S0"] * np.exp(-d["q"] * d["T"]) + d["strikes"] * np.exp(-d["r"] * d["T"])

    err = np.abs(puts - d["ref_puts"]).max()
    assert err < 1e-3, f"CM1999 VG Case 4 max err = {err:.3e}\n puts = {puts}\n ref = {d['ref_puts']}"
