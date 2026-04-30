"""Cross-method agreement for Heston-Kou: COS vs FRFT hi-res oracle.

Parallels :mod:`test_bates_cos_vs_frft_hi_res`. Uses the canonical
Heston-Kou parameter set from
:data:`foureng.refs.paper_refs.HESTON_KOU_REGRESSION_STRIP_V1`.
"""
from __future__ import annotations
import numpy as np

from foureng.pipeline import price_strip
from foureng.utils.grids import FRFTGrid


_STRIP_SLICE = slice(0, 41, 2)  # 21 strikes


def test_heston_kou_cos_vs_frft_hi_res(heston_kou_regression_v1):
    ref = heston_kou_regression_v1
    fwd, p = ref.fwd, ref.params
    K = np.asarray(ref.strikes, dtype=float)[_STRIP_SLICE]

    frft_hi = FRFTGrid(N=16384, eta=0.10, lam=0.0025, alpha=1.5)
    C_frft = price_strip("heston_kou", "frft", K, fwd, p, grid=frft_hi)
    C_cos  = price_strip("heston_kou", "cos",  K, fwd, p)

    err = float(np.max(np.abs(C_cos - C_frft)))
    assert err < 1e-6, (
        f"Heston-Kou COS vs FRFT hi-res max|err| = {err:.3e} (>= 1e-6)"
    )
