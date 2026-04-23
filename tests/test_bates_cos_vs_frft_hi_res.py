"""Cross-method agreement for Bates: COS vs FRFT hi-res oracle.

Using the canonical Bates parameter set from
:data:`foureng.refs.paper_refs.BATES_REGRESSION_STRIP_V1`. The oracle is
an FRFT strip at high resolution; COS is run at the settings the
pipeline uses for production. They price the same model through two
independent Fourier engines, so agreement validates both the Bates CF
and the COS implementation for it.

The measured COS-vs-FRFT-hi gap on this strip is ~1e-9 (limited by the
FRFT grid's truncation error, not COS). We gate at 1e-6 to stay well
above any flakiness threshold while still catching real regressions.
"""
from __future__ import annotations
import numpy as np

from foureng.pipeline import price_strip
from foureng.utils.grids import FRFTGrid


# Mid-strip slice of the full 41-strike reference. Keeps the test runtime
# short while retaining ITM/ATM/OTM coverage.
_STRIP_SLICE = slice(0, 41, 2)  # 21 strikes


def test_bates_cos_vs_frft_hi_res(bates_regression_v1):
    ref = bates_regression_v1
    fwd, p = ref.fwd, ref.params
    K = np.asarray(ref.strikes, dtype=float)[_STRIP_SLICE]

    # FRFT hi-res oracle (matches the oracle grid used to seed the frozen
    # regression strip — same tolerance regime as B4 but with FRFT in the
    # oracle role so we're not comparing COS to itself via CM).
    frft_hi = FRFTGrid(N=16384, eta=0.10, lam=0.0025, alpha=1.5)
    C_frft = price_strip("bates", "frft", K, fwd, p, grid=frft_hi)
    # COS at the pipeline's default production settings (auto-grid, N=256).
    C_cos  = price_strip("bates", "cos",  K, fwd, p)

    err = float(np.max(np.abs(C_cos - C_frft)))
    assert err < 1e-6, (
        f"Bates COS vs FRFT hi-res max|err| = {err:.3e} (>= 1e-6)\n"
        f"  strikes: {K}\n  cos: {C_cos}\n  frft: {C_frft}"
    )
