"""Additional small edge-case tests for project robustness."""
from __future__ import annotations

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid


def test_carr_madan_rejects_nonpositive_strikes():
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    params = BsmParams(sigma=0.2)

    with pytest.raises(ValueError, match="strikes must be > 0"):
        price_strip(
            "bsm",
            "carr_madan",
            np.array([90.0, 0.0, 110.0]),
            fwd,
            params,
            grid=FFTGrid(N=4096, eta=0.25, alpha=1.5),
        )
