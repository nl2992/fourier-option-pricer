"""Shared pytest fixtures.

**Reference prices live in one place now: ``foureng.refs.paper_refs``.**
The fixtures below are thin adapters that expose those anchors in the
dict-shape the pre-existing tests already consume, so no existing test
needed to be rewritten when the constants moved out of this file.

New tests should prefer importing the ``PaperAnchor`` / ``RegressionStrip``
objects directly (they carry the citation, notes, and the read-only
arrays) — see e.g. ``test_bates_regression_strip.py``.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

# Make `foureng` importable without needing `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foureng.refs.paper_refs import (  # noqa: E402  — after sys.path mutation
    CM1999_VG_CASE4 as _CM1999,
    LEWIS_HESTON_STRIP as _LEWIS,
    FO2008_HESTON_ATM as _FO2008,
    OUSV_REGRESSION_STRIP_V1 as _OUSV_V1,
    CGMY_REGRESSION_STRIP_V1 as _CGMY_V1,
    NIG_REGRESSION_STRIP_V1 as _NIG_V1,
    BATES_REGRESSION_STRIP_V1 as _BATES_V1,
    HESTON_KOU_REGRESSION_STRIP_V1 as _HKOU_V1,
    HESTON_CGMY_REGRESSION_STRIP_V1 as _HCGMY_V1,
)


# ---------------------------------------------------------------------------
# Legacy dict-shape fixtures — back-compat with tests that predate the
# PaperAnchor refactor. New tests should depend on the objects directly.
# ---------------------------------------------------------------------------

@pytest.fixture
def cm1999_vg():
    """Carr & Madan (1999) Case 4 VG puts — legacy dict shape."""
    p = _CM1999.params
    return {
        "S0": _CM1999.fwd.S0, "r": _CM1999.fwd.r,
        "q": _CM1999.fwd.q,   "T": _CM1999.fwd.T,
        "sigma": p.sigma, "nu": p.nu, "theta": p.theta,
        "strikes":  np.asarray(_CM1999.strikes, dtype=float),
        "ref_puts": np.asarray(_CM1999.prices,  dtype=float),
    }


@pytest.fixture
def lewis_heston():
    """Lewis (2001) 15-digit Heston calls — legacy dict shape."""
    p = _LEWIS.params
    return {
        "S0": _LEWIS.fwd.S0, "r": _LEWIS.fwd.r,
        "q": _LEWIS.fwd.q,   "T": _LEWIS.fwd.T,
        "kappa": p.kappa, "theta": p.theta, "nu": p.nu,
        "rho": p.rho, "v0": p.v0,
        "strikes":   np.asarray(_LEWIS.strikes, dtype=float),
        "ref_calls": np.asarray(_LEWIS.prices,  dtype=float),
    }


@pytest.fixture
def fo2008_heston():
    """Fang & Oosterlee (2008) Heston ATM call — legacy dict shape."""
    p = _FO2008.params
    return {
        "S0": _FO2008.fwd.S0, "r": _FO2008.fwd.r,
        "q": _FO2008.fwd.q,   "T": _FO2008.fwd.T,
        "kappa": p.kappa, "theta": p.theta, "nu": p.nu,
        "rho": p.rho, "v0": p.v0,
        "K":        float(_FO2008.strikes[0]),
        "ref_call": float(_FO2008.prices[0]),
    }


# ---------------------------------------------------------------------------
# Direct-object fixtures for new tests — carry citations, read-only arrays,
# and the ready-to-use parameter dataclass.
# ---------------------------------------------------------------------------

@pytest.fixture
def ousv_regression_v1():
    return _OUSV_V1


@pytest.fixture
def cgmy_regression_v1():
    return _CGMY_V1


@pytest.fixture
def nig_regression_v1():
    return _NIG_V1


@pytest.fixture
def bates_regression_v1():
    return _BATES_V1


@pytest.fixture
def heston_kou_regression_v1():
    return _HKOU_V1


@pytest.fixture
def heston_cgmy_regression_v1():
    return _HCGMY_V1
