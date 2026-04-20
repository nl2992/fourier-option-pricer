"""Shared pytest fixtures: published reference prices for regression checks."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

# Make `src/foureng` importable without needing `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# --- Carr-Madan 1999, VG, Case 4 ---------------------------------------------
# sigma=0.25, nu=2.0, theta=-0.10, T=0.25, S0=100, r=0.05, q=0.03
# Reference PUT prices at K = 77, 78, 79
CM1999_VG_CASE4 = {
    "S0": 100.0, "r": 0.05, "q": 0.03, "T": 0.25,
    "sigma": 0.25, "nu": 2.0, "theta": -0.10,
    "strikes": np.array([77.0, 78.0, 79.0]),
    "ref_puts": np.array([0.6356, 0.6787, 0.7244]),
}

# --- Lewis (2001) 15-digit Heston --------------------------------------------
# S0=100, r=0.01, q=0.02, T=1, v0=0.04, kappa=4, theta=0.25, nu=1, rho=-0.5
LEWIS_HESTON = {
    "S0": 100.0, "r": 0.01, "q": 0.02, "T": 1.0,
    "kappa": 4.0, "theta": 0.25, "nu": 1.0, "rho": -0.5, "v0": 0.04,
    "strikes": np.array([80.0, 90.0, 100.0, 110.0, 120.0]),
    "ref_calls": np.array([
        26.774758743998854,
        20.933349000596710,
        16.070154917028834,
        12.132211516709845,
        9.024913483457836,
    ]),
}

# --- Fang-Oosterlee (2008) COS Heston (Feller violated) ----------------------
# kappa=1.5768, theta=0.0398, nu=0.5751, v0=0.0175, rho=-0.5711, S0=K=100, T=1
FO2008_HESTON = {
    "S0": 100.0, "K": 100.0, "r": 0.0, "q": 0.0, "T": 1.0,
    "kappa": 1.5768, "theta": 0.0398, "nu": 0.5751, "rho": -0.5711, "v0": 0.0175,
    "ref_call": 5.785155450,
}


@pytest.fixture
def cm1999_vg():
    return CM1999_VG_CASE4


@pytest.fixture
def lewis_heston():
    return LEWIS_HESTON


@pytest.fixture
def fo2008_heston():
    return FO2008_HESTON
