"""Phase 1 tests: MC baselines should agree with BS analytic to MC tolerance."""
from __future__ import annotations
import numpy as np
import pytest

from foureng.mc.black_scholes_mc import european_call_mc, MCSpec
from foureng.mc.heston_conditional_mc import heston_conditional_mc_calls, HestonMCScheme
from foureng.char_func.heston import HestonParams
from foureng.iv.implied_vol import bs_price_from_fwd, BSInputs


def _bs_call(S0, K, T, r, q, vol):
    F = S0 * np.exp((r - q) * T)
    return np.array([bs_price_from_fwd(vol, BSInputs(F0=F, K=float(k), T=T, r=r, q=q)) for k in K])


def test_bs_mc_matches_analytic():
    S0, T, r, q, vol = 100.0, 1.0, 0.05, 0.0, 0.2
    K = np.array([80.0, 100.0, 120.0])
    mc = european_call_mc(S0, K, T, r, q, vol, MCSpec(n_paths=200_000, seed=42))
    ref = _bs_call(S0, K, T, r, q, vol)
    # 3-sigma-ish MC tolerance
    assert np.all(np.abs(mc - ref) < 0.08), f"MC vs BS mismatch: {mc} vs {ref}"


def test_heston_cond_mc_recovers_bs_when_nu_small():
    """As nu -> 0 with v0 = theta, Heston degenerates to BS with vol = sqrt(v0)."""
    S0, T, r, q = 100.0, 1.0, 0.05, 0.0
    v0 = theta = 0.04
    kappa, nu, rho = 10.0, 0.001, 0.0
    K = np.array([80.0, 100.0, 120.0])
    p = HestonParams(kappa=kappa, theta=theta, nu=nu, rho=rho, v0=v0)
    mc = heston_conditional_mc_calls(
        S0, K, T, r, q, p,
        HestonMCScheme(n_paths=200_000, n_steps=100, seed=123, scheme="exact"),
    )
    ref = _bs_call(S0, K, T, r, q, np.sqrt(v0))
    assert np.all(np.abs(mc - ref) < 0.05), f"Heston-MC vs BS mismatch: {mc} vs {ref}"


@pytest.mark.slow
def test_heston_cond_mc_lewis_benchmark(lewis_heston):
    """Conditional MC on Lewis (2001) parameters should be within ~3e-2 at 500k paths."""
    d = lewis_heston
    p = HestonParams(kappa=d["kappa"], theta=d["theta"], nu=d["nu"],
                     rho=d["rho"], v0=d["v0"])
    mc = heston_conditional_mc_calls(
        d["S0"], d["strikes"], d["T"], d["r"], d["q"], p,
        HestonMCScheme(n_paths=500_000, n_steps=200, seed=7, scheme="exact"),
    )
    err = np.abs(mc - d["ref_calls"]).max()
    # MC-level tolerance — tightens Fourier methods later
    assert err < 0.05, f"Heston MC Lewis max err = {err:.3e}"
