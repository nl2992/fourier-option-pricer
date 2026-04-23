"""Phase 1 Monte Carlo baseline: price the same strike strip under

  - Black-Scholes (exact one-step MC)
  - Heston conditional MC (exact chi-square variance + analytic BS conditional)

at a range of sample sizes, and report runtime vs absolute error against a
trusted reference (Carr-Madan FFT at N=16384 for Heston; closed-form Black-76
for BS). Prints a table you can paste into the report.

Usage: python benchmarks/phase1_mc_baseline.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np

# allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cf_form2
from foureng.iv.implied_vol import bs_price_from_fwd, BSInputs
from foureng.mc.black_scholes_mc import european_call_mc, MCSpec
from foureng.mc.heston_conditional_mc import heston_conditional_mc_calls, HestonMCScheme
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.utils.grids import FFTGrid


def main() -> None:
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    # ---- Black-Scholes baseline -----------------------------------------------
    fwd = ForwardSpec(S0=100.0, r=0.02, q=0.0, T=1.0)
    sigma_bs = 0.2
    ref_bs = np.array([
        bs_price_from_fwd(sigma_bs, BSInputs(F0=fwd.F0, K=float(K), T=fwd.T,
                                              r=fwd.r, q=fwd.q, is_call=True))
        for K in strikes
    ])

    print("=" * 70)
    print("Black-Scholes MC  (S0=100, r=0.02, q=0, T=1, sigma=0.2)")
    print(f"Reference prices: {ref_bs}")
    print(f"{'n_paths':>10} {'max err':>12} {'runtime (ms)':>15}")
    for n in (1_000, 10_000, 100_000, 1_000_000):
        t0 = time.perf_counter()
        mc = MCSpec(n_paths=n, seed=42)
        prices_mc = european_call_mc(
            S0=fwd.S0, K=strikes, T=fwd.T, r=fwd.r, q=fwd.q, vol=sigma_bs, mc=mc,
        )
        dt = (time.perf_counter() - t0) * 1e3
        err = float(np.abs(prices_mc - ref_bs).max())
        print(f"{n:>10d} {err:>12.3e} {dt:>15.1f}")

    # ---- Heston conditional MC ------------------------------------------------
    fwd_h = ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
    p = HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)
    phi = lambda u: heston_cf_form2(u, fwd_h, p)
    # CM at N=16384 → trusted to ~1e-7
    ref_h = carr_madan_price_at_strikes(phi, fwd_h, FFTGrid(16384, 0.05, 1.5), strikes)

    print()
    print("=" * 70)
    print("Heston conditional MC  (Lewis parameters)")
    print(f"Reference prices (CM N=16384): {ref_h}")
    print(f"{'n_paths':>10} {'n_steps':>10} {'max err':>12} {'runtime (ms)':>15}")
    for n, steps in [(5_000, 50), (20_000, 50), (50_000, 100), (200_000, 100)]:
        t0 = time.perf_counter()
        mc = HestonMCScheme(n_paths=n, n_steps=steps, seed=42, scheme="exact")
        prices_h = heston_conditional_mc_calls(
            S0=fwd_h.S0, strikes=strikes, T=fwd_h.T,
            r=fwd_h.r, q=fwd_h.q, p=p, mc=mc,
        )
        dt = (time.perf_counter() - t0) * 1e3
        err = float(np.abs(prices_h - ref_h).max())
        print(f"{n:>10d} {steps:>10d} {err:>12.3e} {dt:>15.1f}")


if __name__ == "__main__":
    main()
