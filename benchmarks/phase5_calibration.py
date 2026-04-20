"""Phase 5 demo: calibrate Heston to a synthetic market smile.

We generate a "market" IV surface from a known Heston parameter set (in a real
workflow this comes from SPX option quotes), then calibrate Heston to that
surface starting from a perturbed initial guess. Report runtime and recovered
parameters.

Usage: python benchmarks/phase5_calibration.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foureng.char_func.heston import HestonParams, heston_cf_form2, heston_cumulants
from foureng.surface import SurfaceSpec, model_iv_surface, calibrate_heston


def main() -> None:
    # ---- Build a synthetic "market" surface -----------------------------------
    spec = SurfaceSpec(
        S0=100.0, r=0.02, q=0.0,
        maturities=np.array([0.25, 0.5, 1.0, 2.0]),
        strikes=np.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]),
    )
    truth = HestonParams(kappa=2.5, theta=0.05, nu=0.6, rho=-0.65, v0=0.05)
    cf_true = lambda fwd: (lambda u: heston_cf_form2(u, fwd, truth))
    cum_true = lambda fwd: heston_cumulants(fwd, truth)
    market = model_iv_surface(spec, cf_true, cum_true, N=256, L=10.0)

    print("=" * 78)
    print("Synthetic market IV surface (Heston 'truth'):")
    print(f"  S0={spec.S0}, r={spec.r}, q={spec.q}")
    print(f"  kappa={truth.kappa}, theta={truth.theta}, nu={truth.nu}, "
          f"rho={truth.rho}, v0={truth.v0}")
    print()
    print("  T\\K   " + "  ".join(f"{K:>6.0f}" for K in spec.strikes))
    for i, T in enumerate(spec.maturities):
        row = "  ".join(f"{iv:>6.4f}" for iv in market[i])
        print(f"  {T:>4.2f}  {row}")

    # ---- Calibrate from a perturbed initial guess -----------------------------
    init = HestonParams(kappa=4.0, theta=0.07, nu=0.9, rho=-0.3, v0=0.03)
    t0 = time.perf_counter()
    res = calibrate_heston(spec, market, initial=init, N=192, L=10.0)
    dt = time.perf_counter() - t0

    print()
    print("=" * 78)
    print(f"Calibration (Nelder-Mead on [0,1]^d, N=192 COS points per evaluation):")
    print(f"  success:    {res.success}")
    print(f"  nfev:       {res.nfev}")
    print(f"  final loss: {res.loss:.3e}")
    print(f"  runtime:    {dt*1e3:.1f} ms")
    print()
    print(f"  {'param':>6} {'truth':>10} {'initial':>10} {'fitted':>10} {'|err|':>10}")
    for name, t, i in zip(
        ["kappa", "theta", "nu", "rho", "v0"],
        [truth.kappa, truth.theta, truth.nu, truth.rho, truth.v0],
        [init.kappa, init.theta, init.nu, init.rho, init.v0],
    ):
        v = res.params[name]
        print(f"  {name:>6} {t:>10.4f} {i:>10.4f} {v:>10.4f} {abs(v - t):>10.2e}")

    print()
    print(f"  max IV residual: {np.abs(res.residuals).max():.3e}")
    print(f"  RMS IV residual: {np.sqrt(np.mean(res.residuals**2)):.3e}")


if __name__ == "__main__":
    main()
