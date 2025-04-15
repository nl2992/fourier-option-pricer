"""Phase 3 benchmark: FRFT vs Carr-Madan FFT at matched accuracy.

For a target max-error on the Lewis Heston strike strip, find the smallest N
each method needs and report runtime. FRFT decouples the log-strike step
`lam` from the frequency step `eta`, so it hits a given target with dramatically
smaller N.

Usage: python benchmarks/phase3_frft_vs_cm.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foureng.models.base import ForwardSpec
from foureng.models.heston import HestonParams, heston_cf_form2
from foureng.pricers.carr_madan import carr_madan_price_at_strikes
from foureng.pricers.frft import frft_price_at_strikes
from foureng.utils.grids import FFTGrid, FRFTGrid


def main() -> None:
    fwd = ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
    p = HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)
    phi = lambda u: heston_cf_form2(u, fwd, p)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    # reference
    ref = carr_madan_price_at_strikes(phi, fwd, FFTGrid(16384, 0.05, 1.5), strikes)

    target = 1e-6

    print("=" * 78)
    print(f"Target max error < {target:.0e} on Lewis Heston (5 strikes)")
    print(f"Reference prices (CM N=16384): {ref}")

    def timed(func) -> tuple[float, float]:
        out = func()  # warmup
        t0 = time.perf_counter()
        for _ in range(5):
            out = func()
        dt = (time.perf_counter() - t0) / 5.0 * 1e3
        return float(np.abs(out - ref).max()), dt

    print()
    print("Carr-Madan FFT (eta=0.25, alpha=1.5):")
    print(f"{'N':>8} {'max err':>12} {'runtime (ms)':>15}")
    for N in (512, 1024, 2048, 4096, 8192):
        def f(N=N):
            return carr_madan_price_at_strikes(phi, fwd, FFTGrid(N, 0.25, 1.5), strikes)
        err, dt = timed(f)
        hit = "  HIT" if err < target else ""
        print(f"{N:>8d} {err:>12.3e} {dt:>15.2f}{hit}")

    print()
    print("FRFT (eta=0.25, lam=chosen so N*lam ~ 2, alpha=1.5):")
    print(f"{'N':>8} {'lam':>8} {'max err':>12} {'runtime (ms)':>15}")
    # choose lam so that b = N*lam/2 comfortably covers log(K/F0) in [-0.22, 0.18]
    # Take N*lam ~ 2, so lam ~ 2/N.
    for N in (32, 64, 128, 256):
        lam = 2.0 / N  # keeps b = 1.0, half ~0.9 > 0.22 max
        def f(N=N, lam=lam):
            return frft_price_at_strikes(phi, fwd, FRFTGrid(N, 0.25, lam, 1.5), strikes)
        err, dt = timed(f)
        hit = "  HIT" if err < target else ""
        print(f"{N:>8d} {lam:>8.4f} {err:>12.3e} {dt:>15.2f}{hit}")


if __name__ == "__main__":
    main()
