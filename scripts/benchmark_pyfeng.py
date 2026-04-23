"""Strip-pricing + implied-vol benchmark: foureng engines vs PyFENG.

Prints two tables per model for the six PyFENG-backed models
(``bsm``, ``heston``, ``ousv``, ``vg``, ``cgmy``, ``nig``):

  1. **Price parity** — max |price_ours - price_pyfeng| across a 21-
     strike strip, for each of our Fourier engines (``cos`` / ``frft``
     / ``carr_madan``) evaluated against PyFENG's native
     ``model.price(...)``. Also reports wall-clock.

  2. **Implied-vol parity** — take our strip prices, back out BSM IV
     via :func:`foureng.utils.implied_vol.implied_vol_from_prices`,
     compare with the IV PyFENG itself produces from its own prices
     (so any smile disagreement is traced back to price disagreement,
     not the IV solver).

Run directly from the repo root:

    $ PYTHONPATH=src python scripts/benchmark_pyfeng.py
"""
from __future__ import annotations

import time
import numpy as np

from foureng.models.base import ForwardSpec
from foureng.models.bsm import BsmParams
from foureng.models.heston import HestonParams
from foureng.models.ousv import OusvParams
from foureng.models.variance_gamma import VGParams
from foureng.models.cgmy import CgmyParams
from foureng.models.nig import NigParams
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid, FRFTGrid
from foureng.utils.implied_vol import implied_vol_from_prices


def _time_call(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def _fmt_row(label, err, sec):
    return f"  {label:<14s}  max|err|= {err:.3e}   wall= {1e3*sec:7.1f} ms"


def _run_model(name, fwd, params, K, fft_grid, frft_grid):
    print(f"\n=== {name.upper()}  (S0={fwd.S0}, r={fwd.r}, q={fwd.q}, T={fwd.T}) ===")
    print(f"  strikes: K[0]={K[0]:.1f} … K[-1]={K[-1]:.1f}  (n={len(K)})")

    # --- Price parity ------------------------------------------------------
    pf_price, pf_sec = _time_call(
        price_strip, name, "pyfeng_fft", K, fwd, params
    )

    configs = [
        ("cos",         dict(grid=None)),
        ("frft",        dict(grid=frft_grid)),
        ("carr_madan",  dict(grid=fft_grid)),
    ]
    print(f"  {'pyfeng_fft':<14s}  (oracle)                wall= {1e3*pf_sec:7.1f} ms")
    rows = []
    for method, kwargs in configs:
        C, sec = _time_call(price_strip, name, method, K, fwd, params, **kwargs)
        err = float(np.max(np.abs(C - pf_price)))
        rows.append((method, err, sec, C))
        print(_fmt_row(method, err, sec))

    # --- IV parity ---------------------------------------------------------
    iv_pf = implied_vol_from_prices(pf_price, K, fwd, cp=1)
    print(f"  implied-vol (BSM): pyfeng-price -> IV min/max = "
          f"{np.nanmin(iv_pf):.4f} / {np.nanmax(iv_pf):.4f}")
    for method, _, _, C in rows:
        iv_ours = implied_vol_from_prices(C, K, fwd, cp=1)
        d_iv = np.nanmax(np.abs(iv_ours - iv_pf))
        print(f"  IV diff vs pyfeng: {method:<14s}  max = {d_iv:.3e}")


def main():
    fwd = ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    K = np.linspace(80.0, 120.0, 21)

    fft_grid  = FFTGrid(N=4096, eta=0.25, alpha=1.5)
    frft_grid = FRFTGrid(N=4096, eta=0.25, lam=0.005, alpha=1.5)

    bsm_p    = BsmParams(sigma=0.20)
    heston_p = HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.7, v0=0.04)
    ousv_p   = OusvParams(sigma0=0.2, kappa=2.0, theta=0.2, nu=0.3, rho=-0.5)
    vg_p     = VGParams(sigma=0.2, nu=0.2, theta=-0.1)
    cgmy_p   = CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.7)
    nig_p    = NigParams(sigma=0.2, nu=0.5, theta=-0.10)

    for name, p in [("bsm", bsm_p), ("heston", heston_p),
                    ("ousv", ousv_p), ("vg", vg_p),
                    ("cgmy", cgmy_p), ("nig", nig_p)]:
        _run_model(name, fwd, p, K, fft_grid, frft_grid)


if __name__ == "__main__":
    main()
