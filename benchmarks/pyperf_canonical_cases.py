"""Canonical performance benchmarks for regression tracking.

Run with pyperf, for example:

    python benchmarks/pyperf_canonical_cases.py
    python benchmarks/pyperf_canonical_cases.py -o benchmarks/results/pyperf.json

This script is intentionally kept out of the main CI fast path. It is meant for
manual benchmarking before release or for a dedicated benchmark workflow.
"""

from __future__ import annotations

import numpy as np

import pyperf

import foureng as fe


HESTON_FWD = fe.ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0)
HESTON_PARAMS = fe.HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04)
HESTON_STRIKES = np.linspace(80.0, 120.0, 21)

VG_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.03, T=0.25)
VG_PARAMS = fe.VGParams(sigma=0.25, nu=2.0, theta=-0.10)
VG_STRIKES = np.linspace(77.0, 123.0, 25)

FO2008_FWD = fe.ForwardSpec(S0=100.0, r=0.0, q=0.0, T=1.0)
FO2008_PARAMS = fe.HestonParams(kappa=1.5768, theta=0.0398, nu=0.5751, rho=-0.5711, v0=0.0175)
FO2008_STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=float)


def bench_heston_cos_improved_strip() -> np.ndarray:
    return fe.price_strip("heston", "cos_improved", HESTON_STRIKES, HESTON_FWD, HESTON_PARAMS)


def bench_vg_carr_madan_strip() -> np.ndarray:
    grid = fe.FFTGrid(N=4096, eta=0.25, alpha=1.5)
    return fe.price_strip("vg", "carr_madan", VG_STRIKES, VG_FWD, VG_PARAMS, grid=grid)


def bench_fo2008_heston_cos_improved_case() -> np.ndarray:
    return fe.price_strip("heston", "cos_improved", FO2008_STRIKES, FO2008_FWD, FO2008_PARAMS)


def main() -> None:
    runner = pyperf.Runner()
    runner.bench_func("heston_cos_improved_strip_21", bench_heston_cos_improved_strip)
    runner.bench_func("vg_carr_madan_strip_25", bench_vg_carr_madan_strip)
    runner.bench_func("fo2008_heston_cos_improved_5", bench_fo2008_heston_cos_improved_case)


if __name__ == "__main__":
    main()
