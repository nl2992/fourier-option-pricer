"""Assertion helpers for paper and numerical-reference tests."""

from __future__ import annotations

import numpy as np


def assert_price_table(actual, expected, atol: float, label: str) -> None:
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = np.asarray(expected, dtype=float)
    if actual_arr.shape != expected_arr.shape:
        raise AssertionError(
            f"{label}: shape mismatch actual={actual_arr.shape}, expected={expected_arr.shape}"
        )
    err = np.abs(actual_arr - expected_arr)
    if not np.all(err <= atol):
        i = int(np.argmax(err))
        raise AssertionError(
            f"{label}: max|err|={err[i]:.3e} > {atol:.3e} at row {i}; "
            f"actual={actual_arr[i]:.12g}, expected={expected_arr[i]:.12g}"
        )


def assert_no_arbitrage_call_strip(strikes, calls, fwd, atol: float = 1e-8) -> None:
    strikes_arr = np.asarray(strikes, dtype=float)
    calls_arr = np.asarray(calls, dtype=float)
    if strikes_arr.ndim != 1 or calls_arr.ndim != 1 or strikes_arr.shape != calls_arr.shape:
        raise AssertionError("strikes and calls must be matching one-dimensional arrays")
    if np.any(~np.isfinite(calls_arr)):
        raise AssertionError(f"non-finite call prices: {calls_arr}")
    lower = max(float(fwd.disc) * (float(fwd.F0) - float(np.max(strikes_arr))), 0.0)
    if np.any(calls_arr < -atol):
        raise AssertionError(f"negative call prices: {calls_arr}")
    if np.any(calls_arr > float(fwd.S0) + atol):
        raise AssertionError(f"call price above spot: max={calls_arr.max():.12g}, spot={fwd.S0}")
    intrinsic = float(fwd.disc) * np.maximum(float(fwd.F0) - strikes_arr, 0.0)
    if np.any(calls_arr < intrinsic - atol):
        raise AssertionError("call price below discounted intrinsic value")
    if np.any(np.diff(calls_arr) > atol):
        raise AssertionError(f"call strip is not decreasing in strike: {calls_arr}")
    if len(calls_arr) >= 3 and np.any(np.diff(calls_arr, n=2) < -max(atol, 1e-7)):
        raise AssertionError(f"call strip is not convex in strike: {calls_arr}")
    if lower < -atol:  # pragma: no cover - defensive sanity for unusual fwd objects
        raise AssertionError(f"invalid lower bound {lower}")


def assert_cf_martingale(phi, atol: float = 1e-10) -> None:
    phi0 = np.asarray(phi(np.array([0.0])), dtype=complex)[0]
    phim = np.asarray(phi(np.array([-1j])), dtype=complex)[0]
    if abs(phi0 - 1.0) > atol:
        raise AssertionError(f"phi(0)={phi0!r}, expected 1")
    if abs(phim - 1.0) > atol:
        raise AssertionError(f"phi(-i)={phim!r}, expected 1 in log-forward coordinates")


def assert_grid_convergence(errors, tolerance: float = 1e-12) -> None:
    errors_arr = np.asarray(errors, dtype=float)
    if errors_arr.ndim != 1 or len(errors_arr) < 2:
        raise AssertionError("errors must be a one-dimensional sequence with at least two entries")
    if np.any(~np.isfinite(errors_arr)) or np.any(errors_arr < 0):
        raise AssertionError(f"errors must be finite and nonnegative: {errors_arr}")
    increases = np.diff(errors_arr) > tolerance
    if np.count_nonzero(increases) > 1 or errors_arr[-1] > errors_arr[0] + tolerance:
        raise AssertionError(f"errors do not generally decrease with grid refinement: {errors_arr}")
