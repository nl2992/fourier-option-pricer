from __future__ import annotations
import numpy as np


def interp_linear(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Linear interpolation; returns NaN outside [x.min(), x.max()]."""
    return np.interp(xq, x, y, left=np.nan, right=np.nan)


def interp_cubic(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Natural cubic spline interpolation (scipy CubicSpline, 'natural')."""
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, y, bc_type="natural", extrapolate=False)
    return cs(xq)
