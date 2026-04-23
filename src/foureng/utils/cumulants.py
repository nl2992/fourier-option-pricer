from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Cumulants:
    c1: float
    c2: float
    c4: float


def cos_truncation_interval(c: Cumulants, L: float = 10.0) -> tuple[float, float]:
    """Fang-Oosterlee truncation [a,b] = c1 +/- L*sqrt(c2 + sqrt(|c4|)).

    Uses |c4| (not max(c4, 0)) so that small-negative c4 from numerical noise
    can't silently shrink the interval below what the distribution actually
    needs. For a genuine distribution c4 >= 0 but the FFT-on-circle Cauchy
    estimate can give -1e-13 at the noise floor.
    """
    width = L * np.sqrt(c.c2 + np.sqrt(abs(c.c4)))
    return (c.c1 - width, c.c1 + width)


def cos_centered_half_width(c: Cumulants, L: float = 10.0) -> float:
    """Half-width for a centered COS interval.

    Uses the same Fang-Oosterlee cumulant heuristic as
    :func:`cos_truncation_interval`, but returns only the half-width so the
    caller can form a symmetric interval around the centered variable.
    """
    return float(L * np.sqrt(c.c2 + np.sqrt(abs(c.c4))))


def cos_centered_interval(c: Cumulants, L: float = 10.0) -> tuple[float, float]:
    """Symmetric centered interval [-w, w] for X - E[X]."""
    width = cos_centered_half_width(c, L=L)
    return (-width, width)


def cos_resolution_terms(
    width: float,
    dx_target: float,
    *,
    min_N: int = 32,
    max_N: int = 16384,
) -> int:
    """Choose a power-of-two COS term count from width and target resolution."""
    if width <= 0.0:
        raise ValueError("width must be strictly positive")
    if dx_target <= 0.0:
        raise ValueError("dx_target must be strictly positive")

    raw = width / dx_target
    N = 1 << int(np.ceil(np.log2(max(raw, float(min_N)))))
    return int(min(max(N, int(min_N)), int(max_N)))


def cos_tail_proxy(
    c: Cumulants,
    half_width: float,
    *,
    family: str = "gaussian_like",
) -> float:
    """Cheap tail proxy used by the adaptive interval policy.

    The goal is not to certify a theorem, only to expose a monotone signal the
    policy can use to widen the interval until the support truncation looks
    negligible relative to a requested tolerance.
    """
    if half_width <= 0.0:
        raise ValueError("half_width must be strictly positive")

    sigma = max(float(np.sqrt(max(c.c2, 1e-16))), 1e-8)
    z = half_width / sigma

    if family == "gaussian_like":
        return float(np.exp(-0.5 * z * z))
    if family == "semi_heavy":
        return float(np.exp(-0.75 * z))
    if family == "heavy":
        return float((1.0 + z) ** -4.0)
    raise ValueError(f"unknown tail family {family!r}")


def cumulants_from_cf(
    phi: Callable[[np.ndarray], np.ndarray],
    order: int = 4,
    radius: float = 0.25,
    M: int = 64,
) -> list[float]:
    """Generic cumulants c_1,...,c_order from a characteristic function phi.

    Uses the Cauchy integral formula on a small complex circle of radius r:
      b_n = (1/M) * sum_m K(u_m) * exp(-2*pi*i*n*m/M),   u_m = r*exp(2*pi*i*m/M)
      c_n = n! * b_n / (i^n r^n)     (from K(u) = sum c_n (iu)^n/n! = sum b_n u^n)

    This is robust across models because it only needs phi to be analytic on a
    neighbourhood of 0 and M samples of log phi on the circle. Two independent
    methods (5-point central FD on the imaginary axis and this FFT-on-circle)
    agree for Heston, so we use the FFT-on-circle exclusively here.

    Returns real parts of c_1,...,c_order.
    """
    theta = 2.0 * np.pi * np.arange(M) / M
    u = radius * np.exp(1j * theta)
    K = np.log(phi(u))
    bhat = np.fft.fft(K) / M
    out: list[float] = []
    for n in range(1, order + 1):
        b_n = bhat[n] / radius ** n
        c_n = math.factorial(n) * b_n / (1j) ** n
        out.append(float(np.real(c_n)))
    return out
