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
    """Fang-Oosterlee truncation [a,b] = c1 +/- L*sqrt(c2 + sqrt(|c4|))."""
    width = L * np.sqrt(c.c2 + np.sqrt(max(c.c4, 0.0)))
    return (c.c1 - width, c.c1 + width)


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
