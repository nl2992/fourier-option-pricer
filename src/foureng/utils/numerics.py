from __future__ import annotations
import numpy as np


def simpson_weights(n: int) -> np.ndarray:
    """Composite Simpson's rule weights on an equally spaced grid of length n.

    Returns weights w[0..n-1] such that integral ~= dx * sum(w * f).
    For odd n this is the textbook Simpson's 1/3 rule. For even n we use
    the Carr-Madan convention w = (1/3)*(3 - (-1)^j - delta_{j,0}) which
    sets w[0]=1/3, w[N-1]=1/3 - ((-1)^(N-1))/3 ...  Here we use the simple
    trapezoidal-endpoint Simpson pattern commonly used in CM1999:
        w = [1, 4, 2, 4, 2, ..., 4, 1] / 3
    and let the caller handle the N-even edge case as in CM1999 FFT.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w *= 1.0 / 3.0
    return w


def cm_simpson_weights(n: int) -> np.ndarray:
    """Carr-Madan 1999 FFT Simpson weights (0-indexed form of eq. 24).

    CM1999 writes w_j = (eta/3)*(3 + (-1)^j - delta_{j=1}) with 1-indexed j=1..N,
    which translates (0-indexed) to the pattern:
        w[0]      = 1/3                     (left endpoint, trapezoidal)
        w[odd j]  = 4/3                     (interior odd)
        w[even j] = 2/3                     (interior even)
        w[N-1]    = 1/3                     (right endpoint)
    This function returns the bracket only; caller multiplies by eta.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    w = np.empty(n)
    w[0] = 1.0 / 3.0
    w[-1] = 1.0 / 3.0
    # interior
    idx = np.arange(1, n - 1)
    w[1:-1] = np.where(idx % 2 == 1, 4.0 / 3.0, 2.0 / 3.0)
    return w
