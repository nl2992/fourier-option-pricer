"""Spectral filter weights for the COS option-pricing expansion.

The Fang-Oosterlee (2008) COS method approximates the risk-neutral density
via a truncated cosine series.  When the characteristic function decays
slowly or the payoff is nonsmooth, the finite-series error can exhibit
Gibbs-like oscillations.  Spectral filtering damps the high-frequency COS
coefficients, trading a small amount of truncation bias for a large reduction
in the finite-series oscillation error.

Extension inspired by:
    Ruijter, M. J., Versteegh, M. and Oosterlee, C. W. (2015),
    "On the application of spectral filters in a Fourier option pricing
    technique," Journal of Computational Finance.

Usage
-----
Obtain a weight vector ``sigma`` of length ``N`` and multiply it termwise
against the characteristic-function samples ``A`` *before* the payoff dot
product:

    sigma = cos_filter_weights(N, COSFilterSpec("exponential", order=8))
    A = A * sigma          # broadcast: A[k], sigma[k]

``sigma[0]`` is always 1.0 so the drift / mean term is unaffected.

Available filters
-----------------
- ``"none"``           — identity (no filtering).
- ``"fejer"``          — first-order Cesaro / Fejér summation: σ_k = 1 − k/(N−1).
- ``"lanczos"``        — sinc filter: σ_k = sinc(k/(N−1)).
- ``"raised_cosine"``  — raised-cosine (Hann window): σ_k = ½(1 + cos(π·x)).
- ``"exponential"``    — exponential smoothness filter (order-``p``):
                         σ_k = exp(−α·(k/(N−1))^p).  Default α damps to
                         machine-ε at k = N−1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

FilterName = Literal[
    "none",
    "fejer",
    "lanczos",
    "raised_cosine",
    "exponential",
]


@dataclass(frozen=True)
class COSFilterSpec:
    """Spectral filter specification for the COS expansion.

    Parameters
    ----------
    name :
        Which filter family to apply.  ``"none"`` is the identity.
    order :
        For the ``"exponential"`` filter, the polynomial order ``p`` in
        ``exp(−α·x^p)``.  Higher ``p`` keeps more low-frequency terms
        intact while damping the highest frequencies more aggressively.
        Typical choices: 4 (moderate), 8 (default), 12 (aggressive).
        Ignored by all other filters.
    alpha :
        Exponential filter scale coefficient.  ``None`` (default) uses
        ``−log(ε_machine) ≈ 36.04``, which sends the highest-frequency
        weight to machine-ε.
    """

    name: FilterName = "none"
    order: int = 8
    alpha: float | None = None


def cos_filter_weights(N: int, spec: COSFilterSpec | None = None) -> np.ndarray:
    """Return length-``N`` spectral filter weights for COS coefficients.

    Weights are applied termwise to the cosine-series representation:

        price = disc · Σ_k  σ_k · A_k · V_k

    ``sigma[0]`` is always 1.0; all other weights lie in [0, 1] for the
    implemented filters.

    Parameters
    ----------
    N :
        Number of COS terms (must be ≥ 1).
    spec :
        Filter specification.  ``None`` is identical to
        ``COSFilterSpec("none")`` (identity).

    Returns
    -------
    np.ndarray, shape (N,), dtype float64.
    """
    if N <= 0:
        raise ValueError(f"N must be a positive integer (got {N})")

    if spec is None or spec.name == "none":
        return np.ones(N, dtype=float)

    # Single-term expansion: no high-frequency terms to damp.
    if N == 1:
        return np.ones(1, dtype=float)

    k = np.arange(N, dtype=float)
    x = k / float(N - 1)  # x[0] = 0, x[N-1] = 1

    if spec.name == "fejer":
        # Fejér (first-order Cesàro) summation: σ_k = 1 − k/(N−1)
        w = 1.0 - x

    elif spec.name == "lanczos":
        # Lanczos σ-factor: sinc(k/(N−1)) where np.sinc(x) = sin(πx)/(πx)
        # x[0] = 0 → np.sinc(0) = 1  ✓
        w = np.sinc(x)

    elif spec.name == "raised_cosine":
        # Hann / raised-cosine window
        w = 0.5 * (1.0 + np.cos(np.pi * x))

    elif spec.name == "exponential":
        order = int(spec.order)
        if order <= 0:
            raise ValueError(
                f"exponential filter order must be a positive integer (got {order})"
            )
        alpha = float(
            spec.alpha
            if spec.alpha is not None
            else -np.log(np.finfo(float).eps)   # ≈ 36.04
        )
        w = np.exp(-alpha * x**order)

    else:
        raise ValueError(
            f"unknown COS filter name {spec.name!r}; "
            "choose 'none' | 'fejer' | 'lanczos' | 'raised_cosine' | 'exponential'"
        )

    # Guarantee the k=0 weight is exactly 1.0 (shift/drift must be preserved)
    w[0] = 1.0
    return np.asarray(w, dtype=float)
