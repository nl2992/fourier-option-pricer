"""Finite Moment Log Stable (FMLS) model (Carr & Wu 2003) — native CF.

The FMLS process models log-stock returns as a maximally negatively-skewed
alpha-stable Lévy process.  Because all jumps are downward (β = −1), all
positive moments of S_T are finite — hence "finite moment".

Log-forward CF
--------------
For X_T = log(S_T/F_0), the characteristic function is:

    φ(u; T) = exp(T · σ^α/2 · ((iu)^α − iu))

where (iu)^α is evaluated using the principal branch of z^α = exp(α·log z),
which is analytic in the slit lower-half-plane {Im(z) ≤ 0, z ≠ 0}.

Key properties:
    φ(0) = 1  (normalization)
    φ(−i) = σ^α/2 · (1 − 1) = 0 → φ(−i) = 1  (martingale condition)
    |φ(u)| ≤ 1  for real u  (since cos(πα/2) < 0 for α ∈ (1, 2))

Special case α = 2:
    (iu)^2 = −u², so φ(u;T) = exp(T·σ²/2·(−u² − iu)) = exp(−T·iuσ²/2 − T·σ²u²/2)
    which is the BSM CF for X_T ~ N(−σ²T/2, σ²T).

Cumulants (closed form)
-----------------------
    ω   = −σ^α/2        (linear drift of X_T)
    c1  = T · ω
    c2  = T · σ^α/2 · α · (α−1) · (−1)^{α−2}/2  ... complex formula — use Cauchy
    (numerical Cauchy integration is used instead)

References
----------
* Carr, P. & Wu, L. (2003), "The Finite Moment Log Stable Process and
  Option Pricing", *Journal of Finance*, 58(2), 753–777.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press, §4.5.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class FMLSParams(ModelSpec):
    """Finite Moment Log Stable (FMLS) parameters.

    Parameters
    ----------
    alpha : float
        Stability index. Must be in ``(1, 2]``.  At ``alpha=2`` the model
        reduces to BSM with volatility ``sigma``.
    sigma : float
        Scale parameter (corresponds to BSM volatility when ``alpha=2``).
        Must be ``> 0``.
    """

    alpha: float
    sigma: float

    def __init__(self, alpha: float, sigma: float):
        object.__setattr__(self, "name", "fmls")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "sigma", sigma)


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------

def fmls_cf(u: np.ndarray, fwd: ForwardSpec, p: FMLSParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under the FMLS model.

    φ(u; T) = exp(T · σ^α/2 · ((iu)^α − iu))

    (iu)^α is the principal branch: exp(α · (log|u| + i·(π/2 + k·π))) where
    the branch is chosen continuously from u real positive.

    Parameters
    ----------
    u : array_like (real or complex)
        Frequency grid.
    fwd : ForwardSpec
    p : FMLSParams

    Returns
    -------
    np.ndarray (complex128)
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T

    iu = 1j * u_c
    # Principal branch: (iu)^alpha = exp(alpha * log(iu))
    # log(0) = -inf at u=0, but (iu)^alpha -> 0 there; suppress the benign warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        iu_alpha = np.exp(p.alpha * np.log(iu))
    iu_alpha = np.where(u_c == 0, 0.0 + 0j, iu_alpha)

    exponent = T * (p.sigma ** p.alpha / 2.0) * (iu_alpha - iu)
    return np.exp(exponent)


# ---------------------------------------------------------------------------
# Cumulants — numerical Cauchy integral
# ---------------------------------------------------------------------------

def fmls_cumulants(fwd: ForwardSpec, p: FMLSParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of X_T under the FMLS model.

    Uses numerical Cauchy-circle integration.
    """
    from ..utils.cumulants import cumulants_from_cf

    phi = lambda u: fmls_cf(u, fwd, p)
    c = cumulants_from_cf(phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
