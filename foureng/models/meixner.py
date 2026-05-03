"""Meixner process (Schoutens & Teugels 1998) — native CF.

The Meixner distribution is an infinitely divisible distribution whose
characteristic function has a closed form related to the ratio of cosine
hyperbolic functions.  The Meixner process (a Lévy process with Meixner-
distributed marginals) is used in quantitative finance as an alternative
to variance-gamma for fitting the implied-vol smile.

The CF of X_T = log(S_T/F_0) under the Meixner model is:

    φ(u) = exp(T · [i·u·(r−q+ω) + δ·(log(cos(b/2))² − log(cosh((a·u−i·b)/2))²)])

Wait — the standard form is simpler.  Under the risk-neutral measure the
log-stock return has CF:

    φ(u) = exp(T · [i·u·ω + δ · log( (cos(b/2))² / (cosh((a·u−i·b)/2))² )])

where the drift correction ω ensures E[S_T/F_0] = 1:

    ζ   = (cos(b/2) / cos((b−a)/2))^{2δ} − 1
    ω   = −δ · log((cos(b/2) / cos((b+a)/2))^2)   [log-return drift corr.]

Equivalently, the Lévy exponent of X_T (with martingale correction) is:

    ψ(u) = i·u·ω + 2δ·[log cos(b/2) − log cosh((a·u − i·b)/2)]

so φ(u) = exp(T · ψ(u)).

Here the parameters are:
    a : float, a > 0  — the "scale" parameter (controls tail heaviness).
    b : float, |b| < π — skewness parameter.
    delta : float, delta > 0 — intensity parameter (controls kurtosis).

The one-parameter martingale condition is:
    ω = −2δ · [log cos(b/2) − log cos((b+a)/2)]
      = 2δ · log(cos((b+a)/2) / cos(b/2))

This ensures E[exp(X_T)] = 1 when r=q=0.

Cumulants
---------
Let m1 = a·tan(b/2) and m2 = a²/(2·cos²(b/2)):
    c1  = T·(ω + δ·a·tan(b/2))         = T·(ω + δ·m1)
    c2  = T·δ·a²/(2·cos²(b/2))         = T·δ·m2
    c3  = T·δ·a³·(sin(b/2)/(cos³(b/2))) / ... [not needed]
    c4  = T·δ·a⁴·(2+cos(b))/(4·cos⁴(b/2))  [skip, use Cauchy]

We compute cumulants numerically for COS grid construction.

References
----------
* Schoutens, W. & Teugels, J. L. (1998), "Lévy processes, polynomials
  and martingales", *Communications in Statistics — Stochastic Models*,
  14(1-2), 335–349.
* Schoutens, W. (2002), "The Meixner Process: Theory and Applications in
  Finance", *EUT Report*, EURANDOM, Eindhoven.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press, Chapter 4.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class MeixnerParams(ModelSpec):
    """Meixner process parameters.

    Parameters
    ----------
    a : float
        Scale parameter. Must be ``> 0``.
    b : float
        Skewness parameter. Must satisfy ``|b| < π``.
    delta : float
        Intensity (shape) parameter. Must be ``> 0``.
    """

    a: float
    b: float
    delta: float

    def __init__(self, a: float, b: float, delta: float):
        object.__setattr__(self, "name", "meixner")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "delta", delta)


# ---------------------------------------------------------------------------
# Drift correction
# ---------------------------------------------------------------------------

def _meixner_omega(p: MeixnerParams) -> float:
    """Drift correction so E[exp(X_T)] = 1.

    ω = 2δ · log(cos((b+a)/2) / cos(b/2))
    """
    return 2.0 * p.delta * np.log(np.cos((p.b + p.a) / 2.0) / np.cos(p.b / 2.0))


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------

def meixner_cf(u: np.ndarray, fwd: ForwardSpec, p: MeixnerParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under the Meixner process.

    φ(u) = exp(T · [i·u·ω + 2δ·(log(cos(b/2)) − log(cosh((a·u − i·b)/2)))])

    Parameters
    ----------
    u : array_like (real or complex)
        Frequency grid.
    fwd : ForwardSpec
    p : MeixnerParams

    Returns
    -------
    np.ndarray (complex128)
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T

    omega = _meixner_omega(p)

    log_cos_half_b = np.log(np.cos(p.b / 2.0))
    # cosh((a*u - ib)/2) using complex arithmetic
    arg = (p.a * u_c - 1j * p.b) / 2.0
    log_cosh_arg = np.log(np.cosh(arg))

    levy_exp = 1j * u_c * omega + 2.0 * p.delta * (log_cos_half_b - log_cosh_arg)

    return np.exp(T * levy_exp)


# ---------------------------------------------------------------------------
# Cumulants — numerical Cauchy integral (no simple closed form needed)
# ---------------------------------------------------------------------------

def meixner_cumulants(fwd: ForwardSpec, p: MeixnerParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of X_T under the Meixner process.

    Uses numerical Cauchy-circle integration (same approach as rough Heston).
    """
    from ..utils.cumulants import cumulants_from_cf

    phi = lambda u: meixner_cf(u, fwd, p)
    c = cumulants_from_cf(phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
