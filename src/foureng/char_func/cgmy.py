"""CGMY (Carr-Geman-Madan-Yor 2002) characteristic function — PyFENG-backed.

CGMY is a pure-jump, infinite-activity Lévy process whose Lévy density

    k(x) = C * ( exp(-M*x) / x^{1+Y}  if x > 0,
                 exp(-G*|x|) / |x|^{1+Y} if x < 0 )

gives four-parameter control over activity (C), left/right exponential
tempering (G, M), and stability (Y in [0, 2)). This wrapper exposes
:class:`pyfeng.CgmyFft` — the model **without** a stochastic-variance
factor — in contrast to :mod:`.heston_cgmy` which is Heston ⊗ CGMY.

Parameter conventions
---------------------
CGMY's natural parameterization — ``(C, G, M, Y)`` — is exactly what
PyFENG's :class:`pyfeng.CgmyFft` expects, so the adapter is essentially
a no-op translation plus ``(intr, divr)`` wiring. Martingale correction
(the ``-i*u*psi(-i)`` compensator) is baked into PyFENG's MGF.

References
----------
* Carr, P., Geman, H., Madan, D., Yor, M. (2002), "The Fine Structure
  of Asset Returns: An Empirical Investigation", Journal of Business.
* Ballotta, L., Kyriakou, I. (2014), "Monte Carlo Simulation of the
  CGMY Process and Option Pricing", J. Futures Markets — the exact
  MGF PyFENG uses.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec
from ._pyfeng_backend import build_cached, import_pyfeng


@dataclass(frozen=True)
class CgmyParams(ModelSpec):
    """CGMY Lévy parameters.

    Attributes
    ----------
    C :
        Overall jump intensity / activity level. ``C > 0`` in the
        interior, ``C = 0`` reduces the Lévy measure to zero (pure drift
        — tests/``test_cgmy_reduces_to_zero_jumps.py`` exercises this).
    G :
        Left-tail exponential decay rate (``G > 0``). Larger ``G`` =>
        thinner left tail.
    M :
        Right-tail exponential decay rate (``M > 0``). For the
        martingale correction ``psi(-i)`` to be real and finite, we need
        ``M > 1``.
    Y :
        Stability / fine-structure parameter. ``Y < 2`` is required for
        a well-defined Lévy measure; ``Y in (0, 2) \\ {1}`` is the
        generic regime. ``Y < 0`` gives a compound-Poisson variant
        (finite activity); ``Y = 0`` reduces to Variance Gamma.
    """

    C: float
    G: float
    M: float
    Y: float

    def __init__(self, C: float, G: float, M: float, Y: float):
        object.__setattr__(self, "name", "cgmy")
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "G", G)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "Y", Y)


# ---------------------------------------------------------------------------
# PyFENG-backed CF
# ---------------------------------------------------------------------------

_CGMY_MODEL_CACHE: dict[tuple, object] = {}


def _pyfeng_cgmy_model(fwd: ForwardSpec, p: CgmyParams):
    """Build-and-cache a :class:`pyfeng.CgmyFft` for ``(fwd, p)``."""
    def _factory():
        pf = import_pyfeng()
        return pf.CgmyFft(C=p.C, G=p.G, M=p.M, Y=p.Y,
                           intr=fwd.r, divr=fwd.q)
    return build_cached(_CGMY_MODEL_CACHE, (p, fwd), _factory)


def cgmy_cf(u: np.ndarray, fwd: ForwardSpec, p: CgmyParams) -> np.ndarray:
    """CF of ``X_T = log(S_T / F_0)`` under CGMY — via PyFENG's ``CgmyFft``."""
    m = _pyfeng_cgmy_model(fwd, p)
    u_arr = np.asarray(u)
    return np.asarray(m.charfunc_logprice(u_arr, texp=fwd.T),
                      dtype=np.complex128)


# ---------------------------------------------------------------------------
# Cumulants — closed form from the Lévy exponent
# ---------------------------------------------------------------------------

def cgmy_cumulants(fwd: ForwardSpec, p: CgmyParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of ``X_T`` under CGMY (analytic).

    The CGMY Lévy exponent is

        psi(u) = C * Gamma(-Y) * [(M - i*u)^Y - M^Y + (G + i*u)^Y - G^Y],

    so cumulant ``kappa_n`` of the Lévy process at time ``T`` is
    ``T * (-i)^n * psi^{(n)}(0)`` — which simplifies to

        kappa_n / T = C * Gamma(-Y) * Y * (Y-1) * ... * (Y-n+1)
                      * [ M^{Y-n} - (-1)^n * G^{Y-n} ]  for n >= 1.

    With the martingale correction ``-u * psi(-i) / i`` baked into the
    CF, the first cumulant shifts by ``-T * psi(-i)/i`` (purely real);
    higher cumulants are unchanged.

    Returns ``(c1, c2, c4)`` — the ``c3`` slot is omitted to match the
    project-wide COS auto-grid signature.
    """
    from math import gamma as _gamma

    C, G, M, Y = p.C, p.G, p.M, p.Y
    T = fwd.T
    if C == 0.0:
        return (0.0, 0.0, 0.0)

    # Gamma(-Y) has poles at non-negative integers; CGMY's regime of
    # interest is Y in (0, 2) \ {1}. We don't guard here — PyFENG's CF
    # will itself surface a NaN if the caller picks a degenerate Y.
    gY = _gamma(-Y)
    # Falling factorial Y*(Y-1)*...*(Y-n+1).
    def _ff(n: int) -> float:
        v = 1.0
        for k in range(n):
            v *= (Y - k)
        return v

    # Drift from the martingale correction: E[X_T] = T * (psi(-i)/i),
    # and the CF's compensator is -u * psi(-i)/i, so
    # c1 (mean of X_T) = T * (psi'(0)_no-comp - psi(-i)/i).
    # psi'(0)/i (without comp.) = C * Gamma(-Y) * Y * (M^{Y-1} - G^{Y-1}).
    # psi(-i)/i (the compensator) = C * Gamma(-Y) * [(M-1)^Y - M^Y
    #                                                 + (G+1)^Y - G^Y].
    psi_no = C * gY * Y * (M ** (Y - 1) - G ** (Y - 1))
    psi_comp = C * gY * ((M - 1) ** Y - M ** Y + (G + 1) ** Y - G ** Y)
    c1 = T * (psi_no - psi_comp)
    # Higher cumulants are insensitive to the drift shift.
    c2 = T * C * gY * _ff(2) * (M ** (Y - 2) + G ** (Y - 2))
    c3 = T * C * gY * _ff(3) * (M ** (Y - 3) - G ** (Y - 3))  # noqa: F841
    c4 = T * C * gY * _ff(4) * (M ** (Y - 4) + G ** (Y - 4))
    return float(c1), float(c2), float(c4)
