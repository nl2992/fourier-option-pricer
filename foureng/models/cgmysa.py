"""CGMY-SA: CGMY process with Stochastic Arrival (CIR activity clock).

CGMY-SA runs a CGMY Lévy process on a stochastic CIR activity clock
(analogous to how VGSA runs a VG process on a CIR clock, Carr et al. 2003).
The CGMY process generalises VG (Y→0) and controls both jump activity (C),
tail asymmetry (G, M), and tail heaviness (Y), while the CIR clock adds
persistent volatility clustering and term-structure effects.

Model
-----
    X_T  = CGMY_{Y_T}  − drift correction,   Y_T = integral_0^T y_s ds
    dy_t = kappa*(eta - y_t) dt + lam*sqrt(y_t) dW_t

The drift correction enforces the log-forward martingale convention.

Characteristic function
-----------------------
By the law of total expectation and independence of CGMY increments
given the activity path:

    E[exp(iu X_T)] = E_Y[exp(Y_T * psi_CGMY(u))]
                   = CIR_laplace(-psi_CGMY(u), T, C0, kappa, eta, lam)

with the martingale correction:

    phi(u) = CIR_cf(-i*psi_CGMY(u)) / CIR_cf(-i*psi_CGMY(-i))^{iu}

identical in structure to VGSA (equation 4.9 from Carr et al. 2003).

CGMY Lévy exponent (per unit activity)
---------------------------------------
    psi_CGMY(u) = C * Gamma(-Y) * [(M-iu)^Y - M^Y + (G+iu)^Y - G^Y]

Relationship to VGSA
---------------------
Setting Y → 0 recovers the VGSA model (since the CGMY exponent with Y=0
reduces to the VG exponent in the G, M parameterisation).

Parameters
----------
C0 : float     Initial CIR activity y_0. ``> 0``.
C  : float     CGMY activity / jump intensity. ``> 0``.
G  : float     Left-tail exponential decay rate. ``> 0``.
M  : float     Right-tail exponential decay rate. ``> 1`` (martingale).
Y  : float     Tail index. ``in (0, 2) \\ {1}``.
kappa : float  CIR mean-reversion speed. ``> 0``.
eta : float    CIR long-run activity mean. ``> 0``.
lam : float    CIR vol-of-activity. ``>= 0``.

References
----------
* Carr, P., Geman, H., Madan, D. B. & Yor, M. (2003), "Stochastic
  Volatility for Lévy Processes", *Mathematical Finance*, 13(3), 345-382.
* Carr, P., Geman, H., Madan, D. B. & Yor, M. (2002), "The Fine Structure
  of Asset Returns: An Empirical Investigation", *Journal of Business*,
  75(2), 305-332.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gamma as _gamma_fn

import numpy as np

from .base import ForwardSpec, ModelSpec
from .vgsa import _cir_integrated_cf


@dataclass(frozen=True)
class CGMYSAParams(ModelSpec):
    """CGMY-SA parameters.

    Attributes
    ----------
    C0 : float    Initial CIR activity y_0. ``> 0``.
    C  : float    CGMY jump intensity. ``> 0``.
    G  : float    Left-tail decay rate. ``> 0``.
    M  : float    Right-tail decay rate. ``> 1``.
    Y  : float    Tail index. ``in (0, 2) \\ {1}``.
    kappa, eta : float  CIR mean-reversion params. ``> 0``.
    lam : float   CIR vol-of-activity. ``>= 0``.
    """

    C0: float
    C: float
    G: float
    M: float
    Y: float
    kappa: float
    eta: float
    lam: float

    def __init__(
        self,
        C0: float,
        C: float,
        G: float,
        M: float,
        Y: float,
        kappa: float,
        eta: float,
        lam: float,
    ):
        if not (np.isfinite(C0) and C0 > 0):
            raise ValueError(f"CGMYSAParams: C0 must be > 0; got {C0}")
        if not (np.isfinite(C) and C > 0):
            raise ValueError(f"CGMYSAParams: C must be > 0; got {C}")
        if not (np.isfinite(G) and G > 0):
            raise ValueError(f"CGMYSAParams: G must be > 0; got {G}")
        if not (np.isfinite(M) and M > 1):
            raise ValueError(f"CGMYSAParams: M must be > 1 (martingale); got {M}")
        if not (np.isfinite(Y) and 0.0 < Y < 2.0 and abs(Y - 1.0) > 1e-10):
            raise ValueError(f"CGMYSAParams: Y must be in (0,2) \\ {{1}}; got {Y}")
        if not (np.isfinite(kappa) and kappa > 0):
            raise ValueError(f"CGMYSAParams: kappa must be > 0; got {kappa}")
        if not (np.isfinite(eta) and eta > 0):
            raise ValueError(f"CGMYSAParams: eta must be > 0; got {eta}")
        if not (np.isfinite(lam) and lam >= 0):
            raise ValueError(f"CGMYSAParams: lam must be >= 0; got {lam}")
        object.__setattr__(self, "name", "cgmysa")
        object.__setattr__(self, "C0", C0)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "G", G)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "Y", Y)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "lam", lam)


# ---------------------------------------------------------------------------
# CGMY Lévy exponent (per unit activity)
# ---------------------------------------------------------------------------


def _cgmy_levy_exponent(u: np.ndarray, C: float, G: float, M: float, Y: float) -> np.ndarray:
    """CGMY Lévy exponent at unit activity.

    psi_CGMY(u) = C * Gamma(-Y) * [(M-iu)^Y - M^Y + (G+iu)^Y - G^Y]
    """
    u_c = np.asarray(u, dtype=np.complex128)
    gY = _gamma_fn(-Y)
    return C * gY * (
        (M - 1j * u_c) ** Y - M**Y
        + (G + 1j * u_c) ** Y - G**Y
    )


# ---------------------------------------------------------------------------
# CGMY-SA characteristic function
# ---------------------------------------------------------------------------


def cgmysa_cf(u: np.ndarray, fwd: ForwardSpec, p: CGMYSAParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under CGMY-SA.

    phi(u) = CIR_cf(-i*psi_CGMY(u)) / CIR_cf(-i*psi_CGMY(-i))^{iu}
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T

    psi_u = _cgmy_levy_exponent(u_c, p.C, p.G, p.M, p.Y)
    psi_mi = _cgmy_levy_exponent(np.array([-1j]), p.C, p.G, p.M, p.Y)[0]

    num = _cir_integrated_cf(-1j * psi_u, T, p.C0, p.kappa, p.eta, p.lam)

    den_scalar = _cir_integrated_cf(
        np.array([-1j * psi_mi]), T, p.C0, p.kappa, p.eta, p.lam
    )[0]
    log_den = float(np.real(np.log(den_scalar)))

    return num * np.exp(-1j * u_c * log_den)


# ---------------------------------------------------------------------------
# Cumulants
# ---------------------------------------------------------------------------


def cgmysa_cumulants(fwd: ForwardSpec, p: CGMYSAParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of X_T under CGMY-SA via Cauchy integral."""
    from ..utils.cumulants import cumulants_from_cf

    c = cumulants_from_cf(lambda u: cgmysa_cf(u, fwd, p), order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
