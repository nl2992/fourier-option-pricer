"""Variance Gamma with Stochastic Arrival (VGSA)  -  Carr, Geman, Madan, Yor 2003.

VGSA runs a Variance Gamma (VG) Lévy process on a stochastic CIR activity
clock.  Where the homogeneous VG model struggles with the term structure of
skew and volatility-of-volatility, VGSA adds a persistent clustering of
jump activity through the CIR (Cox-Ingersoll-Ross) process y_t:

    dy_t = kappa*(eta - y_t) dt + lam*sqrt(y_t) dW_t
    X_T  = VG_{Y_T}  − drift correction,   Y_T = ∫_0^T y_s ds

The VG process X is parameterised by (G, M)  -  the exponential tempering
rates of the left and right jump tails  -  at unit activity per unit time.
The CIR process y_t has initial value y_0 = C (the stochastic-activity
analogue of v_0 in Heston), long-run mean eta, speed kappa, and
diffusion lam.  The drift correction keeps X_T in log-forward convention.

Characteristic function
-----------------------
By the law of total expectation and the independence of the VG process
increments given the activity path:

    E[exp(iu X_T)] = E_Y[exp(Y_T * psi_VG(u))]
                   = CIR_laplace(-psi_VG(u), T, C, kappa, eta, lam)

where CIR_laplace(s, T, y0, …) = E[exp(-s Y_T)] is the standard CIR
integrated Laplace transform, and psi_VG(u; G, M) is the VG Lévy
exponent at unit activity.  The martingale correction produces:

    phi_VGSA(u; T) = CIR_laplace(-psi_VG(u)) / CIR_laplace(-psi_VG(-i))^(iu)

which ensures E[exp(X_T)] = 1 (log-forward convention).

Parameter convention
--------------------
* ``C``       -  initial activity y_0 > 0 (NOT the CGMY C parameter)
* ``G``       -  left-tail (negative-jump) tempering rate, G > 0
* ``M``       -  right-tail (positive-jump) tempering rate, M > 1
                (``M > 1`` required for finite E[exp(X_t)])
* ``kappa``   -  CIR mean-reversion speed, > 0
* ``eta``     -  CIR long-run activity mean, > 0
* ``lam``     -  CIR vol-of-activity (diffusion), >= 0
               (``lam = 0``: deterministic clock → VGSA reduces to VG)

Reduction to VG
---------------
When ``lam = 0`` and ``C = eta = 1``, the activity clock is deterministic
and equal to T, so VGSA reduces to a standard VG process at unit activity.

References
----------
* Carr, P., Geman, H., Madan, D. B. & Yor, M. (2003), "Stochastic
  Volatility for Lévy Processes", *Mathematical Finance*, 13(3), 345–382.
* Carr, P., Geman, H., Madan, D. B. & Yor, M. (2002), "The Fine
  Structure of Asset Returns: An Empirical Investigation", *Journal of
  Business*, 75(2), 305–332. (Table 9.2: VGSA quarterly estimates.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class VGSAParams(ModelSpec):
    """VGSA parameters (Carr, Geman, Madan, Yor 2003).

    Attributes
    ----------
    C : float
        Initial CIR activity level y_0. Must be ``> 0``.
    G : float
        Left-tail exponential tempering rate. Must be ``> 0``.
    M : float
        Right-tail exponential tempering rate. Must be ``> 1``
        (required for finite E[exp(X_t)]).
    kappa : float
        CIR mean-reversion speed. Must be ``> 0``.
    eta : float
        CIR long-run activity mean. Must be ``> 0``.
    lam : float
        CIR vol-of-activity (diffusion coefficient). Must be ``>= 0``.
        When ``lam = 0`` the clock is deterministic.
    """

    C: float
    G: float
    M: float
    kappa: float
    eta: float
    lam: float

    def __init__(
        self,
        C: float,
        G: float,
        M: float,
        kappa: float,
        eta: float,
        lam: float,
    ):
        if not (np.isfinite(C) and C > 0):
            raise ValueError(f"VGSAParams: C (initial activity) must be > 0; got {C}")
        if not (np.isfinite(G) and G > 0):
            raise ValueError(f"VGSAParams: G must be > 0; got {G}")
        if not (np.isfinite(M) and M > 1):
            raise ValueError(f"VGSAParams: M must be > 1 (martingale condition); got {M}")
        if not (np.isfinite(kappa) and kappa > 0):
            raise ValueError(f"VGSAParams: kappa must be > 0; got {kappa}")
        if not (np.isfinite(eta) and eta > 0):
            raise ValueError(f"VGSAParams: eta must be > 0; got {eta}")
        if not (np.isfinite(lam) and lam >= 0):
            raise ValueError(f"VGSAParams: lam must be >= 0; got {lam}")
        object.__setattr__(self, "name", "vgsa")
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "G", G)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "lam", lam)


# ---------------------------------------------------------------------------
# VG Lévy exponent in (G, M) parameterisation  -  unit activity
# ---------------------------------------------------------------------------


def _vg_levy_exponent(u: np.ndarray, G: float, M: float) -> np.ndarray:
    """VG Lévy exponent per unit time at unit activity.

    psi_VG(u; G, M) = ln(G*M / ((G + iu) * (M - iu)))

    This is the limit of the CGMY exponent as Y → 0 with C = 1:
        psi_CGMY(u; C=1, Y→0) = -ln(1 - iu*(M-G)/(G*M) + u^2/(G*M))
                               = ln(G*M / ((G + iu)*(M - iu)))

    Parameters
    ----------
    u : np.ndarray (complex)
    G, M : float  (tempering rates, G > 0, M > 1)

    Returns
    -------
    np.ndarray (complex128)
        E[exp(iu * VG_t)] = exp(t * psi_VG(u; G, M)) for the unit-
        activity VG with left-tail rate G, right-tail rate M.
    """
    u_c = np.asarray(u, dtype=np.complex128)
    return np.log(G * M / ((G + 1j * u_c) * (M - 1j * u_c)))


# ---------------------------------------------------------------------------
# CIR integrated Laplace / CF transform
# ---------------------------------------------------------------------------


def _cir_integrated_cf(
    z: np.ndarray,
    T: float,
    y0: float,
    kappa: float,
    eta: float,
    lam: float,
) -> np.ndarray:
    """E[exp(iz * ∫_0^T y_s ds)] for a CIR process y_t.

    Uses the standard affine-diffusion closed form:

        E[exp(-s * Y_T)] = A(s, T) * exp(-B(s, T) * y0)

    substituting s = -iz (so iz = i^2 * ... i.e. we evaluate at complex s).

    When lam = 0 the clock is deterministic::

        y_t = eta + (y0 - eta) * exp(-kappa * t)
        Y_T = eta*T + (y0 - eta) * (1 - exp(-kappa*T)) / kappa

    and this function returns exp(iz * Y_T).

    Parameters
    ----------
    z : np.ndarray (complex)
        Frequency argument; the Laplace argument is s = -iz.
    T : float
        Horizon.
    y0, kappa, eta, lam : float
        CIR parameters. ``lam >= 0``; ``lam = 0`` uses the deterministic
        shortcut.

    Returns
    -------
    np.ndarray (complex128)
    """
    z_c = np.asarray(z, dtype=np.complex128)

    if lam == 0.0:
        # Deterministic clock: Y_T is known analytically.
        if kappa == 0.0:
            Y_T = float(y0 * T)
        else:
            Y_T = float(eta * T + (y0 - eta) * (1.0 - np.exp(-kappa * T)) / kappa)
        return np.exp(1j * z_c * Y_T)

    # Complex Laplace argument s = -iz.
    s = -1j * z_c
    lam2 = lam * lam

    # gamma = sqrt(kappa^2 + 2*lam^2*s)  (complex square root)
    gamma = np.sqrt(kappa**2 + 2.0 * lam2 * s)

    exp_gT = np.exp(gamma * T)
    denom = (kappa + gamma) * (exp_gT - 1.0) + 2.0 * gamma

    # A(s, T) factor  -  raised to 2*kappa*eta/lam^2
    exp_half = np.exp((kappa + gamma) * T / 2.0)
    A_ratio = 2.0 * gamma * exp_half / denom  # scalar-like per z element
    A_exp = 2.0 * kappa * eta / lam2
    A = A_ratio**A_exp

    # B(s, T) factor
    B = 2.0 * s * (exp_gT - 1.0) / denom

    return A * np.exp(-B * y0)


# ---------------------------------------------------------------------------
# VGSA characteristic function
# ---------------------------------------------------------------------------


def vgsa_cf(u: np.ndarray, fwd: ForwardSpec, p: VGSAParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under VGSA (Carr et al. 2003).

    Implements equation 4.9 from Carr, Geman, Madan, Yor (2003):

        phi_VGSA(u; T)
            = CIR_cf(-i * psi_VG(u)) / CIR_cf(-i * psi_VG(-i))^(iu)

    where CIR_cf(z) = E[exp(iz * Y_T)] with Y_T = ∫_0^T y_s ds,
    and psi_VG(u; G, M) = ln(G*M / ((G+iu)*(M-iu))).

    The denominator factor CIR_cf(-i*psi_VG(-i))^(iu) is the martingale
    correction that ensures phi(-i) = 1 (log-forward convention). Since
    CIR_cf(-i*psi_VG(-i)) is real and positive, the power is implemented
    as exp(-iu * ln(CIR_cf(-i*psi_VG(-i)))) to avoid branch cuts.

    Parameters
    ----------
    u : array_like (real or complex)
        Frequency grid.
    fwd : ForwardSpec
    p : VGSAParams

    Returns
    -------
    np.ndarray (complex128)
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T

    # VG Lévy exponent at u and at -i (martingale argument)
    psi_u = _vg_levy_exponent(u_c, p.G, p.M)
    psi_mi = _vg_levy_exponent(np.array([-1j]), p.G, p.M)[0]  # scalar

    # CIR integrated CF
    num = _cir_integrated_cf(-1j * psi_u, T, p.C, p.kappa, p.eta, p.lam)

    # Martingale correction: CIR_cf(-i*psi_mi) is real and positive.
    den_scalar = _cir_integrated_cf(np.array([-1j * psi_mi]), T, p.C, p.kappa, p.eta, p.lam)[0]
    log_den = float(np.real(np.log(den_scalar)))  # safe real log

    # phi = num / den^(iu) = num * exp(-iu * log(den))
    return num * np.exp(-1j * u_c * log_den)


# ---------------------------------------------------------------------------
# Cumulants  -  numerical Cauchy integral
# ---------------------------------------------------------------------------


def vgsa_cumulants(fwd: ForwardSpec, p: VGSAParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of X_T under VGSA.

    Computed numerically via the standard Cauchy-circle FFT on the CF.
    The VGSA CF is analytic near u = 0 for standard parameter regimes,
    so the Cauchy integral converges to machine precision.
    """
    from ..utils.cumulants import cumulants_from_cf

    def _phi(u):
        return vgsa_cf(u, fwd, p)

    c = cumulants_from_cf(_phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
