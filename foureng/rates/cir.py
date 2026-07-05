"""Cox-Ingersoll-Ross (1985) short-rate model  -  closed-form CIR analytics.

Model
-----
    dr_t = kappa * (theta - r_t) dt + sigma * sqrt(r_t) dW_t,   r_0 = r0

CIR is the canonical square-root diffusion: mean-reverting to ``theta`` at
speed ``kappa`` with a state-dependent diffusion coefficient
``sigma * sqrt(r_t)``.  Unlike Vasicek, the rate stays ``>= 0``, and if
the Feller condition ``2*kappa*theta >= sigma^2`` holds the rate stays
strictly positive.

Laplace transform of the integrated rate
----------------------------------------
For any complex ``q`` such that the transform is well-defined:

    E^Q[exp(-q * I_T)] = A_tilde(T, q) * exp(-B_tilde(T, q) * r0),

    gamma        = sqrt(kappa^2 + 2 * sigma^2 * q)
    B_tilde(T,q) = 2 * q * (exp(gamma*T) - 1)
                   / ( (kappa+gamma)(exp(gamma*T)-1) + 2*gamma )
    A_tilde(T,q) = [ 2*gamma * exp((kappa+gamma)*T/2)
                     / ( (kappa+gamma)(exp(gamma*T)-1) + 2*gamma ) ]
                   ^ (2*kappa*theta / sigma^2)

The factor ``q`` in the numerator of ``B_tilde`` is essential: at
``q = 0`` the transform must equal ``1``, which forces ``B_tilde = 0``
and ``A_tilde = 1``.

Zero-coupon bond
----------------
Setting ``q = 1`` recovers the classical CIR bond formula:

    P(0, T) = A_tilde(T, 1) * exp(-B_tilde(T, 1) * r0),

    h    = sqrt(kappa^2 + 2*sigma^2)   (this is gamma at q = 1)
    B(T) = 2 * (exp(h*T) - 1) / ( (kappa + h) * (exp(h*T) - 1) + 2*h )
    A(T) = [ 2*h*exp((kappa+h)*T/2)
             / ( (kappa+h)*(exp(h*T)-1) + 2*h ) ] ^ (2*kappa*theta / sigma^2)

Setting ``q = -i*u`` gives the CF of ``I_T``.

Cumulants of the integrated rate
--------------------------------
Closed forms for higher cumulants of ``I_T`` under CIR are lengthy.  We
expose the mean and variance analytically:

    E[I_T]   = r0 * B_v(T) + theta * (T - B_v(T)),
    with     B_v(T) = (1 - exp(-kappa*T)) / kappa
              (this is the Vasicek-style B-coefficient, which coincides
               with the CIR mean-reversion time-scale integral).

    Var[I_T] = integral form  -  computed numerically from the
               second derivative of the log-Laplace at zero.

Higher cumulants are available on request via ``foureng.utils.cumulants``
by contour integration of the CF.

References
----------
* Cox, J. C., Ingersoll, J. E. & Ross, S. A. (1985), "A theory of the
  term structure of interest rates", *Econometrica*, 53, 385-407.
* Brigo, D. & Mercurio, F. (2006), *Interest Rate Models  -  Theory and
  Practice*, 2nd ed., Springer, §3.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CIRParams:
    """CIR short-rate parameters.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed, must be ``> 0``.
    theta : float
        Long-run mean of the short rate, must be ``>= 0``.
    sigma : float
        Diffusion coefficient, must be ``> 0``.
    r0 : float
        Initial short rate, must be ``>= 0``.

    Notes
    -----
    The Feller condition ``2*kappa*theta >= sigma^2`` is not enforced by
    the constructor  -  it is a sufficient (not necessary) condition for
    strict positivity.  Callers who require it can use
    :meth:`feller_ok` to check.
    """

    kappa: float
    theta: float
    sigma: float
    r0: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.kappa) and self.kappa > 0):
            raise ValueError(f"CIRParams: kappa must be > 0; got {self.kappa}")
        if not (np.isfinite(self.theta) and self.theta >= 0):
            raise ValueError(f"CIRParams: theta must be >= 0; got {self.theta}")
        if not (np.isfinite(self.sigma) and self.sigma > 0):
            raise ValueError(f"CIRParams: sigma must be > 0; got {self.sigma}")
        if not (np.isfinite(self.r0) and self.r0 >= 0):
            raise ValueError(f"CIRParams: r0 must be >= 0; got {self.r0}")

    def feller_ok(self) -> bool:
        """True iff ``2*kappa*theta >= sigma^2`` (strict positivity)."""
        return 2.0 * self.kappa * self.theta >= self.sigma * self.sigma


# ---------------------------------------------------------------------------
# CIR affine coefficients  -  complex-valued to support CF (q = -i u).
# ---------------------------------------------------------------------------


def _cir_affine(
    q: np.ndarray | complex,
    p: CIRParams,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (log_A_tilde, B_tilde) of the CIR Laplace transform.

    ``A_tilde``, ``B_tilde`` are the affine coefficients in
    ``E[exp(-q I_T)] = A_tilde(T, q) * exp(-B_tilde(T, q) * r0)``.

    Returned as complex so callers can plug in ``q = -i u`` to get the CF.
    Numerically stable at ``T = 0`` (returns ``log_A = 0``, ``B = 0``).
    """
    q_arr = np.asarray(q, dtype=np.complex128)

    if T == 0.0:
        return np.zeros_like(q_arr), np.zeros_like(q_arr)

    kappa = complex(p.kappa)
    sigma2 = complex(p.sigma * p.sigma)
    gamma = np.sqrt(kappa * kappa + 2.0 * sigma2 * q_arr)

    exp_gT = np.exp(gamma * T)
    denom = (kappa + gamma) * (exp_gT - 1.0) + 2.0 * gamma

    # Factor of q in the numerator is essential: forces B_tilde(0) = 0.
    B_tilde = 2.0 * q_arr * (exp_gT - 1.0) / denom

    # Take log of A_tilde numerator/denominator separately to keep the
    # branch of the complex logarithm consistent.  A_tilde has an integer
    # power (2*kappa*theta / sigma^2) applied to a *positive real* ratio when
    # q is real; for complex q we track the log to avoid winding-number
    # issues.
    num = 2.0 * gamma * np.exp(0.5 * (kappa + gamma) * T)
    log_ratio = np.log(num) - np.log(denom)
    exponent = 2.0 * p.kappa * p.theta / (p.sigma * p.sigma)
    log_A = exponent * log_ratio

    return log_A, B_tilde


def cir_discount_bond(p: CIRParams, T: float) -> float:
    """Zero-coupon bond price P(0, T) under CIR.

    Parameters
    ----------
    p : CIRParams
        Model parameters.
    T : float
        Maturity in years, must be ``>= 0``.

    Returns
    -------
    float
        Discount factor ``P(0, T)`` in ``(0, 1]`` (CIR guarantees this
        provided ``r0, theta >= 0``).
    """
    if not (np.isfinite(T) and T >= 0):
        raise ValueError(f"cir_discount_bond: T must be >= 0; got {T}")
    if T == 0.0:
        return 1.0

    # Real-valued case: q = 1 (P(0,T) = E[exp(-∫r ds)]).
    log_A, B_tilde = _cir_affine(np.array([1.0 + 0j]), p, T)
    return float(np.real(np.exp(log_A[0] - B_tilde[0] * p.r0)))


def cir_integrated_rate_cf(
    u: np.ndarray | complex | float,
    p: CIRParams,
    T: float,
) -> np.ndarray:
    """Characteristic function of ``I_T = ∫_0^T r_s ds`` under CIR.

    Uses the affine-Laplace transform with ``q = -i u``:

        phi_{I_T}(u) = A_tilde(T, -i u) * exp(-B_tilde(T, -i u) * r0).

    Parameters
    ----------
    u : array-like or scalar
        Fourier argument (real or complex).  Broadcasts elementwise.
    p : CIRParams
        Model parameters.
    T : float
        Maturity in years, ``T >= 0``.

    Returns
    -------
    np.ndarray
        Complex-valued CF evaluated at each ``u``.
    """
    if not (np.isfinite(T) and T >= 0):
        raise ValueError(f"cir_integrated_rate_cf: T must be >= 0; got {T}")

    u_arr = np.asarray(u, dtype=np.complex128)
    if T == 0.0:
        return np.ones_like(u_arr)

    log_A, B_tilde = _cir_affine(-1j * u_arr, p, T)
    return np.exp(log_A - B_tilde * p.r0)


def cir_integrated_rate_cumulants(p: CIRParams, T: float) -> tuple[float, float]:
    """Mean and variance of ``I_T = ∫_0^T r_s ds`` under CIR.

    Both are known in closed form (Brigo-Mercurio 2006, §3.3):

        E[I_T]   = r0 * B_v(T) + theta * (T - B_v(T)),
        Var[I_T] = (sigma^2 / kappa^2) * ( r0 * ( B_v(T) - T*exp(-kappa*T) )
                                           + theta * ( T
                                                       - 2 * B_v(T)
                                                       + T*exp(-kappa*T) ) / kappa )
                                       + higher-order CIR-specific pieces

    where ``B_v(T) = (1 - exp(-kappa*T)) / kappa``.  For robustness we
    compute the variance numerically from the log-CF second derivative
    around ``u = 0`` (central-difference on the analytic CF); this avoids
    a hairy analytic expression while remaining exact to numerical
    tolerance.  The mean is analytic.

    Parameters
    ----------
    p : CIRParams
        Model parameters.
    T : float
        Maturity in years, ``T >= 0``.

    Returns
    -------
    (mean, variance) : tuple of floats
    """
    if not (np.isfinite(T) and T >= 0):
        raise ValueError(f"cir_integrated_rate_cumulants: T must be >= 0; got {T}")
    if T == 0.0:
        return 0.0, 0.0

    # Mean (analytic).
    kT = p.kappa * T
    if kT < 1e-8:
        B_v = T - 0.5 * p.kappa * T * T
    else:
        B_v = (1.0 - np.exp(-kT)) / p.kappa
    mean = p.r0 * B_v + p.theta * (T - B_v)

    # Variance via numerical second derivative of log-CF at u = 0.
    # log(phi(u)) has expansion  i*u*c1 - 0.5*u^2*c2 + O(u^3),
    # so c2 = -Re[d^2 log(phi)/du^2 |_{u=0}].
    h = 1e-4
    log_cf_p = np.log(cir_integrated_rate_cf(np.array([h]), p, T)[0])
    log_cf_m = np.log(cir_integrated_rate_cf(np.array([-h]), p, T)[0])
    log_cf_0 = np.log(cir_integrated_rate_cf(np.array([0.0]), p, T)[0])
    # Central-difference for the second derivative of Re(log phi):
    #   Re(log phi)(u) = -0.5 c2 u^2 + O(u^4)
    #   -> d^2/du^2 Re(log phi)|_0 = -c2.
    d2_re = float(np.real(log_cf_p - 2.0 * log_cf_0 + log_cf_m) / (h * h))
    variance = -d2_re
    return float(mean), float(max(variance, 0.0))
