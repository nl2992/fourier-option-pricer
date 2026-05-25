"""BNS Gamma-OU model  -  Barndorff-Nielsen & Shephard (2001).

The BNS model replaces the Brownian-motion variance driver of Heston with a
non-Gaussian Ornstein-Uhlenbeck (OU) process driven by a Lévy subordinator
(here: a Gamma process).  This gives variance clustering without a diffusive
vol-of-vol term and introduces a leverage effect through the parameter rho_l.

Model dynamics (risk-neutral measure)
--------------------------------------
    d log S_t = (r - q - V_t/2 - lambda_ou * k_Z(rho_l)) dt
                + sqrt(V_t) dW_t + rho_l dZ_{lambda_ou * t}
    dV_t      = -lambda_ou * V_t dt + dZ_{lambda_ou * t}

where W is a standard Brownian motion, Z is a Gamma subordinator with
parameters ``(a, b)`` (mean ``a/b`` per unit time, variance ``a/b^2``),
and ``lambda_ou > 0`` is the OU mean-reversion speed.

The drift correction ``-lambda_ou * k_Z(rho_l)`` in the log-price is the
compensator that maintains the log-forward martingale convention.

Characteristic function
-----------------------
Using the substitution Z_{lambda_ou*T} = V_T - V_0 + lambda_ou*integral(V dt)
(from integrating the OU SDE), the log-forward return X_T = log(S_T/F_0)
can be written as:

    X_T = rho_l*(V_T - V_0) + (rho_l*lambda_ou - 1/2)*integral(V dt)
          + integral(sqrt(V) dW) - lambda_ou*T*k_Z(rho_l)

Taking the CF via the Gamma-OU Laplace transform formula (Barndorff-Nielsen
& Shephard 2001, Nicolato & Venardos 2003):

    log phi(u) = -(iu + u^2)/2 * V_0*(1 - exp(-lambda_ou*T))/lambda_ou
                 - iu * lambda_ou * T * k_Z(rho_l)
                 + lambda_ou * integral_0^T k_Z(g_u(tau)) dtau

where the integration variable is:

    g_u(tau) = iu*rho_l - (iu + u^2)/(2*lambda_ou) * (1 - exp(-lambda_ou*tau))

and the Gamma log-Laplace transform is:

    k_Z(c) = a * log(b / (b - c)) = -a * log(1 - c/b)   for c < b.

The integral is evaluated numerically (200-point trapezoidal rule).

Parameters
----------
V0 : float       Initial variance. ``> 0``.
lambda_ou : float  OU mean-reversion speed. ``> 0``.
rho_l : float    Leverage coefficient (instantaneous correlation between
                 log-price innovations and variance increments). ``< b``
                 for the Gamma log-Laplace to be defined at rho_l.
a : float        Gamma BDLP activity (shape rate). ``> 0``.
b : float        Gamma BDLP rate parameter. ``> rho_l`` (so k_Z(rho_l)
                 is finite) and ``> 0``.

Long-run variance mean
----------------------
The stationary mean of V_t is ``E[V_oo] = a / b`` (mean of the Gamma
increment per unit activity).

References
----------
* Barndorff-Nielsen, O. E. & Shephard, N. (2001), "Non-Gaussian
  Ornstein-Uhlenbeck-based Models and Some of Their Uses in Financial
  Economics", *Journal of the Royal Statistical Society B*, 63, 167-241.
* Nicolato, E. & Venardos, E. (2003), "Option Pricing in Stochastic
  Volatility Models of the Ornstein-Uhlenbeck Type", *Mathematical
  Finance*, 13(4), 445-466.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec

# np.trapezoid was added in NumPy 2.0; np.trapz is deprecated there and
# removed from stubs.  This shim works on both NumPy 1.x and 2.x.
_trapezoid = getattr(np, "trapezoid", np.trapz)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class BNSGammaOUParams(ModelSpec):
    """BNS Gamma-OU parameters (Barndorff-Nielsen & Shephard 2001).

    Attributes
    ----------
    V0 : float         Initial variance (> 0).
    lambda_ou : float  OU mean-reversion speed (> 0).
    rho_l : float      Leverage coefficient.  Must satisfy ``rho_l < b``.
    a : float          Gamma BDLP activity / shape rate (> 0).
    b : float          Gamma BDLP rate parameter (> 0, > rho_l).
    """

    V0: float
    lambda_ou: float
    rho_l: float
    a: float
    b: float

    def __init__(
        self,
        V0: float,
        lambda_ou: float,
        rho_l: float,
        a: float,
        b: float,
    ):
        if not (np.isfinite(V0) and V0 > 0):
            raise ValueError(f"BNSGammaOUParams: V0 must be > 0; got {V0}")
        if not (np.isfinite(lambda_ou) and lambda_ou > 0):
            raise ValueError(f"BNSGammaOUParams: lambda_ou must be > 0; got {lambda_ou}")
        if not np.isfinite(rho_l):
            raise ValueError(f"BNSGammaOUParams: rho_l must be finite; got {rho_l}")
        if not (np.isfinite(a) and a > 0):
            raise ValueError(f"BNSGammaOUParams: a must be > 0; got {a}")
        if not (np.isfinite(b) and b > 0):
            raise ValueError(f"BNSGammaOUParams: b must be > 0; got {b}")
        if rho_l >= b:
            raise ValueError(
                f"BNSGammaOUParams: rho_l must be < b for k_Z(rho_l) to be finite; "
                f"got rho_l={rho_l}, b={b}"
            )
        object.__setattr__(self, "name", "bns_gamma_ou")
        object.__setattr__(self, "V0", V0)
        object.__setattr__(self, "lambda_ou", lambda_ou)
        object.__setattr__(self, "rho_l", rho_l)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)


# ---------------------------------------------------------------------------
# Gamma log-Laplace transform
# ---------------------------------------------------------------------------


def _k_Z_gamma(c: np.ndarray, a: float, b: float) -> np.ndarray:
    """Log-Laplace of the Gamma subordinator Z_1: k_Z(c) = a*log(b/(b-c)).

    Valid for Re(b - c) > 0 (complex c is accepted via complex log).
    """
    c_c = np.asarray(c, dtype=np.complex128)
    return a * np.log(b / (b - c_c))


# ---------------------------------------------------------------------------
# BNS Gamma-OU characteristic function
# ---------------------------------------------------------------------------


def bns_gamma_ou_cf(u: np.ndarray, fwd: ForwardSpec, p: BNSGammaOUParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under the BNS Gamma-OU model.

    Implements the closed-form integral representation (Nicolato-Venardos 2003):

        log phi(u) = -(iu+u^2)/2 * V0*(1-exp(-lambda*T))/lambda
                     - iu*lambda*T*k_Z(rho_l)
                     + lambda * integral_0^T k_Z(g_u(tau)) dtau

        g_u(tau) = iu*rho_l - (iu+u^2)/(2*lambda)*(1-exp(-lambda*tau))
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T
    lam = p.lambda_ou
    V0, a, b, rho_l = p.V0, p.a, p.b, p.rho_l

    # Coefficient (iu + u^2) / 2
    coeff = (1j * u_c + u_c**2) / 2.0

    # Deterministic V0 contribution
    term_V0 = -coeff * V0 * (1.0 - np.exp(-lam * T)) / lam

    # Drift compensator
    k_rho = _k_Z_gamma(np.array([rho_l]), a, b)[0]  # real scalar
    term_drift = -1j * u_c * lam * T * float(np.real(k_rho))

    # Numerical integral: 200 points in [0, T]
    N = 200
    taus = np.linspace(0.0, T, N + 1)  # (N+1,)
    # g_u(tau): shape (M, N+1) where M = len(u_c)
    u_col = u_c[:, np.newaxis]  # (M, 1)
    taus_row = taus[np.newaxis, :]  # (1, N+1)

    g = 1j * rho_l * u_col - coeff[:, np.newaxis] * (1.0 - np.exp(-lam * taus_row)) / lam

    # k_Z(g): complex, shape (M, N+1)
    kZ_g = _k_Z_gamma(g, a, b)

    # Trapezoidal rule
    term_int = lam * _trapezoid(kZ_g, taus_row, axis=1)  # (M,)

    log_phi = term_V0 + term_drift + term_int
    return np.exp(log_phi)


# ---------------------------------------------------------------------------
# Cumulants
# ---------------------------------------------------------------------------


def bns_gamma_ou_cumulants(fwd: ForwardSpec, p: BNSGammaOUParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of X_T under BNS Gamma-OU via Cauchy integral."""
    from ..utils.cumulants import cumulants_from_cf

    c = cumulants_from_cf(lambda u: bns_gamma_ou_cf(u, fwd, p), order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
