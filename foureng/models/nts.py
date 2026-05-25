"""Normal Tempered Stable (NTS) characteristic function.

The NTS Lévy process used here is the *tempered* stable variant of
Kim, Rachev & Rüschendorf (2008).  A drifted Brownian motion is
subordinated to a *tempered* alpha-stable process (not a pure stable),
which guarantees finite exponential moments and a real martingale
correction.

Lévy exponent
-------------
Per unit time, the log-CF (Lévy exponent) is:

    psi(u; lambda, theta, sigma, alpha)
        = -lambda * ((1 - i*theta*u/lambda + sigma^2*u^2/(2*lambda))^alpha - 1)

where the complex power ``z^alpha`` is taken via the principal-branch
logarithm (safe for all real ``u`` provided the existence condition holds).

Parameters
----------
lambda : float  Activity / tempering rate. ``> 0``.
theta  : float  Drift / skewness of the BM in the subordination.
sigma  : float  Diffusion scale of the BM. ``> 0``.
alpha  : float  Stability index. ``in (0, 1)``.

Existence condition
-------------------
    1 + theta/lambda + sigma^2/(2*lambda) > 0
    <==>  lambda + theta + sigma^2/2 > 0

This ensures psi(-i) is real (so the martingale correction is a real
drift) and the process has finite exponential moments.

Martingale correction (log-forward convention)
----------------------------------------------
    psi(-i) = -lambda * ((1 + theta/lambda + sigma^2/(2*lambda))^alpha - 1)

which is real when the existence condition is satisfied.  The risk-neutral
log-forward CF is:

    phi_T(u) = exp(T * (psi(u) - i*u * psi(-i)))

satisfying phi_T(0) = 1, phi_T(-i) = 1, and (for real u) the conjugate
symmetry phi_T(-u) = conj(phi_T(u)).

Special cases
-------------
* alpha → 0:   z^alpha - 1 ≈ alpha*log(z); NTS → VG-like regime.
* alpha = 1/2: similar to NIG structure (but different parameterisation).
* alpha → 1:   psi → -lambda*(z-1) = i*theta*u - sigma^2*u^2/2; reduces
               to BSM-like log-normal drift + diffusion.

References
----------
* Kim, Y. S., Rachev, S. T. & Rüschendorf, L. (2008), "Normal Tempered
  Stable Distribution and its Application to Asset Pricing", *Journal of
  Applied Mathematics and Decision Sciences*.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press, Chapter 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class NTSParams(ModelSpec):
    """Normal Tempered Stable parameters (Kim-Rachev-Rüschendorf 2008).

    Attributes
    ----------
    lam : float   Activity rate (tempering). ``> 0``.
    theta : float Skewness / BM drift.
    sigma : float BM diffusion scale. ``> 0``.
    alpha : float Stability index. ``in (0, 1)``.
    """

    lam: float
    theta: float
    sigma: float
    alpha: float

    def __init__(self, lam: float, theta: float, sigma: float, alpha: float):
        if not (np.isfinite(lam) and lam > 0):
            raise ValueError(f"NTSParams: lam must be > 0; got {lam}")
        if not np.isfinite(theta):
            raise ValueError(f"NTSParams: theta must be finite; got {theta}")
        if not (np.isfinite(sigma) and sigma > 0):
            raise ValueError(f"NTSParams: sigma must be > 0; got {sigma}")
        if not (np.isfinite(alpha) and 0.0 < alpha < 1.0):
            raise ValueError(f"NTSParams: alpha must be in (0, 1); got {alpha}")
        cond = lam + theta + 0.5 * sigma**2
        if cond <= 0.0:
            raise ValueError(
                f"NTSParams: existence condition lam + theta + sigma^2/2 must be > 0; "
                f"got {cond:.6g}"
            )
        object.__setattr__(self, "name", "nts")
        object.__setattr__(self, "lam", lam)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "alpha", alpha)


# ---------------------------------------------------------------------------
# NTS Lévy exponent
# ---------------------------------------------------------------------------


def _nts_levy_exponent(u: np.ndarray, lam: float, theta: float,
                        sigma: float, alpha: float) -> np.ndarray:
    """Per-unit-time Lévy exponent of the NTS process.

    psi(u) = -lam * ((1 - i*theta*u/lam + sigma^2*u^2/(2*lam))^alpha - 1)

    For real u the argument z = 1 - i*theta*u/lam + sigma^2*u^2/(2*lam)
    has positive real part (Re(z) = 1 + sigma^2*u^2/(2*lam) > 1), so the
    principal-branch power z^alpha is safe and conjugate-symmetric.
    """
    u_c = np.asarray(u, dtype=np.complex128)
    z = 1.0 - 1j * theta * u_c / lam + sigma**2 * u_c**2 / (2.0 * lam)
    return -lam * (np.exp(alpha * np.log(z)) - 1.0)


# ---------------------------------------------------------------------------
# NTS characteristic function
# ---------------------------------------------------------------------------


def nts_cf(u: np.ndarray, fwd: ForwardSpec, p: NTSParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under NTS (log-forward convention).

    phi_T(u) = exp(T * (psi(u) - i*u * psi(-i)))
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T

    psi_u = _nts_levy_exponent(u_c, p.lam, p.theta, p.sigma, p.alpha)
    psi_mi = _nts_levy_exponent(np.array([-1j]), p.lam, p.theta, p.sigma, p.alpha)[0]

    return np.exp(T * (psi_u - 1j * u_c * float(np.real(psi_mi))))


# ---------------------------------------------------------------------------
# Cumulants
# ---------------------------------------------------------------------------


def nts_cumulants(fwd: ForwardSpec, p: NTSParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of X_T under NTS via Cauchy integral."""
    from ..utils.cumulants import cumulants_from_cf

    c = cumulants_from_cf(lambda u: nts_cf(u, fwd, p), order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
