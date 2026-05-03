"""Bilateral Gamma (BG) model (Küchler & Tappe 2008) — native CF.

The Bilateral Gamma process is the difference of two independent Gamma
processes:

    X_t = G⁺(t; α⁺, λ⁺) − G⁻(t; α⁻, λ⁻)

where G⁺ and G⁻ are independent Gamma processes with shape rate (α, λ).

A Gamma process G(t; α, λ) has increments G(t) − G(s) ~ Gamma(α(t-s), λ)
with mean α/λ per unit time and variance α/λ² per unit time.

The CF of X_T = log(S_T/F_0) under the BG model is:

    φ(u) = exp(T · [i·u·ω + α⁺·log(λ⁺/(λ⁺ − i·u))
                           + α⁻·log(λ⁻/(λ⁻ + i·u))])

where the drift correction ω ensures E[S_T/F_0] = 1:

    ζ⁺  = α⁺·log(λ⁺/(λ⁺ − 1))   (contribution from the upward process)
    ζ⁻  = α⁻·log(λ⁻/(λ⁻ + 1))   (contribution from the downward process)
    ω   = −(ζ⁺ + ζ⁻)

Equivalently, the Lévy exponent of X_T is:

    ψ(u) = i·u·ω + α⁺·log(λ⁺/(λ⁺ − i·u)) + α⁻·log(λ⁻/(λ⁻ + i·u))

so φ(u) = exp(T · ψ(u)).

Relationship to Variance Gamma
-------------------------------
Setting α⁺ = α⁻ = α/ν and λ⁺ = 1/(θ/σ² + sqrt(1/(σ²ν) + θ²/σ⁴)/...
the BG model reduces to the Variance Gamma model.  The BG parameterisation
is more direct (separate up/down jump intensities and rates).

Cumulants (closed form)
-----------------------
    c1 = T · (ω + α⁺/λ⁺ − α⁻/λ⁻)
    c2 = T · (α⁺/λ⁺² + α⁻/λ⁻²)
    c4 = T · 2 · (α⁺/λ⁺⁴ + α⁻/λ⁻⁴)  [unnormalized 4th cumulant]

References
----------
* Küchler, U. & Tappe, S. (2008), "Bilateral Gamma distributions and
  processes in financial mathematics", *Stochastic Processes and their
  Applications*, 118(2), 261–283.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press, Chapter 4 (Generalized hyperbolic and related).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class BilateralGammaParams(ModelSpec):
    """Bilateral Gamma process parameters.

    Parameters
    ----------
    alpha_p : float
        Shape rate of the upward Gamma process. Must be ``> 0``.
    lambda_p : float
        Rate parameter of the upward Gamma process. Must be ``> 1``
        (required for the martingale condition to hold, i.e., for E[S_T] = F_0
        to be finite at u=-i).
    alpha_m : float
        Shape rate of the downward Gamma process. Must be ``> 0``.
    lambda_m : float
        Rate parameter of the downward Gamma process. Must be ``> 1``
        (required for the martingale condition).
    """

    alpha_p: float
    lambda_p: float
    alpha_m: float
    lambda_m: float

    def __init__(self, alpha_p: float, lambda_p: float,
                 alpha_m: float, lambda_m: float):
        import numpy as _np
        if not (_np.isfinite(alpha_p) and alpha_p > 0):
            raise ValueError(f"alpha_p must be finite and > 0; got {alpha_p}")
        if not (_np.isfinite(lambda_p) and lambda_p > 1):
            raise ValueError(f"lambda_p must be finite and > 1 (martingale condition); got {lambda_p}")
        if not (_np.isfinite(alpha_m) and alpha_m > 0):
            raise ValueError(f"alpha_m must be finite and > 0; got {alpha_m}")
        if not (_np.isfinite(lambda_m) and lambda_m > 1):
            raise ValueError(f"lambda_m must be finite and > 1 (martingale condition); got {lambda_m}")
        object.__setattr__(self, "name", "bilateral_gamma")
        object.__setattr__(self, "alpha_p", alpha_p)
        object.__setattr__(self, "lambda_p", lambda_p)
        object.__setattr__(self, "alpha_m", alpha_m)
        object.__setattr__(self, "lambda_m", lambda_m)


# ---------------------------------------------------------------------------
# Drift correction
# ---------------------------------------------------------------------------

def _bg_omega(p: BilateralGammaParams) -> float:
    """Drift correction ω so E[exp(X_T)] = 1.

    ω = −(α⁺·log(λ⁺/(λ⁺−1)) + α⁻·log(λ⁻/(λ⁻+1)))

    Requires λ⁺ > 1 and λ⁻ > 1.
    """
    zeta_p = p.alpha_p * np.log(p.lambda_p / (p.lambda_p - 1.0))
    zeta_m = p.alpha_m * np.log(p.lambda_m / (p.lambda_m + 1.0))
    return -(zeta_p + zeta_m)


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------

def bilateral_gamma_cf(
    u: np.ndarray, fwd: ForwardSpec, p: BilateralGammaParams
) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under the Bilateral Gamma model.

    φ(u) = exp(T · [i·u·ω + α⁺·log(λ⁺/(λ⁺ − i·u))
                           + α⁻·log(λ⁻/(λ⁻ + i·u))])

    Parameters
    ----------
    u : array_like (real or complex)
        Frequency grid.
    fwd : ForwardSpec
    p : BilateralGammaParams

    Returns
    -------
    np.ndarray (complex128)
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T

    omega = _bg_omega(p)

    # Gamma process CFs:  α·log(λ/(λ − iu))  and  α·log(λ/(λ + iu))
    cf_p = p.alpha_p * np.log(p.lambda_p / (p.lambda_p - 1j * u_c))
    cf_m = p.alpha_m * np.log(p.lambda_m / (p.lambda_m + 1j * u_c))

    exponent = T * (1j * u_c * omega + cf_p + cf_m)
    return np.exp(exponent)


# ---------------------------------------------------------------------------
# Cumulants (closed form)
# ---------------------------------------------------------------------------

def bilateral_gamma_cumulants(
    fwd: ForwardSpec, p: BilateralGammaParams
) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of X_T under the BG model.

    c1 = T·(ω + α⁺/λ⁺ − α⁻/λ⁻)
    c2 = T·(α⁺/λ⁺² + α⁻/λ⁻²)
    c4 = T·2·(α⁺/λ⁺⁴ + α⁻/λ⁻⁴)
    """
    T = fwd.T
    a_p, l_p = p.alpha_p, p.lambda_p
    a_m, l_m = p.alpha_m, p.lambda_m

    omega = _bg_omega(p)

    c1 = T * (omega + a_p / l_p - a_m / l_m)
    c2 = T * (a_p / l_p ** 2 + a_m / l_m ** 2)
    c4 = T * 2.0 * (a_p / l_p ** 4 + a_m / l_m ** 4)
    return float(c1), float(c2), float(c4)
