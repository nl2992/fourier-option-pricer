"""Merton (1976) jump-diffusion model — native CF.

The Merton jump-diffusion (MJD) model adds compound-Poisson log-normal jumps
to a geometric Brownian motion (GBM):

    dS/S = (r - q + ω) dt + σ dW + (exp(Y_k) - 1) dN_t

where:
    N_t  = Poisson process with intensity λ (jumps per year),
    Y_k  ~ Normal(μ_j, σ_j²) = log-jump size (i.i.d.),
    ζ    = E[exp(Y) - 1] = exp(μ_j + σ_j²/2) - 1,
    ω    = -λ ζ         = drift correction (keeps E[S_T/F_0] = 1).

The log-forward return X_T = log(S_T/F_0) has characteristic function:

    φ(u) = exp(T · [i u (ω - σ²/2) - ½ σ² u² + λ (φ_Y(u) - 1)])

where φ_Y(u) = exp(i u μ_j - ½ σ_j² u²) is the CF of the log-jump size.

Relationship to Bates
---------------------
Bates (1996) = Heston (stochastic vol) + MJD jumps. Setting the Heston
diffusion to zero (ν → 0) recovers pure MJD. The jump block here is the
same formula used in :mod:`.bates`.

References
----------
* Merton, R. (1976), "Option pricing when underlying stock returns are
  discontinuous", *Journal of Financial Economics*, 3, 125–144.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press. (Chapter 12 for the CF and pricing formulae.)
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class MertonJDParams(ModelSpec):
    """Merton (1976) jump-diffusion parameters.

    Parameters
    ----------
    sigma : float
        GBM diffusion volatility (annualized). Must be ``>= 0``.
    lam : float
        Jump intensity — expected number of jumps per year. Must be ``>= 0``.
    muj : float
        Mean of the log-jump size (``μ_j``). Can be positive or negative;
        negative values model downside jump risk.
    sigj : float
        Standard deviation of the log-jump size (``σ_j``). Must be ``>= 0``.

    Note
    ----
    Setting ``lam = 0`` reduces the model to pure GBM (BSM).
    """

    sigma: float
    lam: float
    muj: float
    sigj: float

    def __init__(self, sigma: float, lam: float, muj: float, sigj: float):
        if not (np.isfinite(sigma) and sigma >= 0):
            raise ValueError(f"sigma must be finite and >= 0; got {sigma}")
        if not (np.isfinite(lam) and lam >= 0):
            raise ValueError(f"lam must be finite and >= 0; got {lam}")
        if not np.isfinite(muj):
            raise ValueError(f"muj must be finite; got {muj}")
        if not (np.isfinite(sigj) and sigj >= 0):
            raise ValueError(f"sigj must be finite and >= 0; got {sigj}")
        object.__setattr__(self, "name", "merton_jd")
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "lam", lam)
        object.__setattr__(self, "muj", muj)
        object.__setattr__(self, "sigj", sigj)


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------

def merton_jd_cf(u: np.ndarray, fwd: ForwardSpec, p: MertonJDParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under Merton JD.

    phi(u) = exp(T * [i*u*(omega - sigma^2/2) - 1/2*sigma^2*u^2 + lam*(phi_Y(u) - 1)])

    where:
        zeta  = exp(muj + 1/2*sigj^2) - 1
        omega = -lam * zeta
        phi_Y = exp(i*u*muj - 1/2*sigj^2*u^2)

    Parameters
    ----------
    u : array_like (real or complex)
        Frequency grid.
    fwd : ForwardSpec
    p : MertonJDParams

    Returns
    -------
    np.ndarray (complex128)
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T

    # Jump parameters
    zeta = np.expm1(p.muj + 0.5 * p.sigj ** 2)  # exp(muj + sigj^2/2) - 1
    omega = -p.lam * zeta

    # Log-normal jump size CF: phi_Y(u) = E[exp(iu*Y)]
    phi_Y = np.exp(1j * u_c * p.muj - 0.5 * p.sigj ** 2 * u_c ** 2)

    exponent = T * (
        1j * u_c * (omega - 0.5 * p.sigma ** 2)
        - 0.5 * p.sigma ** 2 * u_c ** 2
        + p.lam * (phi_Y - 1.0)
    )
    return np.exp(exponent)


# ---------------------------------------------------------------------------
# Cumulants (closed form)
# ---------------------------------------------------------------------------

def merton_jd_cumulants(fwd: ForwardSpec, p: MertonJDParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of X_T = log(S_T/F_0) under Merton JD.

    For a Lévy process with exponent ψ(u), cumulants follow from the
    log-cumulant generating function κ(s) = T * ψ(-is):

        c1  = T * (omega + lam * muj)
        c2  = T * (sigma^2 + lam * (muj^2 + sigj^2))
        c4  = T * lam * (muj^4 + 6*muj^2*sigj^2 + 3*sigj^4)

    where omega = -lam * zeta and zeta = exp(muj + sigj^2/2) - 1.
    """
    T = fwd.T
    lam, muj, sigj, sigma = p.lam, p.muj, p.sigj, p.sigma

    zeta = np.expm1(muj + 0.5 * sigj ** 2)
    omega = -lam * zeta

    c1 = T * (omega + lam * muj)
    c2 = T * (sigma ** 2 + lam * (muj ** 2 + sigj ** 2))
    c4 = T * lam * (muj ** 4 + 6 * muj ** 2 * sigj ** 2 + 3 * sigj ** 4)
    return float(c1), float(c2), float(c4)
