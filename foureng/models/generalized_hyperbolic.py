"""Generalized Hyperbolic (GH) model (Barndorff-Nielsen 1977)  -  native CF.

The Generalized Hyperbolic distribution is a normal variance-mean mixture
where the mixing distribution is a Generalized Inverse Gaussian (GIG).
The GH family includes several well-known special cases:

    λ = −1/2     → Normal Inverse Gaussian (NIG)
    λ → 0        → Variance Gamma (limit, GH approaches VG)
    λ = 1        → Hyperbolic distribution

CF of a single GH increment X ~ GH(λ, α, β, δ, μ) is:

    φ_X(u) = exp(i·u·μ) · (α² − β²)^{λ/2} / (α² − (β + i·u)²)^{λ/2}
             · K_λ(δ · sqrt(α² − (β + i·u)²)) / K_λ(δ · sqrt(α² − β²))

where K_λ is the modified Bessel function of the second kind.

For the log-forward return X_T = log(S_T/F_0):
    φ(u) = exp(T · ψ(u))

where ψ(u) is the Lévy exponent of the GH Lévy process:

    ψ(u) = i·u·ω + λ·[log(α² − β²) − log(α² − (β + i·u)²)] / 2
            + log(K_λ(δ · sqrt(α² − (β + i·u)²)) / K_λ(δ · sqrt(α² − β²)))

The GH *distribution* is not infinitely divisible in general, but the
GH *Lévy process* (defined by raising the one-period CF to the power T)
is well-posed and extensively used in finance.  Working with the Lévy
exponent directly:

    log φ(u; T) = T · [i·u·ω
                        + λ · log( (α²−β²) / (α²−(β+i·u)²) ) / 2
                        + log K_λ(δ·sqrt(α²−(β+i·u)²)) − log K_λ(δ·sqrt(α²−β²))]

The drift correction ω is:

    ω = −[λ · log( (α²−β²) / (α²−(β+i·(−i))²) ) / 2
           + log K_λ(δ·sqrt(α²−(β+i·(−i))²)) − log K_λ(δ·sqrt(α²−β²))]

    = −[λ · log( (α²−β²) / (α²−(β+1)²) ) / 2
           + log K_λ(δ·sqrt(α²−(β+1)²)) − log K_λ(δ·sqrt(α²−β²))]

Parameter constraints:
    α > |β|      (shape constraint for the GIG mixing)
    δ > 0        (scale parameter)
    −∞ < λ < ∞  (shape index)
    −∞ < μ < ∞  (location / drift, set to 0 for zero-drift convention)
    |β| < α − 1  (required for martingale condition: need α² − (β+1)² > 0)

References
----------
* Barndorff-Nielsen, O. E. (1977), "Exponentially decreasing distributions
  for the logarithm of particle size", *Proceedings of the Royal Society
  of London*, 353(1674), 401–419.
* Eberlein, E. & Keller, U. (1995), "Hyperbolic distributions in finance",
  *Bernoulli*, 1(3), 281–299.
* Cont, R. & Tankov, P. (2004), *Financial Modelling with Jump Processes*,
  CRC Press, Chapter 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import kv  # modified Bessel K function

from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class GHParams(ModelSpec):
    """Generalized Hyperbolic (GH) Lévy model parameters.

    Parameters
    ----------
    lam : float
        Shape index. Special cases: ``lam=-0.5`` gives NIG, ``lam=1``
        gives the hyperbolic distribution. No constraint on sign.
    alpha : float
        Tail heaviness / steepness. Must satisfy ``alpha > |beta|`` and
        ``alpha > |beta| + 1`` for the martingale condition.
    beta : float
        Asymmetry / skewness. Must satisfy ``|beta| < alpha``.
    delta : float
        Scale parameter. Must be ``>= 0``. When ``delta=0`` and ``lam > 0``
        the model reduces to Variance Gamma.
    """

    lam: float
    alpha: float
    beta: float
    delta: float

    def __init__(self, lam: float, alpha: float, beta: float, delta: float):
        import numpy as _np

        if not _np.isfinite(lam):
            raise ValueError(f"lam must be finite; got {lam}")
        if not (_np.isfinite(alpha) and alpha > 0):
            raise ValueError(f"alpha must be finite and > 0; got {alpha}")
        if not _np.isfinite(beta):
            raise ValueError(f"beta must be finite; got {beta}")
        if not (abs(beta) < alpha):
            raise ValueError(f"alpha must be > |beta|; got alpha={alpha}, beta={beta}")
        if not (_np.isfinite(delta) and delta > 0):
            raise ValueError(f"delta must be finite and > 0; got {delta}")
        if not (alpha - abs(beta) > 1):
            raise ValueError(
                f"martingale condition requires alpha - |beta| > 1; "
                f"got alpha={alpha}, |beta|={abs(beta)}, diff={alpha - abs(beta):.4f}"
            )
        object.__setattr__(self, "name", "generalized_hyperbolic")
        object.__setattr__(self, "lam", lam)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "delta", delta)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_kv_ratio(lam: float, z1: np.ndarray, z0: float) -> np.ndarray:
    """log K_λ(z1) − log K_λ(z0), element-wise over z1.

    Uses scipy.special.kv which handles complex arguments.  We take the
    branch of log such that continuity is maintained near the real axis.
    """
    k0 = kv(lam, z0)
    k1 = kv(lam, z1)
    return np.log(k1) - np.log(k0)


def _levy_exponent(u_c: np.ndarray, p: GHParams, z0: float) -> np.ndarray:
    """Lévy exponent ψ(u) without the iu*ω drift term.

    ψ_raw(u) = λ/2 · log((α²−β²)/(α²−(β+iu)²))
               + log K_λ(δ·sqrt(α²−(β+iu)²)) − log K_λ(δ·sqrt(α²−β²))
    """
    alpha2_m_beta2 = p.alpha**2 - p.beta**2
    beta_iu = p.beta + 1j * u_c
    alpha2_m_betaiu2 = p.alpha**2 - beta_iu**2

    # sqrt with principal branch (positive real part)
    sqrt_arg = np.sqrt(alpha2_m_betaiu2)

    log_ratio = 0.5 * p.lam * (np.log(alpha2_m_beta2) - np.log(alpha2_m_betaiu2))
    log_k_ratio = _log_kv_ratio(p.lam, p.delta * sqrt_arg, z0)

    return log_ratio + log_k_ratio


def _gh_omega(p: GHParams) -> complex:
    """Drift correction ω so E[exp(X_T)] = 1.

    Evaluated by plugging u = -i into the raw exponent:
    ω = −ψ_raw(−i)

    Requires α² − (β+1)² > 0 and α² − (β−1)² > 0 for finiteness.
    """
    z0 = p.delta * np.sqrt(p.alpha**2 - p.beta**2)
    psi_raw_neg_i = _levy_exponent(np.array([-1j]), p, float(z0))
    return -float(np.real(psi_raw_neg_i[0]))


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------


def gh_cf(u: np.ndarray, fwd: ForwardSpec, p: GHParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under the GH Lévy model.

    φ(u) = exp(T · [i·u·ω + ψ_raw(u)])

    Parameters
    ----------
    u : array_like (real or complex)
        Frequency grid.
    fwd : ForwardSpec
    p : GHParams

    Returns
    -------
    np.ndarray (complex128)
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T

    z0 = float(p.delta * np.sqrt(p.alpha**2 - p.beta**2))
    omega = _gh_omega(p)

    psi_raw = _levy_exponent(u_c, p, z0)
    exponent = T * (1j * u_c * omega + psi_raw)
    return np.exp(exponent)


# ---------------------------------------------------------------------------
# Cumulants  -  numerical Cauchy integral
# ---------------------------------------------------------------------------


def gh_cumulants(fwd: ForwardSpec, p: GHParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of X_T under the GH Lévy model.

    Uses numerical Cauchy-circle integration (GH has closed-form cumulants
    but they involve Bessel function ratios  -  numerical is simpler and
    equally accurate).
    """
    from ..utils.cumulants import cumulants_from_cf

    def _phi(u):
        return gh_cf(u, fwd, p)

    c = cumulants_from_cf(_phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
