"""Markov regime-switching Black-Scholes model.

The economy switches between ``n`` volatility regimes according to a
continuous-time Markov chain with generator ``Q`` (rows sum to zero,
off-diagonal entries are switching intensities). Conditional on the chain,
the asset is a BSM diffusion with the active regime's volatility and the
per-regime martingale drift ``-sigma_j^2 / 2``, so ``exp(X_t)`` is a
martingale unconditionally and ``phi(-i) = 1`` holds by construction.

Characteristic function (Buffington & Elliott 2002; the matrix-exponential
form used by Kirkby's PROJ regime-switching pricers):

    phi(u) = pi_0' expm( T * (Q + diag(psi_1(u), ..., psi_n(u))) ) 1

where ``psi_j(u) = -0.5 * sigma_j^2 * (u^2 + i u)`` is regime ``j``'s Levy
exponent (including its martingale correction) and ``pi_0`` is the initial
regime distribution.

Cumulants have no convenient closed form (occupation-time moments mix the
regimes), so ``regime_switching_cumulants`` differentiates the cumulant
generating function ``K(s) = log E[e^{s X_T}]`` numerically; the values feed
the COS/PROJ truncation heuristics, which need only a few significant digits.

References
----------
Buffington, J. & Elliott, R.J. (2002). American options with regime
switching. *International Journal of Theoretical and Applied Finance*,
5(5), 497-514.

Kirkby, J.L. & Nguyen, D. (2020). Efficient Asian option pricing under
regime switching jump diffusions and stochastic volatility models.
*Annals of Finance*, 16, 307-351.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class RegimeSwitchingBsmParams(ModelSpec):
    """Markov-modulated BSM parameters.

    sigmas        : per-regime diffusion volatilities (all > 0)
    generator     : Markov-chain generator Q as a nested tuple; rows sum to
                    zero and off-diagonal entries are >= 0
    initial_probs : initial regime distribution pi_0 (sums to 1)
    """

    sigmas: tuple[float, ...]
    generator: tuple[tuple[float, ...], ...]
    initial_probs: tuple[float, ...]

    def __init__(
        self,
        sigmas,
        generator,
        initial_probs,
    ):
        object.__setattr__(self, "name", "regime_switching")
        object.__setattr__(self, "sigmas", tuple(float(s) for s in sigmas))
        object.__setattr__(
            self, "generator", tuple(tuple(float(x) for x in row) for row in generator)
        )
        object.__setattr__(self, "initial_probs", tuple(float(p) for p in initial_probs))
        self.__post_init__()

    def __post_init__(self) -> None:
        n = len(self.sigmas)
        if n == 0:
            raise ValueError("RegimeSwitchingBsmParams: need at least one regime")
        if any(s <= 0.0 or not np.isfinite(s) for s in self.sigmas):
            raise ValueError(f"RegimeSwitchingBsmParams: all sigmas must be > 0; got {self.sigmas}")
        Q = np.asarray(self.generator, dtype=np.float64)
        if Q.shape != (n, n):
            raise ValueError(
                f"RegimeSwitchingBsmParams: generator must be {n}x{n} to match "
                f"{n} regimes; got shape {Q.shape}"
            )
        off_diag = Q - np.diag(np.diag(Q))
        if np.any(off_diag < -1e-12):
            raise ValueError(
                "RegimeSwitchingBsmParams: off-diagonal generator entries must be >= 0"
            )
        if np.any(np.abs(Q.sum(axis=1)) > 1e-8):
            raise ValueError("RegimeSwitchingBsmParams: generator rows must sum to zero")
        p0 = np.asarray(self.initial_probs, dtype=np.float64)
        if p0.shape != (n,):
            raise ValueError(
                f"RegimeSwitchingBsmParams: initial_probs must have length {n}; got {p0.shape}"
            )
        if np.any(p0 < -1e-12) or abs(p0.sum() - 1.0) > 1e-8:
            raise ValueError(
                "RegimeSwitchingBsmParams: initial_probs must be non-negative and sum to 1"
            )

    @property
    def n_regimes(self) -> int:
        return len(self.sigmas)


def _rs_phi_scalar(u: complex, T: float, p: RegimeSwitchingBsmParams) -> complex:
    """phi(u) for one (possibly complex) frequency via the matrix exponential."""
    sig2 = np.asarray(p.sigmas, dtype=np.float64) ** 2
    psi = -0.5 * sig2 * (u * u + 1j * u)
    Q = np.asarray(p.generator, dtype=np.complex128)
    M = expm(T * (Q + np.diag(psi)))
    p0 = np.asarray(p.initial_probs, dtype=np.complex128)
    return complex(p0 @ M @ np.ones(len(p.sigmas), dtype=np.complex128))


def regime_switching_cf(u: np.ndarray, fwd: ForwardSpec, p: RegimeSwitchingBsmParams) -> np.ndarray:
    """CF of X_T = log(S_T/F0) under the Markov regime-switching BSM model.

    Evaluates one small (n x n) matrix exponential per frequency. Accepts
    complex arguments (needed by the Lewis and Hilbert engines).
    """
    u_arr = np.atleast_1d(np.asarray(u, dtype=np.complex128))
    out = np.array([_rs_phi_scalar(complex(ui), fwd.T, p) for ui in u_arr])
    if np.isscalar(u) or np.ndim(u) == 0:
        return out.reshape(())
    return out


def regime_switching_cumulants(
    fwd: ForwardSpec, p: RegimeSwitchingBsmParams
) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of log(S_T/F0) via numeric CGF differentiation.

    K(s) = log(pi_0' expm(T (Q + diag(0.5 sigma_j^2 (s^2 - s)))) 1) is smooth
    in ``s``; central differences with step 0.1 give the handful of digits the
    COS/PROJ interval heuristics require. Single-regime output matches the
    analytic BSM cumulants to that accuracy.
    """
    T = fwd.T
    sig2 = np.asarray(p.sigmas, dtype=np.float64) ** 2
    Q = np.asarray(p.generator, dtype=np.float64)
    p0 = np.asarray(p.initial_probs, dtype=np.float64)
    ones = np.ones(len(p.sigmas), dtype=np.float64)

    def K(s: float) -> float:
        kappa = 0.5 * sig2 * (s * s - s)
        return float(np.log(p0 @ expm(T * (Q + np.diag(kappa))) @ ones))

    # K(0) = 0 exactly; 4th-order central stencils for K' and K'', 2nd-order
    # for K''''.
    h = 0.1
    k_m2, k_m1, k_p1, k_p2 = K(-2 * h), K(-h), K(h), K(2 * h)
    c1 = (-k_p2 + 8.0 * k_p1 - 8.0 * k_m1 + k_m2) / (12.0 * h)
    c2 = (-k_p2 + 16.0 * k_p1 + 16.0 * k_m1 - k_m2) / (12.0 * h * h)
    c4 = (k_p2 - 4.0 * k_p1 - 4.0 * k_m1 + k_m2) / h**4
    return float(c1), float(c2), float(max(c4, 0.0))
