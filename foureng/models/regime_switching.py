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
    """Markov-modulated BSM parameters, optionally with per-regime Merton jumps.

    sigmas          : per-regime diffusion volatilities (all > 0)
    generator       : Markov-chain generator Q as a nested tuple; rows sum to
                      zero and off-diagonal entries are >= 0
    initial_probs   : initial regime distribution pi_0 (sums to 1)
    jump_intensities: optional per-regime Poisson intensities lambda_j >= 0
                      (default: no jumps). When set, each regime carries a
                      Merton compound-Poisson block with log-normal jump sizes
                      N(jump_means[j], jump_stds[j]^2), compensated per regime
                      so the martingale property is preserved (the
                      regime-switching jump-diffusion of Kirkby's RS pricers).
    jump_means      : per-regime mean log-jump sizes (required with intensities)
    jump_stds       : per-regime log-jump standard deviations (>= 0)
    """

    sigmas: tuple[float, ...]
    generator: tuple[tuple[float, ...], ...]
    initial_probs: tuple[float, ...]
    jump_intensities: tuple[float, ...]
    jump_means: tuple[float, ...]
    jump_stds: tuple[float, ...]

    def __init__(
        self,
        sigmas,
        generator,
        initial_probs,
        jump_intensities=None,
        jump_means=None,
        jump_stds=None,
    ):
        object.__setattr__(self, "name", "regime_switching")
        object.__setattr__(self, "sigmas", tuple(float(s) for s in sigmas))
        object.__setattr__(
            self, "generator", tuple(tuple(float(x) for x in row) for row in generator)
        )
        object.__setattr__(self, "initial_probs", tuple(float(p) for p in initial_probs))
        n = len(self.sigmas)
        if jump_intensities is None:
            jump_intensities = (0.0,) * n
        if jump_means is None:
            jump_means = (0.0,) * n
        if jump_stds is None:
            jump_stds = (0.0,) * n
        object.__setattr__(self, "jump_intensities", tuple(float(x) for x in jump_intensities))
        object.__setattr__(self, "jump_means", tuple(float(x) for x in jump_means))
        object.__setattr__(self, "jump_stds", tuple(float(x) for x in jump_stds))
        self.__post_init__()

    def __post_init__(self) -> None:
        n = len(self.sigmas)
        if n == 0:
            raise ValueError("RegimeSwitchingBsmParams: need at least one regime")
        if any(s <= 0.0 or not np.isfinite(s) for s in self.sigmas):
            raise ValueError(f"RegimeSwitchingBsmParams: all sigmas must be > 0; got {self.sigmas}")
        for field_name in ("jump_intensities", "jump_means", "jump_stds"):
            vals = getattr(self, field_name)
            if len(vals) != n:
                raise ValueError(
                    f"RegimeSwitchingBsmParams: {field_name} must have length {n}; got {len(vals)}"
                )
            if not all(np.isfinite(v) for v in vals):
                raise ValueError(f"RegimeSwitchingBsmParams: {field_name} must be finite")
        if any(lam < 0.0 for lam in self.jump_intensities):
            raise ValueError("RegimeSwitchingBsmParams: jump_intensities must be >= 0")
        if any(sj < 0.0 for sj in self.jump_stds):
            raise ValueError("RegimeSwitchingBsmParams: jump_stds must be >= 0")
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


def _rs_psi(u: complex, p: RegimeSwitchingBsmParams) -> np.ndarray:
    """Per-regime Levy exponents psi_j(u), martingale-compensated regime-wise.

    Diffusion block ``-0.5 sigma_j^2 (u^2 + iu)`` plus, when jumps are active,
    the compensated Merton block ``lambda_j (phi_Y(u) - 1 - iu zeta_j)`` with
    ``zeta_j = exp(mu_j + s_j^2/2) - 1``. Each block vanishes at ``u = -i``, so
    phi(-i) = 1 regardless of the regime path.
    """
    sig2 = np.asarray(p.sigmas, dtype=np.float64) ** 2
    psi = -0.5 * sig2 * (u * u + 1j * u)
    lam = np.asarray(p.jump_intensities, dtype=np.float64)
    if np.any(lam > 0.0):
        muj = np.asarray(p.jump_means, dtype=np.float64)
        sj = np.asarray(p.jump_stds, dtype=np.float64)
        zeta = np.expm1(muj + 0.5 * sj * sj)
        phi_y = np.exp(1j * u * muj - 0.5 * sj * sj * u * u)
        psi = psi + lam * (phi_y - 1.0 - 1j * u * zeta)
    return psi


def _rs_phi_scalar(u: complex, T: float, p: RegimeSwitchingBsmParams) -> complex:
    """phi(u) for one (possibly complex) frequency via the matrix exponential."""
    psi = _rs_psi(u, p)
    # Underflow guard: a generator has zero infinity-log-norm, so
    # |phi(u)| <= exp(T * max_j Re(psi_j)). Below the double-precision floor
    # return 0 instead of asking expm for a matrix it may turn into NaNs
    # (older SciPy releases overflow internally on such extreme diagonals).
    if T * float(np.max(psi.real)) < -700.0:
        return 0.0j
    Q = np.asarray(p.generator, dtype=np.complex128)
    M = expm(T * (Q + np.diag(psi)))
    p0 = np.asarray(p.initial_probs, dtype=np.complex128)
    val = complex(p0 @ M @ np.ones(len(p.sigmas), dtype=np.complex128))
    if not (np.isfinite(val.real) and np.isfinite(val.imag)):
        return 0.0j
    return val


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
    Q = np.asarray(p.generator, dtype=np.float64)
    p0 = np.asarray(p.initial_probs, dtype=np.float64)
    ones = np.ones(len(p.sigmas), dtype=np.float64)

    def K(s: float) -> float:
        # kappa_j(s) = psi_j(-is): real for real s, jump blocks included.
        kappa = np.real(_rs_psi(complex(0.0, -s), p))
        return float(np.log(p0 @ expm(T * (Q + np.diag(kappa))) @ ones))

    # K(0) = 0 exactly; 4th-order central stencils for K' and K'', 2nd-order
    # for K''''.
    h = 0.1
    k_m2, k_m1, k_p1, k_p2 = K(-2 * h), K(-h), K(h), K(2 * h)
    c1 = (-k_p2 + 8.0 * k_p1 - 8.0 * k_m1 + k_m2) / (12.0 * h)
    c2 = (-k_p2 + 16.0 * k_p1 + 16.0 * k_m1 - k_m2) / (12.0 * h * h)
    c4 = (k_p2 - 4.0 * k_p1 - 4.0 * k_m1 + k_m2) / h**4
    return float(c1), float(c2), float(max(c4, 0.0))
