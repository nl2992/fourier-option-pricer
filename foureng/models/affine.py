"""Generic affine jump-diffusions (Duffie, Pan & Singleton 2000).

A state ``X`` in ``R^n`` whose first component is the log-price follows

    dX = (K0 + K1 X) dt + sigma(X) dW + dZ,
    sigma sigma^T (X) = H0 + sum_k H1[k] X_k,
    jump intensity l0 + l1 . X, jump-size transform theta(c) = E[e^{c . J}].

For such models ``E[e^{u . X_T}] = exp(alpha(T) + beta(T) . X_0)`` with the
complex Riccati system (Duffie-Pan-Singleton 2000, Prop. 1)

    beta'  = K1^T beta + 1/2 beta^T H1 beta + l1 (theta(beta) - 1),   beta(0) = u,
    alpha' = K0 . beta + 1/2 beta^T H0 beta + l0 (theta(beta) - 1),   alpha(0) = 0,

(``(beta^T H1 beta)_k = beta^T H1[k] beta``), solved here numerically by an
adaptive eighth-order Runge-Kutta scheme (DOP853), vectorised over all
frequencies at once. The CF returned is that of ``log(S_T / F_0)``: the
increment of the first component, normalised by ``E[e^{Y}]`` so that
``phi(-i) = 1``. That normalisation absorbs a *constant* log-price drift
(including the compensator of constant-intensity jumps, as in Bates); a
state-dependent compensator, e.g. ``-l1_k zeta x_k`` for an intensity
``l1 . x`` with ``zeta = theta(e_0) - 1``, must be written into ``K1[0]`` for
the specification to be risk-neutral.

Heston, Bates, double Heston, Merton and Kou jump-diffusions are special cases
(tests check them against the registry's closed forms), and any other affine
specification (e.g. stochastic jump intensity, several variance factors with
correlated jumps) prices without writing a new model. Use the COS or SINC
engines: they need a few hundred frequencies, whereas the contour engine's
quadrature reaches frequencies where each Riccati solve is very expensive.

Reference
---------
Duffie, D., Pan, J. & Singleton, K. (2000), Transform analysis and asset
pricing for affine jump-diffusions, *Econometrica* 68(6), 1343-1376.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .base import ForwardSpec, ModelSpec

_CUMULANT_CACHE: dict = {}


@dataclass(frozen=True)
class AffineParams(ModelSpec):
    """An affine jump-diffusion specification (first state = log-price).

    Parameters
    ----------
    x0 : array_like, shape (n,)
        Initial state; ``x0[0]`` (the log-price level) does not affect the CF.
    K0, K1 : array_like, shapes (n,), (n, n)
        Drift ``K0 + K1 x``.
    H0, H1 : array_like, shapes (n, n), (n, n, n)
        Covariance ``H0 + sum_k H1[k] x_k`` (each matrix symmetric).
    l0, l1 : float, array_like (n,)
        Jump intensity ``l0 + l1 . x`` (default: no jumps).
    jump_transform : callable, optional
        ``theta(c) = E[exp(c . J)]`` for complex ``c`` of shape ``(..., n)``,
        returning shape ``(...)``.
    rtol : float
        Relative tolerance of the Riccati integration.
    """

    x0: tuple
    K0: tuple
    K1: tuple
    H0: tuple
    H1: tuple
    l0: float
    l1: tuple
    jump_transform: Callable | None
    n: int
    rtol: float

    def __init__(self, x0, K0, K1, H0, H1, l0=0.0, l1=None, jump_transform=None, rtol=1e-11):
        x0a = np.asarray(x0, dtype=float)
        n = x0a.size
        shapes = {"K0": (n,), "K1": (n, n), "H0": (n, n), "H1": (n, n, n)}
        arrays = {"K0": K0, "K1": K1, "H0": H0, "H1": H1}
        stored = {}
        for key, val in arrays.items():
            arr = np.asarray(val, dtype=float)
            if arr.shape != shapes[key]:
                raise ValueError(
                    f"AffineParams: {key} must have shape {shapes[key]}; got {arr.shape}"
                )
            stored[key] = tuple(map(float, arr.ravel()))
        l1a = np.zeros(n) if l1 is None else np.asarray(l1, dtype=float)
        if l1a.shape != (n,):
            raise ValueError(f"AffineParams: l1 must have shape ({n},)")
        if (l0 != 0.0 or np.any(l1a != 0.0)) and jump_transform is None:
            raise ValueError("AffineParams: a jump intensity needs a jump_transform")
        object.__setattr__(self, "name", "affine")
        object.__setattr__(self, "x0", tuple(map(float, x0a)))
        for key, val in stored.items():
            object.__setattr__(self, key, val)
        object.__setattr__(self, "l0", float(l0))
        object.__setattr__(self, "l1", tuple(map(float, l1a)))
        object.__setattr__(self, "jump_transform", jump_transform)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "rtol", float(rtol))

    def arrays(self):
        n = self.n
        return (
            np.array(self.x0),
            np.array(self.K0),
            np.array(self.K1).reshape(n, n),
            np.array(self.H0).reshape(n, n),
            np.array(self.H1).reshape(n, n, n),
            self.l0,
            np.array(self.l1),
        )


_CHUNK = 64  # frequencies per ODE solve: one exploding row must not stall the rest
_BLOWUP = 1e60


def _transform(p: AffineParams, u0: np.ndarray, T: float) -> np.ndarray:
    """``log E[exp(u0 X_T[0])]`` for complex ``u0`` (shape (m,)), via the Riccati ODEs.

    Frequencies are integrated in chunks, and a chunk whose solution passes
    ``1e60`` (a moment explosion, e.g. a real exponent beyond the model's
    moment domain) is stopped and returned as NaN rather than integrated with
    ever smaller steps.
    """
    out = np.empty(u0.size, dtype=complex)
    for lo in range(0, u0.size, _CHUNK):
        out[lo : lo + _CHUNK] = _transform_chunk(p, u0[lo : lo + _CHUNK], T)
    return out


def _transform_chunk(p: AffineParams, u0: np.ndarray, T: float) -> np.ndarray:
    x0, K0, K1, H0, H1, l0, l1 = p.arrays()
    n, m = p.n, u0.size
    theta = p.jump_transform

    def rhs(_t, y):
        z = y[: m * (n + 1)] + 1j * y[m * (n + 1) :]
        beta = z[: m * n].reshape(m, n)
        quad = 0.5 * np.einsum("mi,kij,mj->mk", beta, H1, beta)
        d_beta = beta @ K1 + quad  # (K1^T beta)_k = sum_i K1[i, k] beta_i
        d_alpha = beta @ K0 + 0.5 * np.einsum("mi,ij,mj->m", beta, H0, beta)
        if theta is not None:
            jt = np.asarray(theta(beta), dtype=np.complex128) - 1.0
            d_beta = d_beta + jt[:, None] * l1[None, :]
            d_alpha = d_alpha + l0 * jt
        dz = np.concatenate([d_beta.ravel(), d_alpha])
        return np.concatenate([dz.real, dz.imag])

    beta0 = np.zeros((m, n), dtype=complex)
    beta0[:, 0] = u0
    z0 = np.concatenate([beta0.ravel(), np.zeros(m, dtype=complex)])

    def blowup(_t, y):
        return _BLOWUP - np.max(np.abs(y))

    blowup.terminal = True  # type: ignore[attr-defined]
    with np.errstate(over="ignore", invalid="ignore"):
        sol = solve_ivp(
            rhs,
            (0.0, T),
            np.concatenate([z0.real, z0.imag]),
            method="DOP853",
            rtol=p.rtol,
            atol=1e-14,
            events=blowup,
        )
    if sol.status != 0 or not np.all(np.isfinite(sol.y[:, -1])):
        return np.full(m, np.nan + 0j)
    yT = sol.y[:, -1]
    zT = yT[: m * (n + 1)] + 1j * yT[m * (n + 1) :]
    beta_T, alpha_T = zT[: m * n].reshape(m, n), zT[m * n :]
    return alpha_T + beta_T @ x0 - u0 * x0[0]  # increment of the log-price


def affine_cf(u: np.ndarray, fwd: ForwardSpec, p: AffineParams) -> np.ndarray:
    """CF of ``X_T = log(S_T / F_0)`` for an affine jump-diffusion (see module docstring)."""
    w = np.asarray(u, dtype=complex)
    flat = w.ravel()
    log_m1 = _transform(p, np.array([1.0 + 0j]), fwd.T)[0]
    if not np.isfinite(log_m1):
        raise ValueError("affine: E[S_T] is not finite for this specification")
    # Walk the frequencies by increasing |Re u|. Once a whole chunk is below
    # 1e-16, later ones with no larger |Im u| (no stronger exponential tilt) are
    # taken as 0: the ODE at very high frequencies is expensive and irrelevant.
    order = np.argsort(np.abs(flat.real), kind="stable")
    out = np.zeros(flat.shape, dtype=complex)
    tilt_bound = -1.0
    for lo in range(0, order.size, _CHUNK):
        idx = order[lo : lo + _CHUNK]
        z = flat[idx]
        if np.all(np.abs(z.imag) <= tilt_bound):
            continue
        vals = np.exp(_transform(p, 1j * z, fwd.T) - 1j * z * np.real(log_m1))
        out[idx] = vals
        if np.all(np.abs(vals) < 1e-16) and np.min(np.abs(z.real)) > 1.0:
            tilt_bound = max(tilt_bound, float(np.max(np.abs(z.imag))))
    return out.reshape(w.shape)


def affine_cumulants(fwd: ForwardSpec, p: AffineParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` by Cauchy integration of the CF."""
    cached = _CUMULANT_CACHE.get((p, fwd))
    if cached is not None:
        return cached
    from ..utils.cumulants import cumulants_from_cf

    c = cumulants_from_cf(lambda z: affine_cf(z, fwd, p), order=4, radius=0.25, M=64)
    out = (float(c[0]), float(c[1]), float(c[3]))
    _CUMULANT_CACHE[(p, fwd)] = out
    return out
