"""Lifted Heston model (Abi Jaber 2019): a Markovian proxy for rough Heston.

The variance is a weighted sum of ``n`` factors sharing one Brownian motion,

    V_t = g0(t) + sum_i c_i U^i_t,
    dU^i = (-x_i U^i - kappa V) dt + nu sqrt(V) dW,   U^i_0 = 0,
    g0(t) = v0 + kappa theta sum_i c_i (1 - e^{-x_i t}) / x_i,

i.e. a Volterra Heston model whose kernel ``K(t) = t^{alpha - 1}/Gamma(alpha)``
(``alpha = H + 1/2``) is replaced by ``sum_i c_i e^{-x_i t}``. Abi Jaber's
geometric parametrisation (``r_n > 1``, ``n`` factors),

    c_i = (r_n^{1-alpha} - 1) r_n^{(alpha-1)(1+n/2)} / (Gamma(alpha) Gamma(2-alpha))
          * r_n^{(1-alpha) i},
    x_i = (1-alpha)(r_n^{2-alpha} - 1) / ((2-alpha)(r_n^{1-alpha} - 1)) * r_n^{i-1-n/2},

reproduces rough-Heston smiles with ``n = 20``, ``r_20 = 2.5`` while staying
Markovian in ``n`` dimensions. The model is affine: for ``X = log(S_T/F)``,

    log E[e^{u X_T}] = int_0^T F(u, sum_j c_j psi_j(s)) g0(T - s) ds,
    psi_i' = -x_i psi_i + F(u, sum_j c_j psi_j),   psi_i(0) = 0,
    F(u, v) = (u^2 - u)/2 + (rho nu u - kappa) v + nu^2 v^2 / 2.

The Riccati system is stiff (``x_i`` spans several orders of magnitude), so it
is integrated by exponential time differencing (ETDRK4, Cox & Matthews 2002),
with the phi-functions evaluated by the Kassam & Trefethen (2005) contour
average, which is exact through ``x_i = 0``. The convolution with ``g0`` is
turned into extra linear ODE components (``Phi' = F``, ``chi_i' = -x_i chi_i +
Phi``), so nothing has to resolve the ``t^alpha`` boundary layer of ``g0`` by
quadrature. ``n = 1``, ``x = 0``, ``c = 1`` is exactly Heston (tests check it).

References
----------
* Abi Jaber, E. (2019), Lifting the Heston model, *Quantitative Finance*
  19(12), 1995-2013.
* Cox, S. M. & Matthews, P. C. (2002), Exponential time differencing for
  stiff systems, *J. Comput. Phys.* 176, 430-455.
* Kassam, A.-K. & Trefethen, L. N. (2005), Fourth-order time-stepping for
  stiff PDEs, *SIAM J. Sci. Comput.* 26(4), 1214-1233.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as gamma_fn

from .base import ForwardSpec, ModelSpec

_CUMULANT_CACHE: dict = {}
_CONTOUR = np.exp(2j * np.pi * (np.arange(1, 33) - 0.5) / 32)  # Kassam-Trefethen points


def lifted_kernel(H: float, n: int, r_n: float) -> tuple[np.ndarray, np.ndarray]:
    """Abi Jaber's weights ``c_i`` and speeds ``x_i`` for Hurst ``H``, ``n`` factors.

    They discretise the measure ``mu(dx) = x^{-alpha} dx / (Gamma(alpha) Gamma(1-alpha))``
    whose Laplace transform is the fractional kernel, on the geometric partition
    ``eta_i = r_n^{i - n/2}``: ``c_i = mu([eta_{i-1}, eta_i])`` and ``x_i`` its mean.
    """
    alpha = H + 0.5
    i = np.arange(1, n + 1, dtype=float)
    c = (
        (r_n ** (1 - alpha) - 1)
        * r_n ** ((alpha - 1) * (1 + n / 2))
        / (gamma_fn(alpha) * gamma_fn(2 - alpha))
        * r_n ** ((1 - alpha) * i)
    )
    x = (
        (1 - alpha)
        * (r_n ** (2 - alpha) - 1)
        / ((2 - alpha) * (r_n ** (1 - alpha) - 1))
        * r_n ** (i - 1 - n / 2)
    )
    return c, x


@dataclass(frozen=True)
class LiftedHestonParams(ModelSpec):
    """Lifted Heston parameters.

    Parameters
    ----------
    v0, kappa, theta, nu, rho :
        Heston-style initial variance, mean-reversion speed, long-run level,
        vol-of-vol and correlation (drift ``kappa (theta - V)``).
    H : float
        Hurst exponent in ``(0, 1/2)`` for the rough kernel.
    n : int
        Number of factors (default 20).
    r_n : float
        Geometric ratio of the factor speeds (default 2.5, Abi Jaber's ``r_20``).
    weights, speeds : tuple of float, optional
        An explicit kernel ``sum_i weights_i e^{-speeds_i t}`` replacing the
        rough parametrisation (``H, n, r_n`` are then ignored).
    """

    v0: float
    kappa: float
    theta: float
    nu: float
    rho: float
    H: float
    n: int
    r_n: float
    weights: tuple | None
    speeds: tuple | None

    def __init__(
        self,
        v0: float,
        kappa: float,
        theta: float,
        nu: float,
        rho: float,
        H: float = 0.1,
        n: int = 20,
        r_n: float = 2.5,
        weights=None,
        speeds=None,
    ):
        for name, val in (("v0", v0), ("theta", theta), ("nu", nu)):
            if not (np.isfinite(val) and val > 0):
                raise ValueError(f"LiftedHestonParams: {name} must be > 0; got {val}")
        if not (np.isfinite(kappa) and kappa >= 0):
            raise ValueError(f"LiftedHestonParams: kappa must be >= 0; got {kappa}")
        if not (-1.0 < rho < 1.0):
            raise ValueError(f"LiftedHestonParams: rho must be in (-1, 1); got {rho}")
        if (weights is None) != (speeds is None):
            raise ValueError("LiftedHestonParams: pass both weights and speeds, or neither")
        if weights is None:
            if not (0.0 < H < 0.5):
                raise ValueError(f"LiftedHestonParams: H must be in (0, 1/2); got {H}")
            if int(n) < 1 or r_n <= 1.0:
                raise ValueError("LiftedHestonParams: need n >= 1 and r_n > 1")
        else:
            weights, speeds = tuple(map(float, weights)), tuple(map(float, speeds))
            if len(weights) != len(speeds) or not weights:
                raise ValueError("LiftedHestonParams: weights and speeds must match in length")
            if any(s < 0 for s in speeds) or any(w <= 0 for w in weights):
                raise ValueError("LiftedHestonParams: need weights > 0 and speeds >= 0")
        object.__setattr__(self, "name", "lifted_heston")
        for key, val in (
            ("v0", v0), ("kappa", kappa), ("theta", theta), ("nu", nu), ("rho", rho),
            ("H", H), ("n", int(n)), ("r_n", r_n), ("weights", weights), ("speeds", speeds),
        ):  # fmt: skip
            object.__setattr__(self, key, val)

    def kernel(self) -> tuple[np.ndarray, np.ndarray]:
        if self.weights is not None:
            return np.asarray(self.weights), np.asarray(self.speeds)
        return lifted_kernel(self.H, self.n, self.r_n)


def _etd_coefficients(z: np.ndarray, dt: float):
    """ETDRK4 scalars per factor for ``L = z / dt`` (Kassam-Trefethen contour means)."""
    r = z[:, None] + _CONTOUR[None, :]
    e2 = np.exp(r / 2.0)
    er = np.exp(r)
    q = dt * np.real(np.mean((e2 - 1.0) / r, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - r + er * (4.0 - 3.0 * r + r * r)) / r**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + r + er * (r - 2.0)) / r**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * r - r * r + er * (4.0 - r)) / r**3, axis=1))
    return np.exp(z / 2.0), np.exp(z), q, f1, f2, f3


def _integrate(a: np.ndarray, T: float, p: LiftedHestonParams, n_steps: int) -> np.ndarray:
    """``log E[e^{a X_T}]`` for exponent arguments ``a`` (ETDRK4, ``n_steps`` uniform steps)."""
    c, x = p.kernel()
    n = c.size
    kap, nu, rho = p.kappa, p.nu, p.rho

    def nonlinear(Y):
        psi, phi_int = Y[..., :n], Y[..., n]
        v = psi @ c
        f = 0.5 * (a * a - a) + (rho * nu * a - kap) * v + 0.5 * nu * nu * v * v
        return np.concatenate(
            [
                np.repeat(f[..., None], n, axis=-1),
                f[..., None],
                np.repeat(phi_int[..., None], n, axis=-1),
            ],
            axis=-1,
        )

    speeds = np.concatenate([x, [0.0], x])  # psi_i, Phi, chi_i
    Y = np.zeros(a.shape + (2 * n + 1,), dtype=complex)
    dt = T / n_steps
    e2, e1, q, f1, f2, f3 = _etd_coefficients(-speeds * dt, dt)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_steps):
            n_u = nonlinear(Y)
            ya = e2 * Y + q * n_u
            n_a = nonlinear(ya)
            yb = e2 * Y + q * n_a
            n_b = nonlinear(yb)
            yc = e2 * ya + q * (2.0 * n_b - n_u)
            n_c = nonlinear(yc)
            Y = e1 * Y + f1 * n_u + 2.0 * f2 * (n_a + n_b) + f3 * n_c
    return p.v0 * Y[..., n] + kap * p.theta * (Y[..., n + 1 :] @ c)


_STABILITY = 0.1  # explicit nonlinear stages need dt * nu * |u| below ~0.1-0.2


def lifted_heston_cf(
    u: np.ndarray, fwd: ForwardSpec, p: LiftedHestonParams, *, n_steps: int = 512
) -> np.ndarray:
    """CF of ``X_T = log(S_T / F_0)`` under lifted Heston (see module docstring).

    The convolution ``int F(s) g0(T - s) ds`` is rewritten with
    ``Phi' = F`` and ``chi_i' = -x_i chi_i + Phi`` (both zero at 0) as
    ``v0 Phi(T) + kappa theta sum_i c_i chi_i(T)``, so the whole CF comes from
    one stiff ODE system integrated by ETDRK4.

    Frequencies are grouped by magnitude and each group gets at least the
    step count its explicit stages need for stability (``dt nu |u| <= 0.1``).
    Once a whole group of near-real frequencies has ``|phi| < 1e-16``, larger
    near-real frequencies are returned as 0 (the CF of this absolutely
    continuous law tends to zero) instead of being integrated at great cost.
    """
    w = np.asarray(u, dtype=complex)
    flat = w.ravel()
    out = np.empty(flat.shape, dtype=complex)
    T = fwd.T
    need = np.ceil(T * p.nu * np.abs(flat.real) / _STABILITY)
    steps = np.maximum(n_steps, 2.0 ** np.ceil(np.log2(np.maximum(need, 1.0)))).astype(int)
    near_real = np.abs(flat.imag) <= 1.0
    negligible = False
    for count in np.unique(steps):
        idx = np.flatnonzero(steps == count)
        if negligible:
            skip = near_real[idx]
            out[idx[skip]] = 0.0
            idx = idx[~skip]
            if idx.size == 0:
                continue
        out[idx] = np.exp(_integrate(1j * flat[idx], T, p, int(count)))
        real_idx = idx[near_real[idx]]
        if real_idx.size and np.all(np.abs(out[real_idx]) < 1e-16):
            negligible = True
    return out.reshape(w.shape)


def lifted_heston_cumulants(fwd: ForwardSpec, p: LiftedHestonParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` by Cauchy integration of the CF."""
    cached = _CUMULANT_CACHE.get((p, fwd))
    if cached is not None:
        return cached
    from ..utils.cumulants import cumulants_from_cf

    cm = cumulants_from_cf(lambda z: lifted_heston_cf(z, fwd, p), order=4, radius=0.25, M=64)
    out = (float(cm[0]), float(cm[1]), float(cm[3]))
    _CUMULANT_CACHE[(p, fwd)] = out
    return out
