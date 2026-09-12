"""Time-changed Levy models: any Levy base on a stochastic business clock.

Carr, Geman, Madan & Yor (2003) run a Levy process ``L`` with exponent
``psi`` (``E[e^{iu L_t}] = e^{t psi(u)}``) on an absolutely continuous clock
``Y_t = int_0^t y_s ds`` driven by an independent activity rate ``y``. With the
deterministic martingale correction of their eq. (4.9),

    log(S_T / F) = L(Y_T) - log E[e^{L(Y_T)}],
    phi(u) = M_Y(psi(u)) / M_Y(psi(-i))^{iu},     M_Y(w) = E[e^{w Y_T}].

Bases (raw, drift-free exponents per unit clock time)
    ``bsm``  ``-sigma^2 u^2 / 2``
    ``vg``   ``-(1/nu) log(1 - i u theta nu + sigma^2 nu u^2 / 2)``
    ``nig``  ``(1/nu) (1 - sqrt(1 - 2 i u theta nu + sigma^2 nu u^2))``
    ``cgmy`` ``C Gamma(-Y) [(M - iu)^Y - M^Y + (G + iu)^Y - G^Y]``
    ``kou``  ``-sigma^2 u^2/2 + lam (p eta1/(eta1 - iu) + (1-p) eta2/(eta2 + iu) - 1)``
    ``merton_jd`` ``-sigma^2 u^2/2 + lam (e^{i u muj - sigj^2 u^2 / 2} - 1)``

  Each is written out rather than taken as ``log`` of the registry CF, whose
  principal branch would jump; tests pin every exponent to the registry CF.

Clocks
    :class:`CirClock` ``dy = kappa (eta - y) dt + lam sqrt(y) dW`` (CGMY's
    ``CIR`` arrival rate). ``M_Y`` is the Heston-type integrated-CIR transform,
    evaluated in a branch-safe form.
    :class:`GammaOUClock` ``dy = -lam y dt + dz_{lam t}`` with ``z`` compound
    Poisson (rate ``a``, Exp(``b``) jumps), so ``y`` has Gamma(a, b) marginals.
    ``M_Y(w) = exp(w y0 c + lam a / (w - lam b) (b log(b / (b - w c)) - w t))``
    with ``c = (1 - e^{-lam t}) / lam``.

VG on a CIR clock is the registry's ``vgsa`` (tests check the identity).

Reference
---------
Carr, P., Geman, H., Madan, D. B. & Yor, M. (2003), Stochastic volatility for
Levy processes, *Mathematical Finance* 13(3), 345-382.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as gamma_fn

from .base import ForwardSpec, ModelSpec

_BASES = ("bsm", "vg", "nig", "cgmy", "kou", "merton_jd")
_CUMULANT_CACHE: dict = {}


@dataclass(frozen=True)
class CirClock:
    """CIR activity rate ``dy = kappa (eta - y) dt + lam sqrt(y) dW``, ``y(0) = y0``."""

    y0: float
    kappa: float
    eta: float
    lam: float

    def __post_init__(self) -> None:
        for name in ("y0", "kappa", "eta"):
            val = getattr(self, name)
            if not (np.isfinite(val) and val > 0):
                raise ValueError(f"CirClock: {name} must be > 0; got {val}")
        if not (np.isfinite(self.lam) and self.lam >= 0):
            raise ValueError(f"CirClock: lam must be >= 0; got {self.lam}")

    def log_mgf(self, w: np.ndarray, t: float) -> np.ndarray:
        """``log E[exp(w Y_t)]`` for complex ``w`` (``Re w`` below the explosion point)."""
        k, eta, lam, y0 = self.kappa, self.eta, self.lam, self.y0
        if lam == 0.0:
            e = np.exp(-k * t)
            return w * (y0 * (1.0 - e) / k + eta * (t - (1.0 - e) / k))
        g = np.sqrt(k * k - 2.0 * lam * lam * w + 0j)
        e = np.exp(-g * t)
        # cosh(gt/2) + (k/g) sinh(gt/2) = e^{gt/2} ((1 + k/g) + (1 - k/g) e^{-gt}) / 2
        log_den = 0.5 * g * t - np.log(2.0) + np.log((1.0 + k / g) + (1.0 - k / g) * e)
        coth = (1.0 + e) / (1.0 - e)
        return (
            k * k * eta * t / (lam * lam)
            + 2.0 * y0 * w / (k + g * coth)
            - 2.0 * k * eta / (lam * lam) * log_den
        )


@dataclass(frozen=True)
class GammaOUClock:
    """Gamma-OU activity rate ``dy = -lam y dt + dz_{lam t}`` (Gamma(a, b) marginals)."""

    y0: float
    lam: float
    a: float
    b: float

    def __post_init__(self) -> None:
        for name in ("y0", "lam", "b"):
            val = getattr(self, name)
            if not (np.isfinite(val) and val > 0):
                raise ValueError(f"GammaOUClock: {name} must be > 0; got {val}")
        if not (np.isfinite(self.a) and self.a >= 0):
            raise ValueError(f"GammaOUClock: a must be >= 0; got {self.a}")

    def log_mgf(self, w: np.ndarray, t: float) -> np.ndarray:
        lam, a, b = self.lam, self.a, self.b
        c = -np.expm1(-lam * t) / lam
        return w * self.y0 * c + lam * a / (w - lam * b) * (b * np.log(b / (b - w * c)) - w * t)


def _raw_exponent(model: str, p, u: np.ndarray) -> np.ndarray:
    """Drift-free Levy exponent per unit time of the base model."""
    iu = 1j * u
    if model == "bsm":
        return -0.5 * p.sigma**2 * u * u
    if model == "vg":
        return -np.log(1.0 - iu * p.theta * p.nu + 0.5 * p.sigma**2 * p.nu * u * u) / p.nu
    if model == "nig":
        return (1.0 - np.sqrt(1.0 - 2.0 * iu * p.theta * p.nu + p.sigma**2 * p.nu * u * u)) / p.nu
    if model == "cgmy":
        C, G, M, Y = p.C, p.G, p.M, p.Y
        return C * gamma_fn(-Y) * ((M - iu) ** Y - M**Y + (G + iu) ** Y - G**Y)
    if model == "kou":
        jump = p.p * p.eta1 / (p.eta1 - iu) + (1.0 - p.p) * p.eta2 / (p.eta2 + iu) - 1.0
        return -0.5 * p.sigma**2 * u * u + p.lam * jump
    if model == "merton_jd":
        jump = np.exp(iu * p.muj - 0.5 * p.sigj**2 * u * u) - 1.0
        return -0.5 * p.sigma**2 * u * u + p.lam * jump
    raise ValueError(f"time_changed_levy: unsupported base model {model!r}")


@dataclass(frozen=True)
class TimeChangedLevyParams(ModelSpec):
    """A registry Levy model run on a stochastic clock.

    Parameters
    ----------
    base_model : str
        One of ``bsm, vg, nig, cgmy, kou, merton_jd``.
    base_params :
        The base model's parameter dataclass (per unit of clock time).
    clock : CirClock or GammaOUClock
    """

    base_model: str
    base_params: ModelSpec
    clock: CirClock | GammaOUClock

    def __init__(self, base_model: str, base_params, clock):
        if base_model not in _BASES:
            raise ValueError(
                f"TimeChangedLevyParams: base_model must be one of {_BASES}; got {base_model!r}"
            )
        if not isinstance(clock, (CirClock, GammaOUClock)):
            raise TypeError("TimeChangedLevyParams: clock must be a CirClock or GammaOUClock")
        object.__setattr__(self, "name", "time_changed_levy")
        object.__setattr__(self, "base_model", base_model)
        object.__setattr__(self, "base_params", base_params)
        object.__setattr__(self, "clock", clock)


def time_changed_levy_cf(u: np.ndarray, fwd: ForwardSpec, p: TimeChangedLevyParams) -> np.ndarray:
    """CF of ``X_T = log(S_T / F_0)`` (CGMY 2003, eq. 4.9)."""
    u = np.asarray(u, dtype=complex)
    psi = _raw_exponent(p.base_model, p.base_params, u)
    psi_mart = _raw_exponent(p.base_model, p.base_params, np.array([-1j]))
    log_norm = p.clock.log_mgf(psi_mart, fwd.T)[0]
    if not (np.isfinite(log_norm) and abs(np.imag(log_norm)) < 1e-10):
        raise ValueError(
            "time_changed_levy: E[exp(L(Y_T))] is not finite for these parameters "
            "(the clock's moment generating function explodes at psi(-i))"
        )
    return np.exp(p.clock.log_mgf(psi, fwd.T) - 1j * u * np.real(log_norm))


def time_changed_levy_cumulants(
    fwd: ForwardSpec, p: TimeChangedLevyParams
) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` by Cauchy integration of the CF."""
    cached = _CUMULANT_CACHE.get((p, fwd))
    if cached is not None:
        return cached
    from ..utils.cumulants import cumulants_from_cf

    c = cumulants_from_cf(lambda u: time_changed_levy_cf(u, fwd, p), order=4, radius=0.25, M=64)
    out = (float(c[0]), float(c[1]), float(c[3]))
    _CUMULANT_CACHE[(p, fwd)] = out
    return out
