"""Parameter gradients of characteristic functions.

``cf_and_gradient(model, u, fwd, params)`` returns ``phi(u)`` and
``d phi / d theta_j`` for every float field ``theta_j`` of the parameter
dataclass, in field order. Calibration builds its Jacobian from these: prices
are linear in ``phi`` for a fixed COS grid, so a price gradient is one more
COS sum per parameter.

Analytic gradients
    ``bsm, merton_jd, kou, vg, nig, cgmy`` from the Levy exponent: with the
    martingale correction ``log phi = T (psi(u) - i u psi(-i))``, so
    ``d log phi = T (d psi(u) - i u d psi(-i))``.
    ``heston`` and ``bates`` by the chain rule through the "little trap"
    Riccati solution ``log phi = A + B v0`` (the same route as Cui, del Bano
    Rollin & Germano 2017, derived for this parametrisation).

Any other model with plain float parameters
    Central differences of ``phi`` in each parameter at fixed ``u``. The
    frequency grid does not move, so the result is smooth in the parameters
    (accurate to about 1e-10 relative), which is what a Gauss-Newton step needs.

Reference
---------
Cui, Y., del Bano Rollin, S. & Germano, G. (2017), Full and fast calibration of
the Heston stochastic volatility model, *European Journal of Operational
Research* 263(2), 625-638.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace

import numpy as np
from scipy.special import digamma
from scipy.special import gamma as gamma_fn

from .base import ForwardSpec
from .registry import MODEL_REGISTRY


def param_names(model: str) -> tuple[str, ...]:
    """Names of the float parameters of ``model``, in dataclass field order."""
    cls = MODEL_REGISTRY[model].params_cls
    return tuple(f.name for f in dataclasses.fields(cls) if f.name != "name")


def params_from_vector(model: str, x, template=None):
    """Build the parameter dataclass of ``model`` from a vector in field order."""
    names = param_names(model)
    cls = MODEL_REGISTRY[model].params_cls
    kwargs = {n: float(v) for n, v in zip(names, x)}
    if template is not None:  # keep any non-float fields of the template
        for f in dataclasses.fields(cls):
            if f.name != "name" and f.name not in kwargs:
                kwargs[f.name] = getattr(template, f.name)
    return cls(**kwargs)


def params_to_vector(model: str, params) -> np.ndarray:
    return np.array([float(getattr(params, n)) for n in param_names(model)])


# --------------------------------------------------------------------------- Levy exponents
# Each returns (psi(u), [d psi / d theta_j]) for the raw, drift-free exponent per unit time.


def _bsm_psi(u, p):
    return -0.5 * p.sigma**2 * u * u, [-p.sigma * u * u]


def _merton_psi(u, p):
    E = np.exp(1j * u * p.muj - 0.5 * p.sigj**2 * u * u)
    psi = -0.5 * p.sigma**2 * u * u + p.lam * (E - 1.0)
    return psi, [-p.sigma * u * u, E - 1.0, p.lam * 1j * u * E, -p.lam * p.sigj * u * u * E]


def _kou_psi(u, p):
    iu = 1j * u
    up, dn = p.eta1 / (p.eta1 - iu), p.eta2 / (p.eta2 + iu)
    jump = p.p * up + (1.0 - p.p) * dn - 1.0
    psi = -0.5 * p.sigma**2 * u * u + p.lam * jump
    grads = [
        -p.sigma * u * u,
        jump,
        p.lam * (up - dn),
        p.lam * p.p * (-iu) / (p.eta1 - iu) ** 2,
        p.lam * (1.0 - p.p) * iu / (p.eta2 + iu) ** 2,
    ]
    return psi, grads


def _vg_psi(u, p):
    s, nu, th = p.sigma, p.nu, p.theta
    B = 1.0 - 1j * u * th * nu + 0.5 * s * s * nu * u * u
    logB = np.log(B)
    psi = -logB / nu
    d_sigma = -s * u * u / B
    d_nu = logB / nu**2 - (-1j * u * th + 0.5 * s * s * u * u) / (nu * B)
    d_theta = 1j * u / B
    return psi, [d_sigma, d_nu, d_theta]


def _nig_psi(u, p):
    s, nu, th = p.sigma, p.nu, p.theta
    S = np.sqrt(1.0 - 2j * u * th * nu + s * s * nu * u * u)
    psi = (1.0 - S) / nu
    d_sigma = -s * u * u / S
    d_nu = -(1.0 - S) / nu**2 - (-1j * u * th + 0.5 * s * s * u * u) / (nu * S)
    d_theta = 1j * u / S
    return psi, [d_sigma, d_nu, d_theta]


def _cgmy_psi(u, p):
    C, G, M, Y = p.C, p.G, p.M, p.Y
    g = gamma_fn(-Y)
    m, n = M - 1j * u, G + 1j * u
    A = m**Y - M**Y + n**Y - G**Y
    psi = C * g * A
    dC = g * A
    dG = C * g * Y * (n ** (Y - 1.0) - G ** (Y - 1.0))
    dM = C * g * Y * (m ** (Y - 1.0) - M ** (Y - 1.0))
    A_Y = m**Y * np.log(m) - M**Y * np.log(M) + n**Y * np.log(n) - G**Y * np.log(G)
    dY = C * (-g * digamma(-Y) * A + g * A_Y)
    return psi, [dC, dG, dM, dY]


_LEVY: dict[str, Callable] = {
    "bsm": _bsm_psi,
    "merton_jd": _merton_psi,
    "kou": _kou_psi,
    "vg": _vg_psi,
    "nig": _nig_psi,
    "cgmy": _cgmy_psi,
}


def _levy_log_cf_grad(psi_fn, u, T, p):
    """``log phi = T (psi(u) - i u psi(-i))`` and its parameter gradient."""
    psi_u, d_u = psi_fn(u, p)
    psi_m, d_m = psi_fn(np.array([-1j]), p)
    log_phi = T * (psi_u - 1j * u * psi_m[0])
    grads = [T * (du - 1j * u * dm[0]) for du, dm in zip(d_u, d_m)]
    return log_phi, grads


# --------------------------------------------------------------------------- Heston / Bates


def _heston_log_cf_grad(u, T, kappa, theta, nu, rho, v0):
    """``log phi`` and its derivatives in (kappa, theta, nu, rho, v0)."""
    a = -0.5 * (u * u + 1j * u)
    b = kappa - 1j * nu * rho * u
    nu2 = nu * nu
    d = np.sqrt(b * b - 2.0 * nu2 * a)
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    h = b - d
    Q = (1.0 - e) / (1.0 - g * e)
    B = h * Q / nu2
    Lg = np.log((1.0 - g * e) / (1.0 - g))
    A = kappa * theta / nu2 * (h * T - 2.0 * Lg)
    log_phi = A + B * v0

    zero = np.zeros_like(b)
    # (db, dnu) for kappa, theta, nu, rho; v0 and theta enter only linearly
    derivs = {"kappa": (1.0 + zero, 0.0), "nu": (-1j * rho * u, 1.0), "rho": (-1j * nu * u, 0.0)}
    out = {}
    for name, (db, dnu) in derivs.items():
        dd = (b * db - 2.0 * nu * a * dnu) / d
        dg = 2.0 * (db * d - b * dd) / (b + d) ** 2
        de = -T * dd * e
        dh = db - dd
        dQ = (-de * (1.0 - g * e) + (1.0 - e) * (dg * e + g * de)) / (1.0 - g * e) ** 2
        dB = (dh * Q + h * dQ) / nu2 - 2.0 * h * Q * dnu / nu**3
        dL = -(dg * e + g * de) / (1.0 - g * e) + dg / (1.0 - g)
        dkappa = 1.0 if name == "kappa" else 0.0
        dpref = dkappa * theta / nu2 - 2.0 * kappa * theta * dnu / nu**3
        dA = dpref * (h * T - 2.0 * Lg) + kappa * theta / nu2 * (dh * T - 2.0 * dL)
        out[name] = dA + dB * v0
    d_theta = kappa / nu2 * (h * T - 2.0 * Lg)
    grads = [out["kappa"], d_theta, out["nu"], out["rho"], B]
    return log_phi, grads


def _heston_grad(u, T, p):
    return _heston_log_cf_grad(u, T, p.kappa, p.theta, p.nu, p.rho, p.v0)


def _bates_grad(u, T, p):
    log_h, g_h = _heston_log_cf_grad(u, T, p.kappa, p.theta, p.nu, p.rho, p.v0)
    # the jump block is the Merton exponent with no diffusion part
    jumps = SimpleNamespace(sigma=0.0, lam=p.lam_j, muj=p.mu_j, sigj=p.sigma_j)
    log_j, g_j = _levy_log_cf_grad(_merton_psi, u, T, jumps)
    return log_h + log_j, g_h + g_j[1:]


_ANALYTIC: dict[str, Callable] = {
    **{m: partial(_levy_log_cf_grad, fn) for m, fn in _LEVY.items()},
    "heston": _heston_grad,
    "bates": _bates_grad,
}

ANALYTIC_GRADIENT_MODELS = tuple(sorted(_ANALYTIC))


def _fd_gradient(model: str, u, fwd: ForwardSpec, params, rel_step: float = 1e-5):
    entry = MODEL_REGISTRY[model]
    x = params_to_vector(model, params)
    grads = []
    for j in range(len(x)):
        h = rel_step * max(abs(x[j]), 0.1)
        up, dn = x.copy(), x.copy()
        up[j] += h
        dn[j] -= h
        try:
            f_up = entry.cf(u, fwd, params_from_vector(model, up, params))
        except ValueError:
            f_up, up = entry.cf(u, fwd, params), x
        try:
            f_dn = entry.cf(u, fwd, params_from_vector(model, dn, params))
        except ValueError:
            f_dn, dn = entry.cf(u, fwd, params), x
        grads.append((f_up - f_dn) / (up[j] - dn[j]))
    return grads


def cf_and_gradient(model: str, u, fwd: ForwardSpec, params, *, analytic: bool = True):
    """``phi(u)`` and ``d phi / d theta`` (shape ``(n_params, len(u))``) for ``model``.

    Uses the analytic formulas when available (``analytic=True``), otherwise
    central differences of the registry CF at fixed ``u``.
    """
    u = np.asarray(u, dtype=complex)
    if analytic and model in _ANALYTIC:
        log_phi, grads = _ANALYTIC[model](u, float(fwd.T), params)
        phi = np.exp(log_phi)
        return phi, np.array([phi * gj for gj in grads])
    phi = MODEL_REGISTRY[model].cf(u, fwd, params)
    return phi, np.array(_fd_gradient(model, u, fwd, params))


__all__ = [
    "ANALYTIC_GRADIENT_MODELS",
    "cf_and_gradient",
    "param_names",
    "params_from_vector",
    "params_to_vector",
]
