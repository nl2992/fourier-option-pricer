"""3/2 Stochastic Volatility model (Lewis 2000)  -  native closed-form CF.

The 3/2 model is a stochastic-volatility model where the *instantaneous
variance* v_t satisfies a mean-reverting SDE with diffusion term proportional
to v^(3/2) rather than sqrt(v) (Heston). The faster diffusion rate produces
a much larger smile curvature for short maturities, fitting equity skew better
than Heston at the short end.

    dS/S = sqrt(v_t) dW1
    dv   = kappa * v * (theta - v) dt + nu * v^(3/2) dW2,
           <dW1, dW2> = rho dt

Unlike Heston's SDE, the coefficient on dW2 is v^(3/2), giving the "3/2" name.

Characteristic function
-----------------------
``1/v`` is a CIR process, which gives the closed form of Lewis (2000) and
Carr and Sun (2007, eqs. 73-75). For the MGF argument ``s = i u``::

    mu    = 1/2 + (kappa - s rho nu) / nu^2
    delta = sqrt(mu^2 + s (1 - s) / nu^2)
    a     = delta - mu,   b = 1 + 2 delta,   c = b - a
    X     = 2 kappa theta / (nu^2 v0 (exp(kappa theta T) - 1))

    phi(u) = Gamma(c) / Gamma(b) X^a 1F1(a; b; -X)

This is the formula PyFENG's ``Sv32Fft.logp_mgf`` uses, with the same
parameter mapping (``sigma = v0``, ``mr = kappa``, ``theta = theta``,
``vov = nu``), but PyFENG sums 1F1 as a plain 1024-term Taylor series in
``-X``. ``X`` grows like ``2 / (nu^2 v0 T)``, so at short maturities the
alternating series cancels catastrophically (``|phi(1)|`` of about 1e118 at
``T = 0.1`` for ``v0 = 0.04, nu = 1.2``) while ``phi(0) = phi(-i) = 1`` still
hold, because ``a = 0`` there makes every series term vanish.

Here Kummer's transformation ``1F1(a; b; -X) = e^{-X} 1F1(c; b; X)`` turns the
CF into a Poisson mixture::

    phi(u) = sum_k  e^{-X} X^k / k!  *  X^a Gamma(c + k) / Gamma(b + k),

whose terms are bounded and never form ``e^X``. The sum is started at the
term of largest modulus (``|t_k|`` is log-concave in ``k``), whose logarithm
is computed to full precision (Loader's saddle-point Poisson pmf and a
Stirling expansion of the Gamma ratio), and walked outwards in both
directions by the exact term ratio until the terms are negligible. The cost
is ``O(sqrt(X))`` terms per point rather than ``O(X)``. Agreement with
mpmath is about 1e-13 absolute or better for maturities from 0.005 to 10.

References
----------
* Lewis, A. L. (2000), *Option Valuation Under Stochastic Volatility*,
  Finance Press. (Original CF derivation via Laplace transform of ∫v_t dt.)
* Carr, P. and Sun, J. (2007), "A new approach for option pricing under
  stochastic volatility", *Review of Derivatives Research*.

PyFENG benchmark (from docstring)::

    sigma=0.06, mr=20.48, theta=0.218, vov=3.20, rho=-0.99
    strikes=[95, 100, 105], spot=100, texp=0.5
    → prices ≈ [11.7235, 8.9978, 6.7091]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln, loggamma

from ._pyfeng_backend import build_cached, import_pyfeng
from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class Sv32Params(ModelSpec):
    """3/2 SV model parameters.

    Parameters
    ----------
    v0 :
        Initial instantaneous variance (``= sigma`` in PyFENG notation).
        Must be ``> 0``.  Note: this is *variance*, not volatility; the
        initial vol is ``sqrt(v0)``.
    kappa :
        Mean-reversion speed of the variance process (``mr`` in PyFENG).
        Must be ``> 0``.
    theta :
        Long-term mean of the variance process (``theta`` in PyFENG).
        Must be ``> 0``.
    nu :
        Vol-of-vol  -  coefficient on the v^(3/2) diffusion term (``vov``
        in PyFENG).  Must be ``> 0``.
    rho :
        Correlation between the spot and variance Brownian motions.
        Must be in ``(-1, 1)``.
    """

    v0: float
    kappa: float
    theta: float
    nu: float
    rho: float

    def __init__(
        self,
        v0: float,
        kappa: float,
        theta: float,
        nu: float,
        rho: float,
    ):
        if not (np.isfinite(v0) and v0 > 0):
            raise ValueError(f"Sv32Params: v0 must be > 0; got {v0}")
        if not (np.isfinite(kappa) and kappa > 0):
            raise ValueError(f"Sv32Params: kappa must be > 0; got {kappa}")
        if not (np.isfinite(theta) and theta > 0):
            raise ValueError(f"Sv32Params: theta must be > 0; got {theta}")
        if not (np.isfinite(nu) and nu > 0):
            raise ValueError(f"Sv32Params: nu must be > 0; got {nu}")
        if not (np.isfinite(rho) and -1.0 < rho < 1.0):
            raise ValueError(f"Sv32Params: rho must be in (-1, 1); got {rho}")
        object.__setattr__(self, "name", "sv32")
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)


# ---------------------------------------------------------------------------
# Closed-form CF, summed as a Poisson mixture
# ---------------------------------------------------------------------------

_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)
_WALK_TOL = 1e-20  # stop a walk once a term is this small relative to the sum
_MAX_WALK = 50_000_000  # hard cap on terms per walk; about 7e3 * sqrt(X) is typical
_REAL_U_TOL = 1e-9  # |phi(u)| may exceed 1 by this much for real u before we raise


def _log1p_complex(w: np.ndarray) -> np.ndarray:
    """``log(1 + w)`` for complex ``w``, accurate when ``|w|`` is small."""
    re, im = w.real, w.imag
    return 0.5 * np.log1p(2.0 * re + re * re + im * im) + 1j * np.arctan2(im, 1.0 + re)


def _stirling_tail(w: np.ndarray) -> np.ndarray:
    """``log Gamma(w) - (w - 1/2) log w + w - log(2 pi) / 2`` for ``|w| > 20``."""
    w2 = 1.0 / (w * w)
    series = 1 / 12 + w2 * (
        -1 / 360 + w2 * (1 / 1260 + w2 * (-1 / 1680 + w2 * (1 / 1188 + w2 * (-691 / 360360))))
    )
    return series / w


def _log_gamma_ratio(k, a, b, c, X: float) -> np.ndarray:
    """``a log X + log Gamma(c + k) - log Gamma(b + k)``, where ``c = b - a``.

    For large arguments the two log-gammas are each of size ``k log k`` and
    their plain difference would lose ``eps * k log k``; the Stirling form
    below only ever forms quantities of size ``|a|``.
    """
    z, zc = b + k, c + k
    out = np.empty(z.shape, dtype=np.complex128)
    big = (np.abs(z) > 20.0) & (np.abs(zc) > 20.0)
    small = ~big
    out[small] = a[small] * np.log(X) + loggamma(zc[small]) - loggamma(z[small])
    zb, ab, zcb = z[big], a[big], zc[big]
    out[big] = (
        (zb - 0.5) * _log1p_complex(-ab / zb)
        + ab
        - ab * _log1p_complex((zcb - X) / X)
        + _stirling_tail(zcb)
        - _stirling_tail(zb)
    )
    return out


def _log_poisson(k: np.ndarray, X: float) -> np.ndarray:
    """``log(e^{-X} X^k / k!)`` to full relative precision (Loader 2000)."""
    out = np.full(k.shape, -X, dtype=np.float64)
    pos = k > 0
    n = k[pos]
    # stirlerr(n) = log n! - log(sqrt(2 pi n) (n / e)^n)
    n2 = 1.0 / (n * n)
    stirlerr = np.where(
        n < 16,
        gammaln(n + 1.0) - (n + 0.5) * np.log(n) + n - _HALF_LOG_2PI,
        (1 / 12 - n2 * (1 / 360 - n2 * (1 / 1260 - n2 / 1680))) / n,
    )
    # bd0 = n log(n / X) + X - n = X h(x), h(x) = (1 + x) log1p(x) - x
    x = (n - X) / X
    h_series = np.zeros_like(x)
    power = x * x
    for j in range(2, 27):
        h_series += (-1.0) ** j * power / (j * (j - 1))
        power = power * x
    with np.errstate(divide="ignore", invalid="ignore"):
        h_direct = (1.0 + x) * np.log1p(x) - x
    h = np.where(np.abs(x) < 0.1, h_series, h_direct)
    out[pos] = -stirlerr - X * h - 0.5 * np.log(2.0 * np.pi * n)
    return out


def _log_abs_ratio(k, b, c, X: float) -> np.ndarray:
    """``log |t_{k+1} / t_k|`` for the Poisson-mixture terms; decreasing in ``k``."""
    return np.log(np.abs((c + k) / (b + k))) + np.log(X / (k + 1.0))


def _mode_index(b: np.ndarray, c: np.ndarray, X: float) -> np.ndarray:
    """Index of the largest term: the smallest ``k >= 0`` with ``|t_{k+1}| < |t_k|``."""
    hi = 2.0 * (X + np.abs(b) + np.abs(c)) + 10.0
    for _ in range(64):
        rising = _log_abs_ratio(hi, b, c, X) >= 0.0
        if not rising.any():
            break
        hi = np.where(rising, 2.0 * hi, hi)
    hi = np.where(_log_abs_ratio(np.zeros_like(hi), b, c, X) < 0.0, 0.0, hi)
    lo = np.zeros_like(hi)
    while np.any(hi - lo > 1.0):
        mid = np.floor(0.5 * (lo + hi))
        falling = _log_abs_ratio(mid, b, c, X) < 0.0
        hi = np.where(falling, mid, hi)
        lo = np.where(falling, lo, mid)
    return hi


def _walk(total, m, b, c, X: float, forward: bool) -> None:
    """Add the terms on one side of the mode, scaled by the mode term, to ``total``.

    Every point walks from its own mode with the exact ratio
    ``t_{k+1} / t_k = (c + k) / (b + k) * X / (k + 1)`` and drops out once its
    terms are negligible.
    """
    idx = np.flatnonzero(np.ones(m.shape, dtype=bool) if forward else m > 0)
    k, bb, cc = m[idx], b[idx], c[idx]
    t = np.ones(idx.size, dtype=np.complex128)
    for _ in range(_MAX_WALK):
        if idx.size == 0:
            return
        if forward:
            ratio = (cc + k) / (bb + k) * (X / (k + 1.0))
            t = t * ratio
            k = k + 1.0
            edge = np.abs(ratio) < 1.0
        else:
            k = k - 1.0
            t = t / ((cc + k) / (bb + k) * (X / (k + 1.0)))
            edge = k <= 0.0
        s = total[idx] + t
        total[idx] = s
        small = np.abs(t) <= _WALK_TOL * np.abs(s)
        done = ~np.isfinite(t) | (small & edge if forward else small | edge)
        keep = ~done
        idx, k, bb, cc, t = idx[keep], k[keep], bb[keep], cc[keep], t[keep]
    raise RuntimeError(
        "sv32: the 3/2 CF series did not converge; the maturity is too short "
        "for these parameters (X = 2 kappa theta / (nu^2 v0 (exp(kappa theta T) - 1)) "
        f"= {X:.3g})."
    )


def _sv32_mgf(s, T: float, p: Sv32Params) -> np.ndarray:
    """``E[exp(s X_T)]`` for ``X_T = log(S_T / F_0)``; ``s`` real or complex."""
    s_arr = np.asarray(s, dtype=np.complex128)
    if T <= 0.0:
        return np.ones_like(s_arr)
    flat = s_arr.ravel()
    nu2 = p.nu * p.nu
    mu = 0.5 + (p.kappa - flat * p.rho * p.nu) / nu2
    delta = np.sqrt(mu * mu + flat * (1.0 - flat) / nu2)
    a = delta - mu
    b = 1.0 + 2.0 * delta
    c = b - a
    kt = p.kappa * p.theta
    X = 2.0 * kt / (nu2 * p.v0 * np.expm1(kt * T))

    m = _mode_index(b, c, X)
    log_mode_term = _log_poisson(m, X) + _log_gamma_ratio(m, a, b, c, X)
    total = np.ones_like(flat)
    _walk(total, m, b, c, X, forward=True)
    _walk(total, m, b, c, X, forward=False)
    return (np.exp(log_mode_term) * total).reshape(s_arr.shape)


def sv32_cf(u: np.ndarray, fwd: ForwardSpec, p: Sv32Params) -> np.ndarray:
    """CF of ``X_T = log(S_T / F_0)`` under the 3/2 SV model.

    Evaluates the Lewis (2000) / Carr-Sun (2007) closed form as a Poisson
    mixture (see the module docstring), which stays accurate at short
    maturities where PyFENG's ``Sv32Fft.logp_cf`` overflows. The result is
    in the log-forward convention, so ``phi(0) = phi(-i) = 1``.

    Parameters
    ----------
    u : array_like
        Frequency grid (real or complex).
    fwd : ForwardSpec
    p : Sv32Params

    Returns
    -------
    np.ndarray
        Complex CF values, same shape as ``u``.

    Raises
    ------
    FloatingPointError
        If ``|phi(u)| > 1`` (or is not finite) at a real ``u``, which no
        characteristic function can satisfy. This guards against silent
        loss of precision rather than returning prices below intrinsic.
    """
    u_arr = np.asarray(u)
    phi = _sv32_mgf(1j * u_arr, fwd.T, p)
    real_u = np.isreal(u_arr)
    bad = real_u & ~(np.abs(phi) <= 1.0 + _REAL_U_TOL)
    if np.any(bad):
        i = np.flatnonzero(bad.ravel())[0]
        raise FloatingPointError(
            f"sv32_cf: |phi(u)| = {abs(phi.ravel()[i]):.3e} at real u = "
            f"{np.real(u_arr.ravel()[i]):.6g} (T = {fwd.T}); a characteristic function "
            "is bounded by 1, so the 3/2 CF evaluation has lost precision for "
            f"these parameters: {p}."
        )
    return phi


# ---------------------------------------------------------------------------
# PyFENG model for method="pyfeng_fft"
# ---------------------------------------------------------------------------

_SV32_MODEL_CACHE: dict[tuple, object] = {}


def _stable_logp_mgf(self, uu, texp):
    """``logp_mgf`` for a :class:`pyfeng.Sv32Fft` instance, via :func:`_sv32_mgf`."""
    q = Sv32Params(v0=self.sigma, kappa=self.mr, theta=self.theta, nu=self.vov, rho=self.rho)
    return _sv32_mgf(uu, texp, q)


def _pyfeng_sv32_model(fwd: ForwardSpec, p: Sv32Params) -> Any:
    """Build-and-cache a :class:`pyfeng.Sv32Fft` for ``(fwd, p)``.

    The instance keeps PyFENG's FFT pricer but replaces its ``logp_mgf``
    with :func:`_sv32_mgf`, because PyFENG's own 1F1 series overflows at
    short maturities. PyFENG kwarg mapping (the same model convention,
    ``dv = mr v (theta - v) dt + vov v^{3/2} dW``)::

        sigma  <->  p.v0       (initial variance)
        mr     <->  p.kappa    (mean-reversion speed)
        theta  <->  p.theta    (mean-reversion level of the variance)
        vov    <->  p.nu       (vol-of-vol)
        rho    <->  p.rho      (spot-var correlation)
    """

    def _factory():
        pf = import_pyfeng()
        stable_cls = type("Sv32FftStable", (pf.Sv32Fft,), {"logp_mgf": _stable_logp_mgf})
        return stable_cls(
            sigma=p.v0,
            vov=p.nu,
            mr=p.kappa,
            rho=p.rho,
            theta=p.theta,
            intr=fwd.r,
            divr=fwd.q,
        )

    return build_cached(_SV32_MODEL_CACHE, (p, fwd), _factory)


# ---------------------------------------------------------------------------
# Cumulants  -  numerical Cauchy integral
# ---------------------------------------------------------------------------


def sv32_cumulants(fwd: ForwardSpec, p: Sv32Params) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of ``X_T`` under the 3/2 SV model.

    The 3/2 CF has no simple closed-form cumulant formula (the Bessel
    function representation does not differentiate cleanly at u=0), so we
    use the standard numerical Cauchy-circle FFT that all PyFENG-backed
    models fall back to.
    """
    from ..utils.cumulants import cumulants_from_cf

    def _phi(u):
        return sv32_cf(u, fwd, p)

    c = cumulants_from_cf(_phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
