"""Two-asset European options by Fourier inversion of the joint CF.

Write ``X_j = log(S_j(T) / K)`` so that the payoffs scale out of the strike.
If ``P`` has the (complex) Fourier transform ``P^(u) = int e^{-i u . x} P(x) dx``
on the line ``Im u = eps``, then

    V = K e^{-rT} (2 pi)^{-2} int_{R^2 + i eps} Phi_X(u) P^(u) du,

evaluated with the trapezoid rule, which converges exponentially because the
integrand is analytic in a strip around the contour.

Spread ``(e^{x1} - e^{x2} - 1)^+`` (Hurd & Zhou 2010, Theorem 1)
    ``P^(u) = Gamma(i(u1 + u2) - 1) Gamma(-i u2) / Gamma(i u1 + 1)``,
    for ``eps2 > 0`` and ``eps1 + eps2 < -1``.
Call on the minimum ``(min(e^{x1}, e^{x2}) - 1)^+``
    ``P^(u) = -1 / (z1 z2 (1 + z1 + z2))`` with ``z = -i u``, for
    ``eps1 < 0``, ``eps2 < 0`` and ``eps1 + eps2 < -1``. (Split the quadrant
    at ``x1 = x2`` and integrate each half.)
Exchange ``(S1 - S2)^+``
    One dimension is enough: with ``S2`` as numeraire ``Z = log(S1/S2)`` has
    CF ``Phi(w, -w - i)``, and ``(e^z - 1)^+`` has transform ``1 / (z (z + 1))``
    (``z = -i w``, ``Im w < -1``). Vanilla calls on either asset use the same
    one-dimensional formula on the marginal CF.

The rest follows by static replication: ``max = S1 + S2 - min`` gives the call
on the maximum, and parity gives the puts.

The grid is chosen from the CF itself: the step from the width of the damped
density (so that its periodic images do not overlap) and the cutoff from where
the integrand has decayed below ``tol``.

Reference
---------
Hurd, T. R. & Zhou, Z. (2010), A Fourier transform method for spread option
pricing, *SIAM Journal on Financial Mathematics* 1, 142-157.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import loggamma

from ..models.base import ForwardSpec
from ..models.joint import joint_log_cf

_SPREAD_EPS = (-3.0, 1.0)
_MIN_EPS = (-1.5, -1.5)
_EPS_1D = -2.0
# The offset is scaled up by these factors when that needs fewer nodes: a
# larger offset makes the damped payoff decay faster (shorter period) but
# inflates the integrand, so scales whose peak exceeds _MAG_MAX are skipped to
# keep cancellation error near tol.
_EPS_SCALES = (1.0, 1.5, 2.0, 3.0, 4.0)
_MAG_MAX = 1e3


@dataclass(frozen=True)
class Fourier2DGrid:
    """Controls for the two-asset Fourier pricer.

    Parameters
    ----------
    tol : float
        Target size of the neglected tail of the integrand (relative to the
        strike). The cutoff ``u_max`` is set where the integrand drops below it.
    n_max : int
        Cap on nodes per dimension (memory and time grow like ``n_max^2``).
    width : float
        Number of standard deviations of the damped density the grid resolves.
    eps : tuple or None
        Contour offset ``Im u``. By default it is ``(-3, 1)`` for spreads
        (Hurd and Zhou's choice) and ``(-1.5, -1.5)`` for the minimum, scaled
        by up to 4 when that needs fewer nodes. It needs the moment
        ``E[exp(-eps . X)]`` to be finite.

    Notes
    -----
    The trapezoid sum equals the integral plus aliased copies of the damped
    payoff shifted by the period ``2 pi / eta`` in log space. The damped payoff
    decays like ``exp(-a |x|)``, with ``a`` set by ``eps`` (``a = 1`` for the
    spread default, ``1.5`` for the minimum), so the period is at least
    ``log(1 / tol) / a`` as well as wide enough for the density.
    """

    tol: float = 1e-12
    n_max: int = 2048
    width: float = 12.0
    eps: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not (self.tol > 0 and self.n_max >= 16 and self.width > 0):
            raise ValueError("Fourier2DGrid: need tol > 0, n_max >= 16 and width > 0")


class _Joint:
    """Joint CF of ``X_j = log(S_j(T) / K)`` and its moments."""

    def __init__(self, model, fwd: ForwardSpec, params, spot2: float, q2: float, K: float):
        if not (np.isfinite(spot2) and spot2 > 0):
            raise ValueError(f"two-asset pricer: spot2 must be > 0; got {spot2}")
        self.model, self.params, self.T = model, params, float(fwd.T)
        self.F1 = fwd.F0
        self.F2 = spot2 * np.exp((fwd.r - q2) * fwd.T)
        self.disc = fwd.disc
        self.K = K
        # mean and covariance of the log-forward returns (only used to size the grid)
        h = 1e-3
        c1, var = [], []
        for e in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
            lp = self.log_cf0(h * e[0], h * e[1])
            lm = self.log_cf0(-h * e[0], -h * e[1])
            c1.append(float(np.imag(lp - lm)) / (2 * h))
            var.append(float(-np.real(lp + lm)) / (h * h))
        self.c1 = np.array(c1[:2])
        cov12 = 0.5 * (var[2] - var[0] - var[1])
        cov = np.array([[var[0], cov12], [cov12, var[1]]])
        self.c2 = np.maximum(np.diag(cov), 1e-12)
        # principal axes: the CF decays slowest along the narrow one
        # (scaled to unit max-norm, so that t is the half-width of the square grid)
        self.axes = [tuple(v / np.max(np.abs(v))) for v in np.linalg.eigh(cov)[1].T]

    def log_cf0(self, u1, u2):
        return joint_log_cf(self.model, u1, u2, self.T, self.params)

    def cf(self, u1, u2, x0) -> np.ndarray:
        return np.exp(1j * (u1 * x0[0] + u2 * x0[1]) + self.log_cf0(u1, u2))

    def moment_ok(self, eps1: float, eps2: float) -> bool:
        m = self.log_cf0(np.array([1j * eps1]), np.array([1j * eps2]))[0]
        return bool(np.isfinite(m) and abs(np.imag(m)) < 1e-8 * max(1.0, abs(m)))

    def check_moment(self, eps1: float, eps2: float) -> None:
        if not self.moment_ok(eps1, eps2):
            raise ValueError(
                f"two-asset pricer: E[exp(-eps . X)] is not finite for eps = ({eps1}, {eps2}); "
                "pass a Fourier2DGrid with a smaller eps"
            )


def _step(density_half_width: float, x0_max: float, rate: float, tol: float) -> float:
    """Node spacing whose period ``2 pi / eta`` clears both alias sources."""
    period = max(2.0 * density_half_width, (np.log(1.0 / tol) + max(x0_max, 0.0) + 2.0) / rate)
    return 2.0 * np.pi / period


def _cutoff(mag, eta: float, tol: float, dim: int, n_max: int) -> float:
    """Smallest ``U`` beyond which ``mag(t) t^(dim-1)`` stays below ``tol`` (sampled)."""
    t = eta * np.geomspace(1.0, n_max, 400)
    g = mag(t) * t ** (dim - 1)
    above = np.nonzero(~(g < tol))[0]
    return float(t[min(above[-1] + 1, len(t) - 1)]) if len(above) else float(t[0])


def _integrate_2d(integrand, eta: float, n: int, rows: int = 64) -> tuple[float, float]:
    """Trapezoid sum over the grid ``eta k``, ``|k| <= n/2``, in both axes.

    The payoff and the density are real, so the integrand at ``-u`` is the
    conjugate of the one at ``u`` and only the half-plane ``u1 >= 0`` is summed.
    Also returns the sum over the inner 80% of the square, a truncation check.
    """
    m = n // 2
    k_in = int(0.8 * m)
    k2 = np.arange(-m, m + 1)
    u2 = eta * k2[None, :]
    inner_cols = np.abs(k2) <= k_in
    total = inner = 0.0
    for start in range(0, m + 1, rows):
        k1 = np.arange(start, min(start + rows, m + 1))
        vals = np.real(integrand(eta * k1[:, None], u2))
        vals *= np.where(k1 == 0, 1.0, 2.0)[:, None]
        total += float(np.sum(vals))
        inner += float(np.sum(vals[k1 <= k_in][:, inner_cols]))
    c = eta * eta / (4.0 * np.pi**2)
    return total * c, inner * c


def _grid_2d(J: _Joint, x0, eps, rate, pay_hat, grid: Fourier2DGrid):
    """Step, node count (uncapped) and integrand envelope for offset ``eps``."""
    s = np.sqrt(J.c2)
    tilt = np.abs(np.array(eps)) * J.c2
    half = float(np.max(np.abs(x0 + J.c1) + tilt + grid.width * s))
    eta = _step(half, float(np.max(np.abs(x0))), rate, grid.tol)
    dirs = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0), *J.axes]

    def mag(t):
        out = np.zeros_like(t)
        for d in dirs:
            w1 = t * d[0] + 1j * eps[0]
            w2 = t * d[1] + 1j * eps[1]
            val = np.abs(J.cf(w1, w2, x0) * pay_hat(w1, w2))
            out = np.maximum(out, np.nan_to_num(val, nan=np.inf))
        return out

    u_max = _cutoff(mag, eta, grid.tol, 2, 16 * grid.n_max)
    return eta, int(2 * np.ceil(u_max / eta) + 2), mag


def _spread_rate(eps) -> float:
    """Slowest exponential decay rate of the damped spread payoff."""
    return min(-eps[0] - 1.0, eps[1], -1.0 - eps[0] - eps[1])


def _min_rate(eps) -> float:
    return min(-eps[0], -eps[1], -1.0 - eps[0] - eps[1])


def _spread_hat(w1, w2):
    return np.exp(loggamma(1j * (w1 + w2) - 1.0) + loggamma(-1j * w2) - loggamma(1j * w1 + 1.0))


def _min_hat(w1, w2):
    z1, z2 = -1j * w1, -1j * w2
    return -1.0 / (z1 * z2 * (1.0 + z1 + z2))


def _price_2d(J: _Joint, pay_hat, base_eps, rate_of, grid: Fourier2DGrid) -> float:
    x0 = np.log(np.array([J.F1, J.F2]) / J.K)
    if grid.eps is not None:
        J.check_moment(*grid.eps)
        candidates = [tuple(grid.eps)]
    else:
        J.check_moment(*base_eps)
        candidates = [(lam * base_eps[0], lam * base_eps[1]) for lam in _EPS_SCALES]
    # payoff scale in strike units, so the cancellation allowance is relative
    scale = max(1.0, (J.F1 + J.F2) / J.K)
    best = None
    for k, e in enumerate(candidates):
        if k > 0:
            if not J.moment_ok(*e):
                break
            peak = abs(J.cf(np.array([1j * e[0]]), np.array([1j * e[1]]), x0)[0])
            peak *= abs(pay_hat(np.array([1j * e[0]]), np.array([1j * e[1]]))[0])
            if not peak < _MAG_MAX * scale:
                break
        eta_k, n_k, mag_k = _grid_2d(J, x0, e, rate_of(e), pay_hat, grid)
        if best is None or n_k < best[2]:
            best = (e, eta_k, n_k, mag_k)
    assert best is not None
    eps, eta, n, mag = best
    capped = n > grid.n_max
    n = min(n, grid.n_max)

    def integrand(u1, u2):
        w1 = u1 + 1j * eps[0]
        w2 = u2 + 1j * eps[1]
        return J.cf(w1, w2, x0) * pay_hat(w1, w2)

    total, inner = _integrate_2d(integrand, eta, n)
    if capped and abs(total - inner) > grid.tol * scale:
        warnings.warn(
            f"two-asset pricer: the Fourier grid hit n_max = {grid.n_max} before the "
            f"integrand decayed (tail change {abs(total - inner) * J.K:.1e}); "
            "raise Fourier2DGrid.n_max for full accuracy.",
            RuntimeWarning,
            stacklevel=3,
        )
    return J.K * J.disc * total


def _call_1d(log_cf, x0: float, c1: float, c2: float, tol: float) -> float:
    """``E[(e^{x0 + Y} - 1)^+]`` for a log-return ``Y`` with log-CF ``log_cf``."""
    eps = _EPS_1D
    half = abs(x0 + c1) + abs(eps) * c2 + 14.0 * np.sqrt(c2)
    eta = _step(half, abs(x0), abs(eps) - 1.0, tol)

    def integrand(v):
        w = v + 1j * eps
        z = -1j * w
        return np.exp(1j * w * x0 + log_cf(w)) / (z * (z + 1.0))

    u_max = _cutoff(lambda t: np.abs(integrand(t)), eta, tol * 1e-2, 1, 1 << 20)
    v = eta * np.arange(0, int(np.ceil(u_max / eta)) + 1)
    vals = np.real(integrand(v))
    return float(2.0 * np.sum(vals) - vals[0]) * eta / (2.0 * np.pi)


def _vanilla_call(J: _Joint, j: int, K: float, tol: float) -> float:
    x0 = float(np.log((J.F1, J.F2)[j] / K))
    J.check_moment(_EPS_1D if j == 0 else 0.0, _EPS_1D if j == 1 else 0.0)

    def log_cf(w):
        return J.log_cf0(w, 0.0 * w) if j == 0 else J.log_cf0(0.0 * w, w)

    return K * J.disc * _call_1d(log_cf, x0, J.c1[j], J.c2[j], tol)


def _exchange(J: _Joint, tol: float) -> float:
    # Under the asset-2 measure Z = log(S1_T / S2_T) has log-CF Phi(w, -w - i).
    J.check_moment(_EPS_1D, -_EPS_1D - 1.0)

    def log_cf(w):
        return J.log_cf0(w, -w - 1j)

    # moments of Z under the asset-2 measure, again only for the grid size
    h = 1e-3
    lp, lm = log_cf(np.array([h + 0j])), log_cf(np.array([-h + 0j]))
    c1 = float(np.imag(lp - lm)[0]) / (2 * h)
    c2 = max(float(-np.real(lp + lm)[0]) / (h * h), 1e-12)
    x0 = float(np.log(J.F1 / J.F2))
    return J.F2 * J.disc * _call_1d(log_cf, x0, c1, c2, tol)


def fourier_exchange_price(
    model: str,
    fwd: ForwardSpec,
    params,
    *,
    spot2: float,
    q2: float = 0.0,
    grid: Fourier2DGrid | None = None,
) -> float:
    """Exchange option ``(S1_T - S2_T)^+`` under a two-asset model.

    Asset 1 is described by ``fwd`` (spot, rate, dividend yield and maturity),
    asset 2 by ``spot2`` and ``q2``; the dynamics come from ``params``.
    """
    grid = grid or Fourier2DGrid()
    J = _Joint(model, fwd, params, spot2, q2, 1.0)
    return _exchange(J, grid.tol)


def fourier_spread_price(
    model: str,
    fwd: ForwardSpec,
    params,
    *,
    spot2: float,
    strike: float,
    q2: float = 0.0,
    cp: int = 1,
    grid: Fourier2DGrid | None = None,
) -> float:
    """Spread option on ``S1_T - S2_T - K`` (Hurd & Zhou 2010).

    ``strike = 0`` is the exchange option. ``cp = -1`` prices the put by parity.
    """
    if cp not in (1, -1):
        raise ValueError(f"fourier_spread_price: cp must be +1 or -1, got {cp}")
    if not (np.isfinite(strike) and strike >= 0):
        raise ValueError(f"fourier_spread_price: strike must be >= 0, got {strike}")
    grid = grid or Fourier2DGrid()
    if strike == 0.0:
        J = _Joint(model, fwd, params, spot2, q2, 1.0)
        call = _exchange(J, grid.tol)
    else:
        J = _Joint(model, fwd, params, spot2, q2, strike)
        eps = grid.eps or _SPREAD_EPS
        if not (eps[1] > 0 and eps[0] + eps[1] < -1):
            raise ValueError("fourier_spread_price: need eps2 > 0 and eps1 + eps2 < -1")
        call = _price_2d(J, _spread_hat, _SPREAD_EPS, _spread_rate, grid)
    if cp == 1:
        return call
    return call - J.disc * (J.F1 - J.F2 - strike)


def fourier_rainbow_price(
    model: str,
    fwd: ForwardSpec,
    params,
    *,
    spot2: float,
    strike: float,
    q2: float = 0.0,
    cp: int = 1,
    kind: str = "max",
    grid: Fourier2DGrid | None = None,
) -> float:
    """Call or put on ``max(S1_T, S2_T)`` or ``min(S1_T, S2_T)`` with strike ``K > 0``.

    The call on the minimum comes from its 2-D transform; the call on the
    maximum from ``C1 + C2 - C_min``; puts from parity.
    """
    if cp not in (1, -1):
        raise ValueError(f"fourier_rainbow_price: cp must be +1 or -1, got {cp}")
    if kind not in ("max", "min"):
        raise ValueError(f"fourier_rainbow_price: kind must be 'max' or 'min', got {kind!r}")
    if not (np.isfinite(strike) and strike > 0):
        raise ValueError(f"fourier_rainbow_price: strike must be > 0, got {strike}")
    grid = grid or Fourier2DGrid()
    J = _Joint(model, fwd, params, spot2, q2, strike)
    eps = grid.eps or _MIN_EPS
    if not (eps[0] < 0 and eps[1] < 0 and eps[0] + eps[1] < -1):
        raise ValueError("fourier_rainbow_price: need eps1 < 0, eps2 < 0 and eps1 + eps2 < -1")
    call_min = _price_2d(J, _min_hat, _MIN_EPS, _min_rate, grid)
    if kind == "min" and cp == 1:
        return call_min
    # E[min] = F1 - E[(S1 - S2)^+]
    exch = _exchange(J, grid.tol)
    mean_min = J.F1 - exch / J.disc
    if kind == "min":
        return call_min - J.disc * (mean_min - strike)
    call_max = (
        _vanilla_call(J, 0, strike, grid.tol) + _vanilla_call(J, 1, strike, grid.tol) - call_min
    )
    if cp == 1:
        return call_max
    mean_max = J.F1 + J.F2 - mean_min
    return call_max - J.disc * (mean_max - strike)


__all__ = [
    "Fourier2DGrid",
    "fourier_exchange_price",
    "fourier_rainbow_price",
    "fourier_spread_price",
]
