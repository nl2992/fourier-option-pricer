"""Discretely monitored barrier and lookback options by the fast Hilbert transform.

Feng & Linetsky (2008, 2009) price path-dependent options under Levy models
by backward induction in Fourier space: between monitoring dates the value
transform is multiplied by the step CF, and at each date it is *projected* onto
the live region. With the Fourier transform ``F f(xi) = int e^{i xi x} f(x) dx``,

    F[1_{x > l} f](xi) = 1/2 F f(xi) + i/2 e^{i xi l} H[e^{-i eta l} F f](xi),

``H g(xi) = (1/pi) p.v. int g(eta) / (xi - eta) d eta`` the Hilbert transform.
On the sinc grid ``xi_m = m h`` the Hilbert transform becomes the Toeplitz sum
``(H g)_k = sum_{m} g_m (1 - (-1)^{k-m}) / (pi (k - m))``, applied by FFT, with
an error that decays like ``exp(-pi d / h)`` for functions analytic in the
strip ``|Im xi| < d`` (Stenger 1993). Value functions are damped by
``e^{-alpha x}`` so that they are integrable; the damping only shifts the CF
argument, ``phi(-xi) -> phi(-xi - i alpha)``, and commutes with the projection.

Barriers (Feng & Linetsky 2008)
    Knock-outs by projection at each of ``M`` equally spaced monitoring dates
    (maturity included); knock-ins by in-out parity against the European.

Lookbacks (Feng & Linetsky 2009)
    With ``Z = log(M / S)`` (running maximum, including ``S_0``), ``Z`` follows
    the Lindley recursion ``Z' = (Z - Y)^+`` and ``M_T - S_T = S_T (e^{Z_T} - 1)``,
    so under the share measure the put is ``S_0 e^{-qT} (E[e^{Z_T}] - 1)``.
    The law of ``Z`` is propagated as a characteristic function: each step
    projects ``chi phi(-.)`` onto ``z > 0`` and returns the mass below zero as
    an atom at the origin. Two recursions run in lockstep: an untilted one that
    supplies the atom's mass, and an ``e^{z}``-tilted one whose value at
    ``xi = 0`` is ``E[e^{Z_T}]``. The call (``S_T - m_T``) is the mirror image.
    Fixed strikes use Spitzer's duality: the discrete running maximum of the
    log-price walk is distributed as ``Z' = (Z + Y)^+`` under Q, so
    ``E[(S_0 e^{Z_T} - K)^+]`` follows from the ``e^{alpha z}``-tilted CF of
    ``Z_T`` by Parseval (and the minimum mirrors it).

References
----------
* Feng, L. & Linetsky, V. (2008), Pricing discretely monitored barrier options
  and defaultable bonds in Levy process models: a fast Hilbert transform
  approach, *Mathematical Finance* 18(3), 337-384.
* Feng, L. & Linetsky, V. (2009), Computing exponential moments of the discrete
  maximum of a Levy process and lookback options, *Finance and Stochastics*
  13(4), 501-529.
* Stenger, F. (1993), *Numerical Methods Based on Sinc and Analytic Functions*.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY

__all__ = ["hilbert_barrier_price", "hilbert_lookback_price"]

_STRIP = 1.0  # analyticity half-width d of the damped transforms
_H_DEFAULT = np.pi * _STRIP / 32.0  # sinc error ~ exp(-pi d / h) = e^-32
_N_MIN, _N_MAX = 1 << 10, 1 << 17


class _SincHilbert:
    """Discrete Hilbert transform on ``xi_m = (m - N/2) h`` via a padded FFT."""

    def __init__(self, N: int, h: float):
        self.N, self.h = N, h
        self.xi = (np.arange(N) - N // 2) * h
        n = np.arange(-(N - 1), N)
        kernel = np.zeros(n.size)
        odd = n % 2 != 0
        kernel[odd] = 2.0 / (np.pi * n[odd])
        circ = np.zeros(2 * N)
        circ[n % (2 * N)] = kernel
        self._k_hat = np.fft.fft(circ)

    def __call__(self, g: np.ndarray) -> np.ndarray:
        conv = np.fft.ifft(self._k_hat * np.fft.fft(g, 2 * self.N))
        return conv[: self.N]

    def project_above(self, G: np.ndarray, level: float) -> np.ndarray:
        """Transform of ``1_{x > level} f`` from ``G = F f`` on the grid."""
        e = np.exp(1j * self.xi * level)
        return 0.5 * G + 0.5j * e * self(np.conj(e) * G)


def _step_cf(model: str, fwd: ForwardSpec, params, dt: float) -> Callable:
    """CF of the log-return ``log(S_{t+dt}/S_t)`` under Q (drift included)."""
    cf = MODEL_REGISTRY[model].cf
    fwd_dt = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=dt)
    drift = (fwd.r - fwd.q) * dt
    return lambda u: np.exp(1j * u * drift) * np.asarray(cf(u, fwd_dt, params), dtype=complex)


def _grid_size(step_cf: Callable, alpha: float, h: float, n: int | None) -> int:
    """Smallest power of two whose grid edge sees the damped step CF below 1e-15."""
    if n is not None:
        return int(n)
    ref = abs(complex(step_cf(np.array([-1j * alpha]))[0]))
    N = _N_MIN
    while N < _N_MAX:
        edge = np.array([-(N // 2) * h - 1j * alpha])
        if abs(complex(step_cf(edge)[0])) <= 1e-15 * ref:
            break
        N *= 2
    return N


def _exp_integral(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """``int_lo^hi e^{a x} dx`` for complex ``a``; an infinite end needs Re(a) of the right sign.

    Finite intervals use ``e^{a lo} (hi - lo) (e^z - 1)/z`` with ``z = a (hi - lo)``,
    which stays exact through the removable singularity at ``a = 0``.
    """
    if np.isinf(hi):
        return -np.exp(a * lo) / a
    if np.isinf(lo):
        return np.exp(a * hi) / a
    width = hi - lo
    z = a * width
    small = np.abs(z) < 1e-8
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(small, 1.0 + 0.5 * z, np.expm1(z) / np.where(small, 1.0, z))
    return np.exp(a * lo) * width * rel


def _payoff_transform(xi, alpha, lo, hi, k, cp):
    """``int_lo^hi e^{(i xi - alpha) x} cp (e^x - e^k) dx`` (infinite ends allowed)."""
    s = 1j * xi - alpha
    return cp * (_exp_integral(s + 1.0, lo, hi) - np.exp(k) * _exp_integral(s, lo, hi))


def hilbert_barrier_price(
    model: str,
    fwd: ForwardSpec,
    params,
    *,
    strike: float,
    barrier: float,
    maturity: float,
    barrier_type: str,
    cp: int = 1,
    n_monitor: int = 252,
    h: float | None = None,
    N: int | None = None,
) -> float:
    """Discretely monitored single-barrier option by the fast Hilbert transform.

    Parameters
    ----------
    model, fwd, params :
        Registry model (a 1-D Levy model: the increments must be i.i.d.),
        market inputs and parameters.
    strike, barrier, maturity :
        Contract terms.
    barrier_type : {"down_out", "up_out", "down_in", "up_in"}
    cp : int
        ``+1`` call, ``-1`` put.
    n_monitor : int
        Equally spaced monitoring dates ``T/M, 2T/M, ..., T``.
    h, N :
        Sinc step and grid size. Defaults: ``h = pi/32`` (sinc error ~e^-32
        for the unit analyticity strip of the damped transforms) and the
        smallest power of two at which the step CF has decayed to 1e-15.
        Pure-jump models with a slowly decaying short-step CF (e.g. VG with
        ``dt/nu`` small) converge only algebraically.

    Returns
    -------
    float
        Barrier option price.
    """
    from ..pricers.cos_bermudan import _SUPPORTED_MODELS

    if model not in _SUPPORTED_MODELS:
        raise NotImplementedError(
            f"hilbert_barrier: needs a 1-D Levy model {sorted(_SUPPORTED_MODELS)}; got {model!r}"
        )
    if barrier_type not in {"down_out", "up_out", "down_in", "up_in"}:
        raise ValueError(f"hilbert_barrier: unknown barrier_type {barrier_type!r}")
    if cp not in (1, -1):
        raise ValueError(f"cp must be +1 or -1; got {cp}")
    if n_monitor < 1:
        raise ValueError(f"n_monitor must be >= 1; got {n_monitor}")
    down = barrier_type.startswith("down")
    if (down and barrier >= fwd.S0) or (not down and barrier <= fwd.S0):
        raise ValueError("hilbert_barrier: the barrier must be on the far side of the spot")

    if barrier_type.endswith("_in"):
        european = _european(model, fwd, params, strike, maturity, cp)
        out = hilbert_barrier_price(
            model,
            fwd,
            params,
            strike=strike,
            barrier=barrier,
            maturity=maturity,
            barrier_type=barrier_type.replace("_in", "_out"),
            cp=cp,
            n_monitor=n_monitor,
            h=h,
            N=N,
        )
        return float(european - out)

    dt = maturity / n_monitor
    step_cf = _step_cf(model, fwd, params, dt)
    # Damping: live region x > l needs decay at +inf (calls grow like e^x);
    # live region x < u needs decay at -inf.
    alpha = (2.0 if cp == 1 else 1.0) if down else -1.0
    h = h or _H_DEFAULT
    N = _grid_size(step_cf, alpha, h, N)
    ht = _SincHilbert(N, h)
    xi = ht.xi

    level = float(np.log(barrier / fwd.S0))  # x = log(S / S0), spot at x = 0
    k = float(np.log(strike / fwd.S0))
    if down:
        lo, hi = (max(level, k), np.inf) if cp == 1 else (level, max(level, k))
    else:
        lo, hi = (min(level, k), level) if cp == 1 else (-np.inf, min(level, k))
    W = _payoff_transform(xi, alpha, lo, hi, k, cp) if hi > lo else np.zeros(N, dtype=complex)

    mult = np.exp(-fwd.r * dt) * step_cf(-xi - 1j * alpha)
    for _ in range(n_monitor - 1):
        G = mult * W
        above = ht.project_above(G, level)
        W = above if down else G - above
    value = (h / (2.0 * np.pi)) * np.sum(mult * W)
    return float(fwd.S0 * max(np.real(value), 0.0))


def _european(model, fwd, params, strike, maturity, cp) -> float:
    from ..pricers.contour import contour_price_at_strikes

    fwd_T = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=maturity)
    cf = MODEL_REGISTRY[model].cf
    return float(
        contour_price_at_strikes(lambda u: cf(u, fwd_T, params), fwd_T, [strike], cp=cp)[0]
    )


def _lindley_tilted_law(
    incr_cf: Callable, t: float, n_steps: int, h: float | None, N: int | None
) -> tuple[_SincHilbert, np.ndarray, float]:
    """``E[e^{t Z_n} e^{i xi Z_n}]`` on the sinc grid for ``Z' = (Z + D)^+``, ``Z_0 = 0``.

    ``incr_cf`` is the CF of the i.i.d. increment ``D``. An untilted recursion
    supplies the atom ``P(Z + D <= 0)`` that the tilted one returns to the
    origin (where ``e^{t z} = 1``). Also returns the final atom ``P(Z_n = 0)``.
    """
    h = h or _H_DEFAULT
    N = _grid_size(incr_cf, t, h, N)
    ht = _SincHilbert(N, h)
    origin = N // 2  # xi = 0
    chi = np.ones(N, dtype=complex)  # E[e^{i xi Z_0}] with Z_0 = 0
    tilted = np.ones(N, dtype=complex)
    step_plain = incr_cf(ht.xi)
    step_tilt = incr_cf(ht.xi - 1j * t)
    atom = 1.0
    for _ in range(n_steps):
        above = ht.project_above(chi * step_plain, 0.0)
        atom = 1.0 - float(np.real(above[origin]))
        chi = above + atom
        tilted = ht.project_above(tilted * step_tilt, 0.0) + atom
    return ht, tilted, atom


def hilbert_lookback_price(
    model: str,
    fwd: ForwardSpec,
    params,
    *,
    maturity: float,
    cp: int = -1,
    strike_type: str = "floating",
    strike: float | None = None,
    n_monitor: int = 252,
    h: float | None = None,
    N: int | None = None,
) -> float:
    """Discretely monitored lookback option by the fast Hilbert transform.

    The extremum runs over ``t_0 = 0, T/M, ..., T`` (``M = n_monitor``).

    * ``strike_type="floating"``: put (``cp=-1``) pays ``max S - S_T``, call
      pays ``S_T - min S``. ``Z = log(max/S)`` follows the Lindley recursion
      ``Z' = (Z - Y)^+``; under the share measure the put is
      ``S_0 e^{-qT} (E[e^{Z_T}] - 1)`` (the call mirrors it).
    * ``strike_type="fixed"``: call pays ``(max S - K)^+``, put ``(K - min S)^+``.
      By Spitzer's duality the running maximum of the log-price walk has the
      law of ``Z' = (Z + Y)^+`` under Q (the minimum mirrors it), and the
      payoff is integrated against the ``e^{alpha Z}``-tilted CF by Parseval.
    """
    from ..pricers.cos_bermudan import _SUPPORTED_MODELS

    if model not in _SUPPORTED_MODELS:
        raise NotImplementedError(
            f"hilbert_lookback: needs a 1-D Levy model {sorted(_SUPPORTED_MODELS)}; got {model!r}"
        )
    if cp not in (1, -1):
        raise ValueError(f"cp must be +1 or -1; got {cp}")
    if n_monitor < 1:
        raise ValueError(f"n_monitor must be >= 1; got {n_monitor}")
    if strike_type not in ("floating", "fixed"):
        raise ValueError(f"strike_type must be 'floating' or 'fixed'; got {strike_type!r}")

    dt = maturity / n_monitor
    q_cf = _step_cf(model, fwd, params, dt)

    if strike_type == "floating":
        growth = np.exp((fwd.r - fwd.q) * dt)
        s, t = (-1.0, 1.0) if cp == -1 else (1.0, -1.0)
        _, tilted, _ = _lindley_tilted_law(lambda u: q_cf(s * u - 1j) / growth, t, n_monitor, h, N)
        moment = float(np.real(tilted[tilted.size // 2]))  # E^S[e^{t Z_T}]
        share_value = fwd.S0 * np.exp(-fwd.q * maturity)
        return float(share_value * (moment - 1.0) if cp == -1 else share_value * (1.0 - moment))

    if strike is None or strike <= 0.0:
        raise ValueError("fixed-strike lookbacks need strike > 0")
    k = strike / fwd.S0
    disc = np.exp(-fwd.r * maturity)
    # Call: max X ~ (Z + Y)^+, payoff S0 (e^Z - k)^+.  Put: -min X ~ (Z - Y)^+,
    # payoff S0 (k - e^{-Z})^+.  alpha tilts the law so the damped payoff is integrable.
    sign, alpha = (1.0, 2.0) if cp == 1 else (-1.0, 1.0)
    if (cp == 1 and k <= 1.0) or (cp == -1 and k >= 1.0):
        # Payoff linear in e^{+-Z} on the whole support: one exact moment.
        _, tilted, _ = _lindley_tilted_law(lambda u: q_cf(sign * u), sign, n_monitor, h, N)
        moment = float(np.real(tilted[tilted.size // 2]))
        return float(disc * fwd.S0 * ((moment - k) if cp == 1 else (k - moment)))
    incr = lambda u: q_cf(sign * u)  # noqa: E731
    if N is None:
        # The continuous part of Z's law jumps at 0+, so its CF decays only like
        # 1/xi and the Parseval tail error falls like 1/xi_max^2; an 8x wider
        # grid keeps it near 1e-7 of the price.
        N = min(8 * _grid_size(incr, alpha, h or _H_DEFAULT, None), _N_MAX)
    ht, tilted, atom = _lindley_tilted_law(incr, alpha, n_monitor, h, N)
    z_star = float(np.log(k)) if cp == 1 else float(-np.log(k))
    sv = -1j * ht.xi - alpha  # damped payoff transform evaluated at -xi
    if cp == 1:
        payoff = _exp_integral(sv + 1.0, z_star, np.inf) - k * _exp_integral(sv, z_star, np.inf)
    else:
        payoff = k * _exp_integral(sv, z_star, np.inf) - _exp_integral(sv - 1.0, z_star, np.inf)
    # The atom of Z at 0 gives a non-decaying constant in its CF; integrate it
    # exactly (the payoff vanishes there) and sum only the continuous part.
    value = (ht.h / (2.0 * np.pi)) * np.sum((tilted - atom) * payoff)
    return float(disc * fwd.S0 * max(np.real(value), 0.0))
