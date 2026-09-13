"""COS Bermudan option pricing via Fang-Oosterlee (2009) backward induction.

Reference: Fang, F. & Oosterlee, C.W. (2009). Pricing Early-Exercise and
Discrete Barrier Options by Fourier-Cosine Series Expansions.
Numerische Mathematik 114(1), 27-62.

State variable
--------------
x = log(S_t / F_t)  where F_t = S_0 * exp((r-q)*t)

This centering keeps the distribution of x mean-centred for any Lévy model
and lets us reuse the same truncation interval [a, b] at every exercise date.

Key COS formula for the continuation value at position x:
    C(x) = e^{-r Δt} * Σ_k' Re[φ(ω_k) * exp(i ω_k (x−a))] * V_k
         = e^{-r Δt} * Σ_k' [Re(φ_k) cos(ω_k(x−a)) − Im(φ_k) sin(ω_k(x−a))] V_k

where ω_k = kπ/(b−a), ' means k=0 gets half weight, and V_k are the COS
coefficients of the value function.

At x = 0 this simplifies to:
    price = e^{-r Δt} * Σ_k' Re[φ(ω_k) * exp(−i ω_k a)] * V_k

Supported models (1-D Lévy, stationary independent increments):
    bsm, vg, cgmy, nig, kou, merton_jd, bilateral_gamma,
    generalized_hyperbolic, fmls, meixner
"""

from __future__ import annotations

import warnings

import numpy as np

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY
from ..products.american import AmericanOption
from ..products.bermudan import BermudanOption
from .cos import cos_auto_grid

_SUPPORTED_MODELS = frozenset(
    {
        "bsm",
        "vg",
        "cgmy",
        "nig",
        "kou",
        "merton_jd",
        "bilateral_gamma",
        "generalized_hyperbolic",
        "fmls",
        "meixner",
    }
)


def _check_model(model: str) -> None:
    if model not in MODEL_REGISTRY:
        raise ValueError(f"cos_bermudan: unknown model {model!r}")
    if model not in _SUPPORTED_MODELS:
        raise NotImplementedError(
            f"cos_bermudan: model {model!r} is not supported for 1-D COS Bermudan. "
            "For heston, bates and regime_switching use method='cos_ctmc' "
            "(foureng.pricers.cos_ctmc)."
        )


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


def _fo2009_backward(
    cf_dt,
    S0: float,
    r: float,
    q: float,
    K: float,
    cp: int,
    exercise_times: np.ndarray,
    a: float,
    b: float,
    N: int,
) -> float:
    """Fang-Oosterlee (2009) backward induction on the COS coefficients.

    State ``x = log(S_t / F_t)``; ``cf_dt(omega, dt)`` is the CF of the
    increment of ``x`` over ``dt``. At each exercise date the early-exercise
    point ``x*`` solves ``c(x*) = g(x*)`` (safeguarded Newton on the COS series),
    and the value coefficients split exactly at ``x*``: payoff coefficients
    in closed form (``chi``/``psi``) on the exercise region, and continuation
    coefficients ``Re(M u)`` on the rest, where ``M = Hankel + Toeplitz`` is
    applied by FFT in ``O(N log N)``. No spatial grid is involved, so the error
    is controlled by ``N`` and ``[a, b]`` alone.
    """
    bma = b - a
    k = np.arange(N, dtype=float)
    w = k * np.pi / bma
    n_all = np.arange(-(N - 1), 2 * N - 1, dtype=float)  # j + k and j - k indices
    w_all = n_all * np.pi / bma
    nz = n_all != 0.0
    fft_len = _next_pow2(3 * N - 2)

    def fwd_t(t: float) -> float:
        return S0 * np.exp((r - q) * t)

    def chi(c: float, d: float) -> np.ndarray:
        wc, wd = w * (c - a), w * (d - a)
        ec, ed = np.exp(c), np.exp(d)
        return (np.cos(wd) * ed - np.cos(wc) * ec + w * (np.sin(wd) * ed - np.sin(wc) * ec)) / (
            1.0 + w * w
        )

    def psi(c: float, d: float) -> np.ndarray:
        out = np.empty(N)
        out[0] = d - c
        out[1:] = (np.sin(w[1:] * (d - a)) - np.sin(w[1:] * (c - a))) / w[1:]
        return out

    def payoff_coeffs(c: float, d: float, F: float) -> np.ndarray:
        """(2/(b-a)) * int_c^d cp (F e^x - K) cos(w_k (x - a)) dx."""
        if d <= c:
            return np.zeros(N)
        return (2.0 / bma) * cp * (F * chi(c, d) - K * psi(c, d))

    def cont_coeffs(u: np.ndarray, c: float, d: float) -> np.ndarray:
        """(2/(b-a)) * int_c^d Re[sum_j u_j e^{i w_j (x-a)}] cos(w_k (x-a)) dx."""
        if d <= c:
            return np.zeros(N)
        E = np.empty(n_all.size, dtype=complex)
        E[nz] = (np.exp(1j * w_all[nz] * (d - a)) - np.exp(1j * w_all[nz] * (c - a))) / (
            1j * w_all[nz]
        )
        E[~nz] = d - c
        R = np.fft.fft(u[::-1], fft_len)
        hankel = np.fft.ifft(R * np.fft.fft(E[N - 1 :], fft_len))[N - 1 : 2 * N - 1]
        toeplitz = np.fft.ifft(R * np.fft.fft(E[: 2 * N - 1], fft_len))[N - 1 : 2 * N - 1]
        return np.real(hankel + toeplitz[::-1]) / bma

    def exercise_point(u: np.ndarray, disc: float, F: float) -> float:
        """Root of c(x) - g(x) on the side of the strike where exercise can pay."""
        x_k = np.log(K / F)

        def f_and_fp(x: float) -> tuple[float, float]:
            e = u * np.exp(1j * w * (x - a))
            cont = disc * float(np.real(e.sum()))
            dcont = disc * float(np.real((1j * w * e).sum()))
            ex = F * np.exp(x)
            return cont - cp * (ex - K), dcont - cp * ex

        if cp == -1:
            lo, hi = a, min(x_k, b)
            if hi <= a or f_and_fp(a)[0] >= 0.0:
                return a  # never optimal to exercise on [a, b]
            if f_and_fp(hi)[0] < 0.0:
                return hi
        else:
            lo, hi = max(x_k, a), b
            if lo >= b or f_and_fp(b)[0] >= 0.0:
                return b
            if f_and_fp(lo)[0] < 0.0:
                return lo
        x = 0.5 * (lo + hi)
        for _ in range(100):
            fx, fpx = f_and_fp(x)
            if (fx < 0.0) == (cp == -1):
                lo = x
            else:
                hi = x
            x_new = x - fx / fpx if fpx != 0.0 else 0.5 * (lo + hi)
            if not (lo < x_new < hi):
                x_new = 0.5 * (lo + hi)
            if abs(x_new - x) <= 1e-14 * max(1.0, abs(x)) or hi - lo <= 1e-14:
                return x_new
            x = x_new
        return x

    cf_cache: dict[float, np.ndarray] = {}

    def phi(dt: float) -> np.ndarray:
        key = round(dt, 14)
        if key not in cf_cache:
            cf_cache[key] = cf_dt(w, dt)
        return cf_cache[key]

    # Terminal coefficients: the payoff on its support inside [a, b].
    t_last = float(exercise_times[-1])
    x_k = float(np.log(K / fwd_t(t_last)))
    if cp == -1:
        V = payoff_coeffs(a, min(max(x_k, a), b), fwd_t(t_last))
    else:
        V = payoff_coeffs(min(max(x_k, a), b), b, fwd_t(t_last))

    for j in range(len(exercise_times) - 2, -1, -1):
        t_curr = float(exercise_times[j])
        dt = float(exercise_times[j + 1]) - t_curr
        if dt < 1e-12:
            continue
        u = phi(dt) * V
        u[0] *= 0.5
        disc = float(np.exp(-r * dt))
        F = fwd_t(t_curr)
        x_star = exercise_point(u, disc, F)
        if cp == -1:
            V = disc * cont_coeffs(u, x_star, b) + payoff_coeffs(a, x_star, F)
        else:
            V = disc * cont_coeffs(u, a, x_star) + payoff_coeffs(x_star, b, F)

    t_first = float(exercise_times[0])
    u = (phi(t_first) if t_first > 1e-12 else np.ones(N, dtype=complex)) * V
    u[0] *= 0.5
    price = np.exp(-r * t_first) * float(np.real((u * np.exp(-1j * w * a)).sum()))
    return max(price, 0.0)


def _model_cf_dt(entry, params, S0: float, r: float, q: float):
    def cf_dt(omega: np.ndarray, dt: float) -> np.ndarray:
        return entry.cf(omega, ForwardSpec(S0=S0, r=r, q=q, T=dt), params)

    return cf_dt


def cos_bermudan_price(
    model: str,
    fwd: ForwardSpec,
    params,
    product: BermudanOption,
    *,
    grid=None,
    n_spatial: int | None = None,
    N: int = 256,
    L: float = 12.0,
) -> float:
    """Price a Bermudan option via FO2009 COS backward induction.

    Implements Fang & Oosterlee (2009) exactly on the COS coefficients: the
    early-exercise point is found by Newton at each date, payoff coefficients
    are analytic, and continuation coefficients use the Hankel + Toeplitz
    FFT product. Converges exponentially in ``N`` for smooth densities.

    Parameters
    ----------
    model : str
        One of the supported 1-D Lévy models.
    fwd : ForwardSpec
        Market inputs.  ``fwd.T`` is used only for grid sizing.
    params :
        Model parameter dataclass.
    product : BermudanOption
        Bermudan spec (strike, maturity, cp, exercise_times).
    grid :
        Optional pre-built :class:`~foureng.utils.grids.COSGrid`.
    n_spatial : int, optional
        Deprecated and ignored: the pricer no longer uses a spatial grid.
    N : int
        COS terms (used only when ``grid`` is None). Default 256. Short
        exercise intervals need more terms: the transition CF over ``dt``
        must have decayed by ``omega_N = N pi / (b - a)``.
    L : float
        Truncation multiplier (used only when ``grid`` is None).

    Returns
    -------
    float
        Bermudan option price.
    """
    if n_spatial is not None:
        warnings.warn(
            "cos_bermudan_price: n_spatial is deprecated and ignored; the FO2009 "
            "scheme works on COS coefficients without a spatial grid.",
            DeprecationWarning,
            stacklevel=2,
        )
    _check_model(model)
    entry = MODEL_REGISTRY[model]

    T = product.maturity
    exercise_times = np.sort(np.asarray(product.exercise_times, dtype=float))
    # Ensure maturity is included as the final exercise date
    if not np.isclose(exercise_times[-1], T, rtol=1e-8):
        exercise_times = np.sort(np.append(exercise_times, T))

    # Build COS grid from cumulants at maturity T
    if grid is None:
        fwd_T = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=T)
        grid = cos_auto_grid(entry.cumulants(fwd_T, params), N=N, L=L)

    return _fo2009_backward(
        _model_cf_dt(entry, params, fwd.S0, fwd.r, fwd.q),
        fwd.S0,
        fwd.r,
        fwd.q,
        product.strike,
        product.cp,
        exercise_times,
        grid.a,
        grid.b,
        grid.N,
    )


def cos_bermudan_price_strip(
    model: str,
    fwd: ForwardSpec,
    params,
    strikes: np.ndarray,
    maturity: float,
    exercise_times: np.ndarray,
    cp: int = -1,
    *,
    grid=None,
    n_spatial: int | None = None,
    N: int = 256,
    L: float = 12.0,
) -> np.ndarray:
    """Price a strip of Bermudan options at different strikes."""
    strikes = np.asarray(strikes, dtype=float)
    _check_model(model)

    if grid is None:
        fwd_T = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=maturity)
        cums = MODEL_REGISTRY[model].cumulants(fwd_T, params)
        grid = cos_auto_grid(cums, N=N, L=L)

    prices = np.empty(len(strikes))
    for i, K in enumerate(strikes):
        product = BermudanOption(
            strike=float(K),
            maturity=maturity,
            cp=cp,
            exercise_times=exercise_times,
        )
        prices[i] = cos_bermudan_price(
            model,
            fwd,
            params,
            product,
            grid=grid,
            n_spatial=n_spatial,
        )
    return prices


def _terms_for_step(cf_dt, dt: float, a: float, b: float, n_min: int, n_max: int) -> int:
    """Smallest power of two whose top frequency has let the ``dt`` CF decay."""
    n = n_min
    while n < n_max:
        omega = np.array([(n - 1) * np.pi / (b - a)])
        if abs(complex(cf_dt(omega, dt)[0])) <= 1e-12:
            break
        n *= 2
    return n


def cos_american_price(
    model: str,
    fwd: ForwardSpec,
    params,
    product: AmericanOption,
    *,
    base_dates: int = 64,
    grid=None,
    N: int | None = None,
    L: float = 10.0,
    n_max: int = 4096,
) -> float:
    """American option by 4-point Richardson extrapolation of COS Bermudans.

    Fang & Oosterlee (2009), section 5: price Bermudans with ``m, 2m, 4m, 8m``
    equally spaced exercise dates (``m = base_dates``) by the FO2009 backward
    induction and eliminate the ``1/M``, ``1/M^2`` and ``1/M^3`` terms of the
    Bermudan-to-American error,

        v_AM = (64 v(8m) - 56 v(4m) + 14 v(2m) - v(m)) / 21.

    The result is floored at the finest Bermudan and at immediate exercise,
    both of which the American value dominates by definition.

    Parameters
    ----------
    model : str
        One of the supported 1-D Lévy models.
    fwd : ForwardSpec
        Market inputs (``S0``, ``r``, ``q``).
    params :
        Model parameter dataclass.
    product : AmericanOption
        Strike, maturity and ``cp``.
    base_dates : int, default 64
        Exercise dates of the coarsest Bermudan. The extrapolation is only
        asymptotic once ``m`` is moderately large; 64 gives ~1e-6 relative
        accuracy on a BSM put against an extrapolated 40k-step binomial tree.
    grid : COSGrid, optional
        Fixes the truncation interval and number of terms for every level.
    N : int, optional
        COS terms for every level. By default each level picks the smallest
        power of two (up to ``n_max``) at which the CF over its exercise
        interval has decayed below 1e-12.
    L : float, default 10.0
        Truncation multiplier for the cumulant-based interval.
    n_max : int, default 4096
        Cap on the automatic term count (pure-jump models with slowly
        decaying short-horizon CFs converge algebraically and hit the cap).

    Returns
    -------
    float
        American option price.
    """
    _check_model(model)
    if base_dates < 1:
        raise ValueError(f"cos_american_price: base_dates must be >= 1, got {base_dates}")
    entry = MODEL_REGISTRY[model]
    T, K, cp = product.maturity, product.strike, product.cp
    cf_dt = _model_cf_dt(entry, params, fwd.S0, fwd.r, fwd.q)
    if grid is None:
        fwd_T = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=T)
        grid = cos_auto_grid(entry.cumulants(fwd_T, params), N=N or 256, L=L)
        fixed_n = N
    else:
        fixed_n = grid.N
    a, b = grid.a, grid.b

    values = []
    for level in range(4):
        M = base_dates * 2**level
        n_terms = fixed_n or _terms_for_step(cf_dt, T / M, a, b, 256, n_max)
        times = T * np.arange(1, M + 1) / M
        values.append(_fo2009_backward(cf_dt, fwd.S0, fwd.r, fwd.q, K, cp, times, a, b, n_terms))
    v1, v2, v4, v8 = values
    american = (64.0 * v8 - 56.0 * v4 + 14.0 * v2 - v1) / 21.0
    return float(max(american, v8, cp * (fwd.S0 - K), 0.0))
