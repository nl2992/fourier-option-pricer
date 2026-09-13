"""Bermudan, American and discrete barrier options under stochastic volatility.

The variance is replaced by a continuous-time Markov chain (CTMC) on a
non-uniform grid ``v_1 < ... < v_m``, and the log-price is handled by the COS
method, following the hybrid construction of Cui, Kirkby & Nguyen (2018).

Decorrelation
    With ``x = log(S_t / F_t)`` and ``y = x - (rho / nu)(v - v0)``, Heston gives
    ``dy = (-v/2 - (rho/nu) kappa (theta - v)) dt + sqrt(1 - rho^2) sqrt(v) dW``
    with ``W`` independent of the variance. Given the variance path ``y`` is a
    Gaussian process, so on the chain it is a regime-switching Levy process
    with per-state exponents
    ``psi_j(u) = i u (-v_j/2 - (rho/nu) kappa (theta - v_j)) - (1 - rho^2) v_j u^2 / 2``
    (Bates adds its compensated jump exponent to every state).
Transition
    ``Phi_ij(u) = E[exp(i u (y_{t+dt} - y_t)); v_{t+dt} = v_j | v_t = v_i]
    = [expm(dt (Q + diag psi(u)))]_ij`` (Feynman-Kac for the chain).
Backward induction
    The Fang-Oosterlee (2009) COS recursion runs with one coefficient vector per
    variance state: ``U_i(w_k) = sum_j Phi_ij(w_k) V_j(w_k)`` couples the
    states, and in state ``i`` the payoff is ``cp (F e^{y + c_i} - K)`` with
    ``c_i = (rho/nu)(v_i - v0)``. Exercise points and knock-out intervals are
    found per state, and coefficients split there exactly (Hankel + Toeplitz by
    FFT), as in the one-dimensional :mod:`cos_bermudan`.

Generator
    Neighbour rates match the CIR drift ``kappa (theta - v)`` and variance
    ``nu^2 v`` locally; where that would give a negative rate the drift is
    taken upwind. The grid is stretched towards ``v0`` (which is a node) and
    spans the variance distribution over the option's life.

The regime-switching model ``regime_switching`` is already a chain, so it is
priced with no approximation (its own generator and exponents, ``c_i = 0``).

References
----------
Cui, Z., Kirkby, J. L. & Nguyen, D. (2018), A general valuation framework for
SABR and stochastic local volatility models, *SIAM Journal on Financial
Mathematics* 9(2), 520-563.

Fang, F. & Oosterlee, C. W. (2011), A Fourier-based valuation method for
Bermudan and barrier options under Heston's model, *SIAM Journal on Financial
Mathematics* 2, 439-463.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.fft as sfft
from scipy.linalg import expm

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY

TERM_TOL = 1e-10
CTMC_MODELS = ("heston", "bates", "regime_switching")


@dataclass(frozen=True)
class CTMCVarianceGrid:
    """Controls for the variance chain and the COS expansion.

    Parameters
    ----------
    n_states : int
        Number of variance states (ignored for ``regime_switching``).
    richardson : bool
        Also price with ``2 n_states`` states and extrapolate (the chain error
        is second order in the spacing).
    tail : float
        The variance grid spans the ``tail`` to ``1 - tail`` quantiles of ``v_T``.
    N : int or None
        COS terms; by default chosen from the decay of the one-step CF.
    L : float
        COS truncation multiplier.
    n_max : int
        Cap on the automatic number of COS terms.
    """

    n_states: int = 48
    richardson: bool = True
    tail: float = 1e-7
    N: int | None = None
    L: float = 10.0
    n_max: int = 2048


# ----------------------------------------------------------------------------- chain


def cir_chain(v0, kappa, theta, nu, T, n_states, tail=1e-7):
    """Variance grid (containing 0 and ``v0``), generator and the index of ``v0``.

    The grid is uniform in ``sqrt(v)``, the Lamperti coordinate of the CIR
    process in which its diffusion coefficient is constant: ``v_j = (j h)^2``
    with ``h`` chosen so that ``v0`` is a node, reaching the ``1 - tail``
    quantile of ``v_T`` (a scaled noncentral chi-square with a long right
    tail). The node at zero matters when the Feller condition fails and the
    variance reaches zero; there it can only drift up.
    """
    from scipy.stats import ncx2

    e = np.exp(-kappa * T)
    scale = nu * nu * (1.0 - e) / (4.0 * kappa)
    dof, nc = 4.0 * kappa * theta / (nu * nu), v0 * e / scale
    v_hi = max(v0, float(scale * ncx2.ppf(1.0 - tail, dof, nc)))
    s0, s_hi = np.sqrt(v0), np.sqrt(v_hi)
    i0 = int(np.clip(round(s0 / s_hi * (n_states - 1)), 1, n_states - 2))
    h = s0 / i0
    v = (h * np.arange(n_states)) ** 2
    v[i0] = v0  # exactly, not up to rounding
    drift = kappa * (theta - v)
    diff = nu * nu * v
    m = len(v)
    Q = np.zeros((m, m))
    for i in range(m):
        if i == 0:
            kp = v[1] - v[0]
            Q[0, 1] = max(drift[0], 0.0) / kp + diff[0] / (kp * kp)
        elif i == m - 1:
            km = v[-1] - v[-2]
            Q[-1, -2] = max(-drift[-1], 0.0) / km + diff[-1] / (km * km)
        else:
            kp, km = v[i + 1] - v[i], v[i] - v[i - 1]
            up = (diff[i] + drift[i] * km) / (kp * (kp + km))
            dn = (diff[i] - drift[i] * kp) / (km * (kp + km))
            if up < 0.0 or dn < 0.0:  # upwind drift where central rates go negative
                up = diff[i] / (kp * (kp + km)) + max(drift[i], 0.0) / kp
                dn = diff[i] / (km * (kp + km)) + max(-drift[i], 0.0) / km
            Q[i, i + 1], Q[i, i - 1] = up, dn
        Q[i, i] = -Q[i].sum()
    return v, Q, i0


class _Chain:
    """Per-state exponents ``psi(u)`` (shape ``(len(u), m)``), generator and shifts."""

    def __init__(self, model: str, params, T: float, grid: CTMCVarianceGrid):
        if model not in CTMC_MODELS:
            raise NotImplementedError(
                f"cos_ctmc: model {model!r} is not supported; choose from {CTMC_MODELS}"
            )
        self.model = model
        self._cache: dict = {}
        if model == "regime_switching":
            from ..models.regime_switching import _rs_psi

            self.Q = np.asarray(params.generator, dtype=float)
            self.p0 = np.asarray(params.initial_probs, dtype=float)
            self.shift = np.zeros(len(self.p0))
            self._psi = lambda u: np.array([_rs_psi(complex(ui), params) for ui in u])
            self.psi_ref = lambda u: np.max(np.real(self._psi(u)), axis=1)
            return
        kappa, theta, nu, rho, v0 = params.kappa, params.theta, params.nu, params.rho, params.v0
        v, Q, i0 = cir_chain(v0, kappa, theta, nu, T, grid.n_states, grid.tail)
        self.v, self.Q = v, Q
        self.p0 = np.zeros(len(v))
        self.p0[i0] = 1.0
        self.shift = rho / nu * (v - v0)
        drift = -0.5 * v - rho / nu * kappa * (theta - v)
        vol2 = (1.0 - rho * rho) * v
        jump = None
        if model == "bates":
            lam, mu, sj = params.lam_j, params.mu_j, params.sigma_j
            zeta = np.expm1(mu + 0.5 * sj * sj)

            def jump(u):
                return lam * (np.exp(1j * u * mu - 0.5 * sj * sj * u * u) - 1.0 - 1j * u * zeta)

        def psi(u):
            u = np.asarray(u, dtype=complex)[:, None]
            out = 1j * u * drift[None, :] - 0.5 * vol2[None, :] * u * u
            if jump is not None:
                out = out + jump(u)
            return out

        self._psi = psi
        v_ref = 0.5 * min(v0, theta)
        self.psi_ref = lambda u: np.real(
            -0.5 * (1.0 - rho * rho) * v_ref * u * u + (jump(u) if jump is not None else 0.0)
        )

    def psi(self, u) -> np.ndarray:
        return self._psi(np.asarray(u))

    def transition(self, w: np.ndarray, dt: float) -> np.ndarray:
        """``Phi[k] = expm(dt (Q + diag psi(w_k)))``, shape ``(N, m, m)``, cached.

        If the half step is cached with at least as many frequencies, the result
        is its square (the frequencies ``w_k = k pi / (b - a)`` of a shorter
        expansion are a prefix of a longer one's).
        """
        n = len(w)
        key = (round(dt, 14), n)
        if key in self._cache:
            return self._cache[key]
        for (dt_c, n_c), half in self._cache.items():
            if n_c >= n and abs(2.0 * dt_c - dt) <= 1e-14 * dt:
                out = np.matmul(half[:n], half[:n])
                self._cache[key] = out
                return out
        ps = self.psi(w)
        m = self.Q.shape[0]
        dead = dt * np.max(ps.real, axis=1) < -700.0  # every entry has underflowed
        A = dt * (self.Q[None, :, :] + ps[:, :, None] * np.eye(m)[None, :, :])
        A[dead] = 0.0
        out = expm(A)
        out[dead] = 0.0
        out[~np.isfinite(out)] = 0.0
        self._cache[key] = out
        return out


# ----------------------------------------------------------------------------- COS recursion


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


def _backward(
    chain: _Chain,
    S0: float,
    r: float,
    q: float,
    K: float,
    cp: int,
    times: np.ndarray,
    a: float,
    b: float,
    N: int,
    *,
    exercise: bool,
    barrier: tuple[float, float] | None = None,
) -> float:
    """COS backward induction over ``times`` with one coefficient vector per state.

    ``exercise=True`` gives a Bermudan (exercise allowed at every date);
    ``barrier=(L, U)`` knocks out (zero rebate) where ``S`` leaves ``(L, U)``
    at any date, the last one included.
    """
    bma = b - a
    k = np.arange(N, dtype=float)
    w = k * np.pi / bma
    n_all = np.arange(-(N - 1), 2 * N - 1, dtype=float)
    w_all = n_all * np.pi / bma
    nz = n_all != 0.0
    fft_len = _next_pow2(3 * N - 2)
    shift = chain.shift
    m = len(shift)

    def fwd_t(t):
        return S0 * np.exp((r - q) * t)

    def chi(c, d):
        """Rows ``(2/(b-a)) int_c^d e^y cos(w_k (y - a)) dy``-style chi for vectors c, d."""
        wc, wd = w[None, :] * (c[:, None] - a), w[None, :] * (d[:, None] - a)
        ec, ed = np.exp(c)[:, None], np.exp(d)[:, None]
        return (np.cos(wd) * ed - np.cos(wc) * ec + w * (np.sin(wd) * ed - np.sin(wc) * ec)) / (
            1.0 + w * w
        )

    def psi_c(c, d):
        out = np.empty((len(c), N))
        out[:, 0] = d - c
        out[:, 1:] = (np.sin(w[1:] * (d[:, None] - a)) - np.sin(w[1:] * (c[:, None] - a))) / w[1:]
        return out

    def payoff_coeffs(c, d, F):
        """Per state: (2/(b-a)) int_c^d cp (F e^{y + c_i} - K) cos(w_k (y - a)) dy."""
        ok = d > c
        c, d = np.where(ok, c, 0.0), np.where(ok, d, 0.0)
        val = (2.0 / bma) * cp * (F * np.exp(shift)[:, None] * chi(c, d) - K * psi_c(c, d))
        return np.where(ok[:, None], val, 0.0)

    def phases(x):
        """``exp(i w_n (x - a))`` for n = -(N-1) .. 2N-2, by a running product."""
        z = np.exp(1j * np.pi * (x - a) / bma)[:, None]
        P = np.cumprod(np.concatenate((np.ones_like(z), np.repeat(z, 2 * N - 2, 1)), 1), 1)
        return np.concatenate((np.conj(P[:, N - 1 : 0 : -1]), P), axis=1)

    def cont_coeffs(U, c, d):
        """Per state: (2/(b-a)) int_c^d Re[sum_l U_l e^{i w_l (y-a)}] cos(w_k (y-a)) dy."""
        ok = d > c
        c, d = np.where(ok, c, a), np.where(ok, d, a)
        E = (phases(d) - phases(c)) / np.where(nz, 1j * w_all, 1.0)[None, :]
        E[:, ~nz] = (d - c)[:, None]
        R = sfft.fft(U[:, ::-1], fft_len, axis=1)
        EF = sfft.fft(np.concatenate((E[:, N - 1 :], E[:, : 2 * N - 1]), axis=0), fft_len, axis=1)
        both = sfft.ifft(np.concatenate((R, R), axis=0) * EF, axis=1)
        hankel, toeplitz = both[:m], both[m:]
        val = np.real(hankel[:, N - 1 : 2 * N - 1] + toeplitz[:, N - 1 : 2 * N - 1][:, ::-1]) / bma
        return np.where(ok[:, None], val, 0.0)

    def exercise_points(U, disc, F, guess):
        """Per state, the root of C_i(y) - g_i(y) (vectorised safeguarded Newton).

        ``guess`` (the previous date's boundary) starts Newton close to the root,
        and only states that have not converged are evaluated.
        """
        Fi = F * np.exp(shift)
        yk = np.log(K / Fi)

        def f_fp(y, idx):
            # e^{i w_k (y - a)} = z^k by a running product (root finding only)
            z = np.exp(1j * np.pi * (y - a) / bma)[:, None]
            powers = np.cumprod(np.concatenate((np.ones_like(z), np.repeat(z, N - 1, 1)), 1), 1)
            e = U[idx] * powers
            cont = disc * np.real(e.sum(axis=1))
            dcont = disc * np.real((1j * w[None, :] * e).sum(axis=1))
            ex = Fi[idx] * np.exp(y)
            return cont - cp * (ex - K), dcont - cp * ex

        everyone = np.arange(m)
        if cp == -1:
            lo, hi = np.full(m, a), np.minimum(yk, b)
            f_lo, f_hi = f_fp(lo, everyone)[0], f_fp(np.maximum(hi, a), everyone)[0]
            never = (hi <= a) | (f_lo >= 0.0)  # continuation beats exercise everywhere
            always = ~never & (f_hi < 0.0)
            default = np.where(never, a, hi)
        else:
            lo, hi = np.maximum(yk, a), np.full(m, b)
            f_lo, f_hi = f_fp(np.minimum(lo, b), everyone)[0], f_fp(hi, everyone)[0]
            never = (lo >= b) | (f_hi >= 0.0)
            always = ~never & (f_lo < 0.0)
            default = np.where(never, b, lo)
        y = np.where((guess > lo) & (guess < hi), guess, 0.5 * (lo + hi))
        todo = np.nonzero(~(never | always))[0]
        for _ in range(100):
            if todo.size == 0:
                break
            yt, lt, ht = y[todo], lo[todo], hi[todo]
            fx, fpx = f_fp(yt, todo)
            below = (fx < 0.0) == (cp == -1)
            lt = np.where(below, yt, lt)
            ht = np.where(below, ht, yt)
            y_new = np.where(fpx != 0.0, yt - fx / np.where(fpx != 0.0, fpx, 1.0), 0.5 * (lt + ht))
            y_new = np.where((y_new > lt) & (y_new < ht), y_new, 0.5 * (lt + ht))
            # Near smooth pasting C - g has an almost double root and Newton degrades
            # to bisection. Stopping once |C - g| is tiny is enough: a split point off
            # by d then changes the coefficients by at most |C - g| d.
            hit = np.abs(fx) <= 1e-13 * K
            done = hit | (np.abs(y_new - yt) <= 1e-12) | (ht - lt <= 1e-10)
            y[todo], lo[todo], hi[todo] = np.where(hit, yt, y_new), lt, ht
            todo = todo[~done]
        return np.where(never | always, default, y)

    def alive_interval(F):
        lo_b, hi_b = barrier if barrier is not None else (0.0, np.inf)
        Fi = F * np.exp(shift)
        c = np.full(m, a) if lo_b <= 0.0 else np.clip(np.log(lo_b / Fi), a, b)
        d = np.full(m, b) if not np.isfinite(hi_b) else np.clip(np.log(hi_b / Fi), a, b)
        return c, d

    def Phi(dt):
        return chain.transition(w, dt)

    # terminal coefficients: payoff on its support, restricted to the barrier corridor
    F = fwd_t(float(times[-1]))
    yk = np.clip(np.log(K / (F * np.exp(shift))), a, b)
    c, d = alive_interval(F)
    if cp == -1:
        V = payoff_coeffs(c, np.minimum(yk, d), F)
    else:
        V = payoff_coeffs(np.maximum(yk, c), d, F)

    ys = np.full(m, np.nan)
    for j in range(len(times) - 2, -1, -1):
        dt = float(times[j + 1] - times[j])
        if dt < 1e-12:
            continue
        U = np.matmul(Phi(dt), V.T[:, :, None])[:, :, 0].T
        U[:, 0] *= 0.5
        disc = float(np.exp(-r * dt))
        F = fwd_t(float(times[j]))
        c, d = alive_interval(F)
        if exercise:
            ys = exercise_points(U, disc, F, ys)
            if cp == -1:
                V = disc * cont_coeffs(U, np.maximum(ys, c), d) + payoff_coeffs(
                    c, np.minimum(ys, d), F
                )
            else:
                V = disc * cont_coeffs(U, c, np.minimum(ys, d)) + payoff_coeffs(
                    np.maximum(ys, c), d, F
                )
        else:
            V = disc * cont_coeffs(U, c, d)

    t1 = float(times[0])
    if t1 > 1e-12:
        U = np.matmul(Phi(t1), V.T[:, :, None])[:, :, 0].T
    else:
        U = V.astype(complex)
    U[:, 0] *= 0.5
    per_state = np.real(U @ np.exp(-1j * w * a))  # value at y = 0 for each starting state
    price = np.exp(-r * t1) * float(chain.p0 @ per_state)
    return max(price, 0.0)


def _interval(chain: _Chain, model, fwd: ForwardSpec, params, T: float, L: float):
    fwd_T = ForwardSpec(fwd.S0, fwd.r, fwd.q, T)
    c1, c2, c4 = MODEL_REGISTRY[model].cumulants(fwd_T, params)
    spread = float(np.max(chain.shift) - np.min(chain.shift))
    half = L * np.sqrt(max(c2, 1e-12) + np.sqrt(abs(c4))) + spread
    return c1 - half, c1 + half


def _terms(chain: _Chain, a: float, b: float, dt: float, grid: CTMCVarianceGrid) -> int:
    """Smallest multiple of 64 terms at which the ``dt`` CF of a low-variance state has decayed.

    The reference is ``min(v0, theta) / 2`` for the CIR chain (lower states
    matter when the variance can get near zero) and the slowest regime for
    ``regime_switching``.
    """
    if grid.N is not None:
        return grid.N
    n = np.arange(64, grid.n_max + 64, 64)
    decay = dt * np.real(chain.psi_ref((n - 1) * np.pi / (b - a)))
    ok = np.nonzero(decay < np.log(TERM_TOL))[0]
    return int(n[ok[0]]) if len(ok) else grid.n_max


def _priced(model, fwd, params, T, grid, run) -> float:
    """``run(chain, a, b)`` on the chain, extrapolated in the number of states.

    The chain error is second order in the grid spacing, so for the CIR chain
    ``(4 V(2m) - V(m)) / 3`` removes the leading term (this takes the European
    error on the Heston benchmark from about 1e-5 to 1e-7).
    """
    grid = grid or CTMCVarianceGrid()

    def once(g):
        chain = _Chain(model, params, T, g)
        a, b = _interval(chain, model, fwd, params, T, g.L)
        return run(chain, a, b, g)

    if model == "regime_switching" or not grid.richardson:
        return max(once(grid), 0.0)
    coarse = once(grid)
    fine = once(replace(grid, n_states=2 * grid.n_states))
    return max((4.0 * fine - coarse) / 3.0, 0.0)


def _dates(product_times, T):
    t = np.sort(np.asarray(product_times, dtype=float))
    if not np.isclose(t[-1], T, rtol=1e-8):
        t = np.sort(np.append(t, T))
    return t


def cos_ctmc_bermudan_price(model, fwd: ForwardSpec, params, product, *, grid=None) -> float:
    """Bermudan option under ``heston``, ``bates`` or ``regime_switching``."""
    times = _dates(product.exercise_times, product.maturity)
    dt_min = float(np.min(np.diff(np.concatenate(([0.0], times)))))

    def run(chain, a, b, g):
        N = _terms(chain, a, b, dt_min, g)
        return _backward(
            chain, fwd.S0, fwd.r, fwd.q, product.strike, product.cp, times, a, b, N, exercise=True
        )

    return _priced(model, fwd, params, product.maturity, grid, run)


def cos_ctmc_american_price(
    model, fwd: ForwardSpec, params, product, *, base_dates: int = 8, grid=None
) -> float:
    """American option by 4-point Richardson extrapolation of CTMC-COS Bermudans.

    Prices Bermudans with ``m, 2m, 4m, 8m`` dates (``m = base_dates``) and
    removes the ``1/M``, ``1/M^2`` and ``1/M^3`` terms (Fang & Oosterlee 2009).
    """
    if base_dates < 1:
        raise ValueError(f"cos_ctmc_american_price: base_dates must be >= 1, got {base_dates}")
    T, K, cp = product.maturity, product.strike, product.cp

    def run(chain, a, b, g):
        values = []
        for level in range(3, -1, -1):  # finest first: coarser steps square cached ones
            M = base_dates * 2**level
            times = T * np.arange(1, M + 1) / M
            N = _terms(chain, a, b, T / M, g)
            values.append(
                _backward(chain, fwd.S0, fwd.r, fwd.q, K, cp, times, a, b, N, exercise=True)
            )
        v8, v4, v2, v1 = values
        return (64.0 * v8 - 56.0 * v4 + 14.0 * v2 - v1) / 21.0

    american = _priced(model, fwd, params, T, grid, run)
    return float(max(american, cp * (fwd.S0 - K), 0.0))


def cos_ctmc_barrier_price(
    model,
    fwd: ForwardSpec,
    params,
    *,
    strike: float,
    barrier: float,
    maturity: float,
    barrier_type: str = "down_out",
    cp: int = 1,
    n_monitor: int = 252,
    grid=None,
) -> float:
    """Discretely monitored single barrier (``n_monitor`` equally spaced dates, zero rebate).

    Knock-in prices follow from in + out = vanilla, with the vanilla priced by
    the same recursion.
    """
    if barrier_type not in ("down_out", "up_out", "down_in", "up_in"):
        raise ValueError(f"cos_ctmc_barrier_price: unknown barrier_type {barrier_type!r}")
    if cp not in (1, -1):
        raise ValueError(f"cos_ctmc_barrier_price: cp must be +1 or -1, got {cp}")
    if not (np.isfinite(barrier) and barrier > 0):
        raise ValueError(f"cos_ctmc_barrier_price: barrier must be > 0, got {barrier}")
    times = maturity * np.arange(1, n_monitor + 1) / n_monitor
    corridor = (barrier, np.inf) if barrier_type.startswith("down") else (0.0, barrier)

    def run(chain, a, b, g):
        N = _terms(chain, a, b, maturity / n_monitor, g)
        args = (chain, fwd.S0, fwd.r, fwd.q, strike, cp)
        out = _backward(*args, times, a, b, N, exercise=False, barrier=corridor)
        if barrier_type.endswith("out"):
            return out
        vanilla = _backward(*args, times[-1:], a, b, N, exercise=False)
        return vanilla - out

    return _priced(model, fwd, params, maturity, grid, run)


def cos_ctmc_european_price(model, fwd: ForwardSpec, params, *, strike, cp=1, grid=None) -> float:
    """European option through the chain (a check on the variance approximation)."""

    def run(chain, a, b, g):
        N = _terms(chain, a, b, fwd.T, g)
        return _backward(
            chain, fwd.S0, fwd.r, fwd.q, strike, cp, np.array([fwd.T]), a, b, N, exercise=False
        )

    return _priced(model, fwd, params, fwd.T, grid, run)


__all__ = [
    "CTMC_MODELS",
    "CTMCVarianceGrid",
    "cir_chain",
    "cos_ctmc_american_price",
    "cos_ctmc_barrier_price",
    "cos_ctmc_bermudan_price",
    "cos_ctmc_european_price",
]
