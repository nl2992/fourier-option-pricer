"""PROJ (frame-projection) European option pricer — Kirkby (2015, 2017).

This is a faithful Python port of Justin Kirkby's ``PROJ_European.m`` from the
`PROJ_Option_Pricing_Matlab` reference project. PROJ projects the risk-neutral
density of the log-return onto a frame of shifted B-spline generators of a given
``order`` (Haar / linear / quadratic / cubic) and recovers the projection
coefficients ``beta`` from the characteristic function with a single FFT. The
option price is then a dot product of those coefficients with closed-form payoff
coefficients ``G``.

References
----------
1. Kirkby, J.L. (2015), *Efficient Option Pricing by Frame Duality with the Fast
   Fourier Transform.* SIAM J. Financial Math.
2. Kirkby, J.L. (2017), *Robust option pricing with characteristic functions and
   the B-spline order of density projection.* J. Computational Finance.

Convention bridge
-----------------
The package-wide :class:`~foureng.models.base.CharFunc` ``phi`` is the CF of the
**forward** log-return ``X_T = log(S_T / F_0)``. Kirkby's routine works with the
CF of ``log(S_T / S_0)`` and its first cumulant. We convert internally:

    rnCHF(w) = exp(i * w * (r - q) * T) * phi(w)
    c1_proj  = (r - q) * T + c1_forward

so callers continue to pass the same forward CF and forward cumulant ``c1`` used
everywhere else in ``foureng``.
"""

from __future__ import annotations

import numpy as np

from ..models.base import CharFunc, ForwardSpec
from ..utils.grids import ProjGrid

__all__ = [
    "proj_price_at_strikes",
    "proj_auto_grid",
    "proj_bermudan_put",
    "proj_european_price_at_strikes",
]


def proj_auto_grid(
    cumulants: tuple[float, float, float],
    *,
    N: int = 1 << 13,
    L: float = 10.0,
    order: int = 3,
) -> ProjGrid:
    """Build a :class:`ProjGrid` from forward cumulants ``(c1, c2, c4)``.

    Mirrors the COS truncation heuristic: the grid half-width is

        alph = L * sqrt(c2 + sqrt(|c4|))

    which scales with the standard deviation (plus a kurtosis allowance) of the
    log-return so the projection grid covers the tails. ``L = 10`` is a safe
    default; heavier-tailed models may need a larger ``L`` or ``N``.
    """
    _c1, c2, c4 = cumulants
    alph = L * float(np.sqrt(abs(c2) + np.sqrt(abs(c4))))
    if not (np.isfinite(alph) and alph > 0):
        raise ValueError(f"proj_auto_grid: computed non-positive alph={alph} from cumulants")
    return ProjGrid(N=int(N), alph=alph, order=int(order))


def proj_european_price_at_strikes(
    phi: CharFunc,
    fwd: ForwardSpec,
    cumulants: tuple[float, float, float],
    strikes,
    *,
    cp: int = 1,
    N: int = 1 << 13,
    L: float = 10.0,
) -> np.ndarray:
    """European vanilla PROJ price (cumulant-driven auto grid).

    Backward-compatible entry point used by the pipeline dispatcher. Previously
    a COS-backed façade; now routed through the real frame-projection engine via
    a cubic B-spline auto grid built from ``cumulants``.
    """
    grid = proj_auto_grid(cumulants, N=N, L=L)
    return proj_price_at_strikes(phi, fwd, grid, strikes, cp=cp, c1=float(cumulants[0]))


def _proj_single_put_call(
    rn_chf,
    fwd: ForwardSpec,
    grid: ProjGrid,
    W: float,
    c1_proj: float,
) -> tuple[float, float]:
    """Return ``(put, call)`` PROJ prices for one strike ``W``.

    Direct transcription of ``PROJ_European.m``; ``rn_chf`` is the CF of
    ``log(S_T / S_0)`` and ``c1_proj`` its first cumulant.
    """
    N = int(grid.N)
    order = int(grid.order)
    S_0 = fwd.S0
    r, q, T = fwd.r, fwd.q, fwd.T

    dx = 2.0 * grid.alph / (N - 1)
    a = 1.0 / dx

    lws = np.log(W / S_0)
    lam = c1_proj - (N / 2 - 1) * dx
    nbar = int(np.floor(a * (lws - lam) + 1.0))  # 1-based index, MATLAB-style
    if nbar >= N:
        nbar = N - 1
    if nbar < 3:
        # Grid is too narrow / strike too far in the tail for the closed-form
        # payoff stencils (which reach back 2–3 nodes). Widen ``alph`` or ``N``.
        raise ValueError(
            f"PROJ: strike {W} maps to grid index nbar={nbar} < 3; "
            "increase ProjGrid.alph or N so the strike sits inside the grid."
        )
    xmin = lws - (nbar - 1) * dx

    dw = 2.0 * np.pi / (N * dx)
    omega = np.arange(1, N) * dw  # length N-1; w=0 handled explicitly below

    sinc4 = (np.sin(omega / (2.0 * a)) / omega) ** 4
    sinc3 = (np.sin(omega / (2.0 * a)) / omega) ** 3
    sinc2 = (np.sin(omega / (2.0 * a)) / omega) ** 2
    phase = np.exp(-1j * xmin * omega)
    chf = rn_chf(omega)

    if order == 3:
        b0, b1, b2, b3 = 1208 / 2520, 1191 / 2520, 120 / 2520, 1 / 2520
        denom = (
            b0 + b1 * np.cos(omega / a) + b2 * np.cos(2 * omega / a) + b3 * np.cos(3 * omega / a)
        )
        grand = chf * sinc4 / denom
        head = 1.0 / (32 * a**4)
        beta = np.real(np.fft.fft(np.concatenate(([head], phase * grand))))

        G = np.zeros(nbar + 1)
        G[nbar] = W * (
            1 / 24
            - 1
            / 20
            * np.exp(dx)
            * (
                np.exp(-7 / 4 * dx) / 54
                + np.exp(-1.5 * dx) / 18
                + np.exp(-1.25 * dx) / 2
                + 7 * np.exp(-dx) / 27
            )
        )
        G[nbar - 1] = W * (
            0.5
            - 0.05
            * (
                28 / 27
                + np.exp(-7 / 4 * dx) / 54
                + np.exp(-1.5 * dx) / 18
                + np.exp(-1.25 * dx) / 2
                + 14 * np.exp(-dx) / 27
                + 121 / 54 * np.exp(-0.75 * dx)
                + 23 / 18 * np.exp(-0.5 * dx)
                + 235 / 54 * np.exp(-0.25 * dx)
            )
        )
        G[nbar - 2] = W * (
            23 / 24
            - np.exp(-dx)
            / 90
            * (
                (28 + 7 * np.exp(-dx)) / 3
                + (
                    14 * np.exp(dx)
                    + np.exp(-7 / 4 * dx)
                    + 242 * np.cosh(0.75 * dx)
                    + 470 * np.cosh(0.25 * dx)
                )
                / 12
                + 0.25 * (np.exp(-1.5 * dx) + 9 * np.exp(-1.25 * dx) + 46 * np.cosh(0.5 * dx))
            )
        )
        idx = np.arange(0, nbar - 2)  # G(1:nbar-2)
        G[idx] = W - S_0 * np.exp(xmin + dx * idx) / 90 * (
            14 / 3 * (2 + np.cosh(dx))
            + 0.5 * (np.cosh(1.5 * dx) + 9 * np.cosh(1.25 * dx) + 23 * np.cosh(0.5 * dx))
            + 1 / 6 * (np.cosh(7 / 4 * dx) + 121 * np.cosh(0.75 * dx) + 235 * np.cosh(0.25 * dx))
        )
        Cons = 32 * a**4

    elif order == 2:
        denom = 26 * np.cos(omega / a) + np.cos(2 * omega / a) + 33
        grand = chf * sinc3 / denom
        head = 1.0 / (960 * a**3)
        beta = np.real(np.fft.fft(np.concatenate(([head], phase * grand))))

        G = np.zeros(nbar + 1)
        G[nbar] = W * (
            1 / 48
            - np.exp(dx)
            * (
                np.exp(-11 / 8 * dx) / 720
                + np.exp(-1.25 * dx) / 480
                + np.exp(-9 / 8 * dx) / 80
                + 7 / 1440 * np.exp(-dx)
            )
        )
        G[nbar - 1] = W * (
            0.5
            - 0.1
            * (
                7 / 24
                + np.exp(-1.25 * dx) / 9
                + np.exp(-dx) / 6
                + np.exp(-0.75 * dx)
                + 7 / 12 * np.exp(-0.5 * dx)
                + 13 / 12 * np.exp(-3 / 8 * dx)
                + 11 / 24 * np.exp(-dx / 4)
                + 47 / 36 * np.exp(-dx / 8)
            )
        )
        G[nbar - 2] = W * (
            47 / 48
            - np.exp(-dx)
            * 0.1
            * (
                1
                + np.exp(-1.25 * dx) / 9
                + np.exp(-dx) / 6
                + np.exp(-0.75 * dx)
                + 7 / 9 * np.exp(-0.5 * dx)
                + 44 / 9 * np.cosh(dx / 4)
                + 7 / 12 * np.exp(0.5 * dx)
                + 49 / 72 * np.exp(5 / 8 * dx)
                + 3 / 16 * np.exp(0.75 * dx)
                + 25 / 72 * np.exp(7 / 8 * dx)
                + 7 / 144 * np.exp(dx)
            )
        )
        idx = np.arange(0, nbar - 2)
        G[idx] = W - np.exp(xmin + dx * idx) * S_0 * 0.1 * (
            1
            + 2 / 9 * np.cosh(1.25 * dx)
            + np.cosh(dx) / 3
            + 2 * np.cosh(0.75 * dx)
            + 14 / 9 * np.cosh(0.5 * dx)
            + 44 / 9 * np.cosh(0.25 * dx)
        )
        Cons = 960 * a**3

    elif order == 1:
        denom = 2 + np.cos(omega / a)
        grand = chf * sinc2 / denom
        head = 1.0 / (24 * a**2)
        beta = np.real(np.fft.fft(np.concatenate(([head], phase * grand))))

        G = np.zeros(nbar)
        G[nbar - 1] = W * (
            0.5
            - (7 / 6 + 4 / 3 * np.exp(-0.75 * dx) + np.exp(-0.5 * dx) + 4 * np.exp(-0.25 * dx)) / 15
        )
        idx = np.arange(0, nbar - 1)
        G[idx] = W - np.exp(xmin + dx * idx) * S_0 / 15 * (
            7 / 3 + 8 / 3 * np.cosh(0.75 * dx) + 2 * np.cosh(0.5 * dx) + 8 * np.cosh(0.25 * dx)
        )
        Cons = 24 * a**2

    elif order == 0:
        grand = chf * (np.sin(omega / (2.0 * a)) / omega)
        head = 1.0 / (4 * a)
        beta = np.real(np.fft.fft(np.concatenate(([head], phase * grand))))

        G = np.zeros(nbar)
        G[nbar - 1] = W * (0.5 - a * (1 - np.exp(-0.5 * dx)))
        idx = np.arange(0, nbar - 1)
        G[idx] = W - np.exp(xmin + dx * idx) * S_0 * 2 * a * np.sinh(dx / 2)
        Cons = 4 * a

    else:
        raise ValueError(f"PROJ: order must be 0, 1, 2, or 3; got {order}")

    disc = np.exp(-r * T)
    put = Cons * disc / N * float(G @ beta[: G.size])
    call = put + S_0 * np.exp(-q * T) - W * disc
    return max(put, 0.0), max(call, 0.0)


def proj_price_at_strikes(
    phi: CharFunc,
    fwd: ForwardSpec,
    grid: ProjGrid,
    strikes,
    *,
    cp: int = 1,
    c1: float = 0.0,
) -> np.ndarray:
    """Price European calls or puts by the PROJ frame-projection method.

    Parameters
    ----------
    phi
        Characteristic function of the forward log-return
        ``X_T = log(S_T / F_0)`` (the package-wide :class:`CharFunc`).
    fwd
        Market inputs.
    grid
        :class:`ProjGrid` controlling the B-spline order, grid half-width
        ``alph`` and resolution ``N``.
    strikes
        Strike array. PROJ rebuilds the projection per strike (the grid is
        anchored at the log-strike), so the cost scales with the strip length.
    cp
        ``+1`` for calls, ``-1`` for puts.
    c1
        First cumulant of the **forward** log-return ``X_T`` (our convention).
        Pass ``MODEL_REGISTRY[model].cumulants(fwd, params)[0]``. The internal
        routine adds the ``(r - q) * T`` drift to obtain Kirkby's ``c1``.
        Safe to leave at ``0.0`` for short maturities.

    Returns
    -------
    np.ndarray
        Prices at ``strikes``.
    """
    if cp not in (1, -1):
        raise ValueError(f"proj_price_at_strikes: cp must be +1 or -1, got {cp}")

    K = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    if K.size == 0:
        raise ValueError("strikes must be non-empty")
    if np.any(K <= 0.0) or not np.all(np.isfinite(K)):
        raise ValueError("strikes must be finite and > 0")

    drift = (fwd.r - fwd.q) * fwd.T
    c1_proj = drift + float(c1)

    def rn_chf(w: np.ndarray) -> np.ndarray:
        return np.exp(1j * w * drift) * np.asarray(phi(w), dtype=np.complex128)

    out = np.empty(K.size, dtype=np.float64)
    for i, W in enumerate(K):
        put, call = _proj_single_put_call(rn_chf, fwd, grid, float(W), c1_proj)
        out[i] = call if cp == 1 else put
    return out


# ---------------------------------------------------------------------------
# PROJ Bermudan put (1-D Lévy) — port of Kirkby's PROJ_Bermudan_Put.m
# ---------------------------------------------------------------------------


def proj_bermudan_put(
    step_cf,
    *,
    S0: float,
    r: float,
    T: float,
    W: float,
    M: int,
    N: int = 1 << 14,
    alph: float = 0.5,
) -> float:
    """Bermudan **put** price by the PROJ method (Kirkby 2015).

    Faithful port of ``PROJ_Bermudan_Put.m``. The option is exercisable at the
    ``M`` equally spaced monitoring dates ``t = dt, 2 dt, ..., M dt = T`` with
    ``dt = T / M``. Backward induction is carried out with a Toeplitz-FFT
    convolution against the linear-spline density projection at each step.

    Parameters
    ----------
    step_cf
        One-step risk-neutral characteristic function — the CF of
        ``log(S_{t+dt} / S_t)`` under Q over a single step ``dt`` (drift
        included, i.e. Kirkby's ``rnCHF``). For a Lévy model with dividend
        yield ``q`` this is ``exp(i u (r - q) dt) * phi_dt(u)`` where
        ``phi_dt`` is the forward CF over ``dt``.
    S0, r, T, W
        Spot, risk-free rate, maturity, strike.
    M
        Number of monitoring subintervals (``M`` exercise dates).
    N
        Projection / FFT grid size (power of two). ``K = N/2`` working points.
    alph
        Grid half-width; the log-grid spans roughly ``[-alph, alph]``. Size it
        to cover the full-horizon (T) spread of the log-return.

    Returns
    -------
    float
        Bermudan put price at ``t = 0``.
    """
    M = int(M)
    if M < 1:
        raise ValueError("proj_bermudan_put: M must be >= 1")
    N = int(N)
    if N & (N - 1) != 0:
        raise ValueError("proj_bermudan_put: N must be a power of two")

    dt = T / M
    K = N // 2
    Cons3 = 1.0 / 48.0
    Cons4 = 1.0 / 12.0

    dx = 2.0 * alph / (N - 1)
    a = 1.0 / dx

    lws = np.log(W / S0)

    # ---- grid alignment so that x = 0 (log(S0/S0)) is a grid node ----
    nnot = K // 2  # 1-based index of x = 0
    dxtil = 1.0 / a
    nbar = int(np.floor(lws * a + K / 2))  # 1-based grid index of the strike
    if abs(lws) < dxtil:
        dx = dxtil
    elif lws < 0:
        dx = lws / (1 + nbar - K / 2)
        nbar = nbar + 1
    elif lws > 0:
        dx = lws / (nbar - K / 2)

    a = 1.0 / dx
    xmin = (1 - K / 2) * dx

    a2 = a * a
    Cons2 = 24.0 * a2 * np.exp(-r * dt) / N
    zmin = (1 - K) * dx

    dw = 2.0 * np.pi * a / N
    grand = np.arange(1, N) * dw  # length N-1
    grand = (
        np.exp(-1j * zmin * grand)
        * np.asarray(step_cf(grand), dtype=np.complex128)
        * (np.sin(grand / (2.0 * a)) / grand) ** 2
        / (2.0 + np.cos(grand / a))
    )
    beta = Cons2 * np.real(np.fft.fft(np.concatenate(([1.0 / (24.0 * a2)], grand))))  # length N

    # Toeplitz operator (already carries exp(-r*dt) through beta).
    toepM = np.concatenate((np.flip(beta[0:K]), [0.0], np.flip(beta[K : 2 * K - 1])))  # length 2K
    toepM = np.fft.fft(toepM)

    # ---- terminal payoff coefficients ----
    Gs = np.zeros(K)
    Gs[0:nbar] = np.exp(xmin + dx * np.arange(0, nbar)) * S0  # S-values on ITM region

    # Gaussian-quadrature payoff constants
    q_plus = (1 + np.sqrt(3 / 5)) / 2
    q_minus = (1 - np.sqrt(3 / 5)) / 2
    b3 = np.sqrt(15)
    b4 = b3 / 10
    varthet_01 = np.exp(0.5 * dx) * (5 * np.cosh(b4 * dx) - b3 * np.sinh(b4 * dx) + 4) / 18
    varthet_m10 = np.exp(-0.5 * dx) * (5 * np.cosh(b4 * dx) + b3 * np.sinh(b4 * dx) + 4) / 18
    varthet_star = varthet_01 + varthet_m10

    ThetM = np.zeros(K)
    ThetM[nbar - 1] = W * (0.5 - varthet_m10)
    ThetM[0 : nbar - 1] = W - varthet_star * Gs[0 : nbar - 1]
    Gs[0:nbar] = W - Gs[0:nbar]  # put intrinsic on ITM region

    # initial continuation value (one convolution step)
    p = np.fft.ifft(toepM * np.fft.fft(np.concatenate((ThetM[0:K], np.zeros(K)))))
    Cont = np.real(p[0:K])

    Thet = np.zeros(K)
    kstr = nbar + 1  # 1-based

    for _m in range(M - 2, -1, -1):
        while kstr > 1 and Cont[kstr - 1] > Gs[kstr - 1]:
            kstr -= 1

        if kstr >= 2:
            xkstr = xmin + (kstr - 1) * dx
            Ck1 = Cont[kstr - 2]
            Ck2 = Cont[kstr - 1]
            Ck3 = Cont[kstr]
            Gk2 = Gs[kstr - 1]
            Gk3 = Gs[kstr]
            tmp1 = Ck2 - Gk2
            tmp2 = Ck3 - Gk3
            xstrs = ((xkstr + dx) * tmp1 - xkstr * tmp2) / (tmp1 - tmp2)
        else:
            xkstr = xmin
            kstr = 1
            xstrs = xmin
            Ck2 = Cont[kstr - 1]
            Ck1 = Ck2
            Ck3 = Cont[kstr]

        rho = xstrs - xkstr
        zeta = a * rho
        zeta2 = zeta * zeta
        zeta3 = zeta * zeta2
        zeta4 = zeta * zeta3
        zeta_plus = zeta * q_plus
        zeta_minus = zeta * q_minus
        rho_plus = rho * q_plus
        rho_minus = rho * q_minus

        ed1 = np.exp(rho_minus)
        ed2 = np.exp(rho / 2)
        ed3 = np.exp(rho_plus)

        dbar_1 = zeta2 / 2
        dbar_0 = zeta - dbar_1
        d_0 = (
            zeta
            * (5 * ((1 - zeta_minus) * ed1 + (1 - zeta_plus) * ed3) + 4 * (2 - zeta) * ed2)
            / 18
        )
        d_1 = np.exp(-dx) * zeta * (5 * (zeta_minus * ed1 + zeta_plus * ed3) + 4 * zeta * ed2) / 18

        Thet[0 : kstr - 1] = ThetM[0 : kstr - 1]

        Ck4 = Cont[kstr + 1]
        Thet[kstr - 1] = (
            W * (0.5 + dbar_0)
            - S0 * np.exp(xkstr) * (varthet_m10 + d_0)
            + zeta4 / 8 * (Ck1 - 2 * Ck2 + Ck3)
            + zeta3 / 3 * (Ck2 - Ck1)
            + zeta2 / 4 * (Ck1 + 2 * Ck2 - Ck3)
            - zeta * Ck2
            - Ck1 / 24
            + 5 / 12 * Ck2
            + Ck3 / 8
        )
        Thet[kstr] = (
            W * dbar_1
            - S0 * np.exp(xkstr + dx) * d_1
            + zeta4 / 8 * (-Ck2 + 2 * Ck3 - Ck4)
            + zeta3 / 6 * (3 * Ck2 - 4 * Ck3 + Ck4)
            - 0.5 * zeta2 * Ck2
            + Cons4 * (Ck2 + 10 * Ck3 + Ck4)
        )
        Thet[kstr + 1 : K - 1] = Cons4 * (
            Cont[kstr : K - 2] + 10 * Cont[kstr + 1 : K - 1] + Cont[kstr + 2 : K]
        )
        Thet[K - 1] = Cons3 * (13 * Cont[K - 1] + 15 * Cont[K - 2] - 5 * Cont[K - 3] + Cont[K - 4])

        p = np.fft.ifft(toepM * np.fft.fft(np.concatenate((Thet[0:K], np.zeros(K)))))
        Cont = np.real(p[0:K])

    return float(Cont[nnot - 1])
