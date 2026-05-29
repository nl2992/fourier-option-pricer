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

import numpy as np

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY
from ..products.bermudan import BermudanOption
from .cos import cos_auto_grid

_trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]

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
            "Stochastic-volatility and SV+jump models require the 2-D state extension "
            "or Monte Carlo / PDE pricing."
        )


def cos_bermudan_price(
    model: str,
    fwd: ForwardSpec,
    params,
    product: BermudanOption,
    *,
    grid=None,
    n_spatial: int = 2048,
    N: int = 256,
    L: float = 12.0,
) -> float:
    """Price a Bermudan option via FO2009 COS backward induction.

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
    n_spatial : int
        Spatial grid size for backward induction. Default 2048.
    N : int
        COS terms (used only when ``grid`` is None). Default 256.
    L : float
        Truncation multiplier (used only when ``grid`` is None).

    Returns
    -------
    float
        Bermudan option price.
    """
    _check_model(model)
    entry = MODEL_REGISTRY[model]

    K = product.strike
    T = product.maturity
    cp = product.cp
    r = fwd.r
    q = fwd.q
    S0 = fwd.S0

    exercise_times = np.sort(np.asarray(product.exercise_times, dtype=float))
    # Ensure maturity is included as the final exercise date
    if not np.isclose(exercise_times[-1], T, rtol=1e-8):
        exercise_times = np.sort(np.append(exercise_times, T))

    # Build COS grid from cumulants at maturity T
    if grid is None:
        fwd_T = ForwardSpec(S0=S0, r=r, q=q, T=T)
        cums = entry.cumulants(fwd_T, params)
        grid = cos_auto_grid(cums, N=N, L=L)

    a, b, n_cos = grid.a, grid.b, grid.N
    x = np.linspace(a, b, n_spatial)  # spatial grid
    omega = np.arange(n_cos, dtype=float) * np.pi / (b - a)  # frequencies

    # ── Helper: CF for a log-return over interval Δt ────────────────────────
    def _cf(dt: float):
        return entry.cf(omega, ForwardSpec(S0=S0, r=r, q=q, T=dt), params)

    # ── Helper: forward price at time t ─────────────────────────────────────
    def _fwd(t: float) -> float:
        return S0 * np.exp((r - q) * t)

    # ── Helper: intrinsic value at grid points x at time t ──────────────────
    def _intrinsic(t: float) -> np.ndarray:
        S = _fwd(t) * np.exp(x)
        return np.maximum(S - K, 0.0) if cp == 1 else np.maximum(K - S, 0.0)

    # ── Helper: COS coefficients of V on spatial grid ───────────────────────
    def _cos_coeffs(V: np.ndarray) -> np.ndarray:
        """V_k = (2/(b-a)) * ∫_a^b V(x) cos(ω_k(x−a)) dx via trapezoidal rule."""
        cos_kx = np.cos(omega[:, None] * (x[None, :] - a))  # (N, M)
        return (2.0 / (b - a)) * _trapz(V[None, :] * cos_kx, x, axis=1)

    # ── Helper: continuation value on spatial grid ──────────────────────────
    def _continuation(dt: float, V_k: np.ndarray) -> np.ndarray:
        """C(x) = e^{−rΔt} Σ_k' Re[φ(ω_k) exp(iω_k(x−a))] V_k."""
        phi_vals = _cf(dt)  # (N,) complex
        Z = phi_vals * V_k  # (N,) complex: φ_k * V_k
        Z[0] *= 0.5  # half-weight k=0
        z = x - a  # (M,) relative positions
        Re_Z = np.real(Z)  # (N,)
        Im_Z = np.imag(Z)  # (N,)
        cos_z = np.cos(omega[:, None] * z[None, :])  # (N, M)
        sin_z = np.sin(omega[:, None] * z[None, :])  # (N, M)
        raw = (Re_Z[:, None] * cos_z - Im_Z[:, None] * sin_z).sum(axis=0)
        return np.exp(-r * dt) * raw

    # ── Initialise at t_M (final exercise date = maturity) ──────────────────
    V = _intrinsic(float(exercise_times[-1]))

    # ── Backward loop ────────────────────────────────────────────────────────
    for j in range(len(exercise_times) - 2, -1, -1):
        t_next = float(exercise_times[j + 1])
        t_curr = float(exercise_times[j])
        dt = t_next - t_curr
        if dt < 1e-12:
            continue
        V_k = _cos_coeffs(V)
        C = _continuation(dt, V_k)
        V = np.maximum(_intrinsic(t_curr), C)

    # ── Final discount from first exercise date to t = 0 ────────────────────
    t_first = float(exercise_times[0])
    V_k = _cos_coeffs(V)
    if t_first > 1e-12:
        phi_vals0 = _cf(t_first)
        A0 = np.real(phi_vals0 * np.exp(-1j * omega * a))  # Re[φ * exp(−iωa)]
        A0[0] *= 0.5
        disc0 = np.exp(-r * t_first)
        price = disc0 * float(np.dot(A0, V_k))
    else:
        # First exercise is at or before t=0: evaluate at x=0 via COS sum
        phi_vals0 = _cf(1e-10)  # near-zero maturity CF ≈ 1
        A0 = np.real(phi_vals0 * np.exp(-1j * omega * a))
        A0[0] *= 0.5
        price = float(np.dot(A0, V_k))

    return max(float(price), 0.0)


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
    n_spatial: int = 2048,
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
