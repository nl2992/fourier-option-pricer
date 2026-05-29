"""COS digital option pricing via Fang-Oosterlee payoff integrals.

Supports both payoff types for any registered model whose CF is available:

  cash_or_nothing : pays ``cash_amount`` when the exercise condition is met.
  asset_or_nothing: pays S_T when the exercise condition is met.

For a call (cp=+1) the condition is S_T > K; for a put (cp=-1) it is S_T < K.

COS formulas
------------
Let x = log(S_T / F_T), a < x < b the truncation interval, ω_k = kπ/(b-a),
A_k = Re[φ(ω_k) exp(-iω_k a)] with A_0 halved (the density coefficients).

k_star = log(K / F_T)   (log-moneyness of the option)

Cash-or-nothing call payoff integral (cp=+1):
    χ_k = (2/(b-a)) ∫_{k*}^{b} cos(ω_k(x-a)) dx
        = 2/(kπ)  * [sin(ω_k(b-a)) - sin(ω_k(k*-a))]    k > 0
        = 2(b - k*) / (b - a)                              k = 0

Asset-or-nothing call payoff integral (cp=+1):
    ψ_k = (2/(b-a)) F_T * ∫_{k*}^{b} e^x cos(ω_k(x-a)) dx
        = F_T * Re[e^{-iω_k a}/(1+iω_k) * (e^{b(1+iω_k)} - e^{k*(1+iω_k)})]
          * 2/(b-a)

Put variants integrate over (a, k*) instead.

Price = disc * Σ_k' A_k * χ_k  (cash-or-nothing)
Price = disc * Σ_k' A_k * ψ_k  (asset-or-nothing)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY
from ..products.digital import DigitalOption
from .cos import cos_auto_grid

# ── payoff coefficient helpers ─────────────────────────────────────────────


def _chi(omega: np.ndarray, c1: float, c2: float, a: float, b: float) -> np.ndarray:
    """COS indicator coefficients (2/(b-a)) ∫_{c1}^{c2} cos(ω(x-a)) dx.

    Exact result:
        k=0 : 2(c2-c1)/(b-a)
        k≥1 : (2/(b-a)) · [sin(ω(c2-a)) - sin(ω(c1-a))] / ω
    """
    chi = np.empty(len(omega))
    chi[0] = 2.0 * (c2 - c1) / (b - a)
    w = omega[1:]  # ω_k = kπ/(b-a) for k >= 1
    chi[1:] = (2.0 / (b - a)) * (np.sin(w * (c2 - a)) - np.sin(w * (c1 - a))) / w
    return chi


def _psi(omega: np.ndarray, c1: float, c2: float, a: float, b: float, F_T: float) -> np.ndarray:
    """COS asset-weighting coefficients (2/(b-a)) F_T ∫_{c1}^{c2} e^x cos(ω(x-a)) dx.

    Uses the exact integral:
        ∫_{c1}^{c2} e^x cos(ω(x-a)) dx = Re[ e^{-iωa}/(1+iω) * (e^{c2(1+iω)} - e^{c1(1+iω)}) ]
    """
    psi = np.empty(len(omega))
    # k = 0 term: ∫_{c1}^{c2} e^x dx = e^{c2} - e^{c1}
    psi[0] = 2.0 * F_T * (np.exp(c2) - np.exp(c1)) / (b - a)
    k = omega[1:]  # ω_k for k >= 1
    phase = np.exp(-1j * k * a)  # e^{-iωa}
    denom = 1.0 + 1j * k  # 1 + iω
    upper = np.exp(c2 * (1.0 + 1j * k))  # e^{c2(1+iω)}
    lower = np.exp(c1 * (1.0 + 1j * k))  # e^{c1(1+iω)}
    integral = np.real(phase / denom * (upper - lower))
    psi[1:] = 2.0 * F_T * integral / (b - a)
    return psi


# ── main pricer ────────────────────────────────────────────────────────────


def cos_digital_price(
    model: str,
    fwd: ForwardSpec,
    params,
    product: DigitalOption,
    *,
    grid=None,
    N: int = 256,
    L: float = 12.0,
) -> float:
    """Price a digital option via the COS method.

    Parameters
    ----------
    model : str
        Any model registered in ``MODEL_REGISTRY``.
    fwd : ForwardSpec
        Market inputs.
    params :
        Model parameter dataclass.
    product : DigitalOption
        Digital option spec (strike, maturity, cp, payoff_type, cash_amount).
    grid :
        Optional pre-built :class:`~foureng.utils.grids.COSGrid`.
    N : int
        COS terms (used when ``grid`` is None).
    L : float
        Truncation multiplier (used when ``grid`` is None).

    Returns
    -------
    float
        Digital option price.
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(f"cos_digital_price: unknown model {model!r}")

    entry = MODEL_REGISTRY[model]
    K = product.strike
    T = product.maturity
    cp = product.cp
    r, q, S0 = fwd.r, fwd.q, fwd.S0

    fwd_T = ForwardSpec(S0=S0, r=r, q=q, T=T)

    if grid is None:
        cums = entry.cumulants(fwd_T, params)
        grid = cos_auto_grid(cums, N=N, L=L)

    a, b, n_cos = grid.a, grid.b, grid.N
    omega = np.arange(n_cos, dtype=float) * np.pi / (b - a)

    # Characteristic function at grid frequencies
    phi_vals = entry.cf(omega, fwd_T, params)
    A = np.real(phi_vals * np.exp(-1j * omega * a))
    A[0] *= 0.5

    # Forward price and discount
    F_T = S0 * np.exp((r - q) * T)
    disc = np.exp(-r * T)

    # Integration bounds (log-moneyness breakpoint)
    k_star = np.log(K / F_T)
    k_star = float(np.clip(k_star, a, b))  # clamp to [a, b]

    if cp == 1:  # call: exercise when x > k_star
        c1, c2 = k_star, b
    else:  # put: exercise when x < k_star
        c1, c2 = a, k_star

    # Payoff integrals
    if product.payoff_type == "cash_or_nothing":
        V_k = _chi(omega, c1, c2, a, b) * product.cash_amount
    else:  # asset_or_nothing
        V_k = _psi(omega, c1, c2, a, b, F_T)

    # COS sum  (half-weight already in A[0])
    price = disc * float(np.dot(A, V_k))

    # Digital prices are bounded: cash ∈ [0, cash_amount*disc], asset ∈ [0, S0]
    if product.payoff_type == "cash_or_nothing":
        price = float(np.clip(price, 0.0, product.cash_amount))
    else:
        price = float(np.clip(price, 0.0, S0 * np.exp(-q * T) + 1e-6))

    return price


def cos_digital_price_strip(
    model: str,
    fwd: ForwardSpec,
    params,
    strikes: np.ndarray,
    maturity: float,
    cp: int = 1,
    payoff_type: Literal["cash_or_nothing", "asset_or_nothing"] = "cash_or_nothing",
    cash_amount: float = 1.0,
    *,
    grid=None,
    N: int = 256,
    L: float = 12.0,
) -> np.ndarray:
    """Price a strip of digital options at different strikes.

    Builds the grid and evaluates the CF once, then loops over strikes.
    """
    strikes = np.asarray(strikes, dtype=float)
    if model not in MODEL_REGISTRY:
        raise ValueError(f"cos_digital_price_strip: unknown model {model!r}")

    fwd_T = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=maturity)
    if grid is None:
        cums = MODEL_REGISTRY[model].cumulants(fwd_T, params)
        grid = cos_auto_grid(cums, N=N, L=L)

    prices = np.empty(len(strikes))
    for i, K in enumerate(strikes):
        product = DigitalOption(
            strike=float(K),
            maturity=maturity,
            cp=cp,
            payoff_type=payoff_type,
            cash_amount=cash_amount,
        )
        prices[i] = cos_digital_price(model, fwd, params, product, grid=grid)
    return prices
